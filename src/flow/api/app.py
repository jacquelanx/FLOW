"""FastAPI application factory and routes.

Endpoints (all under /api):
  GET  /api/health                         -> liveness + docker/doctor status
  GET  /api/providers                      -> supported providers
  POST /api/projects                       -> create a project
  GET  /api/projects                       -> list projects
  GET  /api/projects/{pid}                 -> project + file profile
  POST /api/projects/{pid}/files           -> upload a file (multipart)
  POST /api/projects/{pid}/config          -> save config.yaml
  GET  /api/projects/{pid}/config          -> read config.yaml (or default)
  GET  /api/projects/{pid}/profile         -> validate/profile the project
  POST /api/runs                           -> start a run (background)
  GET  /api/runs                           -> list runs (history; persisted)
  GET  /api/runs/{rid}                      -> run status + answer
  GET  /api/runs/{rid}/steps               -> live trajectory steps (tail)
  GET  /api/runs/{rid}/notebook            -> notebook.ipynb json
  GET  /api/runs/{rid}/artifacts           -> artifact listing
  GET  /api/runs/{rid}/artifacts/{path}    -> preview/download one artifact
  GET  /api/runs/{rid}/download            -> zip of the whole run

The built web UI is served at ``/`` (see flow.api.webui), so the backend and the app are
one process on one port.

Runs execute in a background thread that calls ``flow.runner.run_analysis`` (Docker).
The host never executes agent code directly.
"""

from __future__ import annotations

import io
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

import yaml

from flow import db
from flow.api.webui import mount_web_ui
from flow.analysis_profile import (
    AnalysisProfile,
    compile_guidance,
    profile_payload,
    read_handwritten,
    save_profile,
    write_handwritten,
)
from flow.config import (
    ConfigError,
    DatasetConfig,
    FlowConfig,
    RuntimeConfig,
    SafetyConfig,
    default_config_yaml,
    load_config,
)
from flow.env.docker_runtime import DEFAULT_IMAGE, doctor
from flow.providers.registry import MODEL_CATALOG, SUPPORTED_PROVIDERS, UI_PROVIDERS, default_model
from flow.runner import run_batch
from flow.trajectory import Trajectory
from flow.validation import (
    MAX_UPLOAD_BYTES,
    UnsafePathError,
    profile_project,
    safe_filename,
    sanitize_filename,
)

# In-memory live status for active runs (DB holds the durable record).
_RUN_STATUS: dict[str, dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()


def _set_status(run_id: str, **fields: Any) -> None:
    with _STATUS_LOCK:
        _RUN_STATUS.setdefault(run_id, {}).update(fields)


def _get_status(run_id: str) -> dict[str, Any]:
    with _STATUS_LOCK:
        return dict(_RUN_STATUS.get(run_id, {}))


def _set_traj_status(batch_id: str, idx: int, **fields: Any) -> None:
    """Update the live status of one trajectory within a batch (thread-safe)."""
    with _STATUS_LOCK:
        batch = _RUN_STATUS.setdefault(batch_id, {})
        trajs = batch.setdefault("trajectories", {})
        trajs.setdefault(str(idx), {}).update(fields)


# ----------------------------------------------------------------- request models
class CreateProjectBody(BaseModel):
    name: str


class SaveConfigBody(BaseModel):
    content: str


class ConfigFormBody(BaseModel):
    """Structured config fields from the friendly form (no raw YAML on the client)."""

    question: str
    description: str = ""
    provider: str = "mock"
    model: str = "mock"
    meta_provider: str = ""
    meta_model: str = ""
    first_run: str = "auto"
    max_steps: int = 30
    per_cell_timeout: float = 120.0
    per_trajectory_timeout: float = 1800.0
    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 256
    allow_network: bool = False


class StartRunBody(BaseModel):
    project_id: str
    question: Optional[str] = None
    provider: str = "mock"
    model: str = "mock"
    # Consensus synthesis model (blank = same as provider/model above).
    meta_provider: str = ""
    meta_model: str = ""
    # First-run scaffold: "auto" (run the project's first_run.py) | "none" ("" = leave config).
    first_run: str = ""
    max_steps: Optional[int] = None
    # Number of independent trajectories to run before the consensus meta-analysis.
    n_trajectories: int = 8


def create_app() -> FastAPI:
    # Load a host-side .env (if present) so provider API keys are available to the backend
    # without exporting them. Existing environment variables are never overridden.
    from flow.envfile import load_dotenv

    load_dotenv()
    app = FastAPI(title="FLOW API", version="1.0.0")
    db.init_db()

    # ------------------------------------------------------------- health
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        rep = doctor(DEFAULT_IMAGE)
        return {
            "status": "ok",
            "docker": {
                "installed": rep.docker_installed,
                "running": rep.docker_running,
                "image_present": rep.image_present,
                "image": rep.image,
                "ready": rep.ok,
                "notes": rep.notes,
            },
        }

    @app.get("/api/providers")
    def providers() -> dict[str, Any]:
        return {
            "providers": SUPPORTED_PROVIDERS,
            "ui_providers": UI_PROVIDERS,
            "catalog": MODEL_CATALOG,
            "default_models": {p: default_model(p) for p in UI_PROVIDERS},
        }

    # ------------------------------------------------------------- projects
    @app.post("/api/projects")
    def create_project(body: CreateProjectBody) -> dict[str, Any]:
        pid = uuid.uuid4().hex[:12]
        proj = db.create_project(pid, body.name or "Untitled project")
        return proj

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": db.list_projects()}

    def _project_or_404(pid: str) -> dict[str, Any]:
        proj = db.get_project(pid)
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found")
        return proj

    @app.get("/api/projects/{pid}")
    def get_project(pid: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        profile = profile_project(proj["path"])
        return {"project": proj, "profile": profile.to_dict()}

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        batches = db.list_batches(pid)
        # Refuse while any run is still active (same guard as deleting a single run).
        for b in batches:
            live = _get_status(b["id"]).get("status", b["status"])
            if live in {"pending", "running", "synthesizing"}:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete a project while one of its runs is in progress.",
                )
        # Remove run artifacts on disk, then the project's own directory.
        for b in batches:
            d = Path(b["artifact_dir"])
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        proj_dir = Path(proj["path"])
        if proj_dir.exists():
            shutil.rmtree(proj_dir, ignore_errors=True)
        # Drop DB rows and any in-memory run status.
        db.delete_project(pid)
        with _STATUS_LOCK:
            for b in batches:
                _RUN_STATUS.pop(b["id"], None)
        return {"ok": True, "deleted": pid}

    @app.post("/api/projects/{pid}/files")
    async def upload_file(pid: str, file: UploadFile = File(...)) -> dict[str, Any]:
        proj = _project_or_404(pid)
        try:
            name = sanitize_filename(file.filename or "")
        except UnsafePathError as e:
            raise HTTPException(status_code=400, detail=str(e))
        project_dir = Path(proj["path"])
        dest = project_dir / name
        size = 0
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File exceeds size limit.")
                out.write(chunk)

        # A .zip is expanded: FCS files become an fcs_dir (named after the zip); other
        # recognized files land at the project root. The zip itself is then removed.
        if name.lower().endswith(".zip"):
            try:
                extracted = _extract_zip(dest, project_dir)
            except zipfile.BadZipFile:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip.")
            dest.unlink(missing_ok=True)
            return {"zip": True, "extracted": extracted, "size": size}

        from flow.validation import validate_file

        report = validate_file(dest)
        return {"file": report.__dict__, "size": size}

    def _common_dir_parts(members: list[zipfile.ZipInfo]) -> list[str]:
        """Directory components shared by every member (the zip's wrapper folder(s))."""
        dirs = [[p for p in Path(m.filename).parent.parts if p not in (".", "/")]
                for m in members]
        if not dirs:
            return []
        common = dirs[0]
        for d in dirs[1:]:
            i = 0
            while i < min(len(common), len(d)) and common[i] == d[i]:
                i += 1
            common = common[:i]
            if not common:
                break
        return common

    def _extract_zip(zip_path: Path, project_dir: Path) -> list[dict[str, Any]]:
        """Safely extract a zip: FCS -> <stem>/ subdir, other files -> project root.

        Writes basenames only (zip-slip safe), skips macOS junk, and caps total expanded
        size to guard against zip bombs.

        Instrument software often exports one folder per acquisition session, reusing the
        same basenames in every folder. Writing those by basename alone silently overwrote
        all but the last, so each FCS keeps the folder it came from as a
        ``<source folder>__`` prefix. The first-run script looks past that prefix, so the
        original filename still drives how the file is interpreted.
        """
        max_total = 2 * 1024 * 1024 * 1024
        total = 0
        try:
            stem = sanitize_filename(zip_path.stem)
        except UnsafePathError:
            stem = "extracted"

        extracted: list[dict[str, Any]] = []
        used: set[str] = set()
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                i for i in zf.infolist()
                if not i.is_dir()
                and Path(i.filename).name
                and "__MACOSX" not in i.filename
                and not Path(i.filename).name.startswith("._")
            ]
            # Drop the directory levels every member shares (the wrapper folder), so a
            # prefix names the meaningful folder ("UPN 25 D14"), not the whole tree.
            common_parts = _common_dir_parts(members)
            for info in members:
                src_path = Path(info.filename)
                try:
                    safe = sanitize_filename(src_path.name)
                except UnsafePathError:
                    continue
                total += info.file_size
                if total > max_total:
                    raise HTTPException(status_code=413, detail="Zip expands beyond size limit.")
                if Path(safe).suffix.lower() == ".fcs":
                    folders = [p for p in src_path.parent.parts if p not in (".", "/")]
                    folders = folders[len(common_parts):]
                    if folders:
                        try:
                            prefix = "-".join(sanitize_filename(p) for p in folders)
                            safe = f"{prefix}__{safe}"
                        except UnsafePathError:
                            pass
                    # Belt-and-braces: never let two members land on one path.
                    base, suffix = safe[: -len(".fcs")], safe[-len(".fcs"):]
                    n = 2
                    while safe.lower() in used:
                        safe = f"{base}-{n}{suffix}"
                        n += 1
                    used.add(safe.lower())
                    sub = project_dir / stem
                    sub.mkdir(exist_ok=True)
                    out_path = sub / safe
                    rel = f"{stem}/{safe}"
                else:
                    out_path = project_dir / safe
                    rel = safe
                with zf.open(info) as src, out_path.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                extracted.append({"name": rel})
        return extracted

    @app.get("/api/projects/{pid}/profile")
    def project_profile(pid: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        return profile_project(proj["path"]).to_dict()

    @app.delete("/api/projects/{pid}/files/{name}")
    def delete_file(pid: str, name: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        seg = name.rstrip("/")  # fcs_dir entries arrive as "samples/"
        try:
            seg = safe_filename(seg)
        except UnsafePathError as e:
            raise HTTPException(status_code=400, detail=str(e))
        base = Path(proj["path"]).resolve()
        target = (base / seg).resolve()
        if not str(target).startswith(str(base)) or not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        return {"ok": True, "deleted": seg}

    def _config_dict(cfg_path: Path) -> dict[str, Any]:
        """Return a structured config dict for the form (parsed file, or defaults)."""
        if cfg_path.exists():
            try:
                return load_config(cfg_path).model_dump()
            except ConfigError:
                pass  # malformed/legacy file -> fall back to defaults below
        return {
            "question": "",
            "dataset": DatasetConfig().model_dump(),
            "runtime": RuntimeConfig().model_dump(),
            "safety": SafetyConfig().model_dump(),
        }

    @app.get("/api/projects/{pid}/config")
    def get_config(pid: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        cfg_path = Path(proj["path"]) / "config.yaml"
        exists = cfg_path.exists()
        content = cfg_path.read_text() if exists else default_config_yaml()
        # `config` powers the friendly form; `content` is the raw YAML (read-only preview).
        return {"content": content, "exists": exists, "config": _config_dict(cfg_path)}

    @app.post("/api/projects/{pid}/config")
    def save_config(pid: str, body: SaveConfigBody) -> dict[str, Any]:
        """Raw-YAML save (kept for power users / API compatibility)."""
        proj = _project_or_404(pid)
        cfg_path = Path(proj["path"]) / "config.yaml"
        cfg_path.write_text(body.content)
        # Validate immediately so the user sees errors (incl. analysis-logic rejection).
        try:
            load_config(cfg_path)
        except ConfigError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    @app.get("/api/projects/{pid}/prompt")
    def get_prompt(pid: str) -> dict[str, Any]:
        """Read only the hand-written part of the prompt.

        prompt.md may also contain a block generated from the Analysis page; that block is
        owned there, so it is excluded here to avoid the two editors clobbering each other.
        """
        proj = _project_or_404(pid)
        txt = Path(proj["path"]) / "prompt.txt"
        if txt.exists():  # legacy plain-text prompt
            return {"content": txt.read_text(), "exists": True, "name": "prompt.txt"}
        content = read_handwritten(proj["path"])
        return {"content": content, "exists": bool(content), "name": "prompt.md"}

    @app.post("/api/projects/{pid}/prompt")
    def save_prompt(pid: str, body: SaveConfigBody) -> dict[str, Any]:
        """Save the hand-written prompt, preserving any Analysis-page generated block."""
        proj = _project_or_404(pid)
        write_handwritten(proj["path"], body.content)
        return {"ok": True}

    # ------------------------------------------------------------- analysis profile
    @app.get("/api/projects/{pid}/analysis")
    def get_analysis(pid: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        return profile_payload(proj["path"])

    @app.post("/api/projects/{pid}/analysis")
    def save_analysis(pid: str, body: AnalysisProfile) -> dict[str, Any]:
        proj = _project_or_404(pid)
        save_profile(proj["path"], body)
        return {"ok": True, "guidance_preview": compile_guidance(body)}

    # ------------------------------------------------------------- first-run script
    @app.get("/api/projects/{pid}/first-run-script")
    def get_first_run_script(
        pid: str, template: str = "example", seed: bool = False
    ) -> dict[str, Any]:
        from flow.firstrun import SCRIPT_FILENAME, anchored_script, example_script

        proj = _project_or_404(pid)
        p = Path(proj["path"]) / SCRIPT_FILENAME
        template_seed = anchored_script() if template == "anchored" else example_script()
        # seed=1 forces the requested template's text (for the "load this template" button),
        # even when a script is already saved.
        if seed:
            return {"content": template_seed, "seeded": True,
                    "filename": SCRIPT_FILENAME, "template": template}
        if p.exists():
            return {"content": p.read_text(), "seeded": False, "filename": SCRIPT_FILENAME}
        # No saved script yet — seed the editor with the requested template.
        return {"content": template_seed, "seeded": True, "filename": SCRIPT_FILENAME,
                "template": template}

    @app.post("/api/projects/{pid}/first-run-script/anchored")
    def use_anchored_template(pid: str) -> dict[str, Any]:
        """Install the reference-anchoring template as this project's first_run.py.

        Writes first_run.py AND copies the vendored ``anchored/`` package into the project so
        ``import anchored`` resolves when the sandbox runs ``python /data/first_run.py``.
        """
        from flow.firstrun import SCRIPT_FILENAME, provision_anchored

        proj = _project_or_404(pid)
        provision_anchored(proj["path"])
        return {"ok": True, "saved": True, "filename": SCRIPT_FILENAME, "template": "anchored"}

    @app.post("/api/projects/{pid}/first-run-script")
    def save_first_run_script(pid: str, body: SaveConfigBody) -> dict[str, Any]:
        from flow.firstrun import SCRIPT_FILENAME

        proj = _project_or_404(pid)
        p = Path(proj["path"]) / SCRIPT_FILENAME
        if body.content.strip():
            p.write_text(body.content)
            return {"ok": True, "saved": True}
        p.unlink(missing_ok=True)  # empty content disables the first-run
        return {"ok": True, "saved": False}

    @app.post("/api/projects/{pid}/config/form")
    def save_config_form(pid: str, body: ConfigFormBody) -> dict[str, Any]:
        """Structured save from the friendly form: builds analysis-free YAML server-side."""
        proj = _project_or_404(pid)
        try:
            cfg = FlowConfig(
                question=body.question,
                dataset=DatasetConfig(root=".", description=body.description),
                runtime=RuntimeConfig(
                    provider=body.provider,
                    model=body.model,
                    meta_provider=body.meta_provider,
                    meta_model=body.meta_model,
                    first_run=body.first_run,
                    max_steps=body.max_steps,
                    per_cell_timeout=body.per_cell_timeout,
                    per_trajectory_timeout=body.per_trajectory_timeout,
                ),
                safety=SafetyConfig(
                    memory=body.memory,
                    cpus=body.cpus,
                    pids_limit=body.pids_limit,
                    allow_network=body.allow_network,
                ),
            )
        except Exception as e:  # e.g. empty question
            return {"ok": False, "error": str(e)}
        yaml_text = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
        cfg_path = Path(proj["path"]) / "config.yaml"
        cfg_path.write_text(yaml_text)
        return {"ok": True, "content": yaml_text}

    # ------------------------------------------------------------- runs (consensus batches)
    # A "run" in the UI is a consensus batch: N independent trajectories + a meta-analysis.
    @app.post("/api/runs")
    def start_run(body: StartRunBody) -> dict[str, Any]:
        proj = _project_or_404(body.project_id)
        data_dir = Path(proj["path"])
        cfg_path = data_dir / "config.yaml"

        # Build config from file (if present) then apply request overrides.
        if cfg_path.exists():
            try:
                cfg = load_config(cfg_path)
            except ConfigError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            if not body.question:
                raise HTTPException(
                    status_code=400, detail="No config.yaml and no question provided."
                )
            cfg = FlowConfig(question=body.question)

        if body.question:
            cfg.question = body.question
        cfg.runtime.provider = body.provider
        cfg.runtime.model = body.model
        cfg.runtime.meta_provider = body.meta_provider
        cfg.runtime.meta_model = body.meta_model
        if body.first_run:
            cfg.runtime.first_run = body.first_run
        if body.max_steps:
            cfg.runtime.max_steps = body.max_steps

        n = max(1, min(int(body.n_trajectories or 1), 16))
        batch_id = uuid.uuid4().hex[:12]
        batch_dir = db.artifacts_root() / batch_id
        db.create_batch(
            batch_id, body.project_id, cfg.question, cfg.runtime.provider,
            cfg.runtime.model, n, str(batch_dir),
        )
        for idx in range(n):
            db.create_trajectory(
                f"{batch_id}-{idx}", batch_id, idx,
                str(batch_dir / "trajectories" / str(idx)),
            )
            _set_traj_status(batch_id, idx, status="pending", steps=0)
        _set_status(batch_id, status="pending", phase="pending", started_at=time.time())

        def _worker() -> None:
            _set_status(batch_id, status="running", phase="running")
            db.update_batch(batch_id, status="running")

            def on_phase(phase: str) -> None:
                if phase == "synthesizing":
                    _set_status(batch_id, status="synthesizing", phase="synthesizing")
                    db.update_batch(batch_id, status="synthesizing")
                else:
                    _set_status(batch_id, phase=phase)

            def on_traj_status(idx: int, status: str, info: dict) -> None:
                _set_traj_status(batch_id, idx, status=status, steps=info.get("steps", 0))
                db.update_trajectory(
                    f"{batch_id}-{idx}",
                    status=status,
                    submitted=1 if info.get("submitted") else 0,
                    failure_reason=info.get("failure_reason"),
                    steps=info.get("steps", 0),
                    finished_at=time.time() if status != "running" else None,
                )

            def on_step(idx: int, rec) -> None:
                _set_traj_status(batch_id, idx, status="running", steps=rec.step)

            try:
                bres = run_batch(
                    data_dir=data_dir, config=cfg, batch_dir=batch_dir,
                    n_trajectories=n, batch_id=batch_id,
                    on_phase=on_phase, on_trajectory_status=on_traj_status, on_step=on_step,
                )
                cons = bres.consensus or {}
                ok = bool(cons.get("consensus"))
                status = "completed" if ok else "failed"
                _set_status(
                    batch_id, status=status, phase="done",
                    consensus_ok=ok, n_submitted=cons.get("n_submitted", 0),
                    failure_reason=cons.get("failure_reason"),
                )
                db.update_batch(
                    batch_id, status=status, finished_at=time.time(),
                    consensus_ok=1 if ok else 0, n_submitted=cons.get("n_submitted", 0),
                    failure_reason=cons.get("failure_reason"),
                )
            except Exception as e:  # Docker missing, etc. — whole batch could not run
                _set_status(batch_id, status="error", phase="done", failure_reason=str(e))
                db.update_batch(
                    batch_id, status="error", finished_at=time.time(), failure_reason=str(e)
                )

        threading.Thread(target=_worker, daemon=True).start()
        return {"run_id": batch_id, "status": "pending", "n_trajectories": n}

    @app.get("/api/runs")
    def list_runs(project_id: Optional[str] = None) -> dict[str, Any]:
        return {"runs": db.list_batches(project_id)}

    def _batch_or_404(rid: str) -> dict[str, Any]:
        batch = db.get_batch(rid)
        if not batch:
            raise HTTPException(status_code=404, detail="Run not found")
        return batch

    def _trajectory_summaries(rid: str, batch: dict[str, Any]) -> list[dict[str, Any]]:
        base = Path(batch["artifact_dir"]) / "trajectories"
        live = _get_status(rid).get("trajectories", {})
        out = []
        for t in db.list_trajectories(rid):
            idx = t["idx"]
            live_t = live.get(str(idx), {})
            traj = Trajectory(base / str(idx))
            # From run.json rather than the DB: no migration needed, and both fields are
            # null unless the project's first-run actually left obligations to check.
            run = traj.read_run_json()
            out.append({
                "idx": idx,
                "status": live_t.get("status", t["status"]),
                "steps": live_t.get("steps", t["steps"]),
                "submitted": bool(t["submitted"]),
                "failure_reason": t["failure_reason"],
                "answer": traj.read_answer(),
                "obligations_total": run.get("obligations_total"),
                "obligations_undischarged": run.get("obligations_undischarged"),
            })
        return out

    def _consensus_text(batch: dict[str, Any]) -> dict[str, Any]:
        """Read the consensus from batch.json (preferred) or consensus.txt."""
        bjson = Path(batch["artifact_dir"]) / "batch.json"
        if bjson.exists():
            import json

            data = json.loads(bjson.read_text())
            c = data.get("consensus") or {}
            return {
                "consensus": c.get("consensus"),
                "synthesized": c.get("synthesized"),
                "n_with_unread_evidence": c.get("n_with_unread_evidence"),
                "n_submitted": c.get("n_submitted"),
                "n_total": c.get("n_total"),
                "failure_reason": c.get("failure_reason"),
            }
        return {"consensus": None, "synthesized": None, "n_submitted": None,
                "n_total": batch["n_trajectories"], "failure_reason": None,
                "n_with_unread_evidence": None}

    @app.get("/api/runs/{rid}")
    def get_run(rid: str) -> dict[str, Any]:
        batch = _batch_or_404(rid)
        live = _get_status(rid)
        return {
            "run": batch,
            "status": live.get("status", batch["status"]),
            "phase": live.get("phase"),
            "trajectories": _trajectory_summaries(rid, batch),
            "consensus": _consensus_text(batch),
        }

    @app.get("/api/runs/{rid}/trajectories")
    def list_trajectories(rid: str) -> dict[str, Any]:
        batch = _batch_or_404(rid)
        return {"trajectories": _trajectory_summaries(rid, batch)}

    def _traj_dir_or_404(batch: dict[str, Any], idx: int) -> Path:
        if idx < 0 or idx >= batch["n_trajectories"]:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        return Path(batch["artifact_dir"]).resolve() / "trajectories" / str(idx)

    @app.get("/api/runs/{rid}/trajectories/{idx}/steps")
    def get_steps(rid: str, idx: int) -> dict[str, Any]:
        batch = _batch_or_404(rid)
        traj = Trajectory(_traj_dir_or_404(batch, idx))
        live = _get_status(rid).get("trajectories", {}).get(str(idx), {})
        return {"steps": traj.read_actions(), "status": live.get("status", "pending")}

    @app.get("/api/runs/{rid}/trajectories/{idx}/notebook")
    def get_notebook(rid: str, idx: int) -> dict[str, Any]:
        batch = _batch_or_404(rid)
        nb = _traj_dir_or_404(batch, idx) / "notebook.ipynb"
        if not nb.exists():
            return {"cells": [], "nbformat": 4}
        import json

        return json.loads(nb.read_text())

    @app.get("/api/runs/{rid}/trajectories/{idx}/artifacts")
    def list_artifacts(rid: str, idx: int) -> dict[str, Any]:
        batch = _batch_or_404(rid)
        return {"artifacts": Trajectory(_traj_dir_or_404(batch, idx)).list_artifacts()}

    @app.get("/api/runs/{rid}/trajectories/{idx}/artifacts/{path:path}")
    def get_artifact(rid: str, idx: int, path: str) -> Response:
        batch = _batch_or_404(rid)
        base = _traj_dir_or_404(batch, idx).resolve()
        target = (base / path).resolve()
        # Traversal guard: the resolved path must stay inside the trajectory dir.
        if not str(target).startswith(str(base)) or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        suffix = target.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            media = "image/svg+xml" if suffix == ".svg" else f"image/{suffix.lstrip('.')}"
            return FileResponse(target, media_type=media)
        if suffix in {".json", ".ipynb"}:
            return Response(target.read_text(), media_type="application/json")
        if suffix in {".csv", ".tsv", ".txt", ".jsonl", ".yaml", ".yml", ".log"}:
            return Response(target.read_text(), media_type="text/plain")
        return FileResponse(target)

    @app.get("/api/runs/{rid}/download")
    def download_run(rid: str) -> StreamingResponse:
        batch = _batch_or_404(rid)
        base = Path(batch["artifact_dir"])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in base.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(base).as_posix())
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="run_{rid}.zip"'},
        )

    @app.delete("/api/runs/{rid}")
    def delete_run(rid: str) -> dict[str, Any]:
        batch = _batch_or_404(rid)
        live = _get_status(rid).get("status", batch["status"])
        if live in {"pending", "running", "synthesizing"}:
            raise HTTPException(
                status_code=409, detail="Cannot delete a run that is still in progress."
            )
        # Remove artifacts on disk, the DB rows, and any in-memory status.
        d = Path(batch["artifact_dir"])
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        db.delete_batch(rid)
        with _STATUS_LOCK:
            _RUN_STATUS.pop(rid, None)
        return {"ok": True, "deleted": rid}

    # --------------------------------------------------------------- web UI
    # Mounted last: it is a catch-all at "/", and Starlette matches in registration
    # order, so every /api route above still wins.
    mount_web_ui(app)

    return app


# Module-level app for `uvicorn flow.api.app:app`.
app = create_app()
