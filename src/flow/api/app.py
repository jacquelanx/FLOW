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

Runs execute in a background thread that calls ``flow.runner.run_analysis`` (Docker).
The host never executes agent code directly.
"""

from __future__ import annotations

import io
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
from flow.runner import run_analysis
from flow.trajectory import Trajectory
from flow.validation import MAX_UPLOAD_BYTES, UnsafePathError, profile_project, safe_filename

# In-memory live status for active runs (DB holds the durable record).
_RUN_STATUS: dict[str, dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()


def _set_status(run_id: str, **fields: Any) -> None:
    with _STATUS_LOCK:
        _RUN_STATUS.setdefault(run_id, {}).update(fields)


def _get_status(run_id: str) -> dict[str, Any]:
    with _STATUS_LOCK:
        return dict(_RUN_STATUS.get(run_id, {}))


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
    max_steps: int = 30
    per_cell_timeout: float = 120.0
    per_trajectory_timeout: float = 1800.0
    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 256
    allow_network: bool = False
    allow_raw_data_to_model: bool = False


class StartRunBody(BaseModel):
    project_id: str
    question: Optional[str] = None
    provider: str = "mock"
    model: str = "mock"
    max_steps: Optional[int] = None


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

    @app.post("/api/projects/{pid}/files")
    async def upload_file(pid: str, file: UploadFile = File(...)) -> dict[str, Any]:
        proj = _project_or_404(pid)
        try:
            name = safe_filename(file.filename or "")
        except UnsafePathError as e:
            raise HTTPException(status_code=400, detail=str(e))
        dest = Path(proj["path"]) / name
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
        from flow.validation import validate_file

        report = validate_file(dest)
        return {"file": report.__dict__, "size": size}

    @app.get("/api/projects/{pid}/profile")
    def project_profile(pid: str) -> dict[str, Any]:
        proj = _project_or_404(pid)
        return profile_project(proj["path"]).to_dict()

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
                    max_steps=body.max_steps,
                    per_cell_timeout=body.per_cell_timeout,
                    per_trajectory_timeout=body.per_trajectory_timeout,
                ),
                safety=SafetyConfig(
                    memory=body.memory,
                    cpus=body.cpus,
                    pids_limit=body.pids_limit,
                    allow_network=body.allow_network,
                    allow_raw_data_to_model=body.allow_raw_data_to_model,
                ),
            )
        except Exception as e:  # e.g. empty question
            return {"ok": False, "error": str(e)}
        yaml_text = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
        cfg_path = Path(proj["path"]) / "config.yaml"
        cfg_path.write_text(yaml_text)
        return {"ok": True, "content": yaml_text}

    # ------------------------------------------------------------- runs
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
            from flow.config import FlowConfig

            if not body.question:
                raise HTTPException(
                    status_code=400,
                    detail="No config.yaml and no question provided.",
                )
            cfg = FlowConfig(question=body.question)

        if body.question:
            cfg.question = body.question
        cfg.runtime.provider = body.provider
        cfg.runtime.model = body.model
        if body.max_steps:
            cfg.runtime.max_steps = body.max_steps

        run_id = uuid.uuid4().hex[:12]
        run_dir = db.artifacts_root() / run_id
        db.create_run(
            run_id, body.project_id, cfg.question, cfg.runtime.provider,
            cfg.runtime.model, str(run_dir),
        )
        _set_status(run_id, status="pending", steps=0, started_at=time.time())

        def _worker() -> None:
            _set_status(run_id, status="running", executing_in_docker=True)
            db.update_run(run_id, status="running")
            try:
                result, _meta = run_analysis(
                    data_dir=data_dir, config=cfg, run_dir=run_dir, run_id=run_id,
                    on_step=lambda rec: _set_status(run_id, steps=rec.step),
                )
                status = "completed" if result.submitted else "failed"
                _set_status(
                    run_id, status=status, executing_in_docker=False,
                    submitted=result.submitted, failure_reason=result.failure_reason,
                )
                db.update_run(
                    run_id, status=status, finished_at=time.time(),
                    submitted=1 if result.submitted else 0,
                    failure_reason=result.failure_reason,
                )
            except Exception as e:  # Docker missing, provider error, etc.
                _set_status(run_id, status="error", executing_in_docker=False,
                            failure_reason=str(e))
                db.update_run(run_id, status="error", finished_at=time.time(),
                              failure_reason=str(e))

        threading.Thread(target=_worker, daemon=True).start()
        return {"run_id": run_id, "status": "pending"}

    @app.get("/api/runs")
    def list_runs(project_id: Optional[str] = None) -> dict[str, Any]:
        return {"runs": db.list_runs(project_id)}

    def _run_or_404(rid: str) -> dict[str, Any]:
        run = db.get_run(rid)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{rid}")
    def get_run(rid: str) -> dict[str, Any]:
        run = _run_or_404(rid)
        live = _get_status(rid)
        traj = Trajectory(run["artifact_dir"])
        run_json = traj.read_run_json()
        return {
            "run": run,
            "live": live,
            "status": live.get("status", run["status"]),
            "answer": traj.read_answer(),
            "run_meta": run_json,
        }

    @app.get("/api/runs/{rid}/steps")
    def get_steps(rid: str) -> dict[str, Any]:
        run = _run_or_404(rid)
        traj = Trajectory(run["artifact_dir"])
        return {"steps": traj.read_actions(), "status": _get_status(rid).get("status", run["status"])}

    @app.get("/api/runs/{rid}/notebook")
    def get_notebook(rid: str) -> dict[str, Any]:
        run = _run_or_404(rid)
        nb = Path(run["artifact_dir"]) / "notebook.ipynb"
        if not nb.exists():
            return {"cells": [], "nbformat": 4}
        import json

        return json.loads(nb.read_text())

    @app.get("/api/runs/{rid}/artifacts")
    def list_artifacts(rid: str) -> dict[str, Any]:
        run = _run_or_404(rid)
        return {"artifacts": Trajectory(run["artifact_dir"]).list_artifacts()}

    @app.get("/api/runs/{rid}/artifacts/{path:path}")
    def get_artifact(rid: str, path: str) -> Response:
        run = _run_or_404(rid)
        base = Path(run["artifact_dir"]).resolve()
        target = (base / path).resolve()
        # Traversal guard: the resolved path must stay inside the run dir.
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
        run = _run_or_404(rid)
        base = Path(run["artifact_dir"])
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

    return app


# Module-level app for `uvicorn flow.api.app:app`.
app = create_app()
