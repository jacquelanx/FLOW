"""Trajectory artifacts — the auditable record of a run.

Every run produces a self-contained directory:
  * notebook.ipynb   — the notebook the agent built (code + outputs).
  * actions.jsonl     — full action/observation log (one JSON object per step).
  * answer.txt        — the submitted conclusion (or empty if the run failed).
  * run.json          — metadata: question, provider/model, image digest, limits,
                        timings, step count, success flag, failure reason.
  * evidence_coverage.json — which of the first-run's obligations the answer discharged, and
                        which diagnostics tables were opened. Written only when the project
                        left a manifest to check against.
  * plots/            — figures the agent saved (PNG), summarized as [image] to the agent.

This is the audit trail required by the safety model.

``evidence_coverage.json`` is part of that audit trail rather than a metric. An answer that
skipped half its evidence and an answer that had none to skip read identically once the
trajectory is over; this file is what tells them apart after the fact.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from flow.agent.react import AgentResult
from flow.env.notebook_env import NotebookEnvironment


@dataclass
class RunMetadata:
    """Audit metadata persisted to run.json."""

    run_id: str
    question: str
    provider: str
    model: str
    image: str
    image_digest: Optional[str]
    max_steps: int
    limits: dict[str, Any]
    started_at: float
    finished_at: Optional[float] = None
    duration_s: Optional[float] = None
    steps_taken: int = 0
    submitted: bool = False
    failure_reason: Optional[str] = None
    network: str = "none"
    # Obligations the answer left undischarged, and whether the check was active at all.
    # ``None`` for both means the question was never asked (no manifest), which is not the
    # same as a clean sweep — see ``evidence_coverage.json``.
    obligations_total: Optional[int] = None
    obligations_undischarged: Optional[int] = None


class Trajectory:
    """Writes and reads a run's artifact directory."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "plots").mkdir(exist_ok=True)
        self._actions_path = self.dir / "actions.jsonl"

    # --- live, incremental writes (so the UI can tail during a run) ---
    def append_action(self, record: dict[str, Any]) -> None:
        with self._actions_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def write_notebook(self, env: NotebookEnvironment) -> None:
        (self.dir / "notebook.ipynb").write_text(
            json.dumps(env.to_notebook(), indent=1)
        )

    # --- finalization ---
    def finalize(
        self, env: NotebookEnvironment, result: AgentResult, meta: RunMetadata
    ) -> None:
        self.write_notebook(env)
        (self.dir / "answer.txt").write_text(result.answer or "")
        meta.finished_at = time.time()
        meta.duration_s = (meta.finished_at - meta.started_at) if meta.started_at else None
        meta.steps_taken = len(result.steps)
        meta.submitted = result.submitted
        meta.failure_reason = result.failure_reason

        coverage = self.write_evidence_coverage(env)
        if coverage.get("enforced"):
            meta.obligations_total = coverage.get("obligations_total")
            meta.obligations_undischarged = coverage.get("obligations_undischarged")
        (self.dir / "run.json").write_text(json.dumps(asdict(meta), indent=2, default=str))

    def write_evidence_coverage(self, env: NotebookEnvironment) -> dict[str, Any]:
        """Persist the run's evidence coverage; returns it (empty dict if unavailable).

        Written only when a manifest was actually in force. Emitting a file that says
        ``0 undischarged`` for a project with nothing to discharge would make an ungated run
        indistinguishable from a thorough one.
        """
        getter = getattr(env, "evidence_coverage", None)
        coverage = getter() if callable(getter) else {}
        if coverage.get("enforced"):
            (self.dir / "evidence_coverage.json").write_text(
                json.dumps(coverage, indent=2, default=str)
            )
        return coverage

    # --- reads for the API ---
    def read_run_json(self) -> dict[str, Any]:
        p = self.dir / "run.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def read_answer(self) -> str:
        p = self.dir / "answer.txt"
        return p.read_text() if p.exists() else ""

    def read_actions(self) -> list[dict[str, Any]]:
        if not self._actions_path.exists():
            return []
        out = []
        for line in self._actions_path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List every artifact file with type + size for the ArtifactBrowser."""
        items: list[dict[str, Any]] = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.dir).as_posix()
                items.append(
                    {
                        "path": rel,
                        "name": p.name,
                        "size": p.stat().st_size,
                        "kind": _classify(p),
                    }
                )
        return items


def _classify(p: Path) -> str:
    suffix = p.suffix.lower()
    if suffix == ".ipynb":
        return "notebook"
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "image"
    if suffix == ".json":
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix == ".jsonl":
        return "log"
    return "text"
