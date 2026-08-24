"""The project-local launcher's write-once guard.

Write-once must protect a COMPLETED canonical run without making a FAILED attempt permanent:
the launcher writes its own project_run_location.json before starting, so a guard keyed on
"directory is non-empty" refused to retry the identical command after a crash and forced the
operator to invent a new run id for a run that never produced a result.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parent.parent / "scripts" / "run_project_first_run.py"


def _launcher():
    spec = importlib.util.spec_from_file_location("run_project_first_run", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invoke(monkeypatch, module, project: Path, data: Path, run_id: str, returncode: int = 0):
    """Run main() with the analysis subprocess stubbed out."""
    calls = []

    class _Completed:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        (out / "outputs").mkdir(parents=True, exist_ok=True)
        (out / "outputs" / "tables.csv").write_text("a\n1\n")
        if returncode == 0:
            (out / "first_run_bundle.json").write_text('{"first_run_bundle_sha256": "abc"}')
        return _Completed(returncode)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.sys, "argv",
        ["run_project_first_run.py", "--project", str(project), "--data", str(data),
         "--run-id", run_id],
    )
    return calls


def test_a_failed_run_can_be_retried_under_the_same_run_id(tmp_path, monkeypatch):
    module = _launcher()
    project = tmp_path / "UPN27"
    project.mkdir()
    data = tmp_path / "snapshot"
    data.mkdir()

    _invoke(monkeypatch, module, project, data, "v1", returncode=1)
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 1

    run_dir = module.project_run_dir(project, "v1")
    assert run_dir.exists() and any(run_dir.iterdir()), "the failed attempt left files behind"

    # The identical command must now succeed rather than being refused.
    _invoke(monkeypatch, module, project, data, "v1", returncode=0)
    module.main()
    assert (run_dir / "first_run_bundle.json").is_file()


def test_the_retry_clears_the_incomplete_attempt_instead_of_merging_into_it(
    tmp_path, monkeypatch
):
    """Stale artifacts from a crashed attempt must not be hashed into the new run's bundle."""
    module = _launcher()
    project = tmp_path / "UPN27"
    project.mkdir()
    data = tmp_path / "snapshot"
    data.mkdir()
    run_dir = module.project_run_dir(project, "v1")
    run_dir.mkdir(parents=True)
    (run_dir / "outputs").mkdir()
    (run_dir / "outputs" / "stale_from_last_attempt.csv").write_text("x\n1\n")

    _invoke(monkeypatch, module, project, data, "v1", returncode=0)
    module.main()

    assert not (run_dir / "outputs" / "stale_from_last_attempt.csv").exists()
    assert (run_dir / "outputs" / "tables.csv").is_file()


def test_a_completed_canonical_run_is_still_write_once(tmp_path, monkeypatch):
    module = _launcher()
    project = tmp_path / "UPN27"
    project.mkdir()
    data = tmp_path / "snapshot"
    data.mkdir()

    _invoke(monkeypatch, module, project, data, "v1", returncode=0)
    module.main()

    calls = _invoke(monkeypatch, module, project, data, "v1", returncode=0)
    with pytest.raises(SystemExit) as refusal:
        module.main()
    assert "refusing to replace a completed canonical run" in str(refusal.value.code)
    assert calls == [], "the analysis must not re-run over a completed canonical result"


def test_a_bad_control_path_does_not_create_the_run_directory(tmp_path, monkeypatch):
    module = _launcher()
    project = tmp_path / "UPN27"
    project.mkdir()
    data = tmp_path / "snapshot"
    data.mkdir()
    monkeypatch.setattr(
        module.sys, "argv",
        ["run_project_first_run.py", "--project", str(project), "--data", str(data),
         "--run-id", "v1", "--compensation-controls", str(tmp_path / "missing")],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not module.project_run_dir(project, "v1").exists()
