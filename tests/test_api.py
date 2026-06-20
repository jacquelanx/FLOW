"""API tests via FastAPI TestClient.

The run endpoint is exercised with a stubbed runner so the API surface is tested WITHOUT
Docker (the runner itself enforces Docker; that is covered elsewhere). Storage roots are
redirected to a tmp dir per test session.
"""

import importlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Storage roots are read from env at call time, so per-test isolation just needs envs.
    monkeypatch.setenv("FLOW_DATA_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("FLOW_ARTIFACTS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FLOW_DB_PATH", str(tmp_path / "flow.sqlite3"))
    import flow.api.app as appmod

    app = appmod.create_app()
    return TestClient(app), appmod


def test_health(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "docker" in r.json()


def test_providers(client):
    c, _ = client
    r = c.get("/api/providers")
    assert "mock" in r.json()["providers"]


def test_project_lifecycle_and_upload(client):
    c, _ = client
    r = c.post("/api/projects", json={"name": "demo"})
    assert r.status_code == 200
    pid = r.json()["id"]

    # Upload each file type.
    files = {
        "metadata.json": b'{"car_channel": "DET-E"}',
        "flow.csv": b"label,date\nBaseline,6/25/2025\n",
        "alc.csv": b"date,alc\n6/25/2025,1.2\n",
        "events.csv": b"sample_id,label,DET-A\nS01,Baseline,1.0\n",
    }
    for name, content in files.items():
        resp = c.post(
            f"/api/projects/{pid}/files",
            files={"file": (name, content, "application/octet-stream")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["file"]["ok"]

    # Profile.
    prof = c.get(f"/api/projects/{pid}/profile").json()
    assert prof["ok"]
    kinds = {f["kind"] for f in prof["files"]}
    assert "metadata" in kinds and "event_table" in kinds


def test_upload_rejects_traversal(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    resp = c.post(
        f"/api/projects/{pid}/files",
        files={"file": ("../evil.csv", b"x", "text/plain")},
    )
    assert resp.status_code == 400


def test_config_save_rejects_analysis_logic(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    bad = 'question: "q"\ngates: ["DET-E > 100"]\n'
    resp = c.post(f"/api/projects/{pid}/config", json={"content": bad})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert "analysis logic" in resp.json()["error"]


def test_config_default_returned(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    resp = c.get(f"/api/projects/{pid}/config")
    assert resp.json()["exists"] is False
    assert "question" in resp.json()["content"]


def test_run_lifecycle_with_stubbed_runner(client, monkeypatch):
    c, appmod = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    c.post(
        f"/api/projects/{pid}/files",
        files={"file": ("events.csv", b"sample_id,label,DET-A\nS01,Baseline,1.0\n", "text/plain")},
    )

    # Stub the Docker-backed runner so the API can be tested without Docker.
    def fake_run_analysis(*, data_dir, config, run_dir, run_id=None, on_step=None, **kw):
        from flow.agent.react import AgentResult, StepRecord
        from flow.trajectory import RunMetadata, Trajectory

        traj = Trajectory(run_dir)
        rec = StepRecord(1, "submit_answer", {"answer": "ok"}, "done", True, 1.0)
        traj.append_action({"step": 1, "tool": "submit_answer", "observation": "done"})
        (Path(run_dir) / "notebook.ipynb").write_text('{"cells": [], "nbformat": 4}')
        (Path(run_dir) / "answer.txt").write_text("Synthetic answer.")
        result = AgentResult(submitted=True, answer="Synthetic answer.", steps=[rec])
        meta = RunMetadata(
            run_id=run_id or "x", question=config.question, provider=config.runtime.provider,
            model=config.runtime.model, image="img", image_digest=None,
            max_steps=config.runtime.max_steps, limits={}, started_at=time.time(),
        )
        traj.finalize_meta = None
        (Path(run_dir) / "run.json").write_text('{"submitted": true}')
        if on_step:
            on_step(rec)
        return result, meta

    monkeypatch.setattr(appmod, "run_analysis", fake_run_analysis)

    start = c.post(
        "/api/runs",
        json={"project_id": pid, "question": "What is the trend?", "provider": "mock", "model": "mock"},
    )
    assert start.status_code == 200
    rid = start.json()["run_id"]

    # Poll until completed.
    for _ in range(50):
        status = c.get(f"/api/runs/{rid}").json()
        if status["status"] in {"completed", "failed", "error"}:
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    assert status["answer"] == "Synthetic answer."

    # Steps, notebook, artifacts.
    steps = c.get(f"/api/runs/{rid}/steps").json()
    assert len(steps["steps"]) >= 1
    nb = c.get(f"/api/runs/{rid}/notebook").json()
    assert "cells" in nb
    arts = c.get(f"/api/runs/{rid}/artifacts").json()["artifacts"]
    assert any(a["name"] == "answer.txt" for a in arts)

    # Run history persists in DB.
    runs = c.get("/api/runs").json()["runs"]
    assert any(rr["id"] == rid for rr in runs)


def test_download_zip(client, monkeypatch):
    c, appmod = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    # Create a run dir manually via DB + filesystem.
    import flow.db as db

    rid = "run123abc"
    run_dir = db.artifacts_root() / rid
    run_dir.mkdir(parents=True)
    (run_dir / "answer.txt").write_text("hi")
    db.create_run(rid, pid, "q", "mock", "mock", str(run_dir))
    resp = c.get(f"/api/runs/{rid}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
