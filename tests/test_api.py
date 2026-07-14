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
    body = r.json()
    assert "mock" in body["providers"]
    # New: model catalog + defaults power the UI dropdowns.
    assert "gemini" in body["catalog"]
    assert body["default_models"]["groq"] == "llama-3.3-70b-versatile"
    assert "google" not in body["ui_providers"]  # alias hidden from UI


def test_config_form_save_and_reload(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    r = c.post(
        f"/api/projects/{pid}/config/form",
        json={
            "question": "How does DET-E change over time?",
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "max_steps": 18,
            "allow_network": True,
        },
    )
    assert r.status_code == 200 and r.json()["ok"] is True
    # GET returns a structured config the form can repopulate from.
    got = c.get(f"/api/projects/{pid}/config").json()
    assert got["exists"] is True
    assert got["config"]["question"] == "How does DET-E change over time?"
    assert got["config"]["runtime"]["provider"] == "groq"
    assert got["config"]["runtime"]["max_steps"] == 18
    assert got["config"]["safety"]["allow_network"] is True


def test_config_form_requires_question(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    r = c.post(f"/api/projects/{pid}/config/form", json={"question": "   "})
    assert r.json()["ok"] is False


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


def test_custom_prompt_save_and_get(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    assert c.get(f"/api/projects/{pid}/prompt").json()["exists"] is False
    c.post(f"/api/projects/{pid}/prompt", json={"content": "my gating rules"})
    got = c.get(f"/api/projects/{pid}/prompt").json()
    assert got["exists"] is True and got["content"] == "my gating rules"
    # Empty content clears it.
    c.post(f"/api/projects/{pid}/prompt", json={"content": "  "})
    assert c.get(f"/api/projects/{pid}/prompt").json()["exists"] is False


def test_config_default_returned(client):
    c, _ = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    resp = c.get(f"/api/projects/{pid}/config")
    assert resp.json()["exists"] is False
    assert "question" in resp.json()["content"]


def test_consensus_batch_lifecycle_with_stubbed_runner(client, monkeypatch):
    c, appmod = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    c.post(
        f"/api/projects/{pid}/files",
        files={"file": ("events.csv", b"sample_id,label,DET-A\nS01,Baseline,1.0\n", "text/plain")},
    )

    # Stub the Docker-backed batch runner so the API surface is tested without Docker.
    # It writes per-trajectory artifacts + batch.json exactly like the real run_batch.
    def fake_run_batch(*, data_dir, config, batch_dir, n_trajectories, batch_id=None,
                       on_phase=None, on_trajectory_status=None, on_step=None, **kw):
        from flow.runner import BatchResult
        from flow.trajectory import Trajectory

        if on_phase:
            on_phase("running")
        results = []
        for idx in range(n_trajectories):
            tdir = Path(batch_dir) / "trajectories" / str(idx)
            traj = Trajectory(tdir)
            traj.append_action({"step": 1, "tool": "submit_answer", "observation": "done"})
            (tdir / "notebook.ipynb").write_text('{"cells": [], "nbformat": 4}')
            (tdir / "answer.txt").write_text(f"Answer from trajectory {idx}.")
            if on_step:
                from flow.agent.react import StepRecord
                on_step(idx, StepRecord(1, "submit_answer", {}, "done", True, 1.0))
            summary = {"idx": idx, "submitted": True, "answer": f"Answer from trajectory {idx}.",
                       "failure_reason": None, "steps": 1, "artifact_dir": str(tdir)}
            results.append(summary)
            if on_trajectory_status:
                on_trajectory_status(idx, "completed", summary)
        if on_phase:
            on_phase("synthesizing")
        consensus = {"consensus": "Synthesized consensus.", "synthesized": True,
                     "n_submitted": n_trajectories, "n_total": n_trajectories, "failure_reason": None}
        (Path(batch_dir) / "consensus.txt").write_text("Synthesized consensus.")
        import json as _json
        (Path(batch_dir) / "batch.json").write_text(_json.dumps(
            {"consensus": consensus, "trajectories": results}))
        if on_phase:
            on_phase("done")
        return BatchResult(batch_id=batch_id or "b", n_trajectories=n_trajectories,
                           temperature=0.7, trajectories=results, consensus=consensus)

    monkeypatch.setattr(appmod, "run_batch", fake_run_batch)

    start = c.post(
        "/api/runs",
        json={"project_id": pid, "question": "What is the trend?", "provider": "mock",
              "model": "mock", "n_trajectories": 3},
    )
    assert start.status_code == 200
    assert start.json()["n_trajectories"] == 3
    rid = start.json()["run_id"]

    for _ in range(60):
        status = c.get(f"/api/runs/{rid}").json()
        if status["status"] in {"completed", "failed", "error"}:
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    assert status["consensus"]["consensus"] == "Synthesized consensus."
    assert status["consensus"]["synthesized"] is True
    assert len(status["trajectories"]) == 3
    assert all(t["submitted"] for t in status["trajectories"])

    # Per-trajectory steps / notebook / artifacts are scoped by index.
    steps = c.get(f"/api/runs/{rid}/trajectories/1/steps").json()
    assert len(steps["steps"]) >= 1
    nb = c.get(f"/api/runs/{rid}/trajectories/2/notebook").json()
    assert "cells" in nb
    arts = c.get(f"/api/runs/{rid}/trajectories/0/artifacts").json()["artifacts"]
    assert any(a["name"] == "answer.txt" for a in arts)
    # Out-of-range trajectory is a 404.
    assert c.get(f"/api/runs/{rid}/trajectories/9/steps").status_code == 404

    # History persists in DB.
    runs = c.get("/api/runs").json()["runs"]
    assert any(rr["id"] == rid for rr in runs)


def test_download_zip(client, monkeypatch):
    c, appmod = client
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    import flow.db as db

    rid = "batch123abc"
    batch_dir = db.artifacts_root() / rid
    (batch_dir / "trajectories" / "0").mkdir(parents=True)
    (batch_dir / "consensus.txt").write_text("hi")
    db.create_batch(rid, pid, "q", "mock", "mock", 1, str(batch_dir))
    resp = c.get(f"/api/runs/{rid}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
