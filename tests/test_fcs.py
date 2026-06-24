"""Tests for FCS handling (parse + zip-of-FCS) and file/run deletion."""

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

flowio = pytest.importorskip("flowio")

from flow.validation import UnsafePathError, fcs_info, sanitize_filename


def _fcs_bytes(n: int = 4, channels=("FSC-A", "SSC-A", "DET-E")) -> bytes:
    buf = io.BytesIO()
    data = [float(i) for i in range(n * len(channels))]
    flowio.create_fcs(buf, data, list(channels))
    return buf.getvalue()


# ----------------------------------------------------------------- unit
def test_sanitize_filename_rejects_traversal_but_cleans_punctuation():
    with pytest.raises(UnsafePathError):
        sanitize_filename("../evil.fcs")
    with pytest.raises(UnsafePathError):
        sanitize_filename("a/b.fcs")
    # Real-world FCS names with spaces/parens are cleaned, not rejected.
    assert sanitize_filename("Sample A (1).fcs") == "Sample_A_1_.fcs"


def test_fcs_info_reads_event_count_and_channels(tmp_path: Path):
    p = tmp_path / "s.fcs"
    p.write_bytes(_fcs_bytes(n=5, channels=("FSC-A", "SSC-A", "DET-E")))
    info = fcs_info(p)
    assert info is not None
    n, chans = info
    assert n == 5
    assert chans == ["FSC-A", "SSC-A", "DET-E"]


def test_validate_file_recognizes_fcs(tmp_path: Path):
    from flow.validation import validate_file

    p = tmp_path / "x.fcs"
    p.write_bytes(_fcs_bytes(n=3))
    r = validate_file(p)
    assert r.kind == "fcs" and r.ok and r.rows == 3 and "FSC-A" in r.columns


# ----------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOW_DATA_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("FLOW_ARTIFACTS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FLOW_DB_PATH", str(tmp_path / "flow.sqlite3"))
    import flow.api.app as appmod

    return TestClient(appmod.create_app())


def test_zip_of_fcs_extracts_to_fcs_dir(client):
    pid = client.post("/api/projects", json={"name": "fcs"}).json()["id"]
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("sampleA.fcs", _fcs_bytes(n=4))
        zf.writestr("nested/sampleB.fcs", _fcs_bytes(n=6))  # flattened by basename
        zf.writestr("flow.csv", b"label,date\nBaseline,6/25/2025\n")  # non-fcs -> root
    resp = client.post(
        f"/api/projects/{pid}/files",
        files={"file": ("samples.zip", zbuf.getvalue(), "application/zip")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["zip"] is True
    assert len(resp.json()["extracted"]) == 3

    prof = client.get(f"/api/projects/{pid}/profile").json()
    kinds = {f["kind"]: f for f in prof["files"]}
    assert "fcs_dir" in kinds
    fcs_dir = kinds["fcs_dir"]
    assert fcs_dir["rows"] == 2  # two FCS files in the samples/ subdir
    assert "FSC-A" in fcs_dir["columns"]  # channels parsed from the first file
    assert "timepoints" in kinds  # flow.csv landed at the project root


def test_delete_file(client):
    pid = client.post("/api/projects", json={"name": "x"}).json()["id"]
    client.post(
        f"/api/projects/{pid}/files",
        files={"file": ("events.csv", b"sample_id,label,DET-A\nS01,Baseline,1.0\n", "text/plain")},
    )
    assert any(f["name"] == "events.csv" for f in client.get(f"/api/projects/{pid}/profile").json()["files"])
    d = client.delete(f"/api/projects/{pid}/files/events.csv")
    assert d.status_code == 200 and d.json()["ok"] is True
    assert not any(f["name"] == "events.csv" for f in client.get(f"/api/projects/{pid}/profile").json()["files"])
    # Deleting a missing file is a 404.
    assert client.delete(f"/api/projects/{pid}/files/nope.csv").status_code == 404


def test_delete_run(client):
    import flow.db as db

    pid = client.post("/api/projects", json={"name": "x"}).json()["id"]
    rid = "batchdel01"
    bdir = db.artifacts_root() / rid
    (bdir / "trajectories" / "0").mkdir(parents=True)
    (bdir / "consensus.txt").write_text("hi")
    db.create_batch(rid, pid, "q", "mock", "mock", 1, str(bdir))
    db.update_batch(rid, status="completed")  # only finished runs are deletable

    resp = client.delete(f"/api/runs/{rid}")
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert not bdir.exists()
    assert client.get(f"/api/runs/{rid}").status_code == 404


def test_delete_run_refuses_while_running(client):
    import flow.db as db

    pid = client.post("/api/projects", json={"name": "x"}).json()["id"]
    rid = "batchrun01"
    bdir = db.artifacts_root() / rid
    bdir.mkdir(parents=True)
    db.create_batch(rid, pid, "q", "mock", "mock", 1, str(bdir))
    db.update_batch(rid, status="running")
    assert client.delete(f"/api/runs/{rid}").status_code == 409
