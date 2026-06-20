"""Tests for upload validation, profiling, and the analysis-free config loader."""

import json
from pathlib import Path

import pytest

from flow.config import ConfigError, default_config_yaml, load_config
from flow.validation import profile_project, validate_file


def _write_project(d: Path) -> None:
    (d / "metadata.json").write_text(json.dumps({"car_channel": "DET-E", "hla_channel": "DET-D"}))
    (d / "flow.csv").write_text("label,date\nBaseline,6/25/2025\nDay14,7/9/2025\n")
    (d / "alc.csv").write_text("date,alc\n6/25/2025,1.2\n7/9/2025,1.1\n")
    (d / "events.csv").write_text("sample_id,label,DET-A,DET-E\nS01,Baseline,1.0,2.0\n")


def test_valid_files(tmp_path):
    _write_project(tmp_path)
    assert validate_file(tmp_path / "metadata.json").ok
    assert validate_file(tmp_path / "flow.csv").ok
    assert validate_file(tmp_path / "alc.csv").ok
    assert validate_file(tmp_path / "events.csv").kind == "event_table"


def test_flow_csv_missing_columns(tmp_path):
    (tmp_path / "flow.csv").write_text("label,when\nBaseline,x\n")
    r = validate_file(tmp_path / "flow.csv")
    assert not r.ok
    assert "missing" in r.detail


def test_profile_project(tmp_path):
    _write_project(tmp_path)
    prof = profile_project(tmp_path)
    assert prof.ok
    kinds = {f.kind for f in prof.files}
    assert "metadata" in kinds and "event_table" in kinds and "timepoints" in kinds
    desc = prof.dataset_description()
    assert "events.csv" in desc and "metadata.json" in desc
    # The description is purely structural — no analysis hints.
    for bad in ["gate", "threshold", "population"]:
        assert bad not in desc.lower()


def test_fcs_directory_detected(tmp_path):
    sub = tmp_path / "fcs"
    sub.mkdir()
    (sub / "a.fcs").write_text("x")
    (sub / "b.fcs").write_text("y")
    prof = profile_project(tmp_path)
    fcs = [f for f in prof.files if f.kind == "fcs_dir"]
    assert fcs and fcs[0].rows == 2


def test_config_loader_accepts_clean_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(default_config_yaml("What is the trend?"))
    cfg = load_config(cfg_path)
    assert cfg.question == "What is the trend?"
    assert cfg.safety.allow_network is False


def test_config_loader_rejects_analysis_logic(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        'question: "q"\ngates:\n  - "DET-E > 100"\n'
    )
    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path)
    assert "analysis logic" in str(ei.value)


def test_config_requires_question(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text('dataset:\n  root: "."\n')
    with pytest.raises(ConfigError):
        load_config(cfg_path)
