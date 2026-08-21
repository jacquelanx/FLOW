from __future__ import annotations

import json

import numpy as np
import pytest

from flow.firstrun.anchored.compensation import (
    CompensationError, acquisition_id, parse_spillover, summarize_compensation,
    verify_control_settings,
)


def _spill(matrix: np.ndarray, channels=("A", "B")) -> str:
    values = ",".join(str(float(value)) for value in matrix.ravel())
    return f"{len(channels)},{','.join(channels)},{values}"


def test_parse_spillover_hash_is_stable_and_auditable():
    parsed = parse_spillover(_spill(np.array([[1.0, 0.1], [0.05, 1.0]])))
    assert parsed.channels == ("A", "B")
    assert parsed.state == "PASS"
    assert parsed.condition_number >= 1
    assert len(parsed.sha256) == 64
    assert json.dumps(parsed.as_audit())


@pytest.mark.parametrize("value", [
    "2,A,B,1,0,0",              # short
    "2,A,A,1,0,0,1",            # duplicate channel
    "2,A,B,1,nan,0,1",          # non-finite
    "2,A,B,1,0,0,0",            # non-positive diagonal
])
def test_malformed_spillover_is_fatal(value):
    with pytest.raises(CompensationError):
        parse_spillover(value)


def test_source_folder_prefix_recovers_acquisition():
    assert acquisition_id("2026-01-01-D3__Specimen_001_D3.fcs") == "2026-01-01-D3"
    assert acquisition_id("Specimen_001_D3.fcs") == "UNASSIGNED"


def test_within_acquisition_matrix_disagreement_blocks():
    summary = summarize_compensation([
        {"file": "a.fcs", "acquisition_id": "A1", "matrix_sha256": "x",
         "compensation_state": "PASS"},
        {"file": "b.fcs", "acquisition_id": "A1", "matrix_sha256": "y",
         "compensation_state": "PASS"},
    ])
    assert summary["state"] == "BLOCKED"
    assert summary["within_acquisition_matrix_disagreement"]["A1"] == ["x", "y"]


def test_controls_must_be_acquisition_linked_and_complete():
    specimen = [{
        "file": "A1__sample.fcs", "acquisition_id": "A1", "matrix_dimension": 2,
        "matrix_channels": ["A", "B"], "matrix_validation_state": "PASS",
        "detector_voltages": [{"channel": "A", "voltage": 250.0},
                              {"channel": "B", "voltage": 300.0}],
    }]
    controls = [
        {"file": "A1__unstained.fcs", "acquisition_id": "A1",
         "detector_voltages": [{"channel": "A", "voltage": 250.0},
                               {"channel": "B", "voltage": 300.0}]},
        {"file": "A1__A stained.fcs", "acquisition_id": "A1",
         "detector_voltages": [{"channel": "A", "voltage": 250.0},
                               {"channel": "B", "voltage": 300.0}]},
        {"file": "A1__B stained.fcs", "acquisition_id": "A1",
         "detector_voltages": [{"channel": "A", "voltage": 250.0},
                               {"channel": "B", "voltage": 300.0}]},
    ]
    inventory = [
        {"file": "Unstained Control.fcs", "acquisition_id": "A1"},
        {"file": "A Stained Control.fcs", "acquisition_id": "A1"},
        {"file": "B Stained Control.fcs", "acquisition_id": "A1"},
    ]
    assert verify_control_settings(specimen, controls, inventory)["state"] == "PASS"
    assert verify_control_settings(specimen, [], [])["state"] == "BLOCKED"


def test_review_matrix_missing_voltages_and_duplicate_stains_cannot_pass():
    specimen = [{
        "file": "A1__sample.fcs", "acquisition_id": "A1", "matrix_dimension": 2,
        "matrix_channels": ["A", "B"], "matrix_validation_state": "REVIEW",
        "compensation_state": "REVIEW_POOR_CONDITIONING",
        "detector_voltages": [{"channel": "A", "voltage": None},
                              {"channel": "B", "voltage": None}],
    }]
    controls = [
        {"file": "A1__unstained.fcs", "acquisition_id": "A1", "detector_voltages": []},
        {"file": "A1__A_1.fcs", "acquisition_id": "A1", "detector_voltages": []},
        {"file": "A1__A_2.fcs", "acquisition_id": "A1", "detector_voltages": []},
    ]
    inventory = [
        {"file": "Unstained Control.fcs", "acquisition_id": "A1"},
        {"file": "A Stained Control.fcs", "acquisition_id": "A1"},
        {"file": "A Stained Control copy.fcs", "acquisition_id": "A1",
         "control_target": "a"},
    ]
    assessment = verify_control_settings(specimen, controls, inventory)
    assert assessment["state"] == "BLOCKED"
    reasons = assessment["acquisitions"][0]["reasons"]
    assert "duplicate_single_stain_controls" in reasons
    assert "detector_voltage_metadata_missing" in reasons
    assert summarize_compensation(specimen)["state"] == "REVIEW"
