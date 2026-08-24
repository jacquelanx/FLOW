from __future__ import annotations

import json

import numpy as np
import pytest

from flow.firstrun.anchored.compensation import (
    CompensationError, _stain_identity, acquisition_id, parse_spillover, summarize_compensation,
    verify_control_settings,
)
from flow.firstrun.anchored.control_compensation import (
    candidate_for_acquisition, compare_specimens_to_control_repository,
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


def test_fcs_date_and_cytometer_serial_override_export_folder_name():
    metadata = {"$DATE": "25-JUN-2025", "$CYTNUM": "H658006R1029"}
    assert acquisition_id("old-export__sample.fcs", metadata) == (
        "2025-06-25|H658006R1029"
    )


def test_stain_identity_preserves_detector_slashes():
    assert _stain_identity("UV 450 L/D-A") == "uv450ld"
    assert _stain_identity("Compensation Controls_UV 450 L,2f,D Stained Control.fcs") == (
        "uv450ld"
    )


def test_reference_controls_do_not_become_run_matched_by_folder_proximity():
    repository = {
        "control_sets": [{
            "acquisition_id": "2024-02-02|FORTESSA",
            "state": "PASS",
            "matrix_sha256": "reference",
            "channels": ["A", "B"],
        }],
    }
    specimen = [{
        "file": "Specimen_001_Baseline.fcs",
        "acquisition_id": "2025-06-25|FORTESSA",
        "acquisition_date": "2025-06-25",
        "cytometer_serial": "FORTESSA",
        "compensation_source": "embedded_fcs_spillover",
        "matrix_sha256": "embedded",
        "matrix_channels": ["A", "B"],
        "matrix_values": [[1.0, 0.1], [0.05, 1.0]],
    }]
    assert candidate_for_acquisition(repository, specimen[0]["acquisition_id"]) is None
    selection = compare_specimens_to_control_repository(specimen, repository)
    assert selection["state"] == "REVIEW"
    assert selection["specimens"][0]["control_comparison_state"] == (
        "REFERENCE_NOT_RUN_MATCHED"
    )
    assert selection["specimens"][0]["recommendation"] == "KEEP_EMBEDDED_PROVISIONAL"


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


def test_stain_identity_matches_a_detector_to_its_own_control_file():
    """The "-A" area suffix is dropped; a fluorochrome's own trailing letter is not.

    Stripping a bare trailing "a" made "Aqua-A" normalize to "aqua" but its own
    "Aqua Stained Control" to "aqu", so a complete LIVE/DEAD Aqua control set was reported as
    a spill channel with no single stain and blocked the run.
    """
    pairs = [
        ("BV510-A", "Compensation Controls_BV510-A Stained Control.fcs"),
        ("APC-A", "Compensation Controls_APC Stained Control.fcs"),
        ("Alexa Fluor 700-A", "Compensation Controls_Alexa Fluor 700 Stained Control.fcs"),
        ("Aqua-A", "Compensation Controls_Aqua Stained Control.fcs"),
        ("LIVE DEAD Aqua-A", "Compensation Controls_LIVE DEAD Aqua Stained Control.fcs"),
        ("CD45RA-A", "Compensation Controls_CD45RA Stained Control.fcs"),
        ("PE-Texas Red-A", "Compensation Controls_PE-Texas Red Stained Control.fcs"),
    ]
    for channel, control_file in pairs:
        # verify_control_settings re-normalizes an inventory row's stored target, so the
        # control side is normalized twice on the real path.
        assert _stain_identity(channel) == _stain_identity(_stain_identity(control_file)), (
            f"{channel} does not match {control_file}"
        )


def test_stain_identity_is_idempotent():
    for value in ("Aqua-A", "BV510-A", "UV 450 L/D-A", "CD45RA-A", "A"):
        once = _stain_identity(value)
        assert _stain_identity(once) == once


def test_complete_control_set_for_an_a_ending_fluorochrome_is_not_blocked():
    voltages = [{"channel": "BV510-A", "voltage": 400.0},
                {"channel": "Aqua-A", "voltage": 400.0}]
    specimen = [{
        "file": "s.fcs", "acquisition_id": "A1", "matrix_dimension": 2,
        "matrix_channels": ["BV510-A", "Aqua-A"], "matrix_validation_state": "PASS",
        "compensation_state": "PROVISIONAL_EMBEDDED_UNVERIFIED",
        "detector_voltages": voltages,
    }]
    inventory = [
        {"file": "Compensation Controls_Unstained Control.fcs", "acquisition_id": "A1",
         "control_target": None, "detector_voltages": voltages},
        {"file": "Compensation Controls_BV510-A Stained Control.fcs", "acquisition_id": "A1",
         "control_target": _stain_identity("Compensation Controls_BV510-A Stained Control.fcs"),
         "detector_voltages": voltages},
        {"file": "Compensation Controls_Aqua Stained Control.fcs", "acquisition_id": "A1",
         "control_target": _stain_identity("Compensation Controls_Aqua Stained Control.fcs"),
         "detector_voltages": voltages},
    ]
    assessment = verify_control_settings(specimen, inventory, inventory)["acquisitions"][0]
    assert assessment["missing_spill_matrix_stain_channels"] == []
    assert assessment["state"] == "PASS", assessment["reasons"]
