from __future__ import annotations

import csv

from flow.firstrun.anchored.flow_outputs import write_unified_cutoffs
from flow.firstrun.anchored.run import _gate_parameter_rows


def _anchor():
    return {
        "reference_timepoint": "Baseline",
        "reference_cuts": {"cd3": 10.0, "cd56": 20.0, "cd4": 30.0,
                           "cd8_from_cd4_neg": True},
        "locked": {"fsc_lo": 1.0, "fsc_hi": 2.0, "ssc_hi": 0.2},
        "channels": {"cd3": "CD3-A", "cd56": "CD56-A", "cd4": "CD4-A",
                     "hla": "HLA-A", "car": "CAR-A"},
        "transfer_policy": {"lineage_on_other_tp": "reestimate"},
        "manual_targets": {},
    }


def test_gate_parameter_rows_record_actual_values_and_cd8_semantics():
    cuts = {
        "cd3": 11.0, "cd56": 22.0, "cd4": 33.0, "cd8": None,
        "cd8_from_cd4_neg": True, "hla": 44.0, "hla_dim": False, "car": 55.0,
        "fsc_lo": 1.1, "fsc_hi": 2.2, "ssc_hi": 0.22,
        "viable_fsc_lo": 0.9, "viable_fsc_hi": 2.4, "viable_ssc_hi": 0.8,
        "ld": 1.0, "cd45": 2.0, "cd14": 3.0, "cd19": 4.0,
    }
    rows = _gate_parameter_rows(
        fname="D7.fcs", timepoint="D7", cuts=cuts, channels=_anchor()["channels"],
        anchor=_anchor(), cut_context={
            "hla_source": "control_or_adaptive", "hla_detail": "adaptive",
            "car_source": "adaptive", "car_detail": "adaptive",
            "lineage_transfer": "reference_nk_dominant",
            "b_fit": "timepoint_manual_refit", "cd14_fit": None,
        },
    )
    indexed = {row["parameter"]: row for row in rows}
    assert indexed["cd3"]["cutoff"] == 11.0
    assert indexed["cd3"]["source"] == "reference_anchor_nk_dominant"
    assert indexed["cd19"]["source"] == "timepoint_manual_refit"
    assert indexed["cd8"]["cutoff"] is None
    assert indexed["cd8"]["operator"] == "derived"
    assert indexed["cd8"]["cd8_is_cd4_negative_derived"] is True
    assert indexed["hla"]["operator"] == ">"


def test_legacy_cutoff_table_calls_values_reference_policy_and_includes_hla_car(tmp_path):
    path = write_unified_cutoffs(_anchor(), tmp_path, applied_policy={
        "hla_cut": 44.0, "hla_source": "control_or_adaptive",
        "car_cut": 55.0, "car_source": "adaptive",
    })
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_marker = {row["marker"]: row for row in rows}
    assert by_marker["CD3"]["cutoff_role"] == "reference anchor value"
    assert "actual per-file value" in by_marker["CD3"]["scope"]
    assert by_marker["HLA (Donor)"]["cutoff"] == "44.0"
    assert by_marker["CAR"]["cutoff"] == "55.0"
    assert by_marker["CD4"]["channel"] == "CD4-A"
