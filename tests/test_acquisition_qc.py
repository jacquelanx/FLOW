from __future__ import annotations

import numpy as np
import pandas as pd

from flow.firstrun.anchored.acquisition_qc import (
    acquisition_cleaning_keep_mask, acquisition_exclusion_reason,
    analyze_acquisition, detector_voltages,
)


def test_missing_time_is_not_evaluable_not_pass():
    result = analyze_acquisition(pd.DataFrame({"FSC-A": np.arange(2000)}))
    assert result["state"] == "NOT_EVALUABLE"
    assert result["event_exclusion_applied"] is False


def test_stable_acquisition_passes_without_removing_events():
    n = 20_000
    result = analyze_acquisition(pd.DataFrame({
        "Time": np.linspace(0, 100, n),
        "FSC-A": np.sin(np.linspace(0, 6, n)) + 100,
        "CD45": np.cos(np.linspace(0, 4, n)) + 10,
    }))
    assert result["state"] == "PASS"
    assert result["candidate_event_fraction"] == 0
    assert result["canonical_keep_count"] == n


def test_concurrent_signal_excursion_is_shadow_only():
    n = 20_000
    time = np.linspace(0, 100, n)
    a = np.sin(time) + 100
    b = np.cos(time) + 50
    spike = (time >= 40) & (time < 45)
    a[spike] += 10_000
    b[spike] += 5_000
    result = analyze_acquisition(pd.DataFrame({"Time": time, "A": a, "B": b}))
    assert result["state"] in {"REVIEW", "FAIL"}
    assert result["signal_spike_proxy_bin_fraction"] > 0
    assert result["event_exclusion_applied"] is False
    assert result["voltage_spike_claim"] == "NOT_MADE"


def test_detector_voltage_metadata_is_distinct_from_signal_proxy():
    rows = detector_voltages({"p1v": "250", "$P2V": "bad"}, ["FSC-A", "CD45"])
    assert rows[0]["voltage"] == 250
    assert rows[0]["voltage_metadata_state"] == "PRESENT"
    assert rows[1]["voltage_metadata_state"] == "NOT_EVALUABLE"


def test_cleaning_mask_replays_predeclared_intervals_without_touching_missing_time():
    qc = {
        "state": "FAIL",
        "candidate_time_intervals": [
            {
                "start": 2.0, "end": 4.0, "end_inclusive": False,
                "rate_burst": True, "signal_spike_proxy": False,
            },
            {
                "start": 8.0, "end": 10.0, "end_inclusive": True,
                "rate_burst": False, "signal_spike_proxy": True,
            },
        ],
    }
    time = np.asarray([1.0, 2.0, 3.999, 4.0, 8.0, 10.0, np.nan])
    keep = acquisition_cleaning_keep_mask(time, qc)
    assert keep.tolist() == [True, False, False, True, False, False, True]
    assert acquisition_exclusion_reason(2.0, qc) == "event_rate_burst"
    assert acquisition_exclusion_reason(9.0, qc) == (
        "concurrent_signal_instability_proxy"
    )
    assert acquisition_exclusion_reason(float("nan"), qc) is None


def test_pass_and_not_evaluable_files_are_never_excluded():
    values = np.asarray([1.0, 2.0, 3.0])
    interval = [{"start": 1.0, "end": 4.0, "end_inclusive": True}]
    assert acquisition_cleaning_keep_mask(
        values, {"state": "PASS", "candidate_time_intervals": interval}
    ).all()
    assert acquisition_cleaning_keep_mask(
        values, {"state": "NOT_EVALUABLE", "candidate_time_intervals": interval}
    ).all()
