from __future__ import annotations

import numpy as np
import pandas as pd

from flow.firstrun.anchored.acquisition_qc import analyze_acquisition, detector_voltages


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
