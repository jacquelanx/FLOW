"""Deterministic acquisition QC using Time and signal-stability proxies.

This module never removes events from the canonical first run. It reports candidate event
intervals and builds an exact, separately labeled sensitivity mask for human review.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


NOT_EVALUABLE = "NOT_EVALUABLE"


def _flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def find_time_channel(columns: Sequence[str]) -> str | None:
    for column in columns:
        if str(column).strip().lower() in {"time", "time-a"}:
            return str(column)
    return None


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    if not math.isfinite(mad) or mad <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return 0.67448975 * (values - med) / mad


def detector_voltages(metadata: Mapping[str, Any], pnn_labels: Sequence[str]) -> list[dict[str, Any]]:
    """Extract static per-detector voltage metadata (``$P#V``), when present."""
    rows = []
    lower = {str(k).lower().lstrip("$"): v for k, v in metadata.items()}
    for index, channel in enumerate(pnn_labels, start=1):
        raw = lower.get(f"p{index}v")
        voltage = None
        if raw not in (None, ""):
            try:
                voltage = float(raw)
            except (TypeError, ValueError):
                voltage = None
        rows.append({
            "detector_index": index,
            "channel": str(channel),
            "voltage": voltage,
            "voltage_metadata_state": "PRESENT" if voltage is not None else NOT_EVALUABLE,
        })
    return rows


def analyze_acquisition(
    raw_df: pd.DataFrame,
    *,
    max_bins: int = 200,
    min_events: int = 1_000,
    signal_z_threshold: float = 8.0,
) -> dict[str, Any]:
    """Measure event-rate instability and concurrent binned signal excursions.

    ``signal_spike_proxy`` is intentionally not called a voltage spike: a static ``$P#V``
    field cannot prove a within-file voltage change.  Candidate intervals are returned as
    time ranges, while event removal remains disabled.
    """
    df = _flat_columns(raw_df)
    time_channel = find_time_channel([str(c) for c in df.columns])
    base = {
        "time_channel": time_channel,
        "event_exclusion_applied": False,
        "shadow_candidate_only": True,
        "voltage_spike_claim": "NOT_MADE",
        "cleaning_policy_id": "time_rate_multichannel_robust_z.v1",
        "cleaning_thresholds": {
            "maximum_bins": int(max_bins),
            "minimum_finite_time_events": int(min_events),
            "event_rate_robust_z": 8.0,
            "signal_median_robust_z": float(signal_z_threshold),
            "minimum_concurrent_signal_channels": 2,
        },
    }
    if time_channel is None:
        return base | {
            "state": NOT_EVALUABLE,
            "reason": "Time channel missing",
            "n_events_evaluated": len(df),
            "candidate_event_fraction": None,
            "candidate_time_intervals": [],
        }
    time = pd.to_numeric(df[time_channel], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(time)
    if int(finite.sum()) < min_events:
        return base | {
            "state": NOT_EVALUABLE,
            "reason": f"fewer than {min_events} finite Time events",
            "n_events_evaluated": int(finite.sum()),
            "candidate_event_fraction": None,
            "candidate_time_intervals": [],
        }
    t = time[finite]
    tmin, tmax = float(np.min(t)), float(np.max(t))
    if not math.isfinite(tmax - tmin) or tmax <= tmin:
        return base | {
            "state": NOT_EVALUABLE,
            "reason": "Time channel has no positive range",
            "n_events_evaluated": int(finite.sum()),
            "candidate_event_fraction": None,
            "candidate_time_intervals": [],
        }

    n_bins = max(20, min(int(max_bins), int(finite.sum() // 500)))
    edges = np.linspace(tmin, tmax, n_bins + 1)
    bin_id = np.clip(np.digitize(t, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.bincount(bin_id, minlength=n_bins).astype(float)
    positive_counts = counts[counts > 0]
    median_count = float(np.median(positive_counts)) if len(positive_counts) else 0.0
    count_z = _robust_z(counts)
    gap_bins = (counts == 0) | (count_z < -8.0)
    burst_bins = count_z > 8.0
    rate_cv = float(np.std(counts) / np.mean(counts)) if np.mean(counts) else math.inf

    excluded = {time_channel}
    signal_columns = [
        str(c) for c in df.columns
        if str(c) not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Evaluate all numeric detectors but report only aggregate multi-channel excursions.
    signal_hits = np.zeros(n_bins, dtype=int)
    signal_peak_z = np.zeros(n_bins, dtype=float)
    if signal_columns:
        signal_frame = df.loc[finite, signal_columns].apply(pd.to_numeric, errors="coerce")
        binned_medians = signal_frame.groupby(bin_id, sort=True).median().reindex(range(n_bins))
    else:
        binned_medians = pd.DataFrame(index=range(n_bins))
    for column in signal_columns:
        z = np.abs(_robust_z(binned_medians[column].to_numpy(dtype=float)))
        z[~np.isfinite(z)] = 0.0
        signal_hits += z >= signal_z_threshold
        signal_peak_z = np.maximum(signal_peak_z, z)
    required_channels = 2 if len(signal_columns) >= 2 else 1
    signal_bins = signal_hits >= required_channels
    candidate_bins = burst_bins | signal_bins
    candidate_for_finite = candidate_bins[bin_id]
    candidate_count = int(candidate_for_finite.sum())
    candidate_fraction = float(candidate_count / finite.sum())
    intervals = [
        {
            "bin_index": int(b),
            "start": float(edges[b]),
            "end": float(edges[b + 1]),
            "end_inclusive": bool(b == n_bins - 1),
            "event_count": int(counts[b]),
            "rate_burst": bool(burst_bins[b]),
            "signal_spike_proxy": bool(signal_bins[b]),
            "signal_channels_flagged": int(signal_hits[b]),
            "max_robust_z": float(signal_peak_z[b]),
        }
        for b in np.flatnonzero(candidate_bins)
    ]
    gap_fraction = float(np.mean(gap_bins))
    burst_fraction = float(np.mean(burst_bins))
    signal_fraction = float(np.mean(signal_bins))
    if candidate_fraction > 0.05 or gap_fraction > 0.10:
        state = "FAIL"
    elif candidate_fraction > 0.005 or gap_fraction > 0.02 or rate_cv > 0.50:
        state = "REVIEW"
    else:
        state = "PASS"
    return base | {
        "state": state,
        "reason": None,
        "n_events_evaluated": int(finite.sum()),
        "n_bins": n_bins,
        "median_events_per_bin": median_count,
        "event_rate_cv": rate_cv,
        "gap_bin_fraction": gap_fraction,
        "burst_bin_fraction": burst_fraction,
        "signal_spike_proxy_bin_fraction": signal_fraction,
        "candidate_event_count": candidate_count,
        "candidate_event_fraction": candidate_fraction,
        "candidate_time_intervals": intervals,
        "signal_channels_evaluated": len(signal_columns),
        "canonical_keep_count": len(df),
    }


def acquisition_cleaning_keep_mask(
    time_values: Sequence[float], acquisition_qc: Mapping[str, Any]
) -> np.ndarray:
    """Return the exact secondary-analysis mask for predeclared candidate intervals.

    Files without evaluable Time, and PASS files with no candidate intervals, retain every
    event. Interval upper bounds are exclusive except for the final acquisition bin, matching
    the bin assignment used by :func:`analyze_acquisition`.
    """
    time = np.asarray(time_values, dtype=float)
    keep = np.ones(len(time), dtype=bool)
    if acquisition_qc.get("state") not in {"REVIEW", "FAIL"}:
        return keep
    for interval in acquisition_qc.get("candidate_time_intervals", []):
        start = float(interval["start"])
        end = float(interval["end"])
        if interval.get("end_inclusive"):
            hit = np.isfinite(time) & (time >= start) & (time <= end)
        else:
            hit = np.isfinite(time) & (time >= start) & (time < end)
        keep[hit] = False
    return keep


def acquisition_exclusion_reason(
    time_value: float, acquisition_qc: Mapping[str, Any]
) -> str | None:
    """Resolve a candidate event to its deterministic interval reason."""
    if not math.isfinite(float(time_value)):
        return None
    for interval in acquisition_qc.get("candidate_time_intervals", []):
        start = float(interval["start"])
        end = float(interval["end"])
        inside = start <= float(time_value) <= end if interval.get("end_inclusive") \
            else start <= float(time_value) < end
        if not inside:
            continue
        reasons = []
        if interval.get("rate_burst"):
            reasons.append("event_rate_burst")
        if interval.get("signal_spike_proxy"):
            reasons.append("concurrent_signal_instability_proxy")
        return "|".join(reasons) or "candidate_time_interval"
    return None
