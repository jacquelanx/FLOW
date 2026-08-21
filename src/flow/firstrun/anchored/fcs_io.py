#!/usr/bin/env python3
"""FCS load + FlowJo-matched compensation / biex transforms."""
from __future__ import annotations

import json
import os
from pathlib import Path

import flowkit as fk
import numpy as np
import pandas as pd
from flowkit import transforms as T

from .acquisition_qc import analyze_acquisition, detector_voltages
from .compensation import CompensationError, acquisition_id, parse_spillover

HERE = Path(__file__).resolve().parent
ANCHOR = json.loads((HERE / "upn18_operator_anchor.json").read_text())
BIEX = {
    k: v for k, v in ANCHOR["transforms"].items()
    if v["type"] == "biex" and not k.startswith("Comp-")
}
BIEX_NORM = {k.replace("_", "/"): v for k, v in BIEX.items()}

DEFAULT_CH = dict(
    cd45="BV510-A",
    cd14="PE-Cy5-A",
    cd19="Qdot 800-A",
    cd3="APC-Cy7-A",
    cd56="PE-Texas Red-A",
    hla="Alexa Fluor 488-A",
    car="Alexa Fluor 647-A",
    cd16="BV650-A",
    ld="UV 450 L/D-A",
)
CH_OPT = dict(cd4="Alexa Fluor 700-A", cd8="PE-Cy5-5-A")


def build_transforms(sample) -> dict:
    xd = {}
    for ch in sample.pnn_labels:
        b = BIEX.get(ch) or BIEX_NORM.get(ch.replace("_", "/"))
        if b:
            xd[ch] = T.WSPBiexTransform(
                negative=b["negative"],
                width=b["width"],
                positive=b["positive"],
                max_value=b["max_value"],
            )
        else:
            xd[ch] = T.LinearTransform(param_t=262144.0, param_a=0.0)
    return xd


def load_xform(
    path: str | Path,
    subsample: int | None = 200_000,
    *,
    return_audit: bool = False,
):
    """Load FCS → compensate → FlowJo biex/linear transform space.

    Returns ``(df, n_total_events)`` by default and appends an audit dictionary when
    ``return_audit=True``. Scatter channels are Linear (0–1 scale after FlowKit
    LinearTransform with param_t=262144).

    A present-but-invalid or non-applicable spillover matrix is fatal. Missing spillover
    is retained as an explicit fluorescence blocker so scatter-only technical QC remains
    possible; it is never described as compensated.
    """
    path = Path(path)
    s = fk.Sample(str(path))
    raw_df = s.as_dataframe(source="raw")
    raw_df.columns = [c[0] if isinstance(c, tuple) else c for c in raw_df.columns]
    acquisition = analyze_acquisition(raw_df)
    raw_time = (
        pd.to_numeric(raw_df[acquisition["time_channel"]], errors="coerce").to_numpy()
        if acquisition.get("time_channel") in raw_df.columns else np.repeat(np.nan, len(raw_df))
    )
    voltages = detector_voltages(s.metadata, s.pnn_labels)
    spill = s.metadata.get("spill") or s.metadata.get("spillover")
    audit = {
        "file": path.name,
        "acquisition_id": acquisition_id(path),
        "event_count": int(s.event_count),
        "compensation_source": "embedded_fcs_spillover" if spill else "missing",
        "compensation_applied": False,
        "compensation_state": "BLOCKED_MISSING" if not spill else "PENDING",
        "compensation_reason": "embedded spillover absent" if not spill else None,
        "matrix_sha256": None,
        "matrix_channels": [],
        "matrix_dimension": None,
        "matrix_condition_number": None,
        "matrix_validation_state": "NOT_EVALUABLE",
        "acquisition_qc": acquisition,
        "detector_voltages": voltages,
    }
    if spill:
        try:
            matrix = parse_spillover(str(spill))
            missing_channels = sorted(set(matrix.channels) - set(map(str, s.pnn_labels)))
            if missing_channels:
                raise CompensationError(
                    "spillover channels absent from sample: " + ", ".join(missing_channels)
                )
            if matrix.state == "BLOCKED":
                raise CompensationError(
                    "spillover matrix failed validation: " + ", ".join(matrix.reasons)
                )
            s.apply_compensation(spill)
        except Exception as exc:
            raise CompensationError(f"{path.name}: compensation failed: {exc}") from exc
        audit.update({
            "compensation_applied": True,
            "compensation_state": (
                "PROVISIONAL_EMBEDDED_UNVERIFIED" if matrix.state == "PASS"
                else "REVIEW_POOR_CONDITIONING"
            ),
            "compensation_reason": "; ".join(matrix.reasons) or None,
            "matrix_sha256": matrix.sha256,
            "matrix_channels": list(matrix.channels),
            "matrix_dimension": len(matrix.channels),
            "matrix_condition_number": matrix.condition_number,
            "matrix_validation_state": matrix.state,
        })
    s.apply_transform(build_transforms(s), include_scatter=True)
    df = s.as_dataframe(source="xform")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Preserve the source-event key and raw Time value for exact membership joins and
    # deterministic shadow masks. Neither column participates in gate estimation.
    df["__event_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["__time_raw"] = raw_time
    ntot = s.event_count
    if subsample and len(df) > subsample:
        df = df.sample(subsample, random_state=0).reset_index(drop=True)
    if return_audit:
        return df, ntot, audit
    return df, ntot


def inspect_fcs_metadata(path: str | Path) -> dict:
    """Read control provenance/settings without transforming or gating control events."""
    path = Path(path)
    sample = fk.Sample(str(path))
    return {
        "file": path.name,
        "acquisition_id": acquisition_id(path),
        "event_count": int(sample.event_count),
        "channels": list(map(str, sample.pnn_labels)),
        "detector_voltages": detector_voltages(sample.metadata, sample.pnn_labels),
        "embedded_spillover_present": bool(
            sample.metadata.get("spill") or sample.metadata.get("spillover")
        ),
    }


def resolve_channels(meta: dict | None = None) -> dict:
    """Merge default channel map with optional metadata overrides."""
    ch = dict(DEFAULT_CH)
    if not meta:
        return ch
    if meta.get("hla_channel"):
        ch["hla"] = meta["hla_channel"]
    if meta.get("car_channel"):
        ch["car"] = meta["car_channel"]
    return ch


def has_full_panel(df: pd.DataFrame, ch: dict) -> bool:
    return all(v in df.columns for v in ch.values())
