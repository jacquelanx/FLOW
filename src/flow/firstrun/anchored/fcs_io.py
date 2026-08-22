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
from .compensation import (
    CompensationError, acquisition_id, acquisition_metadata, parse_spillover,
)

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
    control_derived_candidate: dict | None = None,
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
    acquisition_record = acquisition_metadata(s.metadata)
    resolved_acquisition_id = acquisition_id(path, s.metadata)
    audit = {
        "file": path.name,
        "source_path": str(path.resolve()),
        "acquisition_id": resolved_acquisition_id,
        **acquisition_record,
        "event_count": int(s.event_count),
        "compensation_source": "pending" if spill else "missing",
        "compensation_applied": False,
        "compensation_state": "BLOCKED_MISSING" if not spill else "PENDING",
        "compensation_reason": "embedded spillover absent" if not spill else None,
        "matrix_sha256": None,
        "matrix_channels": [],
        "matrix_values": None,
        "matrix_dimension": None,
        "matrix_condition_number": None,
        "matrix_validation_state": "NOT_EVALUABLE",
        "acquisition_qc": acquisition,
        "detector_voltages": voltages,
        "embedded_spillover_present": bool(spill),
        "embedded_matrix_sha256": None,
        "embedded_matrix_values": None,
        "embedded_matrix_validation_state": "NOT_EVALUABLE",
        "control_derived_candidate_sha256": (
            control_derived_candidate.get("matrix_sha256")
            if control_derived_candidate else None
        ),
        "control_derived_candidate_state": (
            control_derived_candidate.get("state") if control_derived_candidate else None
        ),
    }
    selected_spill = None
    selected_source = None
    matrix = None
    embedded_error = None
    if spill:
        try:
            embedded_matrix = parse_spillover(str(spill))
            audit["embedded_matrix_sha256"] = embedded_matrix.sha256
            audit["embedded_matrix_validation_state"] = embedded_matrix.state
            audit["embedded_matrix_values"] = embedded_matrix.values.tolist()
            matrix = embedded_matrix
            missing_channels = sorted(set(matrix.channels) - set(map(str, s.pnn_labels)))
            if missing_channels:
                raise CompensationError(
                    "spillover channels absent from sample: " + ", ".join(missing_channels)
                )
            if matrix.state == "BLOCKED":
                raise CompensationError(
                    "spillover matrix failed validation: " + ", ".join(matrix.reasons)
                )
        except Exception as exc:
            embedded_error = str(exc)
            matrix = None
        else:
            selected_spill = str(spill)
            selected_source = "embedded_fcs_spillover"

    candidate_eligible = bool(
        control_derived_candidate
        and control_derived_candidate.get("eligible_for_same_acquisition_fallback") is True
        and control_derived_candidate.get("state") == "PASS"
        and control_derived_candidate.get("spillover")
        and control_derived_candidate.get("acquisition_id") == resolved_acquisition_id
    )
    if selected_spill is None and candidate_eligible:
        try:
            matrix = parse_spillover(str(control_derived_candidate["spillover"]))
            missing_channels = sorted(set(matrix.channels) - set(map(str, s.pnn_labels)))
            if missing_channels:
                raise CompensationError(
                    "control-derived spillover channels absent from sample: "
                    + ", ".join(missing_channels)
                )
            if matrix.state != "PASS":
                raise CompensationError(
                    "control-derived spillover matrix is not PASS: "
                    + ", ".join(matrix.reasons)
                )
        except Exception as exc:
            raise CompensationError(
                f"{path.name}: eligible control-derived compensation failed: {exc}"
            ) from exc
        selected_spill = str(control_derived_candidate["spillover"])
        selected_source = "control_derived_run_matched"

    if selected_spill is None and embedded_error is not None:
        raise CompensationError(
            f"{path.name}: embedded compensation failed and no eligible run-matched "
            f"control-derived matrix was available: {embedded_error}"
        )

    if selected_spill is not None:
        try:
            s.apply_compensation(selected_spill)
        except Exception as exc:
            raise CompensationError(f"{path.name}: compensation failed: {exc}") from exc
        audit.update({
            "compensation_applied": True,
            "compensation_source": selected_source,
            "compensation_state": (
                "PROVISIONAL_EMBEDDED_UNVERIFIED"
                if selected_source == "embedded_fcs_spillover" and matrix.state == "PASS"
                else "CONTROL_DERIVED_RUN_MATCHED"
                if selected_source == "control_derived_run_matched" and matrix.state == "PASS"
                else "REVIEW_POOR_CONDITIONING"
            ),
            "compensation_reason": (
                "; ".join(matrix.reasons)
                or (f"embedded matrix unusable: {embedded_error}" if embedded_error else None)
            ),
            "matrix_sha256": matrix.sha256,
            "matrix_channels": list(matrix.channels),
            "matrix_values": matrix.values.tolist(),
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
    spill = sample.metadata.get("spill") or sample.metadata.get("spillover")
    embedded = None
    embedded_error = None
    if spill:
        try:
            embedded = parse_spillover(str(spill))
        except CompensationError as exc:
            embedded_error = str(exc)
    record = acquisition_metadata(sample.metadata)
    return {
        "file": path.name,
        "source_path": str(path.resolve()),
        "acquisition_id": acquisition_id(path, sample.metadata),
        **record,
        "event_count": int(sample.event_count),
        "channels": list(map(str, sample.pnn_labels)),
        "detector_voltages": detector_voltages(sample.metadata, sample.pnn_labels),
        "embedded_spillover_present": bool(spill),
        "embedded_matrix_sha256": embedded.sha256 if embedded else None,
        "embedded_matrix_channels": list(embedded.channels) if embedded else [],
        "embedded_matrix_dimension": len(embedded.channels) if embedded else None,
        "embedded_matrix_values": embedded.values.tolist() if embedded else None,
        "embedded_matrix_validation_state": embedded.state if embedded else "NOT_EVALUABLE",
        "embedded_matrix_error": embedded_error,
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
