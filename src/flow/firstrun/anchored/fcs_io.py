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


def load_xform(path: str | Path, subsample: int | None = 200_000):
    """Load FCS → compensate → FlowJo biex/linear transform space.

    Returns (df, n_total_events). Scatter channels are Linear (0–1 scale after
    FlowKit LinearTransform with param_t=262144).
    """
    s = fk.Sample(str(path))
    spill = s.metadata.get("spill") or s.metadata.get("spillover")
    if spill:
        try:
            s.apply_compensation(spill)
        except Exception:
            pass
    s.apply_transform(build_transforms(s), include_scatter=True)
    df = s.as_dataframe(source="xform")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    ntot = s.event_count
    if subsample and len(df) > subsample:
        df = df.sample(subsample, random_state=0).reset_index(drop=True)
    return df, ntot


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
