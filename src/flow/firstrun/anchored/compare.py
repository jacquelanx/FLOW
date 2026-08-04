#!/usr/bin/env python3
"""Compare automated percentages to manual gating reference CSV."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Map FCS trailing token → manual timepoint label
TOKEN_MAP = {
    "baseline": "Baseline",
    "pre": "Pre",
    "post": "Post",
    "d3": "D3",
    "d7": "D7",
    "d14": "D14",
    "d21": "D21",
    "d28": "D28",
    "week8": "week8",
    "d81": "D81",
    "d86": "D86",
    "d91": "D91",
}

# auto column → manual column
METRIC_MAP = {
    "%B (of lymph)": "b_p",
    "%T (of lymph)": "t_p",
    "%CD4 (of lymph)": "cd4_p",
    "%CD8 (of lymph)": "cd8_p",
    "%NK (of lymph)": "nk_p",
    # Manual d_nk_p is % Donor NK of lymph (same parent as nk_p), not of NK
    "%Donor NK (of lymph)": "d_nk_p",
    "%CAR+ (of Donor NK)": "car_d_p",
    # Optional monocyte / CD14 columns when present in the manual CSV
    "%Monocytes (of live)": "mono_p",
    "%CD14+ (of live)": "cd14_p",
}


def timepoint_from_file(fname: str) -> str | None:
    stem = Path(fname).stem.lower()
    # Specimen_001_Baseline → baseline
    tok = stem.split("_")[-1]
    return TOKEN_MAP.get(tok)


def load_manual(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["timepoint"] = df["timepoint"].astype(str)
    return df


def compare(auto_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    man = manual_df.set_index("timepoint")
    for _, r in auto_df.iterrows():
        tp = r.get("timepoint") or timepoint_from_file(r["file"])
        if tp is None or tp not in man.index:
            continue
        m = man.loc[tp]
        for auto_col, man_col in METRIC_MAP.items():
            if auto_col not in r or man_col not in m:
                continue
            a = float(r[auto_col]) if pd.notna(r[auto_col]) else None
            e = float(m[man_col]) if pd.notna(m[man_col]) else None
            if a is None or e is None:
                continue
            rows.append({
                "timepoint": tp,
                "metric": auto_col,
                "auto": round(a, 3),
                "manual": round(e, 3),
                "delta": round(a - e, 3),
                "abs_delta": round(abs(a - e), 3),
            })
    return pd.DataFrame(rows)


def summarize(cmp: pd.DataFrame) -> dict:
    if cmp.empty:
        return {"n": 0, "mae": None, "within_5pp": None, "within_10pp": None}
    return {
        "n": len(cmp),
        "mae": round(float(cmp["abs_delta"].mean()), 3),
        "median_abs": round(float(cmp["abs_delta"].median()), 3),
        "within_5pp": round(100 * float((cmp["abs_delta"] <= 5).mean()), 1),
        "within_10pp": round(100 * float((cmp["abs_delta"] <= 10).mean()), 1),
        "worst": cmp.loc[cmp["abs_delta"].idxmax()].to_dict(),
    }
