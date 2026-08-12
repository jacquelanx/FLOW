#!/usr/bin/env python3
"""Compare automated percentages to manual gating reference CSV."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Map FCS trailing token → manual timepoint label. Explicit entries win over the generic
# parser below, so a study's manual-gating labels are always reproduced verbatim.
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

# When FLOW flattens a nested upload it keeps each file's provenance as a
# "<source folder>__" prefix (see the app's zip extractor). Classification and timepoint
# parsing look past that prefix so the original filename still drives both.
SRC_SEP = "__"

# Instrument compensation controls — BD FACSDiva writes these as
# "Compensation Controls_<fluor> Stained Control.fcs" / "..._Unstained Control.fcs", one set
# per acquisition folder. They carry the full fluorescence panel (so a panel check passes)
# but they are beads/single-stained tubes, not specimens.
COMP_PAT = re.compile(
    r"(^|[_ ])comp(ensation)?[_ ]|(^|[_ ])(un)?stained[_ ]*control", re.I)

# Pre-infusion / baseline recognition. Defined here (not in calibrate.py) so the timepoint
# parser and the file classifier share one definition; calibrate.py re-exports them.
PRE_PAT = re.compile(r"(baseline|screen|(^|_)pre(\b|_|$))", re.I)
BASE_PAT = re.compile(r"(baseline|screen)", re.I)

# Generic timepoint tokens, tried when TOKEN_MAP has no entry.
_DAY_PAT = re.compile(r"^d[ _-]?(\d+)$", re.I)
_WEEK_PAT = re.compile(r"^(?:week|wk)[ _-]?(\d+)$", re.I)
_MONTH_PAT = re.compile(r"^(?:month|mo)[ _-]?(\d+)$", re.I)
_WORD_TPS = {
    "baseline": "Baseline", "screen": "Screen", "screening": "Screening",
    "pre": "Pre", "post": "Post", "infusion": "Infusion", "eos": "EOS",
}
_PRE_TPS = {"baseline", "screen", "screening", "pre"}
_BASE_TPS = {"baseline", "screen", "screening"}
# Trailing filename boilerplate that is not a specimen qualifier ("D7_Specimen_001" is the
# D7 sample, whereas "D35_Pleural_Fluid" is a distinct specimen from that day).
_BOILERPLATE = {"specimen", "sample", "export", "tube", "data", "fcs", "singlets", "live"}

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


def strip_source_prefix(fname: str) -> str:
    """Drop the ``<source folder>__`` provenance prefix FLOW adds when flattening a tree."""
    name = Path(fname).name
    return name.rsplit(SRC_SEP, 1)[-1] if SRC_SEP in name else name


def is_comp_control(fname: str) -> bool:
    """True for instrument compensation controls (single-stained / unstained tubes)."""
    return bool(COMP_PAT.search(Path(strip_source_prefix(fname)).stem))


def _parse_token(tok: str) -> str | None:
    """One filename segment → canonical timepoint label, or None.

    Handles the explicit TOKEN_MAP, day/week/month numbers in the usual spellings, the
    named pre-infusion tokens, and a trailing specimen qualifier ("D35 Pleural Fluid",
    which is a different specimen from that day's blood draw and must stay distinct).
    """
    tok = tok.strip()
    if not tok:
        return None
    mapped = TOKEN_MAP.get(tok.lower())
    if mapped:
        return mapped
    head, _, qualifier = tok.partition(" ")
    if qualifier.strip():
        base = _parse_token(head)
        return f"{base} {qualifier.strip()}" if base else None
    m = _DAY_PAT.match(tok)
    if m:
        return f"D{int(m.group(1))}"
    m = _WEEK_PAT.match(tok)
    if m:
        return f"week{int(m.group(1))}"
    m = _MONTH_PAT.match(tok)
    if m:
        return f"month{int(m.group(1))}"
    return _WORD_TPS.get(tok.lower())


def timepoint_from_file(fname: str) -> str | None:
    """Canonical timepoint label for a specimen FCS, or None if it carries no timepoint.

    Segments are scanned right-to-left, so both ``Specimen_001_Baseline.fcs`` (the layout
    this pipeline was written against) and ``UPN25_D14_PBMC.fcs`` resolve. Anything after
    the timepoint is kept as a specimen qualifier, so ``Specimen_001_D35 Pleural Fluid.fcs``
    stays distinct from that day's blood draw instead of colliding on "D35" — and it reads
    the same whether or not the upload path replaced the spaces with underscores.

    Compensation controls and the NT-NK / CBMC / Car product controls carry no timepoint
    and return None.
    """
    if is_comp_control(fname):
        return None
    stem = Path(strip_source_prefix(fname)).stem
    parts = stem.split("_")
    for i in range(len(parts) - 1, -1, -1):
        tp = _parse_token(parts[i])
        if tp:
            rest = [p.strip() for p in parts[i + 1:] if p.strip()]
            while rest and (rest[-1].isdigit() or rest[-1].lower() in _BOILERPLATE):
                rest.pop()
            return f"{tp} {' '.join(rest)}" if rest else tp
    return None


_TP_STAGE = {"baseline": 0, "screen": 0, "screening": 0,
             "pre": 1, "infusion": 2, "post": 3, "eos": 9}


def timepoint_sort_key(tp: str | None):
    """Chronological sort key for a timepoint label (weeks/months folded to days).

    Only used to order rows when the project ships no ``flow.csv``; when it does, that
    file's ``label`` order still wins.
    """
    if not tp or tp != tp:  # None / NaN
        return (99, 0.0, "")
    head, _, qualifier = str(tp).partition(" ")
    stage = _TP_STAGE.get(head.lower())
    if stage is not None:
        return (stage, 0.0, qualifier)
    for pat, scale in ((_DAY_PAT, 1.0), (_WEEK_PAT, 7.0), (_MONTH_PAT, 30.0)):
        m = pat.match(head)
        if m:
            return (4, int(m.group(1)) * scale, qualifier)
    return (98, 0.0, str(tp))


def is_pre_infusion(fname: str) -> bool:
    """Pre-infusion sample (Baseline / Screen / Pre) — the donor-HLA negative reference."""
    tp = timepoint_from_file(fname)
    if tp is not None:
        return tp.split(" ")[0].lower() in _PRE_TPS
    return bool(PRE_PAT.search(strip_source_prefix(fname)))


def is_baseline(fname: str) -> bool:
    """Baseline / screening sample — the default reference timepoint for anchoring."""
    tp = timepoint_from_file(fname)
    if tp is not None:
        return tp.split(" ")[0].lower() in _BASE_TPS
    return bool(BASE_PAT.search(strip_source_prefix(fname)))


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
        return {"n": 0, "mae": None, "within_5pp": None, "within_10pp": None,
                "soft_target_ok": None}
    w10 = float((cmp["abs_delta"] <= 10).mean())
    return {
        "n": len(cmp),
        "mae": round(float(cmp["abs_delta"].mean()), 3),
        "median_abs": round(float(cmp["abs_delta"].median()), 3),
        "within_5pp": round(100 * float((cmp["abs_delta"] <= 5).mean()), 1),
        "within_10pp": round(100 * w10, 1),
        "soft_target_ok": w10 >= 0.70,
        "worst": cmp.loc[cmp["abs_delta"].idxmax()].to_dict(),
    }
