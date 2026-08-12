#!/usr/bin/env python3
"""Control-anchored Donor HLA + CAR cutoffs (from NT-NK / Car / pre-infusion)."""
from __future__ import annotations

import re

import numpy as np

from .compare import (  # PRE_PAT/BASE_PAT live with the timepoint parser; re-exported here
    BASE_PAT, PRE_PAT, is_comp_control, strip_source_prefix,
)
from .gates import gate_with, gmm_valley

NT_PAT = re.compile(r"nt[-_ ]?nk", re.I)
CAR_PAT = re.compile(r"(^|_)car(\b|_|\.|$)", re.I)
CTRL_PAT = re.compile(r"(nt[-_ ]?nk|cbmc|(^|_)car(\b|_|\.|$))", re.I)

DONOR_POS_FALLBACK = 2000.0
MIN_PARENT = 50
MIN_POS = 10


def _pool(samples, cap=4000):
    out = []
    for s in samples:
        s = np.asarray(s, float)
        s = s[np.isfinite(s)]
        if len(s) > cap:
            s = np.random.RandomState(0).choice(s, cap, replace=False)
        if len(s):
            out.append(s)
    return np.concatenate(out) if out else np.array([])


def nk_hla_car(df, C, ch):
    """Return (hla_on_NK, car_on_NK) using current cuts."""
    m, g, _, _ = gate_with(df, C, ch)
    nk = m["NK"]
    if nk.sum() < 100:
        nk = m["live"]
    return g.get("hla", np.array([]))[nk], g.get("car", np.array([]))[nk]


def car_cut_from_controls(controls, base_cuts, ch):
    notes = {}
    car_cut = None
    if "ntnk" in controls:
        _, nt_car = nk_hla_car(controls["ntnk"], base_cuts, ch)
        if len(nt_car) > 100:
            # p99.9 — tighter than p99.5 so activated AF647-mid NK aren't called CAR+
            car_cut = float(np.percentile(nt_car, 99.9))
            notes["car_source"] = f"NT-NK p99.9={car_cut:.0f} (n={len(nt_car)})"
    if car_cut is None and "car" in controls:
        _, pr_car = nk_hla_car(controls["car"], base_cuts, ch)
        v = gmm_valley(pr_car, min_gap=100)
        if v is not None:
            car_cut = v
            notes["car_source"] = f"Car product GMM={v:.0f}"
    return car_cut, notes


def donor_cut_from_refs(controls, pre_hla, post_hla, polarity="donor", base_hla=None):
    """FMO-style donor cut. polarity='donor' → Donor = AF488 > cut."""
    notes = {}
    if polarity == "patient":
        notes["hla_dim"] = True
        bright = base_hla if (base_hla is not None and len(base_hla) >= 100) else pre_hla
        if bright is None or len(bright) < 50:
            notes["donor_cut"] = "patient-polarity: no bright ref → donor not gated"
            return None, notes
        b10 = float(np.percentile(bright, 10))
        if post_hla is None or len(post_hla) < 200 or float(np.percentile(post_hla, 50)) >= b10:
            cut = float(np.percentile(bright, 1))
            notes["donor_cut"] = f"patient-polarity: no dim NK → cut {cut:.0f}"
            return cut, notes
        dim_hi = float(np.percentile(post_hla, 90))
        cut = 0.5 * (dim_hi + b10)
        notes["donor_cut"] = f"patient-polarity midpoint dim_p90={dim_hi:.0f} bright_p10={b10:.0f} → {cut:.0f}"
        return cut, notes

    neg = float(np.percentile(pre_hla, 90)) if pre_hla is not None and len(pre_hla) >= 50 else None
    pos_vals, srcs = [], []
    for key in ("ntnk", "cbmc", "car"):
        if key in controls:
            # controls stored as dfs; caller passes precomputed if needed — here we expect
            # controls to already be gated externally. For simplicity accept raw arrays
            # keyed as *_hla in notes path. Real arrays come from process_patient.
            pass
    return neg, notes  # placeholder — real impl in process via arrays


def donor_cut(pre_hla, post_hla, ctrl_hla_medians, polarity="donor", base_hla=None):
    """Bracket donor cut between patient-negative and control-positive HLA.

    Prefers a GMM valley on pooled post-infusion NK when clearly bimodal
    (mixed chimerism), floored above pre-infusion NK p90.
    """
    notes = {}
    if polarity == "patient":
        notes["hla_dim"] = True
        bright = base_hla if (base_hla is not None and len(base_hla) >= 100) else pre_hla
        if bright is None or len(bright) < 50:
            notes["donor_cut"] = "patient-polarity: no bright ref"
            return None, notes
        b10 = float(np.percentile(bright, 10))
        if post_hla is None or len(post_hla) < 200 or float(np.percentile(post_hla, 50)) >= b10:
            cut = float(np.percentile(bright, 1))
            notes["donor_cut"] = f"patient-polarity no dim → {cut:.0f}"
            return cut, notes
        dim_hi = float(np.percentile(post_hla, 90))
        cut = 0.5 * (dim_hi + b10)
        notes["donor_cut"] = f"patient-polarity → {cut:.0f}"
        return cut, notes

    neg = float(np.percentile(pre_hla, 90)) if pre_hla is not None and len(pre_hla) >= 50 else None

    # Mixed chimerism: valley on post-infusion NK HLA
    if post_hla is not None and len(post_hla) >= 300:
        v = gmm_valley(post_hla, min_gap=150, min_n=300)
        if v is not None:
            frac = float(np.mean(post_hla > v))
            if 0.05 <= frac <= 0.95:
                cut = v
                if neg is not None:
                    cut = max(cut, neg)
                notes["donor_cut"] = f"post-NK GMM valley={v:.0f} (floor pre_p90={neg}) → {cut:.0f}"
                return cut, notes

    if ctrl_hla_medians:
        pos = float(np.median(ctrl_hla_medians))
        src = "controls"
    else:
        pos = DONOR_POS_FALLBACK
        src = "cohort fallback"

    if neg is None:
        cut = 0.6 * pos
        notes["donor_cut"] = f"no pre-NK; 0.6*{src}({pos:.0f}) → {cut:.0f}"
        return cut, notes
    if pos <= 1.3 * neg:
        cut = pos
        notes["donor_cut"] = f"poor separation pre_p90={neg:.0f} pos={pos:.0f} → cut at pos"
        return cut, notes
    # Bias toward the positive control so early mixed-chimerism patient NK
    # (intermediate HLA) are not swept into Donor.
    cut = 0.35 * neg + 0.65 * pos
    notes["donor_cut"] = f"weighted pre_p90={neg:.0f} & {src}={pos:.0f} → {cut:.0f}"
    return cut, notes


def adaptive_car_from_donor(loaded, base_cuts, ch, hla_cut, hla_dim=False):
    """GMM valley on pooled Donor-NK AF647 when no NT-NK control."""
    pools = []
    for fn, df in loaded:
        C = dict(base_cuts)
        C["hla"] = hla_cut
        C["hla_dim"] = hla_dim
        if C.get("cd3") is None or C.get("cd56") is None or hla_cut is None:
            continue
        m, g, _, _ = gate_with(df, C, ch)
        pools.append(g["car"][m["Donor"]])
    pool = _pool(pools)
    if len(pool) > 200:
        return gmm_valley(pool, min_gap=100)
    return None


def classify_file(fname: str) -> str:
    """Return 'comp' | 'ntnk' | 'car' | 'cbmc' | 'timepoint'.

    'comp' is an instrument compensation control — it passes a panel check but is not a
    specimen, so callers drop it rather than gating it.
    """
    name = strip_source_prefix(fname)
    if is_comp_control(name):
        return "comp"
    if NT_PAT.search(name):
        return "ntnk"
    if CAR_PAT.search(name):
        return "car"
    if "cbmc" in name.lower():
        return "cbmc"
    return "timepoint"
