"""EXAMPLE first-run script — NK cell-therapy 18-color panel.

THIS IS A STARTING TEMPLATE, NOT PART OF THE FLOW SOFTWARE. It is offered as editor seed
content on the Analysis page; edit it (or replace it entirely) to match YOUR assay, then save.
FLOW runs whatever script the project provides — this file is just a worked example for the
lab's BD LSRFortessa NK-therapy panel, and the harness never imports or depends on it.

Contract (any first-run script must follow this): it is invoked as
    python first_run.py --data <DATA_DIR> --out <OUT_DIR> --plots <PLOTS_DIR>
reads the dataset from --data (FCS files, flow.csv, alc.csv, ... all live there), and writes
result CSVs to --out (and any plots to --plots). Every CSV it writes is auto-loaded into the
agent's notebook so the agent can interpret it.

It runs the ENTIRE flow-cytometry analysis so the agent's job is purely INTERPRETIVE (review,
sanity-check, characterize kinetics, conclude) — it should not re-derive gates or recompute.

It produces five standardized outputs in the ``--out`` directory:
  * ``first_run_results.csv``  — one row per sample: population %s + absolute counts (K/µL).
  * ``qc_report.csv``          — per-sample acquisition/quality QC + pass/warn/fail flags.
  * ``gating_report.csv``      — the full gating hierarchy per sample: event counts, % of parent,
                                 and the exact threshold used at each gate (auditable).
  * ``protein_expression.csv`` — per population × per marker: MFI (median, compensated) and
                                 % positive, at each timepoint.
  * ``cluster_profiles.csv``   — unsupervised k-means clusters per sample with per-marker median
                                 intensities (complements the supervised gating).
Plus per-sample plots (gating, QC, UMAP) and cross-timepoint trend plots under ``--plots``, and a
``first_run_summary.txt`` data dictionary describing every column.

Gating hierarchy (raw + compensated data):
  0. Singlets        : FSC-H/FSC-A ratio within ±3 MAD of median
  1. Debris removal  : FSC-A > 10000
  2. Live cells      : UV 450 L/D-A < adaptive threshold (within CD45+)
  3. CD45+ leukocytes: BV510-A > adaptive valley threshold
  4. Lymphocytes     : automatic FSC/SSC gate (low-scatter k-means cluster)
  5. T cells         : RAW APC-Cy7-A (CD3) > valley
  6. B cells         : COMP Qdot 800-A (CD19) > valley
  7. CD14 dump       : COMP PE-Cy5-A > p90
  8. NK cells        : CD3- CD56+ (PE-Texas Red-A) CD19- CD14-
  9. Donor NK        : NK ∩ HLA-A3 Donor+ (COMP Alexa Fluor 488-A)
 10. CAR+ Donor NK   : Donor NK ∩ CD70+ (COMP PE-Cy7-A)
 11. Patient NK      : NK ∩ HLA-A3 Donor-
 12. CD27+ Patient NK: Patient NK ∩ CD27+ (COMP Alexa Fluor 647-A)

Absolute counts (K/µL) = (pop% / 100) × WBC, nested for sub-populations. The WBC (K/µL) comes
from alc.csv (columns: date, alc), mapped to each timepoint via flow.csv (label -> date). An
optional cbc.csv (label, wbc_kul) overrides the WBC per timepoint.

Usage:
  python nk_panel.py --fcs <dir> --out <dir> [--plots <dir>] [--flow flow.csv]
                     [--alc alc.csv] [--cbc cbc.csv] [--no-unsupervised]
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ── Canonical NK-panel marker → detector map (the validated lab panel) ─────────
# metadata.json's car_channel is only a hint for the GENERIC pipeline; the NK panel uses this
# authoritative map: CAR = CD70 (PE-Cy7-A) gated within Donor NK; Alexa Fluor 647-A = CD27.
MARKER_MAP = {
    "BV510-A": "CD45",
    "UV 450 L/D-A": "Viability(L/D)",
    "APC-Cy7-A": "CD3",
    "Qdot 800-A": "CD19",
    "PE-Cy5-A": "CD14",
    "PE-Texas Red-A": "CD56",
    "Alexa Fluor 488-A": "HLA-A3(Donor)",
    "PE-Cy7-A": "CD70(CAR)",
    "Alexa Fluor 647-A": "CD27",
}
SCATTER_CHANNELS = ["FSC-A", "FSC-H", "SSC-A"]
# Fluorescent detectors used for expression profiling + clustering (viability excluded from
# clustering as it is a gating dye, not a phenotype marker).
PHENOTYPE_DETECTORS = [
    "APC-Cy7-A", "Qdot 800-A", "PE-Cy5-A", "PE-Texas Red-A",
    "Alexa Fluor 488-A", "PE-Cy7-A", "Alexa Fluor 647-A", "BV510-A",
]


# ── FCS I/O ───────────────────────────────────────────────────────────────────
def read_fcs_dual(filepath):
    """Read an FCS 3.x file; return (raw_df, comp_df, kv) — uncompensated, compensated, keywords."""
    with open(filepath, "rb") as f:
        header = f.read(58)
        ts = int(header[10:18].strip())
        te = int(header[18:26].strip())
        dsh = int(header[26:34].strip())
        deh = int(header[34:42].strip())
        f.seek(ts)
        text = f.read(te - ts + 1).decode("latin-1", errors="replace")

    delim = text[0]
    pairs = text[1:].split(delim)
    kv = {}
    for i in range(0, len(pairs) - 1, 2):
        if i + 1 < len(pairs):
            kv[pairs[i].strip().upper()] = pairs[i + 1].strip()

    n_par = int(kv.get("$PAR", 0))
    n_ev = int(kv.get("$TOT", 0))
    dtype = kv.get("$DATATYPE", "F").upper()
    end = "<" if "1,2,3,4" in kv.get("$BYTEORD", "1,2,3,4") else ">"
    ds = dsh if dsh > 0 else int(kv.get("$BEGINDATA", "0"))
    de = deh if deh > 0 else int(kv.get("$ENDDATA", "0"))

    with open(filepath, "rb") as f:
        f.seek(ds)
        raw_bytes = f.read(de - ds + 1)

    fmt = {"F": end + "f4", "D": end + "f8"}.get(dtype, end + "f4")
    data = np.frombuffer(raw_bytes, dtype=fmt).astype(np.float64).reshape(n_ev, n_par)
    chs = [kv.get(f"$P{i + 1}N", f"P{i + 1}") for i in range(n_par)]

    raw_df = pd.DataFrame(data.copy(), columns=chs)

    spill = kv.get("SPILL", kv.get("$SPILLOVER", ""))
    if spill:
        parts = spill.split(",")
        n = int(parts[0])
        sp_ch = parts[1:n + 1]
        S = np.array([float(v) for v in parts[n + 1:n + 1 + n * n]]).reshape(n, n)
        chi = {c: i for i, c in enumerate(chs)}
        sp_idx = [chi[c] for c in sp_ch if c in chi]
        data[:, sp_idx] = data[:, sp_idx] @ np.linalg.inv(S).T

    comp_df = pd.DataFrame(data, columns=chs)
    return raw_df, comp_df, kv


# ── Gate helpers ──────────────────────────────────────────────────────────────
def valley(vals, cf=150, nbins=500, lo=5, hi=95, smooth_w=15):
    """asinh-transform -> find histogram density trough between lo-hi percentiles."""
    t = np.arcsinh(np.clip(vals, -cf * 10, None) / cf)
    p_lo, p_hi = np.percentile(t, lo), np.percentile(t, hi)
    if p_lo >= p_hi:
        return float(np.percentile(vals, (lo + hi) / 2))
    bins = np.linspace(p_lo, p_hi, nbins)
    hist, edges = np.histogram(t, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(hist, np.ones(smooth_w) / smooth_w, mode="same")
    return float(np.sinh(mids[np.argmin(smooth)]) * cf)


def auto_lymphocyte_gate(cd45_raw):
    """Automatically gate the lymphocyte population on FSC/SSC (low-scatter k-means cluster)."""
    fsc = cd45_raw["FSC-A"].values.astype(float)
    ssc = cd45_raw["SSC-A"].values.astype(float)
    n = len(fsc)
    if n < 50:
        ft, st = np.percentile(fsc, 40), np.percentile(ssc, 40)
        return (fsc < ft) & (ssc < st)
    try:
        from sklearn.cluster import KMeans

        X = np.column_stack([fsc, ssc])
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(Xs)
        centers = km.cluster_centers_
        score = centers[:, 1] + 0.5 * centers[:, 0]
        lymph_c = int(np.argmin(score))
        mask = km.labels_ == lymph_c
        print(f"  Auto lymphocyte gate: {mask.sum():,}/{n:,} CD45+live "
              f"({100 * mask.mean():.2f}%) in the low-scatter cluster")
        return mask
    except Exception:
        ft, st = np.percentile(fsc, 40), np.percentile(ssc, 40)
        return (fsc < ft) & (ssc < st)


def smart_cd45_gate(raw_df):
    """Adaptive CD45 (BV510-A) + UV viability thresholds. Returns (cd45_t, uv_t)."""
    bv_vals = raw_df["BV510-A"].values
    uv_vals = raw_df["UV 450 L/D-A"].values
    fsc_vals = raw_df["FSC-A"].values

    cd45_t = valley(bv_vals, lo=27, hi=43)
    if cd45_t < 500:
        cd45_t = valley(bv_vals, lo=88, hi=99)

    g_cd45_prelim = (fsc_vals > 10000) & (bv_vals > cd45_t)
    uv_in_cd45 = uv_vals[g_cd45_prelim]

    if len(uv_in_cd45) > 100:
        uv_t = valley(uv_in_cd45, lo=20, hi=90)
        if uv_t < 500 or uv_t > 20000:
            uv_t = float(np.percentile(uv_in_cd45, 90))
    else:
        uv_t = float(np.percentile(uv_vals, 95))

    return float(cd45_t), float(uv_t)


def singlet_gate(raw_df):
    """Doublet exclusion via FSC-H/FSC-A ratio (±3 MAD of median). Returns boolean mask."""
    if "FSC-H" not in raw_df.columns:
        return np.ones(len(raw_df), dtype=bool)
    fsc_a = raw_df["FSC-A"].values.astype(float)
    fsc_h = raw_df["FSC-H"].values.astype(float)
    above_floor = fsc_a > 1000
    ratio = np.where(above_floor, fsc_h / np.clip(fsc_a, 1, None), np.nan)
    valid = above_floor & np.isfinite(ratio)
    med = float(np.nanmedian(ratio[valid]))
    mad = float(np.nanmedian(np.abs(ratio[valid] - med))) + 1e-9
    return valid & (np.abs(ratio - med) <= 3 * mad)


def cd56_threshold_for_nk(pool_cd56_comp):
    """CD56 (PE-Texas Red-A) threshold in the dump-negative pool."""
    n = len(pool_cd56_comp)
    if n < 10:
        return float(np.percentile(pool_cd56_comp, 95))
    t_v = valley(pool_cd56_comp, lo=50, hi=99.5)
    f_pos = float((pool_cd56_comp > t_v).sum()) / n
    if 0.04 < f_pos < 0.80:
        return t_v
    return float(np.percentile(pool_cd56_comp, 95))


# ── QC module ─────────────────────────────────────────────────────────────────
def qc_sample(raw_df, kv, singlet_mask, cd45_mask, uv_t):
    """Per-sample acquisition/quality QC. Returns a metrics dict + a list of flags.

    Checks: event counts, singlet/debris/viability fractions, acquisition-time stability
    (event-rate CV over time bins, if a Time channel is present), per-channel saturation
    ("margin" events at the top of the detector range), channel completeness, and whether a
    compensation matrix was present. Flags are human-readable warnings for the agent to weigh.
    """
    n_total = len(raw_df)
    flags: list[str] = []

    n_singlet = int(singlet_mask.sum())
    pct_singlet = 100.0 * n_singlet / n_total if n_total else 0.0
    fsc = raw_df["FSC-A"].values if "FSC-A" in raw_df.columns else np.zeros(n_total)
    pct_debris = 100.0 * float(np.mean(fsc <= 10000)) if n_total else 0.0
    n_cd45 = int(cd45_mask.sum())
    pct_cd45 = 100.0 * n_cd45 / n_total if n_total else 0.0
    # Viability: fraction of debris-excluded events below the live/dead threshold.
    live_frac = float("nan")
    if "UV 450 L/D-A" in raw_df.columns and n_total:
        deb = raw_df["FSC-A"].values > 10000
        if deb.sum() > 0:
            live_frac = 100.0 * float(np.mean(raw_df["UV 450 L/D-A"].values[deb] < uv_t))

    # Acquisition-time stability: coefficient of variation of event rate across 10 time bins.
    rate_cv = float("nan")
    tcol = next((c for c in raw_df.columns if str(c).strip().lower() == "time"), None)
    if tcol is not None and n_total > 100:
        t = raw_df[tcol].values.astype(float)
        span = float(np.nanmax(t) - np.nanmin(t))
        if span > 0:
            counts, _ = np.histogram(t, bins=10)
            rate_cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
            if rate_cv > 0.5:
                flags.append(f"unstable acquisition rate (CV={rate_cv:.2f})")

    # Per-channel saturation ("margin" events at the detector max, from $PnR ranges).
    sat = {}
    n_par = int(kv.get("$PAR", 0))
    name_to_range = {}
    for i in range(1, n_par + 1):
        nm = kv.get(f"$P{i}N")
        rng = kv.get(f"$P{i}R")
        if nm and rng:
            try:
                name_to_range[nm] = float(rng)
            except ValueError:
                pass
    for det in PHENOTYPE_DETECTORS:
        if det in raw_df.columns and det in name_to_range and n_total:
            top = name_to_range[det] * 0.99
            frac = 100.0 * float(np.mean(raw_df[det].values >= top))
            sat[det] = frac
            if frac > 5.0:
                flags.append(f"{MARKER_MAP.get(det, det)} saturated ({frac:.1f}% at max)")
    max_sat = max(sat.values()) if sat else 0.0

    # Channel completeness + compensation presence.
    required = set(MARKER_MAP) | {"FSC-A", "SSC-A"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        flags.append(f"missing channels: {', '.join(missing)}")
    comp_present = bool(kv.get("SPILL") or kv.get("$SPILLOVER"))
    if not comp_present:
        flags.append("no compensation matrix in FCS")

    # Low-event warnings.
    if n_total < 10000:
        flags.append(f"low total events ({n_total:,})")
    if n_cd45 < 500:
        flags.append(f"few CD45+ events ({n_cd45:,})")
    if live_frac == live_frac and live_frac < 50:
        flags.append(f"low viability ({live_frac:.0f}%)")

    status = "pass"
    if flags:
        hard = any(("missing channels" in f or "few CD45+" in f or "low viability" in f)
                   for f in flags)
        status = "fail" if hard else "warn"

    return {
        "n_total": n_total,
        "n_singlets": n_singlet,
        "pct_singlets": round(pct_singlet, 2),
        "pct_debris": round(pct_debris, 2),
        "pct_cd45": round(pct_cd45, 2),
        "pct_viable": round(live_frac, 2) if live_frac == live_frac else "",
        "rate_cv": round(rate_cv, 3) if rate_cv == rate_cv else "",
        "max_channel_saturation_pct": round(max_sat, 2),
        "compensation_present": comp_present,
        "n_missing_channels": len(missing),
        "qc_status": status,
        "qc_flags": "; ".join(flags) if flags else "none",
    }


# ── Protein expression ("protein script") ─────────────────────────────────────
def marker_thresholds(cd45_comp):
    """Per-marker positivity thresholds computed once on the CD45+live pool (valley split)."""
    thr = {}
    for det in PHENOTYPE_DETECTORS:
        if det in cd45_comp.columns and len(cd45_comp) >= 20:
            thr[det] = valley(cd45_comp[det].values, lo=30, hi=99)
    return thr


def expression_profile(populations, thresholds, timepoint):
    """For each (population, marker): MFI (median compensated) + % positive.

    ``populations`` maps a population name -> its compensated sub-DataFrame. Returns a list of
    tidy rows (one per population×marker) suitable for a long-format CSV the agent can pivot.
    """
    rows = []
    for pop_name, comp in populations.items():
        n = len(comp)
        for det in PHENOTYPE_DETECTORS:
            if det not in comp.columns:
                continue
            vals = comp[det].values.astype(float)
            if n == 0:
                mfi, pct_pos = float("nan"), float("nan")
            else:
                mfi = float(np.median(vals))
                thr = thresholds.get(det)
                pct_pos = 100.0 * float(np.mean(vals > thr)) if thr is not None else float("nan")
            rows.append({
                "timepoint": timepoint,
                "population": pop_name,
                "n_events": n,
                "marker": MARKER_MAP.get(det, det),
                "detector": det,
                "mfi": round(mfi, 2) if mfi == mfi else "",
                "pct_positive": round(pct_pos, 2) if pct_pos == pct_pos else "",
            })
    return rows


# ── Unsupervised: k-means clustering + UMAP (complements supervised gating) ─────
def unsupervised(cd45_comp, timepoint, plots_dir, max_events=20000, k=8):
    """Cluster CD45+live events and embed with UMAP. Returns per-cluster rows; writes a plot.

    Deterministic (fixed seeds). Subsamples large files for tractability. Fails soft — if
    sklearn/umap are unavailable or the sample is too small, returns [] and writes no plot.
    """
    dets = [d for d in PHENOTYPE_DETECTORS if d in cd45_comp.columns]
    n = len(cd45_comp)
    if n < 200 or len(dets) < 3:
        return []
    try:
        from sklearn.cluster import MiniBatchKMeans

        X = np.arcsinh(cd45_comp[dets].values.astype(float) / 150.0)
        rng = np.random.RandomState(0)
        idx = rng.choice(n, size=min(n, max_events), replace=False) if n > max_events else np.arange(n)
        Xs = X[idx]
        km = MiniBatchKMeans(n_clusters=min(k, max(2, len(Xs) // 50)),
                             n_init=5, random_state=0).fit(Xs)
        labels = km.labels_

        rows = []
        for c in sorted(set(labels)):
            m = labels == c
            row = {
                "timepoint": timepoint,
                "cluster": int(c),
                "n_events": int(m.sum()),
                "pct_of_cd45": round(100.0 * float(m.mean()), 2),
            }
            for det in dets:
                # Report cluster median on the compensated LINEAR scale for interpretability.
                row[f"mfi_{MARKER_MAP.get(det, det)}"] = round(
                    float(np.median(cd45_comp[dets].values[idx][m, dets.index(det)])), 1)
            rows.append(row)

        # UMAP embedding (subsampled) colored by cluster.
        try:
            import umap  # umap-learn

            emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xs)
            os.makedirs(plots_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=2, c=labels, cmap="tab10")
            ax.set(title=f"UMAP (CD45+live) — {timepoint}", xlabel="UMAP1", ylabel="UMAP2")
            ax.legend(*sc.legend_elements(), title="cluster", fontsize=6, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"umap_{_safe(timepoint)}.png"), dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"  UMAP skipped for {timepoint}: {e}")
        return rows
    except Exception as e:
        print(f"  clustering skipped for {timepoint}: {e}")
        return []


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


# ── Gating plots ──────────────────────────────────────────────────────────────
def gating_plot(raw_df, cd45_mask, lymph_idx_in_cd45, thresholds, timepoint, plots_dir):
    """A 4-panel gating overview: FSC/SSC lymph gate, CD45, CD3, and CD56 histograms w/ gates."""
    try:
        os.makedirs(plots_dir, exist_ok=True)
        cd45_raw = raw_df[cd45_mask].reset_index(drop=True)
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
        lym = np.zeros(len(cd45_raw), dtype=bool)
        lym[lymph_idx_in_cd45] = True
        ax[0].scatter(cd45_raw["FSC-A"], cd45_raw["SSC-A"], s=1,
                      c=np.where(lym, "tab:blue", "lightgrey"))
        ax[0].set(xlabel="FSC-A", ylabel="SSC-A", title="Lymphocyte gate (CD45+live)")
        for a, det, key in [(ax[1], "BV510-A", "cd45"), (ax[2], "APC-Cy7-A", "cd3"),
                            (ax[3], "PE-Texas Red-A", "cd56")]:
            if det in raw_df.columns:
                a.hist(np.arcsinh(raw_df[det].values / 150.0), bins=100, color="0.6")
                t = thresholds.get(key)
                if t is not None:
                    a.axvline(np.arcsinh(t / 150.0), color="red", lw=1)
                a.set(title=f"{MARKER_MAP.get(det, det)} ({det})", xlabel="asinh")
        fig.suptitle(f"Gating — {timepoint}", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"gating_{_safe(timepoint)}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  gating plot skipped for {timepoint}: {e}")


# ── Main per-sample analysis ──────────────────────────────────────────────────
def analyze_sample(fcs_path, timepoint, epic_kul, plots_dir, do_unsupervised=True):
    """Full NK-panel analysis of one specimen.

    Returns a dict with keys: summary (population %s + abs), qc, gating (hierarchy rows),
    expression (long rows), clusters (long rows). Also writes per-sample plots.
    """
    print(f"\n{'=' * 60}\n{timepoint}\n{'=' * 60}")
    raw_df, comp_df, kv = read_fcs_dual(fcs_path)
    total = len(raw_df)
    print(f"Total events: {total:,}")

    sing = singlet_gate(raw_df)
    cd45_t, uv_t = smart_cd45_gate(raw_df)
    g_cd45 = ((raw_df["FSC-A"] > 10000) & (raw_df["UV 450 L/D-A"] < uv_t)
              & (raw_df["BV510-A"] > cd45_t)).values
    n_cd45 = int(g_cd45.sum())
    print(f"CD45+live: {n_cd45:,} ({100 * n_cd45 / total:.2f}%)")

    qc = qc_sample(raw_df, kv, sing, g_cd45, uv_t)

    if n_cd45 < 20:
        return {"summary": _empty_result(timepoint, epic_kul, 0.0), "qc": qc,
                "gating": [], "expression": [], "clusters": []}

    cd45_raw = raw_df[g_cd45].reset_index(drop=True)
    cd45_comp = comp_df[g_cd45].reset_index(drop=True)
    t_cd3_global = valley(cd45_raw["APC-Cy7-A"].values, lo=25, hi=75)

    lmask = auto_lymphocyte_gate(cd45_raw)
    lymph_raw = cd45_raw[lmask].reset_index(drop=True)
    lymph_comp = cd45_comp[lmask].reset_index(drop=True)
    n_lymph = len(lymph_raw)
    pct_lymph = 100 * n_lymph / n_cd45 if n_cd45 > 0 else 0
    print(f"Lymphocytes: {n_lymph:,} ({pct_lymph:.2f}%)")

    if n_lymph < 20:
        return {"summary": _empty_result(timepoint, epic_kul, pct_lymph), "qc": qc,
                "gating": [], "expression": [], "clusters": []}

    t_cd3_within = valley(lymph_raw["APC-Cy7-A"].values, lo=25, hi=75)
    t_cd3 = t_cd3_global if (t_cd3_global > 50 and t_cd3_within > t_cd3_global * 1.5) else t_cd3_within
    t_cd19 = valley(lymph_comp["Qdot 800-A"].values, lo=70, hi=99)
    t_cd14 = float(np.percentile(lymph_comp["PE-Cy5-A"], 90))

    g_cd3 = (lymph_raw["APC-Cy7-A"] > t_cd3).values
    g_cd19 = (lymph_comp["Qdot 800-A"] > t_cd19).values
    g_cd14 = (lymph_comp["PE-Cy5-A"] > t_cd14).values
    dump_neg = ~g_cd3 & ~g_cd19 & ~g_cd14
    pct_t = 100 * g_cd3.mean()
    pct_b = 100 * g_cd19.mean()

    pool_cd56 = lymph_comp.loc[dump_neg, "PE-Texas Red-A"].values
    t_cd56 = cd56_threshold_for_nk(pool_cd56)
    g_nk = dump_neg & (lymph_comp["PE-Texas Red-A"] > t_cd56).values
    n_nk = int(g_nk.sum())
    pct_nk = 100 * n_nk / n_lymph if n_lymph > 0 else 0
    print(f"  %T={pct_t:.1f}  %B={pct_b:.1f}  %NK={pct_nk:.2f}")

    nk_comp = lymph_comp[g_nk].reset_index(drop=True)
    # sub-population comp frames for expression profiling
    dnk_comp = pnk_comp = car_comp = cd27_comp = nk_comp.iloc[0:0]
    t_hla = t_cd70 = t_cd27 = None
    if n_nk >= 10:
        af_vals = nk_comp["Alexa Fluor 488-A"].values
        t_hla = valley(af_vals, lo=40, hi=99.5)
        pct_pos = float((af_vals > t_hla).sum()) / n_nk
        if pct_pos < 0.05:
            t_hla = float(np.percentile(af_vals, 99.5))
        elif pct_pos > 0.95:
            t_hla = valley(af_vals, lo=5, hi=60)

        g_dnk = (nk_comp["Alexa Fluor 488-A"] > t_hla).values
        n_dnk = int(g_dnk.sum())
        pct_dnk = 100 * n_dnk / n_nk
        dnk_comp = nk_comp[g_dnk].reset_index(drop=True)
        pnk_comp = nk_comp[~g_dnk].reset_index(drop=True)

        if n_dnk >= 5:
            t_cd70 = valley(dnk_comp["PE-Cy7-A"].values, lo=20, hi=99)
            g_car = (dnk_comp["PE-Cy7-A"] > t_cd70).values
            pct_car = 100 * float(g_car.mean())
            car_comp = dnk_comp[g_car].reset_index(drop=True)
        else:
            pct_car = 0.0

        n_pnk = len(pnk_comp)
        pct_pnk = 100 * n_pnk / n_nk
        if n_pnk >= 5:
            t_cd27 = valley(pnk_comp["Alexa Fluor 647-A"].values, lo=20, hi=90)
            g_cd27 = (pnk_comp["Alexa Fluor 647-A"] > t_cd27).values
            pct_cd27 = 100 * float(g_cd27.mean())
            cd27_comp = pnk_comp[g_cd27].reset_index(drop=True)
        else:
            pct_cd27 = 0.0
        print(f"  DonorNK={pct_dnk:.1f}%  CAR+={pct_car:.1f}%  PatNK={pct_pnk:.1f}%  CD27+PNK={pct_cd27:.1f}%")
    else:
        pct_dnk = pct_car = pct_pnk = pct_cd27 = 0.0

    # Absolute counts (nested).
    lymph_abs = (pct_lymph / 100) * epic_kul
    t_abs = (pct_t / 100) * epic_kul
    b_abs = (pct_b / 100) * epic_kul
    nk_abs = (pct_nk / 100) * epic_kul
    dnk_abs = (pct_dnk / 100) * nk_abs
    car_abs = (pct_car / 100) * dnk_abs
    pnk_abs = (pct_pnk / 100) * nk_abs
    cd27_abs = (pct_cd27 / 100) * pnk_abs

    summary = {
        "timepoint": timepoint, "epic_kul": epic_kul, "pct_lymph": pct_lymph,
        "pct_t": pct_t, "pct_b": pct_b, "pct_nk": pct_nk, "pct_dnk": pct_dnk,
        "pct_car": pct_car, "pct_pnk": pct_pnk, "pct_cd27": pct_cd27,
        "lymph_abs": lymph_abs, "t_abs": t_abs, "b_abs": b_abs, "nk_abs": nk_abs,
        "dnk_abs": dnk_abs, "car_abs": car_abs, "pnk_abs": pnk_abs, "cd27_abs": cd27_abs,
    }

    # Gating hierarchy report (event counts, % of parent, threshold used).
    gating = [
        _grow(timepoint, "Total", 0, total, total, total, ""),
        _grow(timepoint, "CD45+live", 1, n_cd45, total, total,
              f"BV510>{cd45_t:.0f} & UV<{uv_t:.0f} & FSC>10000"),
        _grow(timepoint, "Lymphocytes", 2, n_lymph, n_cd45, total, "auto FSC/SSC low-scatter cluster"),
        _grow(timepoint, "T cells", 3, int(g_cd3.sum()), n_lymph, total, f"CD3(APC-Cy7)>{t_cd3:.0f}"),
        _grow(timepoint, "B cells", 3, int(g_cd19.sum()), n_lymph, total, f"CD19(Qdot800)>{t_cd19:.0f}"),
        _grow(timepoint, "NK cells", 3, n_nk, n_lymph, total,
              f"CD3-CD19-CD14- & CD56>{t_cd56:.0f}"),
        _grow(timepoint, "Donor NK", 4, len(dnk_comp), max(n_nk, 1), total,
              f"HLA-A3(AF488)>{t_hla:.0f}" if t_hla else ""),
        _grow(timepoint, "CAR+ Donor NK", 5, len(car_comp), max(len(dnk_comp), 1), total,
              f"CD70(PE-Cy7)>{t_cd70:.0f}" if t_cd70 else ""),
        _grow(timepoint, "Patient NK", 4, len(pnk_comp), max(n_nk, 1), total,
              "HLA-A3(AF488) negative"),
        _grow(timepoint, "CD27+ Patient NK", 5, len(cd27_comp), max(len(pnk_comp), 1), total,
              f"CD27(AF647)>{t_cd27:.0f}" if t_cd27 else ""),
    ]

    # Protein expression per population.
    thr = marker_thresholds(cd45_comp)
    populations = {
        "CD45+live": cd45_comp, "Lymphocytes": lymph_comp,
        "T cells": lymph_comp[g_cd3].reset_index(drop=True),
        "B cells": lymph_comp[g_cd19].reset_index(drop=True),
        "NK cells": nk_comp, "Donor NK": dnk_comp, "CAR+ Donor NK": car_comp,
        "Patient NK": pnk_comp, "CD27+ Patient NK": cd27_comp,
    }
    expression = expression_profile(populations, thr, timepoint)

    # Plots.
    lymph_idx = np.where(lmask)[0]
    gating_plot(raw_df, g_cd45, lymph_idx,
                {"cd45": cd45_t, "cd3": t_cd3, "cd56": t_cd56}, timepoint, plots_dir)

    clusters = unsupervised(cd45_comp, timepoint, plots_dir) if do_unsupervised else []

    return {"summary": summary, "qc": qc, "gating": gating,
            "expression": expression, "clusters": clusters}


def _grow(timepoint, name, level, n, parent_n, total, threshold):
    """Build one gating-hierarchy row."""
    return {
        "timepoint": timepoint, "level": level, "population": name, "n_events": int(n),
        "pct_of_parent": round(100.0 * n / parent_n, 3) if parent_n else "",
        "pct_of_total": round(100.0 * n / total, 4) if total else "",
        "threshold": threshold,
    }


def _empty_result(timepoint, epic_kul, pct_lymph):
    base = {k: 0.0 for k in ["pct_t", "pct_b", "pct_nk", "pct_dnk", "pct_car",
                             "pct_pnk", "pct_cd27", "lymph_abs", "b_abs", "t_abs",
                             "nk_abs", "dnk_abs", "car_abs", "pnk_abs", "cd27_abs"]}
    base.update({"timepoint": timepoint, "epic_kul": epic_kul, "pct_lymph": pct_lymph})
    return base


# ── flow.csv / WBC loading ────────────────────────────────────────────────────
def _load_flow_map(path):
    labels, dates = [], {}
    if path and os.path.exists(path):
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    lab = (row.get("label") or "").strip()
                    if lab:
                        labels.append(lab)
                        dates[lab] = (row.get("date") or "").strip()
        except Exception:
            pass
    return labels, dates


def _norm_date(s):
    return "".join(str(s).split()).lstrip("0").replace("/0", "/")


def _load_wbc(alc_path, cbc_path):
    by_date, by_label = {}, {}
    if alc_path and os.path.exists(alc_path):
        with open(alc_path, newline="") as f:
            for row in csv.DictReader(f):
                d = (row.get("date") or "").strip()
                try:
                    by_date[_norm_date(d)] = float(row.get("alc") or "nan")
                except ValueError:
                    pass
    if cbc_path and os.path.exists(cbc_path):
        with open(cbc_path, newline="") as f:
            for row in csv.DictReader(f):
                lab = (row.get("label") or "").strip()
                try:
                    by_label[lab] = float(row.get("wbc_kul") or row.get("epic_kul") or "nan")
                except ValueError:
                    pass
    return by_date, by_label


def _timepoint_for(fname, labels):
    stem = os.path.splitext(os.path.basename(fname))[0]
    for lab in labels:
        if lab and lab.lower() in stem.lower():
            return lab
    return stem


def _order_key(tp, labels):
    """Order timepoints by their position in flow.csv (falls back to alphabetical)."""
    try:
        return labels.index(tp)
    except ValueError:
        return len(labels) + 1


# ── Cross-timepoint trend plots ───────────────────────────────────────────────
def trend_plots(summary_rows, plots_dir):
    try:
        os.makedirs(plots_dir, exist_ok=True)
        df = pd.DataFrame(summary_rows)
        if df.empty:
            return
        x = range(len(df))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for col, lab in [("pct_nk", "%NK"), ("pct_dnk", "%Donor NK"),
                         ("pct_car", "%CAR+ DonorNK"), ("pct_pnk", "%Patient NK")]:
            if col in df:
                ax[0].plot(x, pd.to_numeric(df[col], errors="coerce"), marker="o", label=lab)
        ax[0].set(title="Populations over time", xlabel="timepoint", ylabel="% of parent")
        ax[0].set_xticks(list(x))
        ax[0].set_xticklabels(df["timepoint"], rotation=45, ha="right", fontsize=7)
        ax[0].legend(fontsize=7)
        for col, lab in [("car_abs", "CAR+ DonorNK"), ("nk_abs", "NK"), ("dnk_abs", "Donor NK")]:
            if col in df:
                ax[1].plot(x, pd.to_numeric(df[col], errors="coerce"), marker="o", label=lab)
        ax[1].set(title="Absolute counts over time (K/µL)", xlabel="timepoint", ylabel="K/µL")
        ax[1].set_xticks(list(x))
        ax[1].set_xticklabels(df["timepoint"], rotation=45, ha="right", fontsize=7)
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "trends_over_time.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"trend plots skipped: {e}")


# ── Output writers ────────────────────────────────────────────────────────────
def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


DATA_DICTIONARY = """\
FLOW first-run outputs (deterministic; the agent INTERPRETS these — it should not recompute them):

first_run_results.csv  — one row per sample (the headline table).
  file, timepoint                        : sample + mapped timepoint label
  pct_lymph,pct_t,pct_b,pct_nk           : % of parent (lymph=%of CD45+live; T/B/NK=%of lymph)
  pct_dnk                                : % Donor NK   (of NK)
  pct_car                                : % CAR+       (of Donor NK; CAR=CD70 on PE-Cy7-A)
  pct_pnk                                : % Patient NK (of NK)
  pct_cd27                               : % CD27+      (of Patient NK)
  wbc_kul                                : WBC (K/µL) used for absolute counts (from alc.csv)
  *_abs                                  : absolute counts (K/µL), nested (blank if no WBC)

qc_report.csv          — per-sample quality control.
  n_total,n_singlets,pct_singlets,pct_debris,pct_cd45,pct_viable,rate_cv,
  max_channel_saturation_pct,compensation_present,n_missing_channels,qc_status,qc_flags
  qc_status is pass/warn/fail; READ qc_flags before trusting a sample.

gating_report.csv      — the full gating hierarchy per sample (auditable).
  timepoint,level,population,n_events,pct_of_parent,pct_of_total,threshold
  'threshold' is the exact rule used at that gate — review it; refine if it looks wrong.

protein_expression.csv — per population × per marker.
  timepoint,population,n_events,marker,detector,mfi,pct_positive
  mfi = median compensated intensity; pct_positive uses a per-sample CD45-pool threshold.

cluster_profiles.csv   — unsupervised k-means clusters of CD45+live (complements gating).
  timepoint,cluster,n_events,pct_of_cd45, mfi_<marker>...  (per-cluster marker medians)

plots/                 — gating_<tp>.png, umap_<tp>.png (per sample), trends_over_time.png.
"""


def _find_fcs_dir(data_dir):
    """Locate the directory holding the FCS files under the dataset (falls back to data_dir)."""
    for root, _dirs, files in os.walk(data_dir):
        if any(f.lower().endswith(".fcs") for f in files):
            return root
    return data_dir


def main():
    # Standard FLOW first-run contract: --data (dataset dir), --out (results), --plots.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset dir (FCS files, flow.csv, alc.csv).")
    ap.add_argument("--out", required=True, help="Where to write result CSVs.")
    ap.add_argument("--plots", default="", help="Where to write plots.")
    ap.add_argument("--no-unsupervised", action="store_true",
                    help="Skip the clustering/UMAP step (faster).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    plots_dir = args.plots or os.path.join(args.out, "plots")
    # Derive the standard inputs from the dataset directory.
    fcs_dir = _find_fcs_dir(args.data)
    flow_csv = os.path.join(args.data, "flow.csv")
    alc_csv = os.path.join(args.data, "alc.csv")
    cbc_csv = os.path.join(args.data, "cbc.csv")

    labels, label_date = _load_flow_map(flow_csv)
    wbc_by_date, wbc_by_label = _load_wbc(alc_csv, cbc_csv)
    if not wbc_by_date and not wbc_by_label:
        print("NOTE: no WBC found (alc.csv / cbc.csv) — populations are still computed, but "
              "absolute counts are left blank.")

    fcs_files = sorted(
        os.path.join(fcs_dir, f) for f in os.listdir(fcs_dir) if f.lower().endswith(".fcs")
    )
    if not fcs_files:
        print(f"example_nk_panel.py: no FCS files found under {args.data}")
        return

    summary_rows, qc_rows, gating_rows, expr_rows, cluster_rows = [], [], [], [], []
    for path in fcs_files:
        tp = _timepoint_for(path, labels)
        wbc = wbc_by_label.get(tp)
        if wbc is None:
            wbc = wbc_by_date.get(_norm_date(label_date.get(tp, "")), float("nan"))
        has_wbc = wbc == wbc
        epic = wbc if has_wbc else 0.0
        try:
            res = analyze_sample(path, tp, epic, plots_dir,
                                 do_unsupervised=not args.no_unsupervised)
        except Exception as e:
            print(f"  ERROR on {os.path.basename(path)}: {e}")
            res = {"summary": {"timepoint": tp, "error": str(e)[:200]},
                   "qc": {"timepoint": tp, "qc_status": "fail", "qc_flags": f"exception: {e}"[:200]},
                   "gating": [], "expression": [], "clusters": []}
        s = res["summary"]
        s["file"] = os.path.basename(path)
        s["_has_wbc"] = has_wbc
        s["_order"] = _order_key(tp, labels)
        summary_rows.append(s)
        q = res["qc"]
        q["file"] = os.path.basename(path)
        q["timepoint"] = tp
        qc_rows.append(q)
        gating_rows.extend(res["gating"])
        expr_rows.extend(res["expression"])
        cluster_rows.extend(res["clusters"])

    # Order summary by timepoint (flow.csv order) for readability.
    summary_rows.sort(key=lambda r: r.get("_order", 999))

    # --- first_run_results.csv (headline table; stable schema) ---
    def cell(r, k, nd, abs_col=False):
        v = r.get(k)
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if abs_col and not r.get("_has_wbc"):
            return ""
        return round(v, nd) if isinstance(v, (int, float)) else v

    fields = ["file", "timepoint", "pct_lymph", "pct_t", "pct_b", "pct_nk", "pct_dnk",
              "pct_car", "pct_pnk", "pct_cd27", "wbc_kul",
              "lymph_abs", "t_abs", "b_abs", "nk_abs", "dnk_abs", "car_abs", "pnk_abs", "cd27_abs"]
    with open(os.path.join(args.out, "first_run_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in summary_rows:
            w.writerow([
                r.get("file", ""), r.get("timepoint", ""),
                cell(r, "pct_lymph", 2), cell(r, "pct_t", 1), cell(r, "pct_b", 1),
                cell(r, "pct_nk", 2), cell(r, "pct_dnk", 2), cell(r, "pct_car", 2),
                cell(r, "pct_pnk", 1), cell(r, "pct_cd27", 1), cell(r, "epic_kul", 3),
                cell(r, "lymph_abs", 4, True), cell(r, "t_abs", 4, True), cell(r, "b_abs", 4, True),
                cell(r, "nk_abs", 6, True), cell(r, "dnk_abs", 6, True), cell(r, "car_abs", 6, True),
                cell(r, "pnk_abs", 6, True), cell(r, "cd27_abs", 6, True),
            ])

    # --- the four comprehensive tables ---
    _write_csv(os.path.join(args.out, "qc_report.csv"),
               ["file", "timepoint", "n_total", "n_singlets", "pct_singlets", "pct_debris",
                "pct_cd45", "pct_viable", "rate_cv", "max_channel_saturation_pct",
                "compensation_present", "n_missing_channels", "qc_status", "qc_flags"], qc_rows)
    _write_csv(os.path.join(args.out, "gating_report.csv"),
               ["timepoint", "level", "population", "n_events", "pct_of_parent",
                "pct_of_total", "threshold"], gating_rows)
    _write_csv(os.path.join(args.out, "protein_expression.csv"),
               ["timepoint", "population", "n_events", "marker", "detector",
                "mfi", "pct_positive"], expr_rows)
    if cluster_rows:
        cl_fields = ["timepoint", "cluster", "n_events", "pct_of_cd45"] + \
            [k for k in cluster_rows[0] if k.startswith("mfi_")]
        _write_csv(os.path.join(args.out, "cluster_profiles.csv"), cl_fields, cluster_rows)

    trend_plots(summary_rows, plots_dir)

    with open(os.path.join(args.out, "first_run_summary.txt"), "w") as f:
        f.write(DATA_DICTIONARY)
        n_warn = sum(1 for q in qc_rows if q.get("qc_status") == "warn")
        n_fail = sum(1 for q in qc_rows if q.get("qc_status") == "fail")
        f.write(f"\nSamples: {len(summary_rows)} | QC warn: {n_warn} | QC fail: {n_fail}\n")

    print(f"\nFIRST_RUN_OK wrote {len(summary_rows)} samples.")
    print("Outputs: first_run_results.csv, qc_report.csv, gating_report.csv, "
          "protein_expression.csv" + (", cluster_profiles.csv" if cluster_rows else "") +
          ", first_run_summary.txt, plots/")
    print("MARKER MAP: CD45=BV510-A, CD3=APC-Cy7-A, CD19=Qdot 800-A, CD14=PE-Cy5-A, "
          "CD56=PE-Texas Red-A, HLA-A3 Donor=Alexa Fluor 488-A, CAR=CD70=PE-Cy7-A, "
          "CD27=Alexa Fluor 647-A. Lymphocytes AUTO-GATED (low FSC/SSC). WBC (K/µL) from alc.csv.")


if __name__ == "__main__":
    main()
