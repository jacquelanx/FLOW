#!/usr/bin/env python3
"""
Adaptive gating — FSA × SSA FIRST.

Hierarchy
---------
  1. Lymphocytes   FSC-A × SSC-A   (debris / granulocyte removal)  ← FIRST
                   + purity-guided SSC refine (max B+T+NK / CD14-)
  2. Singlets      FSC-A vs FSC-H
  3. Live          UV L/D-
  4. CD45+         BV510
  5. CD14+/−       monocytes (CD14+) reported; lymph = CD14-
  6. CD19+/-       B vs non-B
  7. CD3 × CD56    T / NK
  8. CD4 / CD8     within T (CD8 falls back to CD4− if unimodal)
  9. Donor HLA     AF488  (allele + polarity from metadata)
 10. CAR+          AF647 within Donor NK
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from .fcs_io import CH_OPT


# ── threshold helpers ─────────────────────────────────────────────────────────

def gmm_valley(x, min_gap=200.0, min_weight=0.05, min_n=600):
    """2-component GMM valley; robust to biex floor pile-up."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return None
    floor = np.percentile(x, 1)
    core = x[x > floor + 1.0]
    if len(core) < min_n:
        return None
    g = GaussianMixture(2, random_state=0, n_init=2).fit(core.reshape(-1, 1))
    order = np.argsort(g.means_.ravel())
    mu = g.means_.ravel()[order]
    w = g.weights_[order]
    if mu[1] - mu[0] < min_gap or min(w) < min_weight:
        return None
    grid = np.linspace(mu[0], mu[1], 250)
    dens = np.exp(g.score_samples(grid.reshape(-1, 1)))
    valley = float(grid[np.argmin(dens)])
    if valley < mu[0] + 0.15 * (mu[1] - mu[0]):
        return None
    return valley


def pct_cut(x, p, fallback=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return fallback
    return float(np.percentile(x, p))


def valley_asinh(vals, lo=5, hi=95, cf=150, nbins=500, smooth_w=15):
    """Trough in asinh-smoothed histogram → threshold in original units."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 50:
        return float(np.median(vals)) if len(vals) else None
    v = np.arcsinh(np.clip(vals, -cf * 10, None) / cf)
    plo, phi = np.percentile(v, [lo, hi])
    if plo >= phi:
        return float(np.percentile(vals, (lo + hi) / 2))
    edges = np.linspace(plo, phi, nbins + 1)
    counts, _ = np.histogram(v, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(counts, np.ones(smooth_w) / smooth_w, mode="same")
    return float(np.sinh(centers[int(np.argmin(smooth))]) * cf)


def robust_cd45_cut(vals):
    """CD45 cut: bimodal valley, or a low floor when the sample is unimodal-bright."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return None
    v = gmm_valley(vals)
    if v is not None:
        frac = float(np.mean(vals > v))
        if 0.02 <= frac <= 0.90:
            return v
    p50 = float(np.percentile(vals, 50))
    # Post-infusion NK-bright: nearly all events are CD45-high — keep a low floor
    if p50 > 1400:
        return float(np.percentile(vals, 5))
    # Dim / mixed: search upper half for the leukocyte rise
    v_hi = valley_asinh(vals, lo=50, hi=99)
    if v_hi is not None:
        frac = float(np.mean(vals > v_hi))
        if 0.02 <= frac <= 0.60:
            return v_hi
    return float(np.percentile(vals, 25))


def robust_cd3_cut(vals, cd56_vals=None):
    """CD3 cut that handles T-dominant samples (bulk = T, tiny bright tail).

    GMM valley often splits the bright outlier tail and under-calls T. If the
    resulting CD3+ fraction is <15%, re-estimate looking for the negative
    shoulder — BUT only when CD56 does not already show a substantial NK
    population (rare-T / NK-expanding samples must keep the exclusive GMM cut).
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return None
    v = gmm_valley(vals)
    if v is not None:
        frac = float(np.mean(vals > v))
        if 0.15 <= frac <= 0.90:
            return v
        if frac < 0.15:
            nk_like = False
            if cd56_vals is not None:
                cd56_vals = np.asarray(cd56_vals, float)
                ok = np.isfinite(cd56_vals)
                if ok.sum() >= 100:
                    v56 = gmm_valley(cd56_vals[ok])
                    if v56 is None:
                        v56 = float(np.percentile(cd56_vals[ok], 70))
                    nk_like = float(np.mean(cd56_vals[ok] > v56)) >= 0.15
            if nk_like:
                return v  # T really is rare — keep exclusive cut
            v_lo = valley_asinh(vals, lo=2, hi=40)
            if v_lo is not None and float(np.mean(vals > v_lo)) >= 0.50:
                return v_lo
            return float(np.percentile(vals, 25))
        if frac > 0.90:
            v_hi = valley_asinh(vals, lo=40, hi=95)
            if v_hi is not None and float(np.mean(vals > v_hi)) <= 0.90:
                return v_hi
            return float(np.percentile(vals, 85))
    v_mid = valley_asinh(vals, lo=15, hi=75)
    if v_mid is not None and 0.10 <= float(np.mean(vals > v_mid)) <= 0.90:
        return v_mid
    return float(np.percentile(vals, 40))


def robust_cd56_cut(vals, cd3_vals, t_cd3):
    """CD56 cut preferred inside the CD3− pool; high cut when T-dominant."""
    vals = np.asarray(vals, float)
    cd3_vals = np.asarray(cd3_vals, float)
    parent = np.isfinite(vals) & np.isfinite(cd3_vals)
    if parent.sum() < 80:
        return None

    # T-dominant: CD3− pool is tiny → NK are the bright CD56 tail
    if t_cd3 is not None:
        frac_neg = float(np.mean(cd3_vals[parent] < t_cd3))
        if frac_neg < 0.20:
            v = valley_asinh(vals[parent], lo=70, hi=99.5)
            if v is not None and float(np.mean(vals[parent] > v)) <= 0.40:
                return v
            return float(np.percentile(vals[parent], 90))

    pool = vals[parent & (cd3_vals < t_cd3)] if t_cd3 is not None else vals[parent]
    if len(pool) < 80:
        pool = vals[parent]
    v = gmm_valley(pool)
    if v is not None:
        frac = float(np.mean(pool > v))
        if 0.04 <= frac <= 0.95:
            return v
    v_lo = valley_asinh(pool, lo=5, hi=55)
    if v_lo is not None and 0.10 <= float(np.mean(pool > v_lo)) <= 0.95:
        return v_lo
    v2 = valley_asinh(pool, lo=30, hi=90)
    if v2 is not None:
        return v2
    return pct_cut(pool, 70)


def robust_cd19_cut(vals):
    """CD19 cut; B cells are usually rare — allow up to ~12% for engrafted samples."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return None
    v = gmm_valley(vals, min_gap=80, min_n=300)
    if v is not None:
        frac = float(np.mean(vals > v))
        if 0.001 <= frac <= 0.12:
            return v
    v2 = valley_asinh(vals, lo=70, hi=99.5)
    if v2 is not None and float(np.mean(vals > v2)) <= 0.12:
        return v2
    # Fall back to a high percentile (rare B) rather than rejecting entirely
    return pct_cut(vals, 98)


def robust_ld_cut(vals):
    """L/D live cut; dead cells are the bright tail."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return None
    v = gmm_valley(vals)
    if v is not None:
        # live = below cut; expect majority live
        frac_dead = float(np.mean(vals > v))
        if 0.01 <= frac_dead <= 0.50:
            return v
    return pct_cut(vals, 90)


# ── FSA × SSA lymphocyte gate (FIRST) ─────────────────────────────────────────

def lymph_scatter_gate(fsca, ssca, ssc_cap=None, fsc_lo_pct=2.0):
    """
    Gate lymphocytes on FSC-A × SSC-A BEFORE any fluorescence.

    Strategy (matches typical FlowJo lymph gate):
      - Drop debris: FSC-A > p{fsc_lo_pct}
      - Prefer the LOW-SSC mode via 2–3 component GMM (tight lymph core)
      - Keep mid-FSC lymphocytes: FSC-A within the low-SSC cloud

    In FlowKit LinearTransform space scatter is ~0–1.
    Returns (mask, fsc_lo, fsc_hi, ssc_hi).
    """
    fsca = np.asarray(fsca, float)
    ssca = np.asarray(ssca, float)
    n = len(fsca)
    if n < 50:
        return np.ones(n, bool), 0.0, 1.0, 1.0

    fsc_lo = float(np.percentile(fsca, fsc_lo_pct))
    non_debris = fsca > fsc_lo
    ssc_nd = ssca[non_debris]
    fsc_nd = fsca[non_debris]
    ssc_hi = None
    fsc_hi = None

    # 2D GMM on FSC×SSC: take the lowest-SSC component as the lymph core,
    # then set SSC_hi / FSC bounds from that component's distribution.
    if non_debris.sum() >= 1500:
        try:
            rs = np.random.RandomState(0)
            take = min(8000, int(non_debris.sum()))
            ix = rs.choice(int(non_debris.sum()), take, replace=False)
            X = np.c_[fsc_nd[ix], ssc_nd[ix]]
            n_comp = 3 if take >= 3000 else 2
            gm = GaussianMixture(n_comp, random_state=0, n_init=2).fit(X)
            means = gm.means_
            lymph_k = int(np.argmin(means[:, 1]))  # lowest SSC
            # Bounds: component mean ± k·sqrt(cov) clipped to data
            cov = gm.covariances_[lymph_k]
            if cov.ndim == 2:
                sd_fsc = float(np.sqrt(max(cov[0, 0], 1e-12)))
                sd_ssc = float(np.sqrt(max(cov[1, 1], 1e-12)))
            else:
                sd_fsc = sd_ssc = float(np.sqrt(max(cov, 1e-12)))
            mu_fsc, mu_ssc = float(means[lymph_k, 0]), float(means[lymph_k, 1])
            # Generous enough to keep lymph, tight enough to exclude monocytes/gran
            fsc_lo = max(fsc_lo, mu_fsc - 3.5 * sd_fsc)
            fsc_hi = mu_fsc + 3.5 * sd_fsc
            ssc_hi = mu_ssc + 3.0 * sd_ssc
            # Also never exceed the valley between low and next-higher SSC mode
            order = np.argsort(means[:, 1])
            if len(order) >= 2:
                lo_mu = float(means[order[0], 1])
                hi_mu = float(means[order[1], 1])
                if hi_mu - lo_mu > 0.03:
                    ssc_hi = min(ssc_hi, 0.5 * (lo_mu + hi_mu))
        except Exception:
            ssc_hi = None
            fsc_hi = None

    if ssc_hi is None:
        # 1D SSC fallback
        if len(ssc_nd) >= 800:
            try:
                g1 = GaussianMixture(2, random_state=0, n_init=2).fit(ssc_nd.reshape(-1, 1))
                order = np.argsort(g1.means_.ravel())
                mu = g1.means_.ravel()[order]
                if mu[1] - mu[0] > 0.04:
                    # Bias toward the low mode (not the midpoint) — lymph gate is tight
                    ssc_hi = float(mu[0] + 0.35 * (mu[1] - mu[0]))
            except Exception:
                pass
        if ssc_hi is None:
            ssc_hi = float(np.percentile(ssc_nd, 30)) if len(ssc_nd) else 0.5

    if ssc_cap is not None:
        ssc_hi = min(ssc_hi, float(ssc_cap))

    if fsc_hi is None:
        pool = non_debris & (ssca < ssc_hi)
        fsc_hi = float(np.percentile(fsca[pool], 97)) if pool.sum() >= 200 else float(np.percentile(fsca, 90))

    mask = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < ssc_hi)
    return mask, float(fsc_lo), float(fsc_hi), float(ssc_hi)

def singlet_mask(fsca, fsch, parent):
    """FSC-A vs FSC-H singlets within parent gate."""
    r = fsch / np.maximum(fsca, 1e-9)
    r_p = r[parent]
    if r_p.size < 50:
        return parent.copy()
    md = np.median(r_p)
    mad = np.median(np.abs(r_p - md)) + 1e-9
    return parent & (r > md - 4 * mad) & (r < md + 4 * mad)


def refine_ssc_by_purity(fsca, fsch, ssca, g, fsc_lo, fsc_hi, ssc_hi_init,
                         t_ld=None, t_cd45=None, min_n=600, n_grid=14):
    """Scan SSC-A to maximize lineage purity (B+T+NK of CD14-)."""
    ssc_hi_init = float(ssc_hi_init)
    lo = max(0.045, 0.30 * ssc_hi_init)
    hi = max(ssc_hi_init * 1.05, lo + 0.02)
    best_shi, best_score = ssc_hi_init, -1.0

    for shi in np.linspace(lo, hi, n_grid):
        lymph = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < shi)
        sing = singlet_mask(fsca, fsch, lymph)
        live = sing & (g["ld"] < t_ld) if (t_ld is not None and "ld" in g) else sing
        work = live & (g["cd45"] > t_cd45) if (t_cd45 is not None and "cd45" in g) else live
        if work.sum() < min_n or "cd3" not in g or "cd56" not in g:
            continue

        t14 = None
        if "cd14" in g:
            t14 = gmm_valley(g["cd14"][work]) or pct_cut(g["cd14"][work], 95)
        cd14n = work & (g["cd14"] < t14) if t14 is not None else work
        if cd14n.sum() < min_n:
            continue

        t3 = robust_cd3_cut(g["cd3"][cd14n], g["cd56"][cd14n])
        t56 = robust_cd56_cut(g["cd56"][cd14n], g["cd3"][cd14n], t3)
        if t56 is None and cd14n.sum() >= 100:
            pool = g["cd56"][cd14n]
            if t3 is not None:
                neg = g["cd56"][cd14n & (g["cd3"] < t3)]
                if len(neg) >= 80:
                    pool = neg
            t56 = (float(np.percentile(pool, 10)) if float(np.percentile(pool, 50)) > 1200
                   else (valley_asinh(pool, lo=5, hi=40) or pct_cut(pool, 20)))
        if t3 is None or t56 is None:
            continue

        t19 = robust_cd19_cut(g["cd19"][cd14n]) if "cd19" in g else None
        cd19n = cd14n & (g["cd19"] < t19) if t19 is not None else cd14n
        nB = int((cd14n & (g["cd19"] >= t19)).sum()) if t19 is not None else 0
        nT = int((cd19n & (g["cd3"] > t3) & (g["cd56"] < t56)).sum())
        nNK = int((cd19n & (g["cd3"] < t3) & (g["cd56"] > t56)).sum())
        purity = (nB + nT + nNK) / max(int(cd14n.sum()), 1)
        score = purity + 0.05 * min(int(cd14n.sum()), 15000) / 15000.0
        if score > best_score:
            best_score, best_shi = score, float(shi)

    return best_shi, best_score


def robust_cd4_cd8_cuts(cd4_vals, cd8_vals):
    """CD4/CD8 cuts within T. If CD8 unimodal, cd8_from_cd4_neg=True."""
    t_cd4 = gmm_valley(cd4_vals, min_gap=100, min_n=300)
    if t_cd4 is None:
        t_cd4 = valley_asinh(cd4_vals, lo=10, hi=70)
    t_cd8 = gmm_valley(cd8_vals, min_gap=100, min_n=300)
    cd8_from_cd4_neg = t_cd8 is None
    return t_cd4, t_cd8, cd8_from_cd4_neg


def file_signals(df, ch):
    g = {k: df[v].values for k, v in ch.items() if v in df.columns}
    for opt, name in CH_OPT.items():
        if name in df.columns:
            g[opt] = df[name].values
    fsca = df["FSC-A"].values
    fsch = df["FSC-H"].values if "FSC-H" in df.columns else fsca.copy()
    ssca = df["SSC-A"].values
    return g, fsca, fsch, ssca


def file_cuts(df, ch, ssc_cap=None, refine_ssc=True, locked_scatter=None):
    """Derive cutoffs: FSA×SSA (+purity refine) → singlet → live → CD45 → lineage.

    locked_scatter: optional dict with fsc_lo/fsc_hi/ssc_hi from operator anchor
    (Nature-style transfer — skip adaptive scatter, re-estimate fluor inside lock).
    """
    g, fsca, fsch, ssca = file_signals(df, ch)
    if locked_scatter is not None:
        fsc_lo = float(locked_scatter["fsc_lo"])
        fsc_hi = float(locked_scatter["fsc_hi"])
        ssc_hi = float(locked_scatter["ssc_hi"])
        if ssc_cap is not None:
            ssc_hi = min(ssc_hi, float(ssc_cap))
        lymph = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < ssc_hi)
        refine_ssc = False
    else:
        lymph, fsc_lo, fsc_hi, ssc_hi = lymph_scatter_gate(fsca, ssca, ssc_cap=ssc_cap)
    sing = singlet_mask(fsca, fsch, lymph)

    t_ld = robust_ld_cut(g["ld"][sing]) if "ld" in g else None
    live = sing & (g["ld"] < t_ld) if t_ld is not None else sing.copy()
    t_cd45 = robust_cd45_cut(g["cd45"][live]) if "cd45" in g else None

    if refine_ssc:
        ssc_hi, _ = refine_ssc_by_purity(
            fsca, fsch, ssca, g, fsc_lo, fsc_hi, ssc_hi,
            t_ld=t_ld, t_cd45=t_cd45,
        )
        if ssc_cap is not None:
            ssc_hi = min(ssc_hi, float(ssc_cap))
        lymph = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < ssc_hi)
        sing = singlet_mask(fsca, fsch, lymph)
        live = sing & (g["ld"] < t_ld) if t_ld is not None else sing.copy()

    cd45p = live & (g["cd45"] > t_cd45) if t_cd45 is not None else live.copy()

    t_cd14 = gmm_valley(g["cd14"][cd45p]) if "cd14" in g else None
    if t_cd14 is not None and "cd14" in g:
        frac = float(np.mean(g["cd14"][cd45p] > t_cd14))
        # Monocytes are usually a minority of CD45+; reject over-inclusive cuts
        if not (0.01 <= frac <= 0.45):
            t_cd14 = None
    if t_cd14 is None and "cd14" in g and cd45p.sum() >= 100:
        t_cd14 = valley_asinh(g["cd14"][cd45p], lo=80, hi=99.5) or pct_cut(g["cd14"][cd45p], 97)
        if t_cd14 is not None and float(np.mean(g["cd14"][cd45p] > t_cd14)) > 0.45:
            t_cd14 = pct_cut(g["cd14"][cd45p], 97)
    # Cap CD14+ of live (~monocytes); engrafted samples can look CD14-dim/noisy
    if t_cd14 is not None and "cd14" in g and live.sum() >= 200:
        frac_live = float(np.mean(g["cd14"][live] >= t_cd14))
        if frac_live > 0.15:
            t_cd14 = pct_cut(g["cd14"][live], 97)
    cd14n = cd45p & (g["cd14"] < t_cd14) if t_cd14 is not None else cd45p.copy()

    t_cd19 = robust_cd19_cut(g["cd19"][cd14n]) if "cd19" in g else None
    cd19n = cd14n & (g["cd19"] < t_cd19) if t_cd19 is not None else cd14n.copy()

    t_cd3 = robust_cd3_cut(g["cd3"][cd19n], g["cd56"][cd19n]) if "cd3" in g else None
    t_cd56 = (
        robust_cd56_cut(g["cd56"][cd19n], g["cd3"][cd19n], t_cd3)
        if "cd56" in g else None
    )
    if t_cd56 is None and "cd56" in g and cd19n.sum() >= 100:
        pool = g["cd56"][cd19n]
        if t_cd3 is not None:
            neg = g["cd56"][cd19n & (g["cd3"] < t_cd3)]
            if len(neg) >= 80:
                pool = neg
        t_cd56 = (float(np.percentile(pool, 10)) if float(np.percentile(pool, 50)) > 1200
                  else (valley_asinh(pool, lo=5, hi=40) or pct_cut(pool, 20)))

    t_cd4 = t_cd8 = None
    cd8_from_cd4_neg = False
    if t_cd3 is not None and t_cd56 is not None and "cd4" in g:
        Tc = cd19n & (g["cd3"] > t_cd3) & (g["cd56"] < t_cd56)
        if Tc.sum() >= 400:
            if "cd8" in g:
                t_cd4, t_cd8, cd8_from_cd4_neg = robust_cd4_cd8_cuts(
                    g["cd4"][Tc], g["cd8"][Tc])
            else:
                t_cd4 = gmm_valley(g["cd4"][Tc], min_gap=100, min_n=300) or valley_asinh(
                    g["cd4"][Tc], lo=10, hi=70)

    return dict(
        ld=t_ld, cd45=t_cd45, cd14=t_cd14, cd19=t_cd19,
        cd3=t_cd3, cd56=t_cd56, cd4=t_cd4, cd8=t_cd8,
        cd8_from_cd4_neg=cd8_from_cd4_neg,
        fsc_lo=fsc_lo, fsc_hi=fsc_hi, ssc_hi=ssc_hi,
    )


def gate_with(df, C, ch, ssc_cap=None):
    """Apply cuts. Lymph denom = CD45+ CD14−. CD14+ monocytes reported when present."""
    g, fsca, fsch, ssca = file_signals(df, ch)

    fsc_lo, fsc_hi, ssc_hi = C.get("fsc_lo"), C.get("fsc_hi"), C.get("ssc_hi")
    if fsc_lo is None or fsc_hi is None or ssc_hi is None:
        lymph, fsc_lo, fsc_hi, ssc_hi = lymph_scatter_gate(fsca, ssca, ssc_cap=ssc_cap)
    else:
        if ssc_cap is not None:
            ssc_hi = min(ssc_hi, float(ssc_cap))
        lymph = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < ssc_hi)

    sing = singlet_mask(fsca, fsch, lymph)
    live = sing & (g["ld"] < C["ld"]) if (C.get("ld") is not None and "ld" in g) else sing.copy()
    cd45p = live & (g["cd45"] > C["cd45"]) if (C.get("cd45") is not None and "cd45" in g) else live.copy()

    if C.get("cd14") is not None and "cd14" in g:
        CD14p = cd45p & (g["cd14"] >= C["cd14"])
        cd14n = cd45p & (g["cd14"] < C["cd14"])
    else:
        CD14p = np.zeros(len(df), bool)
        cd14n = cd45p.copy()

    mono_ssc = cd45p & (ssca >= ssc_hi) & (ssca < min(max(ssc_hi * 1.8, ssc_hi + 0.05), 0.95))
    mono = CD14p | mono_ssc
    work = cd14n  # lymphocytes = CD45+ CD14−

    if C.get("cd19") is not None and "cd19" in g:
        cd19n = work & (g["cd19"] < C["cd19"])
        B = work & (g["cd19"] >= C["cd19"])
    else:
        cd19n = work.copy()
        B = np.zeros(len(df), bool)

    if C.get("cd3") is None or C.get("cd56") is None:
        NK = np.zeros(len(df), bool)
        Tc = np.zeros(len(df), bool)
    else:
        NK = cd19n & (g["cd3"] < C["cd3"]) & (g["cd56"] > C["cd56"])
        Tc = cd19n & (g["cd3"] > C["cd3"]) & (g["cd56"] < C["cd56"])

    hla_dim = bool(C.get("hla_dim"))
    if C.get("hla") is not None and "hla" in g:
        hla_donor = (g["hla"] < C["hla"]) if hla_dim else (g["hla"] > C["hla"])
        Donor = NK & hla_donor
    else:
        hla_donor = np.zeros(len(df), bool)
        Donor = np.zeros(len(df), bool)

    if C.get("car") is not None and C.get("hla") is not None and "car" in g:
        CAR = NK & hla_donor & (g["car"] > C["car"])
    else:
        CAR = np.zeros(len(df), bool)

    masks = dict(
        lymph_scatter=lymph, sing=sing, live=live, cd45p=cd45p,
        lympho=work, mono=mono, CD14p=CD14p, cd14n=cd14n, cd19n=cd19n,
        B=B, NK=NK, T=Tc, Donor=Donor, CAR=CAR,
    )
    cuts = dict(
        ld=C.get("ld"), cd45=C.get("cd45"), cd14=C.get("cd14"), cd19=C.get("cd19"),
        cd3=C.get("cd3"), cd56=C.get("cd56"), hla=C.get("hla"), car=C.get("car"),
        fsc_lo=fsc_lo, fsc_hi=fsc_hi, ssc_hi=ssc_hi, hla_dim=hla_dim,
        cd8_from_cd4_neg=bool(C.get("cd8_from_cd4_neg")),
    )

    if "cd4" in g and C.get("cd4") is not None:
        masks["CD4"] = Tc & (g["cd4"] > C["cd4"])
        cuts["cd4"] = C["cd4"]
        if C.get("cd8") is not None and "cd8" in g and not C.get("cd8_from_cd4_neg"):
            masks["CD8"] = Tc & (g["cd8"] > C["cd8"]) & (g["cd4"] <= C["cd4"])
            cuts["cd8"] = C["cd8"]
        else:
            masks["CD8"] = Tc & (g["cd4"] <= C["cd4"])
            cuts["cd8"] = None
            cuts["cd8_from_cd4_neg"] = True

    return masks, g, (fsca, ssca), cuts


MIN_PARENT = 50
MIN_POS = 10


def percentages(m):
    """B/T/NK/CD4/CD8 of CD45+CD14− lymph; CD14+ of CD45+ when present."""
    def pc(a, b):
        return round(100.0 * a / b, 3) if b else 0.0

    lymph = int(m["lympho"].sum())
    cd45 = int(m["cd45p"].sum())
    cd19n = int(m["cd19n"].sum())
    nk = int(m["NK"].sum())
    donor = int(m["Donor"].sum())
    car = int(m["CAR"].sum())
    live = int(m["live"].sum())
    sing = int(m["sing"].sum())
    scat = int(m["lymph_scatter"].sum())

    out = {
        "%Lymph_scatter (of total)": pc(scat, len(m["lymph_scatter"])),
        "%Singlets (of scatter)": pc(sing, scat),
        "%Live (of singlets)": pc(live, sing),
        "%CD45+ (of live)": pc(cd45, live),
        "%Lymphocytes (of CD45+)": pc(lymph, cd45),
        "%Lymphocytes (of live)": pc(lymph, live),
        "%B (of lymph)": pc(int(m["B"].sum()), lymph),
        "%T (of lymph)": pc(int(m["T"].sum()), lymph),
        "%NK (of lymph)": pc(nk, lymph),
        # Manual QC reports Donor NK on the lymph denominator (same as NK/B/T)
        "%Donor NK (of lymph)": pc(donor, lymph),
        "%Donor NK (of NK)": pc(donor, nk) if nk >= MIN_PARENT else 0.0,
        "%CAR+ (of Donor NK)": pc(car, donor) if donor >= MIN_PARENT else 0.0,
        "%B (of CD14-)": pc(int(m["B"].sum()), int(m["cd14n"].sum())),
        "%T (of CD19-)": pc(int(m["T"].sum()), cd19n),
        "%NK (of CD19-)": pc(nk, cd19n),
        "lineage_purity": pc(int(m["B"].sum()) + int(m["T"].sum()) + nk, lymph),
        "donor_reliable": nk >= MIN_PARENT and donor >= MIN_POS,
        "car_reliable": donor >= MIN_PARENT and car >= MIN_POS,
    }
    if "CD14p" in m:
        out["%CD14+ (of CD45+)"] = pc(int(m["CD14p"].sum()), cd45)
        out["%CD14+ (of live)"] = pc(int(m["CD14p"].sum()), live)
        out["%Monocytes (of live)"] = out["%CD14+ (of live)"]
        out["n_CD14p"] = int(m["CD14p"].sum())
    if "mono" in m:
        out["%Mono (of CD45+)"] = pc(int(m["mono"].sum()), cd45)
    if "CD4" in m:
        out["%CD4 (of lymph)"] = pc(int(m["CD4"].sum()), lymph)
        out["%CD4 (of T)"] = pc(int(m["CD4"].sum()), int(m["T"].sum()))
    if "CD8" in m:
        out["%CD8 (of lymph)"] = pc(int(m["CD8"].sum()), lymph)
        out["%CD8 (of T)"] = pc(int(m["CD8"].sum()), int(m["T"].sum()))
    return out


def calibrate_patient(loaded, ch, ssc_cap=None, locked_scatter=None):
    """loaded: list of (fname, df). Median of per-file valleys."""
    perfile = {}
    keys = ["ld", "cd45", "cd14", "cd19", "cd3", "cd56", "cd4", "cd8",
            "fsc_lo", "fsc_hi", "ssc_hi"]
    valid = {k: [] for k in keys}
    for fn, df in loaded:
        c = file_cuts(df, ch, ssc_cap=ssc_cap, locked_scatter=locked_scatter)
        perfile[fn] = c
        for k in keys:
            if c.get(k) is not None:
                valid[k].append(c[k])
    med = {k: (float(np.median(v)) if v else None) for k, v in valid.items()}
    return perfile, med
