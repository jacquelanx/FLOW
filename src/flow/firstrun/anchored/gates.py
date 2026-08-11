#!/usr/bin/env python3
"""
Adaptive gating — hierarchical, literature-aligned.

Hierarchy (biological meaning)
------------------------------
  1. Viable cells   FSC-A × SSC-A density island (connected high-density
                    cellular cloud; debris excluded). UNITO Nat Commun 2025;
                    flowDensity / OpenCyto — NOT a rectangle or quadrant.
  2. Singlets       SSC-W × SSC-H within viable (Rico Cytometry A 2023).
  3. Live           UV L/D− within singlets.
  4. CD45+          Leukocytes within live.
  5. CD14±          Monocytes (CD14+) vs lymph parent (CD45+ CD14−).
                    Grans are CD14−/dim — lymph reporting also applies a
                    low-SSC lymph ceiling (flowDensity / OpenCyto lymph island)
                    so the panel-1 viable cloud stays generous while %B/T/NK
                    use a clean lymph denominator.
  6. CD19           B = CD19+ of CD14− lymph; non-B continue.
  7. CD3 × CD56     T (CD3+ CD56−) vs NK (CD3− CD56+) on CD19−.
  8. CD4 / CD8      within T (CD8 := CD4− when CD8 unimodal).
  9. Donor HLA / CAR within NK.

Lymph denominator for B/T/NK/Donor = CD45+ CD14− ∩ low-SSC lymph island.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter, label as nd_label

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


def robust_cd14_cut(vals, ssca=None, max_frac=0.40):
    """CD14 monocyte cut within CD45+.

    Prefer a GMM / asinh valley so CD14− lymph denom stays clean. When SSC is
    available, seed the cut from the mid/high-SSC monocyte cloud (grans are
    CD14−/dim — fluorescence alone cannot drop them).
    """
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    if ok.sum() < 100:
        return None

    seed = vals[ok]
    if ssca is not None:
        ssca = np.asarray(ssca, float)
        mid = ok & np.isfinite(ssca) & (ssca >= 0.12)
        if mid.sum() >= 200:
            seed = vals[mid]

    v = gmm_valley(seed, min_gap=80, min_n=300)
    if v is not None:
        frac = float(np.mean(vals[ok] > v))
        if 0.01 <= frac <= max_frac:
            return float(v)

    v2 = valley_asinh(seed, lo=55, hi=99.0)
    if v2 is not None:
        frac = float(np.mean(vals[ok] > v2))
        if 0.01 <= frac <= max_frac:
            return float(v2)

    # Rare-mono fallback: high percentile of all CD45+
    v3 = pct_cut(vals[ok], 92)
    if v3 is not None and float(np.mean(vals[ok] > v3)) <= max_frac:
        return float(v3)
    return pct_cut(vals[ok], 97)


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


# ── FSA × SSA lymphocyte gate (FIRST) — heat-density ──────────────────────────

def heat_density_2d(fsca, ssca, bins=220, smooth=2.2, subsample=120_000, seed=0,
                    range_=None):
    """
    Build a smoothed FSC×SSC heat-density grid (like a FlowJo density plot).
    Returns H[ny,nx], xc, yc, xe, ye (bin edges).

    range_ : optional ((xmin, xmax), (ymin, ymax)) so the grid spans a full
    display / instrument scale instead of a tight data percentile window.
    """
    fsca = np.asarray(fsca, float)
    ssca = np.asarray(ssca, float)
    ok = np.isfinite(fsca) & np.isfinite(ssca)
    x, y = fsca[ok], ssca[ok]
    if len(x) < 50:
        return None
    if len(x) > subsample:
        idx = np.random.RandomState(seed).choice(len(x), subsample, replace=False)
        x, y = x[idx], y[idx]
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=range_)
    H = gaussian_filter(H.T.astype(float), sigma=smooth)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H, xc, yc, xe, ye


def _events_in_grid_mask(fsca, ssca, xe, ye, mask2d):
    """Map events into a boolean density-grid mask."""
    ix = np.clip(np.digitize(fsca, xe) - 1, 0, mask2d.shape[1] - 1)
    iy = np.clip(np.digitize(ssca, ye) - 1, 0, mask2d.shape[0] - 1)
    # digitize edges: values == xe[-1] go to len(xe); clip handles it
    return mask2d[iy, ix]


def _ssc_valley_above_peak(H, yc, peak_iy, peak_ix, search_half=8):
    """Find density valley on SSC above the lymph peak (toward mono/gran)."""
    ny, nx = H.shape
    x0 = max(0, peak_ix - search_half)
    x1 = min(nx, peak_ix + search_half + 1)
    profile = H[:, x0:x1].mean(axis=1)
    # Search upward from peak for a local minimum before the next rise
    best = None
    for i in range(peak_iy + 2, min(ny - 2, peak_iy + max(8, ny // 3))):
        if profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]:
            # Prefer valleys that sit between two higher shoulders
            left = float(np.max(profile[peak_iy:i + 1]))
            right = float(np.max(profile[i:min(ny, i + 12)]))
            if left > profile[i] * 1.15 and right > profile[i] * 1.05:
                best = float(yc[i])
                break
    return best


def _density_lymph_peak(fsca, ssca, fsc_lo_pct=2.0):
    """Locate the low-SSC lymph density peak. Returns dens grid + peak info or None."""
    from scipy.ndimage import maximum_filter

    fsca = np.asarray(fsca, float)
    ssca = np.asarray(ssca, float)
    dens = heat_density_2d(fsca, ssca)
    if dens is None:
        return None
    H, xc, yc, xe, ye = dens
    ny, nx = H.shape
    ok = np.isfinite(fsca) & np.isfinite(ssca)
    Xc, Yc = np.meshgrid(xc, yc)

    fsc_lo_seed = max(float(np.percentile(fsca[ok], max(fsc_lo_pct, 5))), 0.10)
    fsc_hi_seed = float(np.percentile(fsca[ok], 95))
    ssc_hi_search = max(0.30, float(np.percentile(ssca[ok], 70)))
    search = (
        (Yc >= 0.045) & (Yc <= ssc_hi_search)
        & (Xc >= fsc_lo_seed) & (Xc <= fsc_hi_seed)
        & (H > 0)
    )
    if not search.any():
        search = (Yc >= 0.03) & (Yc <= ssc_hi_search) & (H > 0)

    Hr = np.where(search, H, 0.0)
    if Hr.max() <= 0:
        return None

    footprint = max(9, min(ny, nx) // 22)
    if footprint % 2 == 0:
        footprint += 1
    local_max = (Hr == maximum_filter(Hr, size=footprint)) & (Hr > 0)
    candidates = []
    cell_mass = float(Hr.sum()) + 1e-12
    for iy, ix in np.argwhere(local_max):
        fx, sy = float(xc[ix]), float(yc[iy])
        if sy < 0.045 or fx < fsc_lo_seed:
            continue
        y0, y1 = max(0, iy - footprint), min(ny, iy + footprint + 1)
        x0, x1 = max(0, ix - footprint), min(nx, ix + footprint + 1)
        mass = float(Hr[y0:y1, x0:x1].sum())
        if mass < 0.005 * cell_mass:
            continue
        score = mass * (0.35 + fx) / (1.0 + 6.0 * max(0.0, sy - 0.05))
        candidates.append((score, mass, iy, ix, float(Hr[iy, ix]), fx, sy))

    if not candidates:
        peak_iy, peak_ix = np.unravel_index(np.argmax(Hr), Hr.shape)
        peak_val = float(Hr[peak_iy, peak_ix])
    else:
        off_wall = [c for c in candidates if c[5] >= fsc_lo_seed + 0.08]
        pool = off_wall or candidates
        lymph_like = [c for c in pool if c[6] <= 0.20]
        pool = lymph_like or pool
        pool.sort(reverse=True)
        _, _, peak_iy, peak_ix, peak_val, _, _ = pool[0]

    return dict(
        H=H, xc=xc, yc=yc, xe=xe, ye=ye, Xc=Xc, Yc=Yc, ok=ok,
        peak_iy=peak_iy, peak_ix=peak_ix, peak_val=peak_val,
        peak_ssc=float(yc[peak_iy]), peak_fsc=float(xc[peak_ix]),
        fsc_lo_seed=fsc_lo_seed, fsc_hi_seed=fsc_hi_seed,
    )


def _bbox_from_mask(fsca, ssca, mask, fallback=(0.05, 0.95, 0.32)):
    """Descriptive extents of a density island (metadata only — not the gate)."""
    if mask is None or not np.any(mask):
        return fallback
    x = np.asarray(fsca, float)[mask]
    y = np.asarray(ssca, float)[mask]
    if len(x) < 20:
        return fallback
    flo = float(np.percentile(x, 1))
    fhi = float(np.percentile(x, 99))
    shi = float(np.percentile(y, 99))
    return flo, fhi, shi


def _connected_density_island(H, xe, ye, Xc, Yc, seed_iy, seed_ix, level,
                              *, ssc_lo=0.02, ssc_hi=None, fsc_lo=None,
                              merge_mass_frac=0.12, max_extra=2):
    """Threshold density → connected component(s) seeded at the cellular peak.

    Returns boolean grid mask. Never a rectangle: the gate follows the contour.
    """
    core = (H >= level) & (Yc >= ssc_lo) & np.isfinite(H)
    if ssc_hi is not None:
        core &= (Yc <= float(ssc_hi))
    if fsc_lo is not None:
        core &= (Xc >= float(fsc_lo))
    if not core.any():
        return core

    labeled, nlab = nd_label(core)
    lab = int(labeled[seed_iy, seed_ix])
    if lab <= 0:
        # Seed fell just outside — take nearest labeled cell, else largest
        iy, ix = np.unravel_index(np.argmax(np.where(core, H, 0.0)), H.shape)
        lab = int(labeled[iy, ix])
    if lab <= 0:
        return core

    island = labeled == lab
    main_mass = float(H[island].sum()) + 1e-12
    if merge_mass_frac > 0 and nlab > 1 and max_extra > 0:
        extras = []
        for k in range(1, nlab + 1):
            if k == lab:
                continue
            comp = labeled == k
            mass = float(H[comp].sum())
            if mass < merge_mass_frac * main_mass:
                continue
            cy = float(np.average(Yc[comp], weights=H[comp] + 1e-12))
            cx = float(np.average(Xc[comp], weights=H[comp] + 1e-12))
            # Only merge other cellular clouds (not debris floor)
            if cx < 0.06 or cy < ssc_lo:
                continue
            if ssc_hi is not None and cy > float(ssc_hi):
                continue
            extras.append((mass, k))
        extras.sort(reverse=True)
        for _, k in extras[:max_extra]:
            island |= (labeled == k)
    return island


def lymph_scatter_gate(fsca, ssca, ssc_cap=None, fsc_lo_pct=2.0,
                       density_frac=0.22):
    """
    Gate *lymphocytes* on FSC-A × SSC-A as a density island (UNITO / flowDensity).

    Build 2D heat density → threshold → connected component around the low-SSC
    lymph peak. SSC stops at the density valley toward monocytes (≤0.32).
    Returns (mask, fsc_lo, fsc_hi, ssc_hi) where flo/fhi/shi are descriptive
    extents of the island (soft-lock metadata), not a rectangle gate.
    """
    fsca = np.asarray(fsca, float)
    ssca = np.asarray(ssca, float)
    n = len(fsca)
    if n < 50:
        return np.ones(n, bool), 0.0, 1.0, 1.0

    peak = _density_lymph_peak(fsca, ssca, fsc_lo_pct=fsc_lo_pct)
    if peak is None:
        flo = float(np.percentile(fsca, fsc_lo_pct))
        shi = float(np.percentile(ssca, 35))
        fhi = float(np.percentile(fsca, 90))
        # Last-resort fallback still prefers a soft density-like band, not quads
        dens = heat_density_2d(fsca, ssca)
        if dens is None:
            mask = (fsca > flo) & (fsca < fhi) & (ssca < shi) & (ssca > 0.02)
            return mask, flo, fhi, shi
        H, xc, yc, xe, ye = dens
        ok = np.isfinite(fsca) & np.isfinite(ssca)
        level = max(float(H.max()) * 0.25, 1e-9)
        Xc, Yc = np.meshgrid(xc, yc)
        core = (H >= level) & (Yc <= shi) & (Yc >= 0.02) & (Xc >= flo * 0.8)
        labeled, _ = nd_label(core)
        iy, ix = np.unravel_index(np.argmax(np.where(core, H, 0.0)), H.shape)
        lab = int(labeled[iy, ix])
        island = labeled == lab if lab > 0 else core
        mask = _events_in_grid_mask(fsca, ssca, xe, ye, island) & ok
        return mask, *_bbox_from_mask(fsca, ssca, mask, (flo, fhi, shi))

    H, xc, yc, xe, ye = peak["H"], peak["xc"], peak["yc"], peak["xe"], peak["ye"]
    Xc, Yc, ok = peak["Xc"], peak["Yc"], peak["ok"]
    peak_iy, peak_ix = peak["peak_iy"], peak["peak_ix"]
    peak_val, peak_ssc = peak["peak_val"], peak["peak_ssc"]
    ny, nx = H.shape
    debris_fsc = max(0.04, float(peak["fsc_lo_seed"]) * 0.7)

    x0b = max(0, peak_ix - 6)
    x1b = min(nx, peak_ix + 7)
    profile = H[:, x0b:x1b].mean(axis=1)

    ssc_hi = None
    valley = _ssc_valley_above_peak(H, yc, peak_iy, peak_ix, search_half=10)
    if valley is not None and valley > peak_ssc + 0.015:
        ssc_hi = valley
    if ssc_hi is None:
        target = peak_val * 0.20
        for i in range(peak_iy + 1, min(ny - 1, peak_iy + ny // 2)):
            if profile[i] <= target:
                ssc_hi = float(yc[i])
                break
    if ssc_hi is None:
        ssc_hi = peak_ssc + 0.07
    ssc_hi = float(np.clip(ssc_hi, peak_ssc + 0.028, 0.32))
    if ssc_cap is not None:
        ssc_hi = min(ssc_hi, float(ssc_cap))

    level = peak_val * density_frac
    island = _connected_density_island(
        H, xe, ye, Xc, Yc, peak_iy, peak_ix, level,
        ssc_lo=max(0.02, peak_ssc - 0.06), ssc_hi=ssc_hi, fsc_lo=debris_fsc,
        merge_mass_frac=0.0, max_extra=0,
    )
    if not island.any():
        island = _connected_density_island(
            H, xe, ye, Xc, Yc, peak_iy, peak_ix, peak_val * 0.12,
            ssc_lo=0.02, ssc_hi=ssc_hi, fsc_lo=debris_fsc,
            merge_mass_frac=0.0, max_extra=0,
        )
    dens_mask = _events_in_grid_mask(fsca, ssca, xe, ye, island) & ok & (ssca <= ssc_hi)
    flo, fhi, _ = _bbox_from_mask(
        fsca, ssca, dens_mask,
        (peak["peak_fsc"] - 0.08, peak["peak_fsc"] + 0.15, ssc_hi),
    )
    return dens_mask, float(flo), float(fhi), float(ssc_hi)


def viable_scatter_gate(fsca, ssca, fsc_lo_pct=2.0, density_frac=0.08):
    """
    Panel-1 / cleanup gate: density-island cellular cloud (lymph + mono + gran).

    UNITO / flowDensity style — NOT a FSC×SSC rectangle or quadrant:
      1) build 2D heat density on FSC-A × SSC-A
      2) threshold at a fraction of the cellular peak
      3) keep the connected high-density island(s), excluding debris floor

    Lineage reporting still uses lymph_scatter_gate (tighter low-SSC island).
    Returns (mask, fsc_lo, fsc_hi, ssc_hi) with flo/fhi/shi = island extents.
    """
    fsca = np.asarray(fsca, float)
    ssca = np.asarray(ssca, float)
    n = len(fsca)
    if n < 50:
        return np.ones(n, bool), 0.0, 1.0, 1.0

    ok = np.isfinite(fsca) & np.isfinite(ssca)
    peak = _density_lymph_peak(fsca, ssca, fsc_lo_pct=fsc_lo_pct)
    if peak is None:
        dens = heat_density_2d(fsca, ssca)
        if dens is None:
            flo = max(0.04, float(np.percentile(fsca[ok], fsc_lo_pct)))
            fhi = float(np.percentile(fsca[ok], 99))
            shi = min(0.80, float(np.percentile(ssca[ok], 97)))
            mask = (fsca > flo) & (fsca < fhi) & (ssca < shi) & (ssca > 0.02) & ok
            return mask, flo, fhi, shi
        H, xc, yc, xe, ye = dens
        Xc, Yc = np.meshgrid(xc, yc)
        peak_iy, peak_ix = np.unravel_index(np.argmax(H), H.shape)
        peak_val = float(H[peak_iy, peak_ix])
        debris_fsc = max(0.04, float(np.percentile(fsca[ok], max(fsc_lo_pct, 5))))
    else:
        H, xc, yc, xe, ye = peak["H"], peak["xc"], peak["yc"], peak["xe"], peak["ye"]
        Xc, Yc = peak["Xc"], peak["Yc"]
        peak_iy, peak_ix = peak["peak_iy"], peak["peak_ix"]
        peak_val = peak["peak_val"]
        debris_fsc = max(0.04, float(peak["fsc_lo_seed"]) * 0.65)

    ssc_hi_cap = float(np.clip(np.percentile(ssca[ok], 98), 0.40, 0.85))
    level = max(peak_val * density_frac, float(H[H > 0].min()) if (H > 0).any() else 1e-9)
    island = _connected_density_island(
        H, xe, ye, Xc, Yc, peak_iy, peak_ix, level,
        ssc_lo=0.02, ssc_hi=ssc_hi_cap, fsc_lo=debris_fsc,
        merge_mass_frac=0.10, max_extra=2,
    )
    if not island.any() or float(H[island].sum()) < 0.02 * float(H.sum() + 1e-12):
        island = _connected_density_island(
            H, xe, ye, Xc, Yc, peak_iy, peak_ix, peak_val * 0.05,
            ssc_lo=0.02, ssc_hi=ssc_hi_cap, fsc_lo=debris_fsc,
            merge_mass_frac=0.15, max_extra=3,
        )
    dens_mask = _events_in_grid_mask(fsca, ssca, xe, ye, island) & ok
    flo, fhi, shi = _bbox_from_mask(
        fsca, ssca, dens_mask, (0.05, 0.95, ssc_hi_cap))
    return dens_mask, float(flo), float(fhi), float(shi)


def soft_lock_scatter(fsca, ssca, locked, density_frac=0.22):
    """
    Soft reference transfer for *lymph* FSA×SSA (HIPC / OpenCyto-style).

    Hard-locking Baseline FSC fails after engraftment when lymph FSC shifts
    (D28 peak can sit entirely left of a Baseline lock). Literature practice
    is hierarchical templates with data-driven gates per sample (Finak 2014
    OpenCyto; Finak 2016 HIPC Sci Rep) plus optional reference SSC ceiling.

    Policy
    ------
      • SSC_hi: max(locked.ssc_hi, density lymph ssc) clipped to ≤0.32
      • Mask: density island re-estimated per file, clipped to SSC_hi
        (NOT a hard FSC×SSC rectangle)
      • Panel-1 viable cloud is separate (viable_scatter_gate)

    Returns (mask, fsc_lo, fsc_hi, ssc_hi, meta).
    """
    locked = {k: float(locked[k]) for k in ("fsc_lo", "fsc_hi", "ssc_hi")}
    dens_mask, flo, fhi, shi = lymph_scatter_gate(
        fsca, ssca, density_frac=density_frac)
    ssc_hi = float(np.clip(max(locked["ssc_hi"], shi), locked["ssc_hi"], 0.32))
    # Re-estimate density island under the soft-locked SSC ceiling so the
    # contour can expand/contract with engraftment FSC shifts.
    dens_mask2, flo2, fhi2, shi2 = lymph_scatter_gate(
        fsca, ssca, ssc_cap=ssc_hi, density_frac=density_frac)
    mask = dens_mask2 & (ssca <= ssc_hi) & np.isfinite(fsca) & np.isfinite(ssca)
    if mask.sum() < 200:
        mask = dens_mask & (ssca <= ssc_hi) & np.isfinite(fsca) & np.isfinite(ssca)
        flo2, fhi2 = flo, fhi
    meta = dict(
        mode="soft_lock_density",
        locked_ssc_hi=locked["ssc_hi"],
        density_ssc_hi=float(shi),
        density_fsc=(float(flo2), float(fhi2)),
    )
    return mask, float(flo2), float(fhi2), float(ssc_hi), meta


def singlet_mask(sscw, ssch, parent, n_mad=6.0, mode="percentile"):
    """SSC-W × SSC-H singlets within parent gate.

    Panel convention: width vs height on SSC (Rico et al. Cytometry A 2023;
    Bio-Rad / ISAC doublet guidance). Doublets elevate SSC-W while SSC-H stays
    similar — so a harsh symmetric MAD box on both axes over-rejects.

    Modes
    -----
      percentile (default): keep W ≤ p99 of parent and H in [p0.5, p99.5]
      mad: asymmetric MAD — looser on height, tighter on high-W only
    """
    if sscw is None or ssch is None:
        return parent.copy()
    w = np.asarray(sscw, float)
    h = np.asarray(ssch, float)
    n = int(parent.sum())
    if n < 50:
        return parent.copy()
    wp, hp = w[parent], h[parent]
    ok = np.isfinite(wp) & np.isfinite(hp)
    if ok.sum() < 50:
        return parent.copy()
    wp, hp = wp[ok], hp[ok]

    if mode == "percentile":
        w_hi = float(np.percentile(wp, 99.0))
        h_lo, h_hi = [float(x) for x in np.percentile(hp, [0.5, 99.5])]
        return (
            parent
            & np.isfinite(w) & np.isfinite(h)
            & (w <= w_hi)
            & (h >= h_lo) & (h <= h_hi)
        )

    mw, mh = float(np.median(wp)), float(np.median(hp))
    mad_w = float(np.median(np.abs(wp - mw))) + 1e-9
    mad_h = float(np.median(np.abs(hp - mh))) + 1e-9
    # Asymmetric: allow low-W; cap high-W; wider H band
    return (
        parent
        & np.isfinite(w) & np.isfinite(h)
        & (w > mw - (n_mad + 2.0) * mad_w) & (w < mw + n_mad * mad_w)
        & (h > mh - (n_mad + 1.0) * mad_h) & (h < mh + (n_mad + 1.0) * mad_h)
    )


def refine_ssc_by_purity(fsca, sscw, ssch, ssca, g, fsc_lo, fsc_hi, ssc_hi_init,
                         t_ld=None, t_cd45=None, min_n=600, n_grid=14):
    """Scan SSC-A to maximize lineage purity (B+T+NK of CD14-).

    Keeps SSC near the density-derived lymph upper bound — does not collapse
    onto a tiny ultra-pure island (important for engrafted / NK-dominant files).
    """
    ssc_hi_init = float(ssc_hi_init)
    lo = max(0.095, 0.70 * ssc_hi_init)
    hi = max(ssc_hi_init * 1.08, lo + 0.02)
    hi = min(hi, 0.32)
    best_shi, best_score = ssc_hi_init, -1.0

    for shi in np.linspace(lo, hi, n_grid):
        lymph = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < shi) & (ssca > 0.02)
        sing = singlet_mask(sscw, ssch, lymph)
        live = sing & (g["ld"] < t_ld) if (t_ld is not None and "ld" in g) else sing
        work = live & (g["cd45"] > t_cd45) if (t_cd45 is not None and "cd45" in g) else live
        if work.sum() < min_n or "cd3" not in g or "cd56" not in g:
            continue

        t14 = None
        if "cd14" in g:
            t14 = robust_cd14_cut(g["cd14"][work], ssca=ssca[work])
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
        n_lymph = max(int(cd14n.sum()), 1)
        purity = (nB + nT + nNK) / n_lymph
        # Prefer purity but keep enough events (avoid tiny ultra-pure SSC)
        score = purity + 0.18 * min(n_lymph, 20000) / 20000.0
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
    ssca = df["SSC-A"].values
    sscw = df["SSC-W"].values if "SSC-W" in df.columns else None
    ssch = df["SSC-H"].values if "SSC-H" in df.columns else None
    return g, fsca, ssca, sscw, ssch


def file_cuts(df, ch, ssc_cap=None, refine_ssc=True, locked_scatter=None):
    """Derive cutoffs: viable → singlet → live → CD45 → CD14 → lymph-SSC → lineage.

    locked_scatter locks the *lymph reporting* SSC floor (soft-lock FSC per file).
    Viable-cloud bounds are always re-estimated for panel-1 display/cleanup.
    """
    g, fsca, ssca, sscw, ssch = file_signals(df, ch)
    viable, vflo, vfhi, vshi = viable_scatter_gate(fsca, ssca)
    scatter_method = "density"
    if locked_scatter is not None:
        lymph, fsc_lo, fsc_hi, ssc_hi, _meta = soft_lock_scatter(
            fsca, ssca, locked_scatter)
        scatter_method = "soft_lock"
        if ssc_cap is not None:
            ssc_hi = min(ssc_hi, float(ssc_cap))
            lymph, fsc_lo, fsc_hi, ssc_hi = lymph_scatter_gate(
                fsca, ssca, ssc_cap=ssc_hi)
        refine_ssc = True
    else:
        lymph, fsc_lo, fsc_hi, ssc_hi = lymph_scatter_gate(fsca, ssca, ssc_cap=ssc_cap)

    # Cleanup hierarchy on the *viable* cloud (UNITO/OpenCyto pre-gates)
    sing = singlet_mask(sscw, ssch, viable)
    t_ld = robust_ld_cut(g["ld"][sing]) if "ld" in g else None
    live = sing & (g["ld"] < t_ld) if t_ld is not None else sing.copy()
    t_cd45 = robust_cd45_cut(g["cd45"][live]) if "cd45" in g else None

    if refine_ssc:
        ssc_hi, _ = refine_ssc_by_purity(
            fsca, sscw, ssch, ssca, g, fsc_lo, fsc_hi, ssc_hi,
            t_ld=t_ld, t_cd45=t_cd45,
        )
        if locked_scatter is not None:
            ssc_hi = max(float(ssc_hi), float(locked_scatter["ssc_hi"]))
        if ssc_cap is not None:
            ssc_hi = min(ssc_hi, float(ssc_cap))
        # Re-apply density island under refined SSC ceiling (never a rectangle)
        lymph, fsc_lo, fsc_hi, _ = lymph_scatter_gate(fsca, ssca, ssc_cap=ssc_hi)

    cd45p = live & (g["cd45"] > t_cd45) if t_cd45 is not None else live.copy()

    t_cd14 = None
    if "cd14" in g and cd45p.sum() >= 100:
        t_cd14 = robust_cd14_cut(g["cd14"][cd45p], ssca=ssca[cd45p])
    if t_cd14 is not None and "cd14" in g and live.sum() >= 200:
        frac_live = float(np.mean(g["cd14"][live] >= t_cd14))
        if frac_live > 0.20:
            t_cd14 = pct_cut(g["cd14"][live], 96)

    # Lymph denom = CD45+ CD14− ∩ density lymph island
    cd14n = cd45p & lymph & (g["cd14"] < t_cd14) if t_cd14 is not None else (
        cd45p & lymph)

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
        viable_fsc_lo=vflo, viable_fsc_hi=vfhi, viable_ssc_hi=vshi,
        scatter_method=scatter_method,
    )


def gate_with(df, C, ch, ssc_cap=None):
    """
    Apply cuts in hierarchy order:
      viable scatter → singlets → live → CD45+ → CD14±
      lymph path: CD45+ CD14− ∩ lymph-SSC island → CD19 → CD3×CD56
      mono path:  CD14+ of CD45+

    Panel-1 `lymph_scatter` mask is the *viable* cellular cloud. Lymph
    reporting SSC (`ssc_hi`) stays separate so B/T/NK % are not diluted by
    CD14− granulocytes.
    """
    g, fsca, ssca, sscw, ssch = file_signals(df, ch)

    # --- Viable cloud: ALWAYS density island (never reconstruct a rectangle) ---
    viable, vflo, vfhi, vshi = viable_scatter_gate(fsca, ssca)

    # --- Lymph reporting island: density / soft-lock density ---
    fsc_lo, fsc_hi, ssc_hi = C.get("fsc_lo"), C.get("fsc_hi"), C.get("ssc_hi")
    locked = C.get("_anchor_scatter") in ("locked", "soft_lock")
    use_density = True
    locked_scatter = C.get("_locked_scatter")
    if locked_scatter is None and locked and None not in (fsc_lo, fsc_hi, ssc_hi):
        locked_scatter = {"fsc_lo": fsc_lo, "fsc_hi": fsc_hi, "ssc_hi": ssc_hi}
    if locked_scatter is not None:
        lymph, fsc_lo, fsc_hi, ssc_hi, _ = soft_lock_scatter(
            fsca, ssca, locked_scatter)
        use_density = False
        if ssc_cap is not None:
            ssc_hi = min(float(ssc_hi), float(ssc_cap))
            lymph, fsc_lo, fsc_hi, ssc_hi = lymph_scatter_gate(
                fsca, ssca, ssc_cap=ssc_hi)
    else:
        cap = ssc_cap
        if ssc_hi is not None:
            cap = float(ssc_hi) if cap is None else min(float(ssc_hi), float(cap))
        lymph, fsc_lo, fsc_hi, ssc_hi = lymph_scatter_gate(
            fsca, ssca, ssc_cap=cap)

    # 2–4. Singlets → live → CD45+ on viable cloud
    sing = singlet_mask(sscw, ssch, viable)
    live = sing & (g["ld"] < C["ld"]) if (C.get("ld") is not None and "ld" in g) else sing.copy()
    cd45p = live & (g["cd45"] > C["cd45"]) if (C.get("cd45") is not None and "cd45" in g) else live.copy()

    # 5. CD14 branch within CD45+
    if C.get("cd14") is not None and "cd14" in g:
        CD14p = cd45p & (g["cd14"] >= C["cd14"])
        cd14n_all = cd45p & (g["cd14"] < C["cd14"])
    else:
        CD14p = np.zeros(len(df), bool)
        cd14n_all = cd45p.copy()

    # Lymph denom = CD14− ∩ lymph SSC island (excludes CD14− grans)
    cd14n = cd14n_all & lymph
    mono_ssc = cd45p & (ssca >= float(ssc_hi)) & (
        ssca < min(max(float(ssc_hi) * 1.8, float(ssc_hi) + 0.05), 0.95))
    mono = CD14p | mono_ssc
    work = cd14n

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
        lymph_scatter=viable,  # panel-1 shows full cellular cloud
        lymph_report=lymph,    # low-SSC island used for B/T/NK denom
        sing=sing, live=live, cd45p=cd45p,
        lympho=work, mono=mono, CD14p=CD14p, cd14n=cd14n, cd19n=cd19n,
        B=B, NK=NK, T=Tc, Donor=Donor, CAR=CAR,
    )
    cuts = dict(
        ld=C.get("ld"), cd45=C.get("cd45"), cd14=C.get("cd14"), cd19=C.get("cd19"),
        cd3=C.get("cd3"), cd56=C.get("cd56"), hla=C.get("hla"), car=C.get("car"),
        fsc_lo=fsc_lo, fsc_hi=fsc_hi, ssc_hi=ssc_hi,
        viable_fsc_lo=vflo, viable_fsc_hi=vfhi, viable_ssc_hi=vshi,
        hla_dim=hla_dim,
        cd8_from_cd4_neg=bool(C.get("cd8_from_cd4_neg")),
        scatter_method=C.get("scatter_method") or (
            "density" if use_density else "soft_lock" if locked else "density"
        ),
        scatter_gate="density_island",
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

    return masks, g, (fsca, ssca, sscw, ssch), cuts


MIN_PARENT = 50
MIN_POS = 10


def percentages(m):
    """B/T/NK/CD4/CD8 of CD45+CD14− lymph; CD14+ monocytes of CD45+ (and of live)."""
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
            "fsc_lo", "fsc_hi", "ssc_hi",
            "viable_fsc_lo", "viable_fsc_hi", "viable_ssc_hi"]
    valid = {k: [] for k in keys}
    for fn, df in loaded:
        c = file_cuts(df, ch, ssc_cap=ssc_cap, locked_scatter=locked_scatter)
        perfile[fn] = c
        for k in keys:
            if c.get(k) is not None:
                valid[k].append(c[k])
    med = {k: (float(np.median(v)) if v else None) for k, v in valid.items()}
    return perfile, med
