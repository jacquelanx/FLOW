"""Statistical primitives for the cutoff diagnostics — PURE NUMPY, no scipy/sklearn.

Every quantity a cytometrist reads off a histogram has an estimator here. The module is
deliberately dependency-light (numpy only) for two reasons:

  1. It is unit-testable in FLOW's own dev environment, which has numpy but none of the
     container-only scientific stack. The statistical core is the part most worth testing,
     so it must not be locked behind flowkit/pandas/sklearn imports.
  2. It stays portable into any sandbox image.

Design rules that hold throughout:

  * **Deterministic.** Every resampling routine takes an explicit integer seed and uses a
    local ``np.random.default_rng``. The anchored pipeline is deterministic; its diagnostics
    must be too, or a re-run would "discover" different flags.
  * **Honest missingness.** An estimator that cannot be computed returns ``None`` (or NaN for
    array-valued returns), never a silent 0.0. A false "no drift" is worse than a gap.
  * **Binned density.** Kernel density is evaluated by histogram-then-convolve, which is
    O(n + m log m) rather than O(n·m). Fluorescence arrays here run to 2e5 events per file
    and a direct KDE outer product would allocate gigabytes.

Nothing in this module knows anything about cytometry — it takes arrays of numbers.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

# Consistent MAD -> sigma scaling for a Gaussian.
_MAD_TO_SD = 1.4826022185056018
# Grid resolution for binned density estimates. 1024 keeps the grid spacing well below any
# plausible bandwidth, so the binning error is negligible relative to the smoothing.
_DENSITY_GRID = 1024


# ── basic cleaning / robust scale ──────────────────────────────────────────────
def finite(x) -> np.ndarray:
    """Return ``x`` as a 1-D float array with non-finite values removed."""
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def robust_sd(x, min_n: int = 8) -> Optional[float]:
    """MAD-based standard deviation (MAD x 1.4826), or None if too few events.

    Preferred over ``np.std`` everywhere in the diagnostics: a fluorescence distribution
    with a bright tail has an inflated ordinary SD, which would make every "distance in SD
    units" metric read as safe when it is not.
    """
    a = finite(x)
    if a.size < min_n:
        return None
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad <= 0:
        # Degenerate (>=50% identical values, e.g. a biex floor pile-up). Fall back to a
        # trimmed SD so the caller gets a usable scale instead of a zero divisor.
        lo, hi = np.percentile(a, [10, 90])
        trimmed = a[(a >= lo) & (a <= hi)]
        sd = float(np.std(trimmed)) if trimmed.size >= min_n else 0.0
        return sd if sd > 0 else None
    return mad * _MAD_TO_SD


def iqr(x) -> Optional[float]:
    """Interquartile range, or None if fewer than 4 finite values."""
    a = finite(x)
    if a.size < 4:
        return None
    q1, q3 = np.percentile(a, [25, 75])
    return float(q3 - q1)


def half_sample_mode(x, min_n: int = 4) -> Optional[float]:
    """Robust mode via the recursive half-sample method (Bickel/Robertson-Cryer HSM).

    Repeatedly keeps the densest half of the (sorted) sample. Used as an independent
    cross-check on the KDE mode: when the two disagree the peak is ill-defined, which is
    itself worth reporting.
    """
    a = np.sort(finite(x))
    n = a.size
    if n < min_n:
        return float(np.median(a)) if n else None
    while n > 3:
        k = int(math.ceil(n / 2.0))
        widths = a[k - 1 : n] - a[0 : n - k + 1]
        if widths.size == 0:
            break
        j = int(np.argmin(widths))
        a = a[j : j + k]
        if a.size == n:  # no progress; stop rather than loop forever
            break
        n = a.size
    if n == 3:
        # Midpoint of whichever adjacent pair is tighter.
        return float((a[0] + a[1]) / 2 if (a[1] - a[0]) < (a[2] - a[1]) else (a[1] + a[2]) / 2)
    return float(np.mean(a))


# ── binned kernel density ──────────────────────────────────────────────────────
def silverman_bandwidth(x, min_n: int = 8) -> Optional[float]:
    """Silverman's rule-of-thumb bandwidth using a robust scale estimate.

    ``h = 0.9 * min(sd, IQR/1.34) * n**(-1/5)``. The robust ``min(...)`` guards against a
    long positive tail inflating the bandwidth and smoothing away a real valley.
    """
    a = finite(x)
    if a.size < min_n:
        return None
    sd = float(np.std(a))
    q = iqr(a)
    scale = sd if not q else min(sd, q / 1.34)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return float(0.9 * scale * a.size ** (-0.2))


def density_profile(
    x,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    bandwidth: Optional[float] = None,
    n_grid: int = _DENSITY_GRID,
    min_n: int = 20,
) -> Optional[dict]:
    """Gaussian kernel density on a regular grid, computed by histogram-then-convolve.

    Returns ``{"grid", "density", "bandwidth", "n", "lo", "hi"}`` or None. ``density``
    integrates to ~1 over the grid, so density values are comparable across markers only
    after normalizing by a peak height (which every consumer here does).
    """
    a = finite(x)
    if a.size < min_n:
        return None
    if lo is None or hi is None:
        p_lo, p_hi = np.percentile(a, [0.1, 99.9])
        lo = float(p_lo) if lo is None else float(lo)
        hi = float(p_hi) if hi is None else float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    bw = bandwidth if bandwidth is not None else silverman_bandwidth(a)
    if bw is None or bw <= 0:
        return None

    # Pad the histogram range by 4 bandwidths so the convolution does not wrap mass at the
    # edges (np.convolve 'same' would otherwise truncate the kernel asymmetrically).
    pad = 4.0 * bw
    edges = np.linspace(lo - pad, hi + pad, n_grid + 1)
    counts, _ = np.histogram(a, bins=edges)
    step = float(edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Gaussian kernel sampled on the same step, truncated at 4 sigma.
    half = max(1, int(math.ceil(4.0 * bw / step)))
    offs = np.arange(-half, half + 1) * step
    kern = np.exp(-0.5 * (offs / bw) ** 2)
    ksum = kern.sum()
    if ksum <= 0:
        return None
    kern = kern / ksum

    dens = np.convolve(counts.astype(float), kern, mode="same")
    total = dens.sum() * step
    if total > 0:
        dens = dens / total  # normalize to a density
    return {
        "grid": centers,
        "density": dens,
        "bandwidth": float(bw),
        "n": int(a.size),
        "lo": float(lo),
        "hi": float(hi),
        "step": step,
    }


def density_at(profile: dict, points) -> np.ndarray:
    """Linear interpolation of a ``density_profile`` at arbitrary points."""
    pts = np.asarray(points, dtype=float).ravel()
    return np.interp(pts, profile["grid"], profile["density"], left=0.0, right=0.0)


def local_maxima(profile: dict, rel_prominence: float = 0.02) -> list[dict]:
    """Local maxima of a density profile, as ``[{"loc", "density"}, ...]`` sorted by loc.

    ``rel_prominence`` drops bumps whose height is under that fraction of the global peak,
    which suppresses ripple in the tails without needing a second smoothing pass.

    Fully vectorized on purpose: this runs inside every bootstrap replicate of every
    marker x timepoint mode estimate, so a Python loop over the 1024-point grid would
    dominate the runtime of the whole diagnostics pass.
    """
    d = profile["density"]
    g = profile["grid"]
    if d.size < 3:
        return []
    peak = float(d.max())
    if peak <= 0:
        return []
    # Strict interior maxima. A plateau is resolved to its LAST point (rising edge admitted
    # by >=, falling edge required by >), which is deterministic and good enough given that
    # the grid step is far below the smoothing bandwidth.
    prev, cur, nxt = d[:-2], d[1:-1], d[2:]
    sel = (cur >= prev) & (cur > nxt) & (cur >= rel_prominence * peak)
    idx = np.flatnonzero(sel) + 1
    return [{"loc": float(g[i]), "density": float(d[i])} for i in idx]


def count_modes(profile: dict, rel_prominence: float = 0.02) -> int:
    """Number of local maxima in a density profile."""
    return len(local_maxima(profile, rel_prominence=rel_prominence))


def two_mode_structure(
    x,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    bandwidth: Optional[float] = None,
    rel_prominence: float = 0.02,
) -> Optional[dict]:
    """Describe a distribution as (lower mode, trough, upper mode) without using any cutoff.

    Mode locations are estimated from the data alone — deliberately NOT conditioned on the
    cutoff being audited, which would make every placement metric circular.

    Selection rule: the lower mode is the leftmost qualifying maximum (the negative peak in
    a fluorescence histogram); the upper mode is the tallest maximum strictly to its right.
    Returns None when a density cannot be estimated; returns ``n_modes == 1`` with null
    upper/trough fields when the distribution has a single peak.
    """
    prof = density_profile(x, lo=lo, hi=hi, bandwidth=bandwidth)
    if prof is None:
        return None
    peaks = local_maxima(prof, rel_prominence=rel_prominence)
    res = {
        "n_modes": len(peaks),
        "bandwidth": prof["bandwidth"],
        "lower_mode": None,
        "lower_density": None,
        "upper_mode": None,
        "upper_density": None,
        "trough_loc": None,
        "trough_density": None,
        "trough_depth_ratio": None,
        "gap": None,
        "profile": prof,
    }
    if not peaks:
        return res

    lower = peaks[0]
    res["lower_mode"] = lower["loc"]
    res["lower_density"] = lower["density"]
    if len(peaks) < 2:
        return res

    upper = max(peaks[1:], key=lambda p: p["density"])
    res["upper_mode"] = upper["loc"]
    res["upper_density"] = upper["density"]
    res["gap"] = float(upper["loc"] - lower["loc"])

    g, d = prof["grid"], prof["density"]
    seg = (g > lower["loc"]) & (g < upper["loc"])
    if seg.any():
        idx = np.flatnonzero(seg)
        j = idx[int(np.argmin(d[idx]))]
        res["trough_loc"] = float(g[j])
        res["trough_density"] = float(d[j])
        denom = math.sqrt(max(lower["density"], 0.0) * max(upper["density"], 0.0))
        if denom > 0:
            # 0 = perfectly separated; -> 1 = no valley at all.
            res["trough_depth_ratio"] = float(d[j] / denom)
    return res


def half_width_half_max(profile: dict, mode_loc: float) -> Optional[float]:
    """Half-width at half-maximum of the peak containing ``mode_loc``.

    A contamination-robust width for the negative population: unlike the MAD of
    "events below the cutoff", it does not change when the cutoff is misplaced, so it can
    be compared across timepoints while auditing the cutoff itself.
    """
    g, d = profile["grid"], profile["density"]
    if g.size < 3:
        return None
    i = int(np.argmin(np.abs(g - mode_loc)))
    peak = float(d[i])
    if peak <= 0:
        return None
    target = peak / 2.0

    left = None
    for j in range(i, 0, -1):
        if d[j] <= target:
            # Linear interpolation between j and j+1.
            d0, d1 = d[j], d[j + 1]
            left = float(g[j] + (target - d0) / (d1 - d0) * (g[j + 1] - g[j])) if d1 != d0 else float(g[j])
            break
    right = None
    for j in range(i, d.size - 1):
        if d[j] <= target:
            d0, d1 = d[j], d[j - 1]
            right = float(g[j] - (target - d0) / (d1 - d0) * (g[j] - g[j - 1])) if d1 != d0 else float(g[j])
            break
    if left is None or right is None:
        return None
    return float(max(right - left, 0.0) / 2.0)


# ── separation / positivity ────────────────────────────────────────────────────
def stain_index(negative, positive, min_n: int = 20) -> Optional[float]:
    """Standard flow separation metric: ``(median_pos - median_neg) / (2 * robust_sd_neg)``.

    The field's conventional resolution number, so it needs no translation in review.
    """
    neg = finite(negative)
    pos = finite(positive)
    if neg.size < min_n or pos.size < min_n:
        return None
    sd = robust_sd(neg)
    if not sd:
        return None
    return float((np.median(pos) - np.median(neg)) / (2.0 * sd))


def percentile_rank(x, value: float) -> Optional[float]:
    """Percentile rank (0-100) of ``value`` within ``x``.

    Doubles as a non-invasive derivation fingerprint: a cutoff whose rank is exactly a round
    number (98.00, 25.00, 5.00) with no valley present is a percentile fallback, not a
    discovered boundary — recoverable without instrumenting the pipeline.
    """
    a = finite(x)
    if a.size < 2 or not np.isfinite(value):
        return None
    return float(100.0 * np.mean(a <= value))


def fraction_above(x, cut: float) -> Optional[float]:
    """Fraction of finite values strictly above ``cut``."""
    a = finite(x)
    if a.size == 0 or not np.isfinite(cut):
        return None
    return float(np.mean(a > cut))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion, as fractions (not percentages).

    Used instead of the normal approximation because the interesting denominators here are
    small (late-timepoint donor NK can be a few dozen events) and Wald intervals misbehave
    badly near 0, which is exactly where the control checks live.
    """
    if n <= 0:
        return None, None
    k = max(0, min(int(k), int(n)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = centre - half, centre + half
    # At k=0 (or k=n) the exact bound is 0 (or 1); floating point leaves ~1e-18 residue,
    # which would render as "0.00000000000000000347" in the emitted tables.
    snap = 1e-12
    lo = 0.0 if lo < snap else lo
    hi = 1.0 if hi > 1.0 - snap else hi
    return float(max(0.0, lo)), float(min(1.0, hi))


def overton_positive(sample, reference_negative, min_n: int = 50) -> Optional[float]:
    """Overton-style positive fraction: cutoff-free positivity via CDF subtraction.

    ``max_t (F_ref(t) - F_sample(t))`` — the classic Overton subtraction, estimating the
    positive proportion from the largest vertical gap between the reference-negative CDF and
    the sample CDF. Reported alongside the cutoff-based fraction: when the two disagree, the
    cutoff (not the biology) is doing the work.
    """
    s = np.sort(finite(sample))
    r = np.sort(finite(reference_negative))
    if s.size < min_n or r.size < min_n:
        return None
    allv = np.union1d(s, r)
    f_s = np.searchsorted(s, allv, side="right") / s.size
    f_r = np.searchsorted(r, allv, side="right") / r.size
    return float(max(0.0, float(np.max(f_r - f_s))))


# ── distribution comparison ────────────────────────────────────────────────────
def wasserstein1d(x, y, min_n: int = 20) -> Optional[float]:
    """Exact 1-D earth-mover's distance, in the units of the inputs.

    ``W1 = integral |F_x(t) - F_y(t)| dt``, evaluated on the merged support. One number, in
    axis units, for "how far did this distribution move".
    """
    a = np.sort(finite(x))
    b = np.sort(finite(y))
    if a.size < min_n or b.size < min_n:
        return None
    allv = np.concatenate([a, b])
    allv.sort(kind="mergesort")
    deltas = np.diff(allv)
    if deltas.size == 0:
        return 0.0
    cdf_a = np.searchsorted(a, allv[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, allv[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


def permutation_pvalue(
    x,
    y,
    statistic: Callable[[np.ndarray, np.ndarray], Optional[float]] = None,
    n_perm: int = 200,
    seed: int = 0,
    max_events: int = 4000,
) -> Optional[dict]:
    """Two-sample permutation test. Returns ``{"observed", "p_value", "n_perm", "null_median"}``.

    Both samples are capped at ``max_events`` (deterministic draw) before permuting: the
    permutation cost is O(n_perm * n) and a 2e5-event array would dominate the whole
    diagnostics pass while adding no resolution to a 200-draw null.
    """
    stat = statistic or wasserstein1d
    a = finite(x)
    b = finite(y)
    if a.size < 20 or b.size < 20:
        return None
    rng = np.random.default_rng(seed)
    if a.size > max_events:
        a = rng.choice(a, max_events, replace=False)
    if b.size > max_events:
        b = rng.choice(b, max_events, replace=False)

    observed = stat(a, b)
    if observed is None:
        return None
    pooled = np.concatenate([a, b])
    n_a = a.size
    hits = 0
    null_vals = []
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        s = stat(perm[:n_a], perm[n_a:])
        if s is None:
            continue
        null_vals.append(s)
        if s >= observed:
            hits += 1
    if not null_vals:
        return None
    # Add-one correction: a p-value of exactly 0 is not supportable from a finite null.
    return {
        "observed": float(observed),
        "p_value": float((hits + 1) / (len(null_vals) + 1)),
        "n_perm": len(null_vals),
        "null_median": float(np.median(null_vals)),
    }


# ── resampling ─────────────────────────────────────────────────────────────────
def bootstrap(
    x,
    statistic: Callable[[np.ndarray], Optional[float]],
    n_boot: int = 200,
    seed: int = 0,
    max_events: int = 20000,
    alpha: float = 0.05,
) -> Optional[dict]:
    """Nonparametric bootstrap of ``statistic``.

    Returns ``{"estimate", "se", "ci_low", "ci_high", "n_boot"}`` or None. The standard
    error is the key output for the drift rules: it converts "the peak moved 0.38" into
    "the peak moved 1.4x further than this estimator's own noise", which is a
    self-calibrating threshold rather than a number someone picked.
    """
    a = finite(x)
    if a.size < 20:
        return None
    rng = np.random.default_rng(seed)
    if a.size > max_events:
        a = rng.choice(a, max_events, replace=False)
    est = statistic(a)
    if est is None:
        return None
    vals = []
    n = a.size
    for _ in range(n_boot):
        s = statistic(a[rng.integers(0, n, n)])
        if s is not None and np.isfinite(s):
            vals.append(float(s))
    if len(vals) < max(10, n_boot // 10):
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "estimate": float(est),
        "se": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
        "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_boot": int(arr.size),
    }


# ── unimodality tests ──────────────────────────────────────────────────────────
def critical_bandwidth(x, n_grid: int = 512, tol: float = 1e-4, max_iter: int = 60) -> Optional[float]:
    """Smallest Gaussian KDE bandwidth at which the density becomes unimodal.

    Silverman's critical bandwidth. Found by bisection on h; mode counting is exact on a
    fixed grid. Large h_crit relative to the data's scale = strong multimodal structure.
    """
    a = finite(x)
    if a.size < 30:
        return None
    scale = robust_sd(a) or float(np.std(a))
    if not scale or scale <= 0:
        return None
    lo_h, hi_h = scale * 1e-3, scale * 10.0

    def modes_at(h: float) -> Optional[int]:
        prof = density_profile(a, bandwidth=h, n_grid=n_grid)
        return None if prof is None else count_modes(prof)

    n_hi = modes_at(hi_h)
    if n_hi is None:
        return None
    # Expand upward if even the widest trial bandwidth is still multimodal.
    tries = 0
    while n_hi is not None and n_hi > 1 and tries < 8:
        hi_h *= 2.0
        n_hi = modes_at(hi_h)
        tries += 1
    if n_hi is None or n_hi > 1:
        return None
    if (modes_at(lo_h) or 2) <= 1:
        return float(lo_h)  # unimodal even at a tiny bandwidth

    for _ in range(max_iter):
        mid = math.sqrt(lo_h * hi_h)  # geometric bisection: h spans orders of magnitude
        m = modes_at(mid)
        if m is None:
            return None
        if m > 1:
            lo_h = mid
        else:
            hi_h = mid
        if (hi_h - lo_h) / hi_h < tol:
            break
    return float(hi_h)


def silverman_multimodality_test(
    x, n_boot: int = 100, seed: int = 0, max_events: int = 4000, n_grid: int = 512
) -> Optional[dict]:
    """Silverman's bootstrap test of H0: the distribution is unimodal.

    Smoothed bootstrap from the KDE at the critical bandwidth; the null distribution is the
    critical bandwidth of resamples. A SMALL p-value is evidence against unimodality, i.e.
    evidence that a valley really exists and a valley-derived cutoff is meaningful.

    Non-parametric, so it makes no Gaussian assumption about either population — important
    for biex-transformed fluorescence, which is not Gaussian in either mode.
    """
    a = finite(x)
    if a.size < 50:
        return None
    rng = np.random.default_rng(seed)
    if a.size > max_events:
        a = rng.choice(a, max_events, replace=False)
    h_crit = critical_bandwidth(a, n_grid=n_grid)
    if h_crit is None:
        return None
    sd = float(np.std(a, ddof=1))
    if sd <= 0:
        return None
    n = a.size
    # Silverman's smoothed bootstrap, rescaled so the resample variance matches the data.
    scale = 1.0 / math.sqrt(1.0 + (h_crit ** 2) / (sd ** 2))
    mean = float(np.mean(a))
    exceed = 0
    valid = 0
    for _ in range(n_boot):
        base = a[rng.integers(0, n, n)]
        draw = mean + scale * (base - mean + h_crit * rng.standard_normal(n))
        h_b = critical_bandwidth(draw, n_grid=n_grid)
        if h_b is None:
            continue
        valid += 1
        if h_b > h_crit:
            exceed += 1
    if valid < max(10, n_boot // 5):
        return None
    return {
        "critical_bandwidth": float(h_crit),
        "critical_bandwidth_over_robust_sd": (
            float(h_crit / robust_sd(a)) if robust_sd(a) else None
        ),
        "p_value_unimodal": float((exceed + 1) / (valid + 1)),
        "n_boot": int(valid),
        "n_events_used": int(n),
    }


def _gaussian_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-12)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def fit_gmm2_1d(
    x, n_iter: int = 200, tol: float = 1e-7, seed: int = 0, max_events: int = 20000
) -> Optional[dict]:
    """Two-component 1-D Gaussian mixture by EM. Returns means/sds/weights/loglik/bic.

    Hand-rolled rather than sklearn so the estimator core stays numpy-only. Initialized
    deterministically from the 25th/75th percentiles, which for a bimodal fluorescence
    histogram lands one component in each mode and converges in a few dozen iterations.
    """
    a = finite(x)
    if a.size < 50:
        return None
    rng = np.random.default_rng(seed)
    if a.size > max_events:
        a = rng.choice(a, max_events, replace=False)
    n = a.size
    sd0 = float(np.std(a, ddof=1))
    if sd0 <= 0:
        return None
    mu = np.array(np.percentile(a, [25, 75]), dtype=float)
    if mu[1] - mu[0] <= 0:
        mu = np.array([a.min(), a.max()], dtype=float)
    sd = np.array([sd0, sd0], dtype=float)
    w = np.array([0.5, 0.5], dtype=float)

    prev = -np.inf
    loglik = -np.inf
    for _ in range(n_iter):
        p0 = w[0] * _gaussian_pdf(a, mu[0], sd[0])
        p1 = w[1] * _gaussian_pdf(a, mu[1], sd[1])
        tot = p0 + p1
        tot = np.where(tot <= 0, 1e-300, tot)
        loglik = float(np.sum(np.log(tot)))
        r0 = p0 / tot
        r1 = 1.0 - r0
        n0, n1 = float(r0.sum()), float(r1.sum())
        if n0 < 2 or n1 < 2:
            break
        mu = np.array([float((r0 * a).sum() / n0), float((r1 * a).sum() / n1)])
        sd = np.array(
            [
                math.sqrt(max(float((r0 * (a - mu[0]) ** 2).sum() / n0), 1e-12)),
                math.sqrt(max(float((r1 * (a - mu[1]) ** 2).sum() / n1), 1e-12)),
            ]
        )
        w = np.array([n0 / n, n1 / n])
        if abs(loglik - prev) < tol * max(1.0, abs(prev)):
            break
        prev = loglik

    order = np.argsort(mu)
    k_params = 5  # 2 means + 2 sds + 1 weight
    return {
        "means": [float(mu[order][0]), float(mu[order][1])],
        "sds": [float(sd[order][0]), float(sd[order][1])],
        "weights": [float(w[order][0]), float(w[order][1])],
        "loglik": float(loglik),
        "bic": float(k_params * math.log(n) - 2.0 * loglik),
        "n_events_used": int(n),
    }


def _gauss1_loglik(a: np.ndarray) -> tuple[float, float]:
    """(loglik, bic) of the single-Gaussian MLE for ``a``."""
    n = a.size
    mu = float(np.mean(a))
    sd = float(np.std(a, ddof=0))
    sd = max(sd, 1e-12)
    ll = float(np.sum(np.log(_gaussian_pdf(a, mu, sd))))
    return ll, float(2 * math.log(n) - 2.0 * ll)


def bimodality_lrt(
    x, n_boot: int = 60, seed: int = 0, max_events: int = 20000
) -> Optional[dict]:
    """Parametric-bootstrap likelihood-ratio test of 1 vs 2 Gaussian components.

    Returns the observed LR statistic, delta-BIC (positive favours two components), and a
    bootstrap p-value under the single-Gaussian null. Complements
    ``silverman_multimodality_test``: this one is parametric and sensitive to a small
    well-separated second component; Silverman's is non-parametric and sensitive to shape.
    Reporting both means neither has to be trusted alone.
    """
    a = finite(x)
    if a.size < 50:
        return None
    rng = np.random.default_rng(seed)
    if a.size > max_events:
        a = rng.choice(a, max_events, replace=False)
    ll1, bic1 = _gauss1_loglik(a)
    fit2 = fit_gmm2_1d(a, seed=seed, max_events=max_events)
    if fit2 is None:
        return None
    lr = 2.0 * (fit2["loglik"] - ll1)

    mu = float(np.mean(a))
    sd = float(np.std(a, ddof=0))
    n = a.size
    exceed = 0
    valid = 0
    for i in range(n_boot):
        draw = mu + sd * rng.standard_normal(n)
        ll1_b, _ = _gauss1_loglik(draw)
        f2_b = fit_gmm2_1d(draw, seed=seed + 1 + i, max_events=max_events)
        if f2_b is None:
            continue
        valid += 1
        if 2.0 * (f2_b["loglik"] - ll1_b) >= lr:
            exceed += 1
    p = float((exceed + 1) / (valid + 1)) if valid else None
    sep = None
    if fit2["sds"][0] > 0 and fit2["sds"][1] > 0:
        pooled = math.sqrt(0.5 * (fit2["sds"][0] ** 2 + fit2["sds"][1] ** 2))
        if pooled > 0:
            sep = float((fit2["means"][1] - fit2["means"][0]) / pooled)
    return {
        "lr_statistic": float(lr),
        "delta_bic_favouring_two": float(bic1 - fit2["bic"]),
        "p_value_unimodal": p,
        "n_boot": int(valid),
        "component_means": fit2["means"],
        "component_sds": fit2["sds"],
        "component_weights": fit2["weights"],
        "standardized_separation": sep,
        "n_events_used": int(fit2["n_events_used"]),
    }


# ── plot-space helpers ─────────────────────────────────────────────────────────
def shared_axis_range(series: Sequence[np.ndarray], lo_pct: float = 0.5, hi_pct: float = 99.5):
    """Pooled (lo, hi) percentile range across several arrays.

    Mirrors how the overlay figures fix their shared x-axis, so diagnostics reported "in
    plot space" are checkable against the rendered PNG rather than merely adjacent to it.
    """
    pooled = [finite(s) for s in series]
    pooled = [s for s in pooled if s.size]
    if not pooled:
        return None, None
    allv = np.concatenate(pooled)
    lo, hi = np.percentile(allv, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    return float(lo), float(hi)


def bin_width(lo: float, hi: float, n_bins: int = 120) -> Optional[float]:
    """Width of one histogram bin — the precision floor of anything read off the figure."""
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return float((hi - lo) / float(n_bins))


def coefficient_of_variation(x) -> Optional[float]:
    """CV (sd/mean) of positive-valued data; None when the mean is non-positive."""
    a = finite(x)
    if a.size < 3:
        return None
    m = float(np.mean(a))
    if m <= 0:
        return None
    return float(np.std(a, ddof=1) / m)


def binned_medians(values, order_by, n_bins: int = 10, min_per_bin: int = 20):
    """Median of ``values`` within equal-count bins of ``order_by``.

    The engine behind the acquisition-stability metric: pass fluorescence and the Time
    channel to see whether a marker's central tendency slid during the run.
    Returns ``(bin_centres_of_order_by, medians, counts)`` or None.
    """
    v = np.asarray(values, dtype=float).ravel()
    t = np.asarray(order_by, dtype=float).ravel()
    ok = np.isfinite(v) & np.isfinite(t)
    v, t = v[ok], t[ok]
    if v.size < n_bins * min_per_bin:
        return None
    idx = np.argsort(t, kind="mergesort")
    v, t = v[idx], t[idx]
    splits = np.array_split(np.arange(v.size), n_bins)
    centres, meds, counts = [], [], []
    for s in splits:
        if s.size < min_per_bin:
            continue
        centres.append(float(np.median(t[s])))
        meds.append(float(np.median(v[s])))
        counts.append(int(s.size))
    if len(meds) < 3:
        return None
    return np.asarray(centres), np.asarray(meds), np.asarray(counts)
