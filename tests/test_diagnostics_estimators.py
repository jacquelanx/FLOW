"""Tests for the diagnostics statistical core.

``estimators.py`` is deliberately numpy-only so exactly this can happen: the part of the
cutoff diagnostics most worth testing is checked against KNOWN ground truth in FLOW's own dev
environment, without needing the container-only scientific stack.

Every test here asserts against an analytic or constructed truth (a Gaussian's HWHM, a
Wasserstein distance equal to a known shift, a mixture with a known positive fraction) rather
than against a previously observed output, so the suite catches real regressions instead of
freezing whatever the code happened to do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flow.firstrun.anchored.diagnostics import estimators as E


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)


# ── robust scale ──────────────────────────────────────────────────────────────
def test_robust_sd_recovers_gaussian_sigma(rng):
    assert E.robust_sd(rng.normal(0, 1, 20000)) == pytest.approx(1.0, abs=0.05)


def test_robust_sd_resists_a_bright_tail(rng):
    """The whole reason robust_sd exists: a bright tail must not inflate the scale.

    Every "distance in SD units" metric would read as safe if it did.
    """
    core = rng.normal(0, 1, 20000)
    contaminated = np.concatenate([core, rng.normal(20, 3, 2000)])
    assert E.robust_sd(contaminated) == pytest.approx(1.0, abs=0.15)
    assert np.std(contaminated) > 3.0  # the naive estimate really is inflated


def test_robust_sd_survives_a_biex_floor_pileup(rng):
    """>50% identical values makes the MAD zero; a zero divisor must not escape."""
    piled = np.concatenate([np.zeros(1200), rng.normal(5, 1, 400)])
    sd = E.robust_sd(piled)
    assert sd is not None and sd > 0


def test_robust_sd_reports_missing_rather_than_guessing():
    assert E.robust_sd([1.0, 2.0]) is None


# ── mode estimation ───────────────────────────────────────────────────────────
def test_half_sample_mode_picks_the_dominant_mode(rng):
    mix = np.concatenate([rng.normal(0, 1, 8000), rng.normal(10, 1, 2000)])
    assert E.half_sample_mode(mix) == pytest.approx(0.0, abs=0.5)


def test_density_profile_is_a_normalized_density(rng):
    prof = E.density_profile(rng.normal(0, 1, 20000))
    integral = prof["density"].sum() * prof["step"]
    assert integral == pytest.approx(1.0, abs=0.02)
    peak = prof["grid"][int(np.argmax(prof["density"]))]
    assert peak == pytest.approx(0.0, abs=0.15)
    # A standard normal's peak density is 1/sqrt(2*pi).
    assert prof["density"].max() == pytest.approx(1 / math.sqrt(2 * math.pi), abs=0.02)


def test_two_mode_structure_finds_both_modes_and_the_trough(rng):
    bim = np.concatenate([rng.normal(0, 1, 6000), rng.normal(8, 1, 4000)])
    s = E.two_mode_structure(bim)
    assert s["n_modes"] == 2
    assert s["lower_mode"] == pytest.approx(0.0, abs=0.4)
    assert s["upper_mode"] == pytest.approx(8.0, abs=0.4)
    assert s["lower_mode"] < s["trough_loc"] < s["upper_mode"]
    assert s["gap"] == pytest.approx(8.0, abs=0.6)
    assert s["trough_depth_ratio"] < 0.15  # a deep, clean valley


def test_two_mode_structure_reports_one_mode_without_inventing_a_second(rng):
    s = E.two_mode_structure(rng.normal(0, 1, 10000))
    assert s["n_modes"] == 1
    assert s["upper_mode"] is None
    assert s["trough_depth_ratio"] is None  # missing, NOT zero


def test_trough_depth_ratio_separates_deep_from_shallow_valleys(rng):
    deep = E.two_mode_structure(
        np.concatenate([rng.normal(0, 1, 5000), rng.normal(9, 1, 5000)]))
    shallow = E.two_mode_structure(
        np.concatenate([rng.normal(0, 1, 5000), rng.normal(2.6, 1, 5000)]))
    assert deep["trough_depth_ratio"] < 0.15
    # An overlapping pair either shows a shallow trough or merges to one mode; both are
    # correct answers, and both must be distinguishable from the deep case.
    if shallow["trough_depth_ratio"] is not None:
        assert shallow["trough_depth_ratio"] > 0.5
    else:
        assert shallow["n_modes"] == 1


def test_half_width_half_max_matches_the_gaussian_identity(rng):
    sigma = 2.0
    prof = E.density_profile(rng.normal(0, sigma, 40000))
    expect = sigma * math.sqrt(2 * math.log(2))  # 1.1774 sigma
    assert E.half_width_half_max(prof, 0.0) == pytest.approx(expect, rel=0.08)


def test_local_maxima_is_vectorized_but_agrees_with_a_naive_loop(rng):
    """The vectorized form runs inside every bootstrap replicate; it must stay equivalent."""
    prof = E.density_profile(
        np.concatenate([rng.normal(0, 1, 4000), rng.normal(7, 1, 3000)]))
    d, g = prof["density"], prof["grid"]
    peak = float(d.max())
    naive = [
        float(g[i]) for i in range(1, d.size - 1)
        if d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] >= 0.02 * peak
    ]
    assert [m["loc"] for m in E.local_maxima(prof)] == pytest.approx(naive)


# ── separation and positivity ─────────────────────────────────────────────────
def test_stain_index_matches_its_definition(rng):
    """A 6-sigma gap between unit-variance populations is a stain index of 3."""
    si = E.stain_index(rng.normal(0, 1, 5000), rng.normal(6, 1, 5000))
    assert si == pytest.approx(3.0, abs=0.2)


def test_percentile_rank_recovers_a_known_percentile(rng):
    a = rng.normal(0, 1, 20000)
    assert E.percentile_rank(a, float(np.percentile(a, 98))) == pytest.approx(98.0, abs=0.5)


@pytest.mark.parametrize("true_p", [0.10, 0.30, 0.60])
def test_overton_recovers_a_known_positive_fraction(rng, true_p):
    """Cutoff-free positivity must land on the constructed truth."""
    ref_neg = rng.normal(0, 1, 8000)
    n, k = 10000, int(10000 * true_p)
    sample = np.concatenate([rng.normal(0, 1, n - k), rng.normal(8, 1, k)])
    assert E.overton_positive(sample, ref_neg) == pytest.approx(true_p, abs=0.04)


# ── Wilson intervals ──────────────────────────────────────────────────────────
def test_wilson_is_exact_at_the_boundaries():
    """k=0 must give exactly 0, not 3e-18 rendered into a report."""
    lo, hi = E.wilson_interval(0, 100)
    assert lo == 0.0
    assert 0.01 < hi < 0.05
    lo, hi = E.wilson_interval(100, 100)
    assert hi == 1.0


def test_wilson_brackets_the_estimate_and_widens_for_small_denominators():
    lo, hi = E.wilson_interval(50, 100)
    assert lo < 0.5 < hi
    wide_lo, wide_hi = E.wilson_interval(5, 10)
    assert (wide_hi - wide_lo) > (hi - lo)


def test_wilson_reports_missing_for_an_empty_denominator():
    assert E.wilson_interval(0, 0) == (None, None)


# ── distribution comparison ───────────────────────────────────────────────────
@pytest.mark.parametrize("shift", [0.0, 0.5, 3.0])
def test_wasserstein_equals_the_known_shift(rng, shift):
    x = rng.normal(0, 1, 6000)
    y = rng.normal(shift, 1, 6000)
    assert E.wasserstein1d(x, y) == pytest.approx(shift, abs=0.12)


def test_wasserstein_matches_a_closed_form_uniform_case(rng):
    """W1(U(0,1), U(0,2)) = 0.5 exactly."""
    assert E.wasserstein1d(rng.uniform(0, 1, 20000),
                           rng.uniform(0, 2, 20000)) == pytest.approx(0.5, abs=0.03)


def test_permutation_test_discriminates_identical_from_shifted(rng):
    same = E.permutation_pvalue(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000),
                                n_perm=120, seed=1)
    diff = E.permutation_pvalue(rng.normal(0, 1, 3000), rng.normal(1.5, 1, 3000),
                                n_perm=120, seed=1)
    assert same["p_value"] > 0.15
    assert diff["p_value"] < 0.02
    # A finite null can never support a p-value of exactly zero.
    assert diff["p_value"] > 0.0


# ── resampling ────────────────────────────────────────────────────────────────
def test_bootstrap_se_of_the_mean_matches_the_analytic_value(rng):
    n = 5000
    bs = E.bootstrap(rng.normal(0, 1, n), lambda v: float(np.mean(v)), n_boot=200, seed=3)
    assert bs["se"] == pytest.approx(1.0 / math.sqrt(n), rel=0.15)
    assert bs["ci_low"] < 0 < bs["ci_high"]


# ── unimodality evidence ──────────────────────────────────────────────────────
def test_silverman_test_discriminates_unimodal_from_bimodal(rng):
    uni = E.silverman_multimodality_test(rng.normal(0, 1, 3000), n_boot=60, seed=5)
    bim = E.silverman_multimodality_test(
        np.concatenate([rng.normal(0, 1, 1500), rng.normal(7, 1, 1500)]), n_boot=60, seed=5)
    assert uni["p_value_unimodal"] > 0.10
    assert bim["p_value_unimodal"] < 0.05
    assert bim["critical_bandwidth"] > 2 * uni["critical_bandwidth"]


def test_mixture_fit_recovers_known_components(rng):
    """The FIT is what tier 1 relies on; its p-value is only descriptive (see below)."""
    fit = E.fit_gmm2_1d(
        np.concatenate([rng.normal(0, 1, 3000), rng.normal(5, 2, 2000)]), seed=11)
    assert fit["means"][0] < fit["means"][1]
    assert fit["means"][0] == pytest.approx(0.0, abs=0.4)
    assert fit["means"][1] == pytest.approx(5.0, abs=0.5)
    assert fit["weights"][0] == pytest.approx(0.6, abs=0.08)
    assert sum(fit["weights"]) == pytest.approx(1.0)


def test_mixture_separation_discriminates_where_the_lrt_pvalue_does_not(rng):
    """Regression guard for a real design error found during end-to-end validation.

    The mixture LRT's null is a SINGLE GAUSSIAN, which biex-transformed fluorescence never
    is, so its p-value pinned to the bootstrap floor for every marker and carried no
    information. Tier 1 therefore uses the fit's SEPARATION, not its p-value. This asserts
    that separation really does discriminate, so the criterion rests on something that works.
    """
    unimodal = E.bimodality_lrt(rng.normal(0, 1, 4000), n_boot=30, seed=7)
    bimodal = E.bimodality_lrt(
        np.concatenate([rng.normal(0, 1, 2000), rng.normal(7, 1, 2000)]), n_boot=30, seed=7)
    assert unimodal["standardized_separation"] < 1.5
    assert bimodal["standardized_separation"] > 5.0
    assert unimodal["delta_bic_favouring_two"] < 0 < bimodal["delta_bic_favouring_two"]


# ── determinism ───────────────────────────────────────────────────────────────
def test_resampling_is_deterministic_for_a_fixed_seed(rng):
    """The pipeline is deterministic; a re-run must not "discover" different flags."""
    a = np.concatenate([rng.normal(0, 1, 2000), rng.normal(6, 1, 1500)])
    assert (E.silverman_multimodality_test(a, n_boot=30, seed=42)
            == E.silverman_multimodality_test(a, n_boot=30, seed=42))
    assert (E.bootstrap(a, lambda v: float(np.mean(v)), n_boot=50, seed=42)
            == E.bootstrap(a, lambda v: float(np.mean(v)), n_boot=50, seed=42))
    assert E.bimodality_lrt(a, n_boot=20, seed=42) == E.bimodality_lrt(a, n_boot=20, seed=42)


# ── plot-space helpers ────────────────────────────────────────────────────────
def test_bin_width_uses_intervals_not_edges():
    """The overlay figures pass 120 EDGES to linspace, which is 119 bins."""
    assert E.bin_width(0.0, 119.0, 119) == pytest.approx(1.0)
    assert E.bin_width(5.0, 1.0) is None


def test_shared_axis_range_spans_every_series(rng):
    lo, hi = E.shared_axis_range([rng.normal(0, 1, 5000), rng.normal(4, 1, 5000)])
    assert lo < -1.5 and hi > 5.5
    assert E.shared_axis_range([]) == (None, None)


# ── acquisition stability ─────────────────────────────────────────────────────
def test_binned_medians_detects_drift_and_ignores_stability(rng):
    t = np.sort(rng.uniform(0, 100, 5000))
    _c, stable, _n = E.binned_medians(rng.normal(10, 1, 5000), t, n_bins=10)
    _c2, drifting, _n2 = E.binned_medians(10 + 0.05 * t + rng.normal(0, 1, 5000), t, n_bins=10)
    assert (stable.max() - stable.min()) < 0.5
    assert (drifting[-1] - drifting[0]) > 3.5


# ── honest missingness ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "call",
    [
        lambda: E.two_mode_structure([]),
        lambda: E.wasserstein1d([1.0], [2.0]),
        lambda: E.stain_index([1.0], [2.0]),
        lambda: E.overton_positive([1.0], [2.0]),
        lambda: E.silverman_multimodality_test([1.0, 2.0]),
        lambda: E.bimodality_lrt([1.0, 2.0]),
        lambda: E.density_profile([1.0, 2.0]),
        lambda: E.binned_medians([1, 2, 3], [1, 2, 3]),
    ],
)
def test_too_little_data_returns_none_never_a_fabricated_zero(call):
    """A false "no drift" is worse than a gap, so every estimator must return None."""
    assert call() is None


def test_finite_strips_non_finite_values():
    assert E.finite([1.0, np.nan, np.inf, -np.inf, 2.0]).tolist() == [1.0, 2.0]


# ── tier 0: the time axis ─────────────────────────────────────────────────────
# Every other tier audits how a cutoff was derived or transferred; none looks at the resulting
# series. A real run exploited that gap: two timepoints whose %NK read ~0.08% between
# neighbours of 19% and 15% passed all six tiers, and the agent dropped them from its answer
# without naming them. These tests pin the rule that closes it.
from flow.firstrun.anchored.diagnostics import adequacy, flags, record, thresholds  # noqa: E402


class _Ctx:
    def __init__(self, labels):
        self._labels = labels

    def timepoint_labels(self):
        return self._labels


def _counting_rows(metric, series, halfwidth=0.5):
    """series: [(timepoint, value_pct)] -> tier-0 counting rows shaped like adequacy._counting."""
    return [{"timepoint": tp, "metric": metric, "value_pct": v, "n_parent": 5000,
             "n_numerator": None if v is None else int(v * 50),
             "counting_ci_halfwidth_pp": halfwidth}
            for tp, v in series]


def _run_disc(series, metric="%NK (of lymph)", halfwidth=0.5, th=None):
    labels = [tp for tp, _ in series]
    book, rec = flags.FlagBook(), record.MetricRecorder()
    rows = adequacy._discontinuity(_counting_rows(metric, series, halfwidth), _Ctx(labels),
                                   rec, book, th or thresholds.Thresholds())
    return rows, book


def test_a_steep_monotone_trend_is_never_off_trend():
    """A trend is a STEP: every interior value lies between its neighbours, however steep."""
    rows, book = _run_disc([("T0", 1.0), ("T1", 20.0), ("T2", 55.0), ("T3", 90.0), ("T4", 99.0)])
    assert len(book) == 0
    interior = [r for r in rows if r.get("assessable")]
    assert interior and all(not r["discontinuous"] for r in interior)
    assert all(r["direction"] == "within envelope" for r in interior)


def test_a_spike_between_consistent_neighbours_is_caught():
    """The artifact signature: far outside both neighbours, which agree with each other."""
    rows, book = _run_disc([("T0", 30.0), ("T1", 0.08), ("T2", 28.0)])
    hit = [r for r in rows if r.get("discontinuous")]
    assert len(hit) == 1 and hit[0]["timepoint"] == "T1"
    assert hit[0]["direction"] == "below both neighbours"
    assert hit[0]["excursion_in_tolerances"] > 1.0
    assert len(book) == 1
    f = book.flags[0]
    assert f.code == "TEMPORAL_DISCONTINUITY" and f.tier == 0
    # The rule must say why a trend cannot trigger it, and the hint must warn about runs.
    assert "spike, not a step" in f.rule
    assert "RUN" in f.resolution_hint
    assert "panel" in f.resolution_hint


def test_the_tolerance_is_the_data_s_own_counting_noise():
    """The same excursion must NOT fire when the counting intervals are wide enough to explain it."""
    series = [("T0", 30.0), ("T1", 20.0), ("T2", 28.0)]
    tight, _ = _run_disc(series, halfwidth=0.2)
    wide, _ = _run_disc(series, halfwidth=8.0)
    assert any(r.get("discontinuous") for r in tight)
    assert not any(r.get("discontinuous") for r in wide)


def test_endpoints_are_reported_as_not_assessable_rather_than_omitted():
    """Silence about an unchecked timepoint reads as a pass; say it was not assessed."""
    rows, _ = _run_disc([("T0", 10.0), ("T1", 11.0), ("T2", 12.0)])
    ends = [r for r in rows if not r.get("assessable")]
    assert {r["timepoint"] for r in ends} == {"T0", "T2"}
    assert all(r["reason"] for r in ends)
    assert any("no earlier neighbour" in r["reason"] for r in ends)
    assert any("no later neighbour" in r["reason"] for r in ends)
    # Every timepoint appears exactly once, assessed or not.
    assert len(rows) == 3


def test_an_unmeasurable_value_or_interval_is_not_treated_as_zero():
    labels = ["T0", "T1", "T2"]
    rows_in = _counting_rows("%NK (of lymph)", [("T0", 30.0), ("T1", None), ("T2", 28.0)])
    rows_in[1]["counting_ci_halfwidth_pp"] = None
    book, rec = flags.FlagBook(), record.MetricRecorder()
    rows = adequacy._discontinuity(rows_in, _Ctx(labels), rec, book, thresholds.Thresholds())
    mid = [r for r in rows if r["timepoint"] == "T1"][0]
    assert mid["assessable"] is False
    assert "could not be measured" in mid["reason"]
    assert len(book) == 0          # a gap is not a finding


def test_the_threshold_is_configurable_and_declares_its_provenance():
    th = thresholds.Thresholds()
    assert th.discontinuity_ci_multiple == 3.0
    described = {d["threshold"]: d for d in th.describe()}
    row = described["discontinuity_ci_multiple"]
    assert row["provenance"] == "self-calibrating"
    assert "no biological rate of change" in row["rationale"].lower()
    # Raising it must be able to silence a marginal excursion.
    series = [("T0", 30.0), ("T1", 10.0), ("T2", 28.0)]   # gap 18 vs tolerance 3*(1+1)=6
    strict, _ = _run_disc(series, halfwidth=1.0)
    loose, _ = _run_disc(series, halfwidth=1.0,
                         th=thresholds.Thresholds(discontinuity_ci_multiple=50.0))
    assert any(r.get("discontinuous") for r in strict)
    assert not any(r.get("discontinuous") for r in loose)
