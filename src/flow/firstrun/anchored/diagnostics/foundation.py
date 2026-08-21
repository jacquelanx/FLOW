"""Tier 1 — is the cutoff well founded at the reference timepoint?

This is the "is the ruler well made" tier. It runs at the reference (pre-infusion) timepoint,
because that is where the anchored method derives every locked cutoff.

With ONE exception, and it exists because a reference pool can be empty by construction. The
CAR cutoff is applied inside Donor NK, and a pre-infusion reference has no donor cells — two
events on the run this was written against. Tier 1 returned nothing for it, tier 2 needs tier
1's reference structure so it returned nothing either, and tier 3 dropped the marker at every
timepoint for want of an offset scale. The one reported number whose cutoff was never audited
and never perturbed was ``%CAR+ (of Donor NK)`` — the study's headline. So when the reference
cannot carry a marker's audit, ``_audit_with_fallback`` takes it at the first timepoint that
can, labels it as a different measurement, suppresses the derivation-specific checks, and
raises ``REFERENCE_AUDIT_UNAVAILABLE`` so the substitution is never silent.

Two design decisions carry most of the scientific weight:

**Mode locations are estimated without reference to the cutoff.** If the negative mode were
estimated from "events below the cutoff", every placement metric would be circular — a
badly placed cutoff would move the population it is being judged against. So modes come from
a kernel density estimate of the whole pool, and the negative/positive split used for the
stain index comes from the density TROUGH, not from the cutoff under audit.

**"Does a valley exist" is decided structurally, not by a p-value.** Both available tests have
a known bias, and an end-to-end run on data with injected defects showed both of them clearly:

  * The mixture LRT's null is a SINGLE GAUSSIAN, which biex-transformed fluorescence never is.
    It rejected unconditionally — the identical bootstrap-floor p-value for all eight markers.
    Rejecting that null means "not Gaussian", not "bimodal", so its p-value is reported for
    completeness and does NOT enter the criterion. The FIT is kept, because component
    separation and delta-BIC discriminate cleanly.
  * Silverman's bootstrap has the right null (unimodal, any shape) but is known to lose power
    on UNEQUAL mixtures. It called the viability channel unimodal at p=0.70 despite a real 10%
    dead population.

So the criterion is a stated structural rule — two density modes with a real trough at
conventional smoothing, OR a well-separated mixture component of non-trivial weight — and the
row records which route decided it. Both p-values travel alongside with their caveats.

The derivation fingerprint deserves a note. When ``gmm_valley`` finds nothing, ``gates.py``
falls back to a percentile (``pct_cut(vals, 98)`` for CD19, the 25th for CD3, the 5th for
CD45 in NK-bright samples). A percentile fallback asserts "this fraction is positive" rather
than discovering a boundary, and ``unified_cutoffs.csv`` cannot currently distinguish the
two. Recovering it needs no instrumentation of the pipeline: a cutoff whose percentile rank
in its own pool lands exactly on one of those round numbers, with no valley evidence, is a
fallback. The check is skipped when the anchor was calibrated to manual gating, where a rank
coincidence would mean nothing.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np

from . import estimators as E
from .context import DiagnosticsContext, SampleState
from .flags import (
    F_ASSERTED_CUTOFF,
    F_CUTOFF_INSIDE_POPULATION,
    F_FRACTION_OUT_OF_BAND,
    F_LOW_PLACEMENT_MARGIN,
    F_MODE_ESTIMATORS_DISAGREE,
    F_NO_VALLEY_EVIDENCE,
    F_PERCENTILE_FALLBACK,
    F_REFERENCE_AUDIT_UNAVAILABLE,
    F_SHALLOW_TROUGH,
    F_STRUCTURE_ONLY_IN_MIXTURE_FIT,
    FlagBook,
)
from .markers import HEADLINE_METRICS, MarkerSpec
from .record import (
    U_FRACTION,
    U_GAP_FRACTION,
    U_INDEX,
    U_PVALUE,
    U_RATIO,
    U_ROBUST_SD,
    U_TRANSFORM,
    MetricRecorder,
    rounded,
    safe_ratio,
)

# HWHM -> sigma for a Gaussian peak.
_HWHM_TO_SIGMA = 1.0 / math.sqrt(2.0 * math.log(2.0))

# Percentile fallbacks that appear literally in gates.py. Used only as a fingerprint.
_FALLBACK_PERCENTILES = (1.0, 5.0, 10.0, 20.0, 25.0, 40.0, 70.0, 85.0, 90.0, 92.0,
                         96.0, 97.0, 98.0, 99.0, 99.5, 99.9)
_FALLBACK_RANK_TOL = 0.15  # percentile-rank units

# The overlay figures build bins with np.linspace(lo, hi, 120) -> 120 EDGES -> 119 bins.
OVERLAY_EDGES = 120
OVERLAY_BINS = OVERLAY_EDGES - 1


def run(ctx: DiagnosticsContext, rec: MetricRecorder, book: FlagBook) -> dict[str, Any]:
    """Compute tier 1 at the reference timepoint. Returns a JSON-ready dict."""
    ref = ctx.reference
    if ref is None:
        return {
            "available": False,
            "reason": (
                f"no sample matches the reference timepoint "
                f"{ctx.reference_timepoint!r}; the locked cutoffs cannot be audited where "
                "they were derived"
            ),
            "markers": {},
        }

    manual_calibrated = bool(ctx.anchor.get("manual_targets"))
    axis = _overlay_axes(ctx)
    out: dict[str, Any] = {
        "available": True,
        "reference_timepoint": ref.timepoint,
        "reference_file": ref.filename,
        "anchor_derivation": (
            "calibrated to manual gating at the reference timepoint" if manual_calibrated
            else "negative-population valley at the reference timepoint"
        ),
        "markers": {},
    }

    for spec in ctx.marker_specs():
        cut = ctx.cut_value(spec.key)
        res = _audit_with_fallback(
            ctx, ref, spec, cut, axis.get(spec.key), manual_calibrated, rec, book
        )
        if res is not None:
            out["markers"][spec.key] = res
    return out


# ── which sample carries the audit ────────────────────────────────────────────
def _fallback_order(
    ctx: DiagnosticsContext, ref: SampleState, spec: MarkerSpec
) -> list[SampleState]:
    """Non-reference samples, best candidate for a fallback audit first.

    Two properties are wanted and they pull against each other. The fallback becomes tier 2's
    drift BASELINE, so it should sit as close to the reference as possible — the less biology
    that has intervened, the more a later drift means. But it also has to carry a density
    estimate, and the pipeline's ``MIN_PARENT`` of 50 is a COUNTING floor: enough events to
    quote a percentage, nowhere near enough for a mode, a trough depth and a bootstrap SE. On
    the run this was written against, "earliest that clears 50" picked D3 with 51 Donor NK
    events, which would have made every CAR drift number a comparison against noise.

    So: earliest timepoint clearing ``fallback_audit_min_events``, and only if none do, the
    largest pool available. Pool sizes are counted directly rather than by attempting the
    audit, because an attempt writes measurements as soon as it succeeds and there would be no
    way to un-write the ones taken on a candidate that was then rejected.
    """
    floor = int(ctx.thresholds.fallback_audit_min_events)
    sized: list[tuple[SampleState, int]] = []
    for s in ctx.samples:
        if s is ref:
            continue
        vals = s.marker_values(spec)
        sized.append((s, 0 if vals is None else int(E.finite(vals).size)))
    # ``ctx.samples`` is already in the pipeline's timepoint order, so "earliest" is just the
    # first qualifying entry.
    qualifying = [s for s, n in sized if n >= floor]
    remainder = sorted((p for p in sized if p[1] < floor), key=lambda p: -p[1])
    return qualifying + [s for s, _n in remainder]


def _audit_with_fallback(
    ctx: DiagnosticsContext,
    ref: SampleState,
    spec: MarkerSpec,
    cut: Optional[float],
    axis: Optional[dict[str, Any]],
    manual_calibrated: bool,
    rec: MetricRecorder,
    book: FlagBook,
) -> dict[str, Any]:
    """Audit ``spec`` at the reference; if the reference cannot carry it, audit it elsewhere.

    The reference timepoint is where every cutoff is DERIVED, so it is where "is the ruler
    well made" belongs. But a reference pool can be empty by construction rather than by
    accident: the CAR cutoff is applied inside Donor NK, and at a pre-infusion reference there
    are no donor cells — two events on the run this was written against. Tier 1 then returned
    ``available: false``, tier 2 returned a single unavailable row (it needs tier 1's reference
    structure), and tier 3 dropped the marker at every timepoint for want of an offset scale.
    The result was a reported number — ``%CAR+ (of Donor NK)`` — whose own cutoff was the one
    parameter never audited and never perturbed, while the uncertainty column still named
    "cutoff" as its dominant source, computed by moving OTHER cutoffs.

    So when the reference cannot carry the audit, it is taken at the first timepoint that can.
    That is a different measurement and it is labelled as one: ``audited_at_reference`` is
    false, ``reference_audit_unavailable_reason`` says why the reference could not, and the
    derivation-specific checks are suppressed (see ``is_reference_audit`` in
    ``_audit_marker``) because a locked cutoff's percentile rank in some other timepoint's
    pool is a coincidence, not a derivation fingerprint. What survives is what a cytometrist
    reads off the overlay: where the line sits relative to the negative population, how deep
    the valley is under it, how far the modes are apart.

    A fallback is never silent. ``F_REFERENCE_AUDIT_UNAVAILABLE`` fires whenever the reference
    could not carry the audit — with or without a usable fallback — for any marker that moves
    a headline number, because "no tier-1 row" is indistinguishable from "tier 1 found nothing
    wrong" to anyone reading the table.
    """
    at_ref = _audit_marker(ctx, ref, spec, cut, axis, manual_calibrated, rec, book,
                           is_reference_audit=True)
    if at_ref.get("available"):
        at_ref["audited_at_timepoint"] = ref.timepoint
        at_ref["audited_at_reference"] = True
        return at_ref

    # The reference cannot carry it. Candidates are ordered by ``_fallback_order`` — earliest
    # timepoint with a pool big enough for a density estimate to mean something, then the
    # remainder largest-first — and the first that audits successfully wins. Every unavailable
    # branch of ``_audit_marker`` returns before it writes a measurement or a flag, so a
    # candidate that fails leaves no trace of having been tried.
    ref_reason = at_ref.get("reason")
    result = at_ref
    for s in _fallback_order(ctx, ref, spec):
        cand = _audit_marker(ctx, s, spec, cut, axis, manual_calibrated, rec, book,
                             is_reference_audit=False)
        if cand.get("available"):
            cand["audited_at_timepoint"] = s.timepoint
            cand["audited_at_reference"] = False
            cand["reference_audit_unavailable_reason"] = ref_reason
            cand["audit_scope_note"] = (
                f"The reference timepoint {ref.timepoint!r} could not carry this marker's "
                f"audit ({ref_reason}), so these numbers judge the SAME locked cutoff against "
                f"the {s.timepoint!r} {spec.pool} pool instead. They say where the cutoff sits "
                "in that distribution; they do NOT verify how it was derived, and the "
                "derivation-fingerprint checks were skipped here for that reason."
            )
            result = cand
            break
    else:
        result["audited_at_timepoint"] = None
        result["audited_at_reference"] = False
        result["reference_audit_unavailable_reason"] = ref_reason
        result["audit_scope_note"] = (
            "No timepoint could carry this marker's audit, so there is NO measurement of how "
            "well this cutoff is placed anywhere. Treat it as unaudited, not as sound."
        )

    headline = tuple(m for m in HEADLINE_METRICS if m in spec.downstream)
    if headline:
        fallback_tp = result.get("audited_at_timepoint")
        book.add(
            code=F_REFERENCE_AUDIT_UNAVAILABLE, tier=1,
            subject=f"{spec.key} @ {ref.timepoint} (reference)",
            rule=(f"the reference timepoint's {spec.pool} pool could not carry this cutoff's "
                  f"tier-1 audit, and {len(headline)} reported headline metric(s) move with "
                  "this cutoff"),
            measured={
                "reference_timepoint": ref.timepoint,
                "reason": ref_reason,
                "pool": spec.pool,
                "cutoff": rounded(cut, 3),
                "audited_at_timepoint": fallback_tp,
                "headline_metrics_affected": list(headline),
                "valley_derived_by_pipeline": spec.valley_derived,
            },
            resolution_hint=(
                (f"The audit was taken at {fallback_tp} instead — read that row, and read it "
                 "as placement in THAT timepoint's distribution rather than as a check on how "
                 "the cutoff was derived. Tier 2's drift for this marker is measured against "
                 f"{fallback_tp}, not against the reference.")
                if fallback_tp else
                ("No timepoint could carry it, so this cutoff has no placement measurement at "
                 "all. The only external checks on it are tier 4's internal controls and tier "
                 "5's operator concordance; say which of those you are relying on.")
            ),
        )
    return result


# ── plot-space axis, reproduced from histogram_overlays ────────────────────────
def _overlay_axes(ctx: DiagnosticsContext) -> dict[str, dict[str, Any]]:
    """Per-marker (lo, hi, bin width) matching the overlay PNGs' shared axis.

    Reproduced from ``flow_outputs.histogram_overlays``: pooled 0.5/99.5 percentiles over the
    UNGATED events of every timepoint file, controls excluded. Recomputing it here is what
    makes a reported drift checkable against the rendered figure — the bin width is the
    precision floor of anything a human can read off that PNG.
    """
    axes: dict[str, dict[str, Any]] = {}
    for spec in ctx.marker_specs():
        detector = ctx.channels.get(spec.key)
        if detector is None:
            continue
        series = []
        for s in ctx.samples:
            if s.df is None or detector not in s.df.columns:
                continue
            series.append(np.asarray(s.df[detector].values, dtype=float))
        if len(series) < 2:
            continue
        lo, hi = E.shared_axis_range(series, 0.5, 99.5)
        if lo is None:
            continue
        axes[spec.key] = {
            "lo": lo,
            "hi": hi,
            "bin_width": E.bin_width(lo, hi, OVERLAY_BINS),
            "n_bins": OVERLAY_BINS,
        }
    return axes


# ── per-marker audit ──────────────────────────────────────────────────────────
def _audit_marker(
    ctx: DiagnosticsContext,
    sample: SampleState,
    spec: MarkerSpec,
    cut: Optional[float],
    axis: Optional[dict[str, Any]],
    manual_calibrated: bool,
    rec: MetricRecorder,
    book: FlagBook,
    is_reference_audit: bool = True,
) -> dict[str, Any]:
    """Audit one locked cutoff against ``sample``'s own distribution.

    ``is_reference_audit`` gates the checks that are about DERIVATION rather than placement —
    valley evidence, the mixture-only structure note, the percentile-rank fingerprint and the
    accepted-fraction band. All four compare the cutoff against the pool it was derived in, so
    running them somewhere else answers a question nobody asked: a locked cutoff's percentile
    rank in a later timepoint's pool lands where the biology put it, and reporting that as a
    "suspected percentile fallback" would be a fabricated provenance claim. Placement, valley
    depth and separation stay on, because those are the quantities a cytometrist reads off the
    overlay for whatever distribution is in front of them.
    """
    th = ctx.thresholds
    subject = (f"{spec.key} @ {sample.timepoint} (reference)" if is_reference_audit
               else f"{spec.key} @ {sample.timepoint}")
    pool = sample.marker_values(spec)
    if pool is None or cut is None:
        return {
            "available": False,
            "reason": "cutoff or parent pool unavailable at the reference timepoint",
            "pool": spec.pool,
            "cutoff": cut,
        }
    pool = E.finite(pool)
    if pool.size < 50:
        return {
            "available": False,
            "reason": f"parent pool has only {int(pool.size)} events",
            "pool": spec.pool,
            "cutoff": cut,
        }

    lo = axis["lo"] if axis else None
    hi = axis["hi"] if axis else None
    struct = E.two_mode_structure(pool, lo=lo, hi=hi)
    if struct is None:
        return {
            "available": False,
            "reason": "density could not be estimated for the parent pool",
            "pool": spec.pool,
            "cutoff": cut,
        }
    prof = struct["profile"]

    res: dict[str, Any] = {
        "available": True,
        "label": spec.label,
        "channel": ctx.channels.get(spec.key),
        "cutoff": rounded(cut, 3),
        "pool": spec.pool,
        "applied_pool": spec.applied_pool,
        "pool_events": int(pool.size),
        "positive_side": "above cutoff" if spec.positive_above else "below cutoff",
        "positive_label": spec.positive_label,
        "valley_derived_by_pipeline": spec.valley_derived,
        "n_modes": struct["n_modes"],
        "negative_mode": rounded(struct["lower_mode"], 3),
        "positive_mode": rounded(struct["upper_mode"], 3),
        "mode_gap": rounded(struct["gap"], 3),
        "trough_location": rounded(struct["trough_loc"], 3),
        "trough_depth_ratio": rounded(struct["trough_depth_ratio"], 4),
        "kde_bandwidth": rounded(struct["bandwidth"], 4),
        "overlay_axis": axis,
    }

    # ---- negative-population scale ------------------------------------------
    sd_hwhm = None
    if struct["lower_mode"] is not None:
        hw = E.half_width_half_max(prof, struct["lower_mode"])
        if hw:
            sd_hwhm = hw * _HWHM_TO_SIGMA
    sd_mad = None
    if struct["trough_loc"] is not None:
        below = pool[pool < struct["trough_loc"]]
        sd_mad = E.robust_sd(below)
    sd_neg = sd_hwhm if sd_hwhm else sd_mad
    res["negative_sd_from_hwhm"] = rounded(sd_hwhm, 4)
    res["negative_sd_from_mad_below_trough"] = rounded(sd_mad, 4)
    res["negative_sd_used"] = rounded(sd_neg, 4)
    res["negative_sd_source"] = "hwhm" if sd_hwhm else ("mad_below_trough" if sd_mad else None)

    # ---- mode estimator cross-check -----------------------------------------
    hsm = E.half_sample_mode(pool[pool < struct["trough_loc"]]) if struct["trough_loc"] is not None else None
    res["negative_mode_half_sample"] = rounded(hsm, 3)
    if hsm is not None and struct["lower_mode"] is not None and sd_neg:
        disagreement = abs(hsm - struct["lower_mode"]) / sd_neg
        res["mode_estimator_disagreement_in_sd"] = rounded(disagreement, 4)
        rec.add(
            tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
            metric="mode_estimator_disagreement",
            value=rounded(abs(hsm - struct["lower_mode"]), 3), unit=U_TRANSFORM,
            normalized_value=rounded(disagreement, 4), normalized_unit=U_ROBUST_SD,
            note="KDE peak vs half-sample mode of the negative population",
        )
        book.raise_if(
            disagreement > 1.0,
            code=F_MODE_ESTIMATORS_DISAGREE, tier=1, subject=subject,
            rule="KDE peak and half-sample mode of the negative population differ by > 1 robust SD",
            measured={
                "kde_mode": rounded(struct["lower_mode"], 3),
                "half_sample_mode": rounded(hsm, 3),
                "disagreement_in_sd": rounded(disagreement, 4),
            },
            resolution_hint=(
                "The negative peak is not well defined, so every 'distance from the negative "
                "mode' number for this marker is soft. Inspect the overlay PNG for a "
                "shoulder or a third population before relying on the placement metrics."
            ),
            value=disagreement, threshold=1.0,
        )

    # ---- placement -----------------------------------------------------------
    placement_z = None
    if struct["lower_mode"] is not None and sd_neg:
        signed = (cut - struct["lower_mode"]) if spec.positive_above else (struct["lower_mode"] - cut)
        placement_z = signed / sd_neg
    res["placement_in_negative_sd"] = rounded(placement_z, 4)
    gap_pos = safe_ratio(
        None if struct["lower_mode"] is None else cut - struct["lower_mode"], struct["gap"]
    )
    res["normalized_gap_position"] = rounded(gap_pos, 4)
    cut_to_trough = None
    if struct["trough_loc"] is not None:
        cut_to_trough = safe_ratio(abs(cut - struct["trough_loc"]), struct["gap"])
    res["cutoff_to_trough_distance_gap_fraction"] = rounded(cut_to_trough, 4)
    depth_at_cut = None
    if struct["lower_density"] and struct["upper_density"]:
        denom = math.sqrt(struct["lower_density"] * struct["upper_density"])
        if denom > 0:
            depth_at_cut = float(E.density_at(prof, [cut])[0] / denom)
    res["density_at_cutoff_ratio"] = rounded(depth_at_cut, 4)

    rec.add(
        tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
        metric="placement_above_negative_mode",
        value=rounded(None if struct["lower_mode"] is None else cut - struct["lower_mode"], 3),
        unit=U_TRANSFORM,
        normalized_value=rounded(placement_z, 4), normalized_unit=U_ROBUST_SD,
        note="how far the cutoff sits clear of the negative population",
    )
    rec.add(
        tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
        metric="normalized_gap_position", value=rounded(gap_pos, 4), unit=U_GAP_FRACTION,
        note="0 = at the negative mode, 1 = at the positive mode, ~0.5 = mid-valley",
    )
    rec.add(
        tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
        metric="density_at_cutoff_ratio", value=rounded(depth_at_cut, 4), unit=U_RATIO,
        note="density at the cutoff / geometric mean of the two modal densities; ~0 = deep valley",
    )

    book.raise_if(
        placement_z is not None and placement_z < th.placement_z_min,
        code=F_LOW_PLACEMENT_MARGIN, tier=1, subject=subject,
        rule=f"cutoff sits < {th.placement_z_min} robust SD clear of the negative mode",
        measured={
            "placement_in_negative_sd": rounded(placement_z, 4),
            "cutoff": rounded(cut, 3),
            "negative_mode": rounded(struct["lower_mode"], 3),
            "negative_sd": rounded(sd_neg, 4),
        },
        resolution_hint=(
            "Small cutoff moves will reassign negative events. Check the sensitivity tier for "
            "how many percentage points this margin is worth on the reported populations."
        ),
        value=placement_z, threshold=th.placement_z_min, higher_is_worse=False,
    )
    if gap_pos is not None:
        # Distance outside the acceptable band, expressed as a fraction of the band's own
        # width, so a single dimensionless number orders "how far inside a population".
        outside_by = max(0.0, gap_pos - th.gap_position_high, th.gap_position_low - gap_pos)
        band_width = max(th.gap_position_high - th.gap_position_low, 1e-9)
        if outside_by > 0:
            book.add(
                code=F_CUTOFF_INSIDE_POPULATION, tier=1, subject=subject,
                rule=(f"normalized gap position outside [{th.gap_position_low}, "
                      f"{th.gap_position_high}] — the cutoff is inside one of the two populations"),
                measured={
                    "normalized_gap_position": rounded(gap_pos, 4),
                    "outside_band_by": rounded(outside_by, 4),
                    "which_side": "positive population" if gap_pos > th.gap_position_high
                                  else "negative population",
                    "negative_mode": rounded(struct["lower_mode"], 3),
                    "positive_mode": rounded(struct["upper_mode"], 3),
                    "cutoff": rounded(cut, 3),
                },
                resolution_hint=(
                    "Compare against the trough location in the same row: if the trough is far "
                    "from the cutoff the valley finder did not place this cutoff, a fallback did."
                ),
                exceedance=1.0 + outside_by / band_width,
            )
    book.raise_if(
        depth_at_cut is not None and depth_at_cut > th.trough_depth_max,
        code=F_SHALLOW_TROUGH, tier=1, subject=subject,
        rule=(f"density at the cutoff > {th.trough_depth_max} x the geometric mean of the "
              "two modal densities — a shoulder, not a valley"),
        measured={
            "density_at_cutoff_ratio": rounded(depth_at_cut, 4),
            "trough_depth_ratio": rounded(struct["trough_depth_ratio"], 4),
            "trough_location": rounded(struct["trough_loc"], 3),
            "cutoff": rounded(cut, 3),
        },
        resolution_hint=(
            "A shallow separation means the reported fraction is set by where the line was "
            "drawn as much as by the biology. Tier 3's range for this marker quantifies it."
        ),
        value=depth_at_cut, threshold=th.trough_depth_max,
    )

    # ---- separation ----------------------------------------------------------
    stain = None
    if struct["trough_loc"] is not None:
        neg = pool[pool < struct["trough_loc"]]
        pos = pool[pool >= struct["trough_loc"]]
        stain = E.stain_index(neg, pos)
    res["stain_index"] = rounded(stain, 4)
    rec.add(
        tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
        metric="stain_index", value=rounded(stain, 4), unit=U_INDEX,
        note="(median_pos - median_neg) / (2 x robust SD_neg), split at the density trough",
    )

    # ---- unimodality: two independent tests ----------------------------------
    silv = E.silverman_multimodality_test(
        pool, n_boot=th.n_unimodality_boot, seed=th.random_seed
    )
    lrt = E.bimodality_lrt(
        pool, n_boot=max(20, th.n_unimodality_boot // 2), seed=th.random_seed
    )
    res["unimodality_silverman"] = silv
    res["unimodality_mixture_lrt"] = lrt
    p_silv = (silv or {}).get("p_value_unimodal")
    p_lrt = (lrt or {}).get("p_value_unimodal")
    for name, p in (("silverman", p_silv), ("mixture_lrt", p_lrt)):
        rec.add(
            tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
            metric=f"p_unimodal_{name}", value=rounded(p, 4), unit=U_PVALUE,
            note="small p = evidence AGAINST unimodality = a valley really exists",
        )

    # Deciding "does a valley exist" from either p-value alone would be wrong, and the
    # end-to-end run made both failure modes visible:
    #
    #   * The mixture LRT's null is a SINGLE GAUSSIAN. Biex-transformed fluorescence is never
    #     one, so it rejects unconditionally — its p-value came out identical (the bootstrap
    #     floor) for all eight markers. Rejecting that null means "not Gaussian", not
    #     "bimodal", so the p-value is demoted to descriptive here. The FIT is still useful:
    #     component separation and delta-BIC discriminate cleanly.
    #   * Silverman's bootstrap has the right null (unimodal, any shape) but is known to lose
    #     power on UNEQUAL mixtures. On this data it called the viability channel unimodal
    #     (p=0.70) despite a real 10% dead population.
    #
    # So the criterion is structural and stated plainly: either the density shows two modes
    # with a real trough at conventional smoothing, or the mixture fit finds a well-separated
    # second component with non-trivial weight. Both routes are reported, and which one
    # carried the decision is recorded.
    sep = (lrt or {}).get("standardized_separation")
    weights = (lrt or {}).get("component_weights") or []
    min_weight = min(weights) if weights else None
    d_bic = (lrt or {}).get("delta_bic_favouring_two")

    structural = (
        struct["n_modes"] is not None and struct["n_modes"] >= 2
        and struct["trough_depth_ratio"] is not None
        and struct["trough_depth_ratio"] <= th.trough_depth_max
    )
    mixture = (
        sep is not None and sep >= th.mixture_separation_min
        and min_weight is not None and min_weight >= th.mixture_min_component_weight
        and d_bic is not None and d_bic > 0
    )
    valley_supported = bool(structural or mixture)
    res["valley_supported"] = valley_supported
    res["valley_evidence"] = {
        "structural_two_modes_with_trough": structural,
        "mixture_well_separated_component": mixture,
        "decided_by": ("structural" if structural else ("mixture_fit" if mixture else "neither")),
        "standardized_separation": rounded(sep, 4),
        "smallest_component_weight": rounded(min_weight, 5),
        "delta_bic_favouring_two": rounded(d_bic, 2),
        "rule": (
            f"valley supported if (n_modes >= 2 AND trough depth <= {th.trough_depth_max}) OR "
            f"(component separation >= {th.mixture_separation_min} pooled SD AND smallest "
            f"component weight >= {th.mixture_min_component_weight} AND delta-BIC > 0)"
        ),
        "p_value_caveats": (
            "Silverman's p-value tests the right null (unimodal, any shape) but loses power on "
            "unequal mixtures. The mixture LRT's p-value tests a single-Gaussian null, which "
            "biex fluorescence always violates, so it is reported for completeness only and "
            "does NOT enter this rule."
        ),
    }

    # A cutoff the pipeline does NOT derive from a valley (HLA from control brackets, CAR from
    # the NT-NK control's p99.9) is silent in every check above: they all ask whether a found
    # boundary was found well, and this one was never claimed to be found. So the audit table
    # recorded ``valley_supported=false`` with a separation of 0.92 pooled SD on the cutoff
    # that decides "%Donor NK", and no flag surfaced it — while the write-ups called the
    # cutoffs "reasonable". The finding is not that the cutoff is wrong; it is that its
    # placement metrics cannot answer the question, so the answer has to come from tier 4's
    # internal controls and tier 5's operator concordance, and the reader has to be told that.
    if not spec.valley_derived:
        book.raise_if(
            not valley_supported,
            code=F_ASSERTED_CUTOFF, tier=1, subject=subject,
            rule=("the pipeline does not derive this cutoff from a valley, and no valley "
                  "evidence exists in the pool it is applied to either — so the cutoff "
                  "ASSERTS a positive fraction rather than finding a boundary"),
            measured={
                "valley_derived_by_pipeline": False,
                "valley_supported": valley_supported,
                "n_modes": struct["n_modes"],
                "standardized_separation": rounded(sep, 4),
                "trough_depth_ratio": rounded(struct["trough_depth_ratio"], 4),
                "placement_in_negative_sd": rounded(placement_z, 4),
                "pool": spec.pool,
                "pool_events": int(pool.size),
                "derivation_notes": _derivation_notes(ctx, spec),
            },
            resolution_hint=(
                "Placement metrics for this marker are DESCRIPTIVE, not a pass/fail: there is "
                "no boundary in this pool for the cutoff to have found. The only external "
                "checks on it are tier 4's internal controls (a population with a known "
                "answer) and tier 5's operator concordance. Report which of those you relied "
                "on, and report it as unmeasured if neither was available."
            ),
        )

    if spec.valley_derived and is_reference_audit:
        book.raise_if(
            not valley_supported,
            code=F_NO_VALLEY_EVIDENCE, tier=1, subject=subject,
            rule=("no valley evidence by either route (density structure or mixture "
                  "separation), yet the pipeline derives this cutoff from a valley"),
            measured={
                "n_modes": struct["n_modes"],
                "trough_depth_ratio": rounded(struct["trough_depth_ratio"], 4),
                "standardized_separation": rounded(sep, 4),
                "smallest_component_weight": rounded(min_weight, 5),
                "p_unimodal_silverman": rounded(p_silv, 4),
                "critical_bandwidth_over_robust_sd":
                    (silv or {}).get("critical_bandwidth_over_robust_sd"),
            },
            resolution_hint=(
                "No detectable valley: this cutoff is asserting a positive fraction rather "
                "than finding a boundary. Judge it against the internal controls (tier 4) and "
                "the manual concordance (tier 5), not against its placement."
            ),
        )
        book.raise_if(
            valley_supported and mixture and not structural,
            code=F_STRUCTURE_ONLY_IN_MIXTURE_FIT, tier=1, subject=subject,
            rule=("a well-separated second component exists in the mixture fit but does NOT "
                  "appear as a separate density mode at conventional (Silverman) smoothing"),
            measured={
                "n_modes": struct["n_modes"],
                "standardized_separation": rounded(sep, 4),
                "smallest_component_weight": rounded(min_weight, 5),
                "positive_fraction": None,
            },
            resolution_hint=(
                "Typically a real but SMALL or unequal second population — the valley is there "
                "and the rule-of-thumb bandwidth smooths it away. Placement metrics that "
                "depend on a mode gap are unavailable for this marker; read the overlay PNG, "
                "where a small population is usually visible on a density axis."
            ),
        )

    # ---- fraction positive vs the pipeline's own band ------------------------
    frac = E.fraction_above(pool, cut)
    if frac is not None and not spec.positive_above:
        frac = 1.0 - frac
    res["positive_fraction"] = rounded(frac, 5)
    n_pos = None if frac is None else int(round(frac * pool.size))
    ci_lo, ci_hi = E.wilson_interval(n_pos or 0, int(pool.size)) if frac is not None else (None, None)
    res["positive_fraction_ci"] = [rounded(ci_lo, 5), rounded(ci_hi, 5)]
    res["pipeline_accepted_fraction"] = list(spec.accepted_fraction) if spec.accepted_fraction else None

    band_pos = None
    if spec.accepted_fraction and frac is not None:
        b_lo, b_hi = spec.accepted_fraction
        if b_hi > b_lo:
            band_pos = (frac - b_lo) / (b_hi - b_lo)
    res["position_within_accepted_band"] = rounded(band_pos, 4)
    rec.add(
        tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
        metric="positive_fraction", value=rounded(frac, 5), unit=U_FRACTION,
        normalized_value=rounded(band_pos, 4),
        normalized_unit="position within the pipeline's accepted band (0=low edge, 1=high edge)",
        note=f"'{spec.positive_label}' side; band from gates.py",
    )
    # Derivation-only: gates.py applies this band when CHOOSING the cutoff at the reference.
    # A locked cutoff transferred to a post-infusion sample is supposed to leave the band —
    # that is what NK expansion looks like — so firing it on a fallback audit would report the
    # biology as a provenance failure.
    if spec.accepted_fraction and frac is not None and is_reference_audit:
        b_lo, b_hi = spec.accepted_fraction
        outside_by = max(0.0, frac - b_hi, b_lo - frac)
        if outside_by > 0:
            band_width = max(b_hi - b_lo, 1e-12)
            book.add(
                code=F_FRACTION_OUT_OF_BAND, tier=1, subject=subject,
                rule=(f"positive fraction outside the pipeline's own accepted band "
                      f"[{b_lo}, {b_hi}] from gates.py"),
                measured={
                    "positive_fraction": rounded(frac, 5),
                    "band": [b_lo, b_hi],
                    "outside_band_by": rounded(outside_by, 5),
                    "which_side": "above the band" if frac > b_hi else "below the band",
                    "n_pool": int(pool.size),
                },
                resolution_hint=(
                    "gates.py rejects candidate cutoffs that produce a fraction outside this "
                    "band, so a value outside it means the accepted cutoff came from a fallback "
                    "path. Cross-check the percentile-rank fingerprint in this same row."
                ),
                exceedance=1.0 + outside_by / band_width,
            )

    # ---- derivation fingerprint ---------------------------------------------
    rank = E.percentile_rank(pool, cut)
    res["cutoff_percentile_rank_in_pool"] = rounded(rank, 3)
    # Only meaningful for cutoffs the pipeline derives from a valley, IN THE POOL IT DERIVED
    # THEM IN. For HLA/CAR (control- or fit-derived) a round percentile rank is a coincidence,
    # and so is one measured in a timepoint the cutoff was merely transferred to — reporting
    # either as a "suspected fallback" would be actively misleading.
    suspected = (_suspected_percentile(rank)
                 if (spec.valley_derived and is_reference_audit) else None)
    res["suspected_percentile_fallback"] = suspected
    rec.add(
        tier=1, subject_type="marker", marker=spec.key, timepoint=sample.timepoint,
        metric="cutoff_percentile_rank", value=rounded(rank, 3), unit="percentile",
        note="exact round rank + no valley evidence = a percentile fallback, not a found boundary",
    )
    if spec.valley_derived and not manual_calibrated and is_reference_audit:
        no_valley = (valley_supported is False) or (
            depth_at_cut is not None and depth_at_cut > th.trough_depth_max
        )
        book.raise_if(
            suspected is not None and no_valley,
            code=F_PERCENTILE_FALLBACK, tier=1, subject=subject,
            rule=(f"cutoff percentile rank is within {_FALLBACK_RANK_TOL} of {suspected}, a "
                  "literal fallback percentile in gates.py, and there is no valley evidence"),
            measured={
                "cutoff_percentile_rank": rounded(rank, 3),
                "suspected_percentile": suspected,
                "valley_supported": valley_supported,
                "density_at_cutoff_ratio": rounded(depth_at_cut, 4),
            },
            resolution_hint=(
                "unified_cutoffs.csv reports this cutoff as a negative-population valley. If "
                "it is a percentile fallback, that provenance is wrong for this marker and "
                "the cutoff carries less evidential weight than the table implies."
            ),
        )
    elif manual_calibrated:
        res["fingerprint_note"] = (
            "Percentile-fingerprint check skipped: the anchor was calibrated to manual "
            "gating, so a round percentile rank would carry no information."
        )
    elif not is_reference_audit:
        res["fingerprint_note"] = (
            "Percentile-fingerprint check skipped: this audit was taken away from the "
            "reference timepoint, and a locked cutoff's rank in a pool it was only "
            "transferred to says nothing about how it was derived."
        )
    return res


def _derivation_notes(ctx: DiagnosticsContext, spec: MarkerSpec) -> list[str]:
    """The replay's own notes about how this cutoff was derived, if it recorded any.

    ``calibrate`` explains a control-derived cutoff in prose ("CAR cutoff: NT-NK p99.9=1229
    (n=490)") and ``context`` collects those into ``ctx.notes``. Matching on the marker's key
    and the words of its display label keeps this generic: a marker added to ``markers.py``
    picks up its own note without a mapping here. Empty when the anchor supplied the cutoff
    directly, in which case there is nothing to quote.
    """
    words = {spec.key.lower()}
    words |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", spec.label) if len(w) >= 3}
    return [n for n in (ctx.notes or [])
            if any(re.search(rf"(?<![0-9a-z]){re.escape(w)}(?![0-9a-z])", n.lower())
                   for w in words)]


def _suspected_percentile(rank: Optional[float]) -> Optional[float]:
    """The gates.py fallback percentile this rank matches, or None."""
    if rank is None:
        return None
    for p in _FALLBACK_PERCENTILES:
        if abs(rank - p) <= _FALLBACK_RANK_TOL:
            return p
    return None
