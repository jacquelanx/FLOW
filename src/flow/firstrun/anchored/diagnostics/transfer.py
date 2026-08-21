"""Tier 2 — does the locked cutoff still hold at every other timepoint?

This is the scientific heart of the diagnostics, because it is the one risk the anchored
method uniquely takes on. One cutoff is derived at the reference (pre-infusion) timepoint and
transferred to D7, D28, week8. The failure mode is that the distribution drifts while the
line stays put.

**The discriminator.** Every drift measurement here is anchored on the NEGATIVE population,
because the negative population is the internal reference that biology should not move. Post
infusion NK expansion moves *mass across* a stationary cutoff; staining or instrument drift
moves *the negative population itself*. Measuring both separately is what turns an ambiguous
"CD56 looks different at D28" into one of four named patterns:

    negative stable  + positive changed  -> mass moved across a stationary cutoff
    negative shifted + positive changed  -> confounded; not attributable from this marker
    negative shifted + positive stable   -> technical drift, possibly absorbed
    negative stable  + positive stable   -> nothing moved

The pattern's documented reading is emitted alongside it, clearly labelled as a lookup
against a stated rule — not as a verdict about the biology.

**Comparability.** Modes at every timepoint are estimated with the bandwidth and axis fixed
to the reference's, so a difference between timepoints reflects the data and not a
re-tuned estimator.

**"Positive changed" needs no invented threshold.** A change counts as resolved only when it
exceeds the combined Wilson counting intervals of the two timepoints — i.e. when it is
bigger than the sampling noise of the two numbers being compared.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..gates import lymph_scatter_gate
from . import estimators as E
from .context import DiagnosticsContext, SampleState
from .flags import (
    F_BOUNDARY_MASS,
    F_NEG_DISTRIBUTION_SHIFT,
    F_NEG_MODE_DRIFT,
    F_NEG_WIDTH_CHANGE,
    F_NEGATIVE_UNDEFINED,
    F_OVERTON_DISAGREES,
    F_PLACEMENT_MARGIN_LOST,
    F_QUADRANT_MARGIN,
    F_SCATTER_CONTAINMENT,
    FlagBook,
)
from .markers import MarkerSpec
from .record import (
    U_FRACTION,
    U_GAP_FRACTION,
    U_PVALUE,
    U_RATIO,
    U_ROBUST_SD,
    U_SE_MULTIPLE,
    U_TRANSFORM,
    MetricRecorder,
    rounded,
    safe_ratio,
)

_HWHM_TO_SIGMA = 1.0 / math.sqrt(2.0 * math.log(2.0))

# Pattern labels and the documented reading of each. These readings are a LOOKUP against a
# stated rule, offered as reference material — not a conclusion. The agent is free to
# disagree, and the rule that produced the label travels with it.
PATTERN_READINGS = {
    "STABLE_NEGATIVE_STABLE_POSITIVE": (
        "Neither the negative population nor the reported positive fraction moved beyond "
        "measurement noise. Nothing to attribute."
    ),
    "STABLE_NEGATIVE_SHIFTED_POSITIVE": (
        "The negative population held still while mass moved across a stationary cutoff. "
        "This is the pattern a real biological change produces, and the pattern the anchored "
        "method exists to make visible."
    ),
    "SHIFTED_NEGATIVE_SHIFTED_POSITIVE": (
        "The negative population moved AND the positive fraction changed. The two effects are "
        "confounded; a change in the reported number cannot be attributed to biology from "
        "this marker alone at this timepoint."
    ),
    "SHIFTED_NEGATIVE_STABLE_POSITIVE": (
        "The negative population moved but the reported fraction did not. The locked cutoff "
        "may now sit in the wrong place even though the headline number looks steady — check "
        "the placement margin in the same row."
    ),
    "UNDETERMINED": (
        "The negative population could not be located at this timepoint, so drift is "
        "undefined (not zero) and no pattern can be assigned."
    ),
}


def run(ctx: DiagnosticsContext, rec: MetricRecorder, book: FlagBook,
        tier1: dict[str, Any]) -> dict[str, Any]:
    """Compute tier 2 for every marker x timepoint. Needs tier 1's reference structure."""
    ref = ctx.reference
    out: dict[str, Any] = {
        "available": ref is not None,
        "reference_timepoint": None if ref is None else ref.timepoint,
        "markers": {},
        "gate_geometry": {},
        "pattern_readings": PATTERN_READINGS,
        "note": (
            "Drift is measured on the NEGATIVE population only — the internal reference that "
            "biology should not move. Pattern labels are a lookup against the stated rule in "
            "each row, not a conclusion about the biology."
        ),
    }
    if ref is None:
        out["reason"] = (
            f"no sample matches the reference timepoint {ctx.reference_timepoint!r}; "
            "transfer validity cannot be assessed"
        )
    else:
        ref_markers = (tier1 or {}).get("markers", {})
        for spec in ctx.marker_specs():
            t1m = ref_markers.get(spec.key) or {}
            # Drift is measured against whatever sample tier 1 could actually audit. For all
            # but the pathological cases that IS the reference; when it is not, comparing
            # against a reference pool tier 1 already rejected as too small would produce
            # drift numbers with no baseline behind them.
            base = ctx.sample_for_timepoint(t1m.get("audited_at_timepoint")) or ref
            res = _marker_across_timepoints(ctx, base, spec, t1m, rec, book)
            if res is not None:
                out["markers"][spec.key] = res

    # 2-D gate integrity does not depend on the reference sample being present.
    for s in ctx.samples:
        tp = s.timepoint or s.filename
        out["gate_geometry"][tp] = {
            "nk_quadrant": _nk_quadrant(ctx, s, rec, book, tp),
            "donor_car_quadrant": _donor_car_quadrant(ctx, s, rec, book, tp),
            "locked_scatter": _scatter_containment(ctx, s, rec, book, tp),
        }
    return out


# ── 1-D marker transfer ───────────────────────────────────────────────────────
def _mode_statistic(lo, hi, bw):
    """Bootstrap statistic: the lower (negative) mode, with the bandwidth held fixed.

    Fixing the bandwidth matters — letting Silverman's rule re-tune it on every resample
    would fold bandwidth variability into the standard error and inflate the drift threshold.
    """
    def f(v):
        st = E.two_mode_structure(v, lo=lo, hi=hi, bandwidth=bw)
        return None if st is None else st["lower_mode"]
    return f


def _marker_across_timepoints(
    ctx: DiagnosticsContext,
    ref: SampleState,
    spec: MarkerSpec,
    tier1_marker: Optional[dict[str, Any]],
    rec: MetricRecorder,
    book: FlagBook,
) -> Optional[dict[str, Any]]:
    """Drift of one marker across every timepoint, measured against ``ref``.

    ``ref`` is the sample tier 1 audited — normally the reference timepoint, and otherwise the
    fallback tier 1 fell back to (see ``foundation._audit_with_fallback``). Every row records
    which, because "drift vs the reference" and "drift vs D7" are different statements and a
    reader who cannot tell them apart has been handed the wrong one.
    """
    th = ctx.thresholds
    cut = ctx.cut_value(spec.key)
    if cut is None:
        return None
    if not tier1_marker or not tier1_marker.get("available"):
        return {
            "available": False,
            "reason": (
                "no timepoint could carry this marker's tier-1 audit, so there is no baseline "
                "distribution to measure drift against — the transfer of this cutoff is "
                "UNMEASURED, not verified"
            ),
        }
    baseline_is_reference = bool(tier1_marker.get("audited_at_reference", True))

    axis = tier1_marker.get("overlay_axis") or {}
    lo, hi = axis.get("lo"), axis.get("hi")
    bw = tier1_marker.get("kde_bandwidth")
    ref_mode = tier1_marker.get("negative_mode")
    ref_sd = tier1_marker.get("negative_sd_used")
    ref_placement = tier1_marker.get("placement_in_negative_sd")
    # Magnitude scale for normalizing drift. Preferred: the measured negative-to-positive mode
    # gap. When the pool is unimodal there is no gap, and without a fallback the magnitude rule
    # silently drops out -- leaving the significance rule to fire alone on drifts far too small
    # to matter. 4 x the negative SD is the standard proxy for a resolved neg->pos distance
    # (a stain index of 2 means exactly that), so the rule always has a scale.
    ref_gap = tier1_marker.get("mode_gap")
    gap_source = "measured_mode_gap"
    if not ref_gap and ref_sd:
        ref_gap = 4.0 * float(ref_sd)
        gap_source = "proxy_4x_negative_sd (pool is unimodal: no measured mode gap)"

    ref_pool = E.finite(ref.marker_values(spec))
    ref_negative = ref_pool[ref_pool < cut] if ref_pool.size else ref_pool
    # Cutoff-independent definition of the reference negative population, for the
    # cutoff-free (Overton) comparison. Falls back to the cutoff-truncated subset when the
    # reference pool showed no trough.
    ref_trough = tier1_marker.get("trough_location")
    if ref_trough is not None and ref_pool.size:
        ref_negative_pop = ref_pool[ref_pool < float(ref_trough)]
        ref_negative_source = "reference density trough (cutoff-independent)"
    else:
        ref_negative_pop = ref_negative
        ref_negative_source = "below the cutoff (no reference trough was found)"
    ref_se = None
    if bw and ref_pool.size >= 50:
        bs = E.bootstrap(ref_pool, _mode_statistic(lo, hi, bw),
                         n_boot=th.n_bootstrap, seed=th.random_seed)
        ref_se = (bs or {}).get("se")

    out: dict[str, Any] = {
        "available": True,
        "label": spec.label,
        "cutoff": rounded(cut, 3),
        # Which sample every drift below is measured against, and whether that is the anchor's
        # own reference timepoint. Carried onto every ROW by ``emit.transfer_rows`` as well:
        # a reader filtering this table by marker must not have to join back to find out.
        "baseline_timepoint": ref.timepoint,
        "baseline_is_reference": baseline_is_reference,
        "baseline_note": (
            None if baseline_is_reference else
            f"Tier 1 could not audit this marker at the reference timepoint "
            f"({tier1_marker.get('reference_audit_unavailable_reason')}), so every drift in "
            f"this block is measured against {ref.timepoint!r}. A stable row means 'unchanged "
            f"since {ref.timepoint}', NOT 'unchanged since the cutoff was derived'."
        ),
        "reference_negative_mode": ref_mode,
        "reference_mode_gap": rounded(ref_gap, 3),
        "reference_mode_gap_source": gap_source,
        "reference_negative_sd": ref_sd,
        "reference_mode_bootstrap_se": rounded(ref_se, 4),
        "reference_placement_in_negative_sd": ref_placement,
        "timepoints": {},
    }

    for s in ctx.samples:
        tp = s.timepoint or s.filename
        row = _one_timepoint(
            ctx, spec, s, tp, cut, lo, hi, bw, ref_mode, ref_gap, gap_source, ref_sd,
            ref_placement, ref_se, ref_pool, ref_negative,
            ref_negative_pop, ref_negative_source, rec, book,
        )
        row["is_audit_baseline"] = bool(s is ref)
        row["baseline_timepoint"] = ref.timepoint
        row["baseline_is_reference"] = baseline_is_reference
        out["timepoints"][tp] = row
    return out


def _one_timepoint(
    ctx, spec: MarkerSpec, s: SampleState, tp: str, cut: float,
    lo, hi, bw, ref_mode, ref_gap, gap_source, ref_sd, ref_placement, ref_se,
    ref_pool: np.ndarray, ref_negative: np.ndarray,
    ref_negative_pop: np.ndarray, ref_negative_source: str,
    rec: MetricRecorder, book: FlagBook,
) -> dict[str, Any]:
    th = ctx.thresholds
    subject = f"{spec.key} @ {tp}"
    row: dict[str, Any] = {"is_reference": bool(s.is_reference)}

    pool = E.finite(s.marker_values(spec))
    if pool.size < 50:
        row["available"] = False
        row["reason"] = f"parent pool has only {int(pool.size)} events"
        return row
    row["available"] = True
    row["pool_events"] = int(pool.size)

    struct = E.two_mode_structure(pool, lo=lo, hi=hi, bandwidth=bw)
    if struct is None or struct["lower_mode"] is None:
        row["negative_mode"] = None
        row["negative_population_defined"] = False
        row["pattern"] = "UNDETERMINED"
        row["pattern_reading"] = PATTERN_READINGS["UNDETERMINED"]
        book.add(
            code=F_NEGATIVE_UNDEFINED, tier=2, subject=subject,
            rule="no negative-population mode could be located at this timepoint",
            measured={"pool_events": int(pool.size),
                      "n_modes": None if struct is None else struct["n_modes"]},
            resolution_hint=(
                "Common in NK-bright post-infusion samples, where gates.py itself drops to a "
                "low percentile because no negative population exists. Drift is UNDEFINED "
                "here, not zero — do not read the absence of a drift flag as stability."
            ),
        )
        return row

    row["negative_population_defined"] = True
    row["n_modes"] = struct["n_modes"]
    row["negative_mode"] = rounded(struct["lower_mode"], 3)
    row["positive_mode"] = rounded(struct["upper_mode"], 3)

    # ---- mode drift -----------------------------------------------------------
    drift = None if ref_mode is None else float(struct["lower_mode"] - ref_mode)
    drift_gap_frac = safe_ratio(None if drift is None else abs(drift), ref_gap)
    tp_se = None
    if bw:
        bs = E.bootstrap(pool, _mode_statistic(lo, hi, bw),
                         n_boot=th.n_bootstrap, seed=th.random_seed)
        tp_se = (bs or {}).get("se")
    combined_se = None
    if ref_se is not None and tp_se is not None:
        combined_se = math.sqrt(float(ref_se) ** 2 + float(tp_se) ** 2)
    elif tp_se is not None:
        combined_se = float(tp_se)
    drift_in_se = safe_ratio(None if drift is None else abs(drift), combined_se)

    # Magnitude in units of the negative population's OWN width. This is the scale that
    # actually governs whether a locked cutoff is still correctly placed, and unlike the mode
    # gap it exists for a unimodal pool too. It is the magnitude rule; the gap fraction is
    # kept as a second, reported normalization.
    drift_in_neg_sd = safe_ratio(None if drift is None else abs(drift), ref_sd)
    row.update(
        {
            "negative_mode_drift": rounded(drift, 3),
            "negative_mode_drift_in_negative_sd": rounded(drift_in_neg_sd, 4),
            "negative_mode_drift_gap_fraction": rounded(drift_gap_frac, 4),
            "negative_mode_bootstrap_se": rounded(tp_se, 4),
            "negative_mode_drift_in_se": rounded(drift_in_se, 3),
        }
    )
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="negative_mode_drift_in_negative_sd", value=rounded(drift_in_neg_sd, 4),
        unit=U_ROBUST_SD,
        note=("drift as a multiple of the negative population's own SD — the magnitude scale "
              "that governs whether the locked cutoff is still correctly placed"),
    )
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="negative_mode_drift", value=rounded(drift, 3), unit=U_TRANSFORM,
        normalized_value=rounded(drift_gap_frac, 4), normalized_unit=U_GAP_FRACTION,
        note="movement of the internal reference (negative) population vs the anchor timepoint",
    )
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="negative_mode_drift_in_se", value=rounded(drift_in_se, 3), unit=U_SE_MULTIPLE,
        note="self-calibrating: drift relative to the bootstrap SE of the mode estimates",
    )

    exceeds_noise = drift_in_se is not None and drift_in_se > th.drift_se_multiple
    exceeds_magnitude = (
        drift_in_neg_sd is not None and drift_in_neg_sd > th.drift_negative_sd_max
    )
    row["drift_exceeds_noise_rule"] = exceeds_noise
    row["drift_exceeds_magnitude_rule"] = exceeds_magnitude
    # "Significant but immaterial" is recorded as a FIELD, not a flag. At tens of thousands of
    # events the bootstrap SE of the mode is tiny, so nearly any drift clears the noise rule --
    # on the synthetic run this produced flags for 3-unit drifts on a ~1800-unit axis. Keeping
    # the information without the flag stops those burying the drifts that matter.
    row["drift_significant_but_immaterial"] = bool(exceeds_noise and not exceeds_magnitude)

    if not s.is_reference:
        book.raise_if(
            exceeds_noise and exceeds_magnitude,
            code=F_NEG_MODE_DRIFT, tier=2, subject=subject,
            rule=(f"negative-mode drift exceeds BOTH rules: > {th.drift_se_multiple} x the "
                  "bootstrap SE of the mode estimate (self-calibrating — larger than the "
                  f"measurement's own noise) AND > {th.drift_negative_sd_max} x the negative "
                  "population's own SD (magnitude). Both are required, because at this event "
                  "count significance alone is met by trivially small drifts"),
            measured={
                "drift": rounded(drift, 3),
                "drift_in_negative_sd": rounded(drift_in_neg_sd, 4),
                "drift_in_se": rounded(drift_in_se, 3),
                "drift_gap_fraction": rounded(drift_gap_frac, 4),
                "combined_se": rounded(combined_se, 4),
                "reference_negative_sd": ref_sd,
                "reference_mode_gap": rounded(ref_gap, 3),
                "reference_mode_gap_source": gap_source,
                "reference_negative_mode": ref_mode,
                "timepoint_negative_mode": rounded(struct["lower_mode"], 3),
            },
            resolution_hint=(
                "The internal reference moved, so the locked cutoff is no longer in the same "
                "position relative to the negative population it was derived against. Read "
                "the pattern label in this row to see whether the reported fraction moved too."
            ),
            value=drift_in_neg_sd, threshold=th.drift_negative_sd_max,
        )

    # ---- negative width -------------------------------------------------------
    hw = E.half_width_half_max(struct["profile"], struct["lower_mode"])
    tp_sd = hw * _HWHM_TO_SIGMA if hw else None
    width_ratio = safe_ratio(tp_sd, ref_sd)
    row["negative_sd"] = rounded(tp_sd, 4)
    row["negative_width_ratio_vs_reference"] = rounded(width_ratio, 4)
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="negative_width_ratio", value=rounded(width_ratio, 4), unit=U_RATIO,
        note="HWHM-derived negative-population SD relative to the reference timepoint",
    )
    if not s.is_reference and width_ratio is not None:
        # Widening and narrowing are both informative; flag either beyond a factor implied by
        # the drift-magnitude rule, so one settable number governs both shape rules.
        tol = 1.0 + max(th.drift_negative_sd_max, 0.1)
        excursion = max(width_ratio, 1.0 / width_ratio) if width_ratio > 0 else None
        book.raise_if(
            excursion is not None and excursion > tol,
            code=F_NEG_WIDTH_CHANGE, tier=2, subject=subject,
            rule=(f"negative-population width changed by more than a factor of {round(tol, 3)} "
                  f"vs the reference (derived from drift_negative_sd_max = "
                  f"{th.drift_negative_sd_max}, so one settable number governs both the "
                  "location and the width rules)"),
            measured={
                "width_ratio": rounded(width_ratio, 4),
                "reference_negative_sd": ref_sd,
                "timepoint_negative_sd": rounded(tp_sd, 4),
            },
            resolution_hint=(
                "Compensation change or staining variability can widen the negative population "
                "without moving its mode, which invalidates a locked cutoff just as surely. "
                "Mode drift alone would miss this."
            ),
            value=excursion, threshold=tol,
        )

    # ---- distributional shift: full pool vs negative subset -------------------
    tp_negative = pool[pool < cut]
    emd_neg = E.wasserstein1d(ref_negative, tp_negative)
    emd_full = E.wasserstein1d(ref_pool, pool)
    perm = None
    if not s.is_reference and ref_negative.size >= 20 and tp_negative.size >= 20:
        perm = E.permutation_pvalue(
            ref_negative, tp_negative, n_perm=th.n_permutation, seed=th.random_seed
        )
    row.update(
        {
            "emd_negative_subset": rounded(emd_neg, 4),
            "emd_negative_subset_gap_fraction": rounded(safe_ratio(emd_neg, ref_gap), 4),
            "emd_full_pool": rounded(emd_full, 4),
            "emd_full_pool_gap_fraction": rounded(safe_ratio(emd_full, ref_gap), 4),
            "emd_negative_permutation_p": rounded((perm or {}).get("p_value"), 4),
            "emd_negative_permutation_null_median": rounded((perm or {}).get("null_median"), 4),
        }
    )
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="emd_negative_subset", value=rounded(emd_neg, 4), unit=U_TRANSFORM,
        normalized_value=rounded(safe_ratio(emd_neg, ref_gap), 4), normalized_unit=U_GAP_FRACTION,
        note="earth-mover's distance of the sub-cutoff (negative) events vs the reference",
    )
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="emd_full_pool", value=rounded(emd_full, 4), unit=U_TRANSFORM,
        normalized_value=rounded(safe_ratio(emd_full, ref_gap), 4), normalized_unit=U_GAP_FRACTION,
        note="whole-pool movement: technical drift AND biology together",
    )
    if perm:
        rec.add(
            tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
            metric="emd_negative_permutation_p", value=rounded(perm.get("p_value"), 4),
            unit=U_PVALUE,
            note=("permutation null; note EMD has a positive small-sample floor, which is "
                  "exactly why significance is judged against a permuted null rather than 0"),
        )
        book.raise_if(
            perm.get("p_value") is not None and perm["p_value"] < th.drift_pvalue_max,
            code=F_NEG_DISTRIBUTION_SHIFT, tier=2, subject=subject,
            rule=(f"the sub-cutoff (negative) distributions at this timepoint and the "
                  f"reference are distinguishable at p < {th.drift_pvalue_max} "
                  "(permutation test on the earth-mover's distance)"),
            measured={
                "emd_negative_subset": rounded(emd_neg, 4),
                "permutation_p": rounded(perm.get("p_value"), 4),
                "null_median": rounded(perm.get("null_median"), 4),
                "n_perm": perm.get("n_perm"),
            },
            resolution_hint=(
                "Caveat by construction: the negative subset is truncated at the cutoff, so a "
                "large drift is UNDER-stated by this statistic. Read it together with the "
                "mode drift, which is truncation-free."
            ),
            # Order by how many times the observed EMD exceeds its own permuted null, which is
            # a meaningful effect size. Ordering by 1/p would let one tiny p-value dominate
            # the whole tier's ordering without being a bigger effect.
            value=emd_neg, threshold=perm.get("null_median"),
        )

    # ---- placement at this timepoint ------------------------------------------
    placement = None
    if tp_sd:
        signed = (cut - struct["lower_mode"]) if spec.positive_above else (struct["lower_mode"] - cut)
        placement = signed / tp_sd
    # A RATIO of placements is unsafe: placement is signed (negative when the cutoff sits below
    # the negative mode), so a margin going from -0.27 to -1.43 — a 1.16 SD LOSS — produces a
    # ratio of 5.27, which reads as an improvement. The validation run hit exactly that case.
    # The signed DIFFERENCE is well-behaved regardless of sign, so it drives the rule; the
    # ratio is kept as a field only where the reference margin was positive.
    placement_change = (
        None if (placement is None or ref_placement is None)
        else float(placement) - float(ref_placement)
    )
    placement_ratio = (
        safe_ratio(placement, ref_placement)
        if (ref_placement is not None and float(ref_placement) > 0) else None
    )
    row["placement_in_negative_sd"] = rounded(placement, 4)
    row["placement_change_in_negative_sd"] = rounded(placement_change, 4)
    row["placement_ratio_vs_reference"] = rounded(placement_ratio, 4)
    row["placement_ratio_note"] = (
        None if placement_ratio is not None
        else "ratio omitted: the reference margin is not positive, so a ratio is not meaningful"
    )
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="placement_in_negative_sd", value=rounded(placement, 4), unit=U_ROBUST_SD,
        normalized_value=rounded(placement_change, 4),
        normalized_unit="change vs the same cutoff's margin at the reference timepoint, in SD",
        note="the locked cutoff re-expressed in THIS timepoint's negative-population scale",
    )
    if not s.is_reference:
        lost_margin = (
            placement_change is not None
            and placement_change < -abs(th.placement_loss_sd_max)
        )
        fell_below_floor = (
            placement is not None and ref_placement is not None
            and float(ref_placement) >= th.placement_z_min
            and float(placement) < th.placement_z_min
        )
        book.raise_if(
            lost_margin or fell_below_floor,
            code=F_PLACEMENT_MARGIN_LOST, tier=2, subject=subject,
            rule=(f"the cutoff's margin above the negative mode LOST more than "
                  f"{th.placement_loss_sd_max} SD vs the reference timepoint, or fell below "
                  f"the {th.placement_z_min} SD floor having been above it at the reference"),
            measured={
                "placement_here": rounded(placement, 4),
                "placement_at_reference": ref_placement,
                "placement_change_in_sd": rounded(placement_change, 4),
                "lost_more_than_threshold": lost_margin,
                "fell_below_floor": fell_below_floor,
            },
            resolution_hint=(
                "The anchor was sound where it was derived and is not sound where it was "
                "transferred. This identifies WHICH timepoint would need another iteration — "
                "the question the analysis specification asks."
            ),
            value=abs(placement_change) if placement_change is not None else None,
            threshold=th.placement_loss_sd_max,
        )

    # ---- positive fraction + the discriminator --------------------------------
    frac = E.fraction_above(pool, cut)
    if frac is not None and not spec.positive_above:
        frac = 1.0 - frac
    ref_frac = E.fraction_above(ref_pool, cut)
    if ref_frac is not None and not spec.positive_above:
        ref_frac = 1.0 - ref_frac
    overton = E.overton_positive(pool, ref_negative_pop)
    row.update(
        {
            "positive_fraction": rounded(frac, 5),
            "reference_positive_fraction": rounded(ref_frac, 5),
            "overton_positive_vs_reference_negative": rounded(overton, 5),
            "overton_reference_negative_definition": ref_negative_source,
            "overton_reference_negative_events": int(ref_negative_pop.size),
        }
    )
    overton_gap = None if (overton is None or frac is None) else overton - frac
    row["overton_minus_cutoff_fraction"] = rounded(overton_gap, 5)
    rec.add(
        tier=2, subject_type="marker", marker=spec.key, timepoint=tp,
        metric="overton_positive", value=rounded(overton, 5), unit=U_FRACTION,
        normalized_value=rounded(overton_gap, 5),
        normalized_unit="difference from the cutoff-based fraction",
        note=("cutoff-free positivity against the REFERENCE negative; when this disagrees with "
              "the cutoff-based fraction, the cutoff is doing the work"),
    )
    # Overton subtraction estimates the positive proportion from the largest gap between the
    # reference-negative CDF and this sample's CDF, using no cutoff at all. A material
    # disagreement with the cutoff-based fraction means the reported number is being set by
    # where the line was drawn rather than by the distributions.
    if not s.is_reference and overton_gap is not None:
        book.raise_if(
            abs(overton_gap) > th.overton_disagreement_max,
            code=F_OVERTON_DISAGREES, tier=2, subject=subject,
            rule=(f"|cutoff-free (Overton) positive fraction - cutoff-based fraction| > "
                  f"{th.overton_disagreement_max}"),
            measured={
                "overton_positive": rounded(overton, 5),
                "cutoff_based_fraction": rounded(frac, 5),
                "difference": rounded(overton_gap, 5),
                "direction": ("cutoff under-calls positives" if overton_gap > 0
                              else "cutoff over-calls positives"),
            },
            resolution_hint=(
                "Two independent estimates of the same quantity disagree, and only one of them "
                "depends on the cutoff. Read the sign: it says whether the locked cutoff is "
                "over- or under-calling at this timepoint. Overton itself assumes the "
                "reference negative is still the right negative, so a drift flag in this same "
                "row weakens both estimates rather than adjudicating between them."
            ),
            value=abs(overton_gap), threshold=th.overton_disagreement_max,
        )

    # "Positive changed" is resolved against the two timepoints' combined counting noise —
    # no invented threshold.
    resolved = None
    combined_halfwidth = None
    if frac is not None and ref_frac is not None:
        n_tp, n_ref = int(pool.size), int(ref_pool.size)
        lo_tp, hi_tp = E.wilson_interval(int(round(frac * n_tp)), n_tp)
        lo_rf, hi_rf = E.wilson_interval(int(round(ref_frac * n_ref)), n_ref)
        if None not in (lo_tp, hi_tp, lo_rf, hi_rf):
            combined_halfwidth = 0.5 * ((hi_tp - lo_tp) + (hi_rf - lo_rf))
            resolved = abs(frac - ref_frac) > combined_halfwidth
    row["positive_change"] = rounded(
        None if (frac is None or ref_frac is None) else frac - ref_frac, 5)
    row["positive_change_resolvable"] = resolved
    row["positive_change_noise_floor"] = rounded(combined_halfwidth, 5)

    neg_stable = None if (drift_in_se is None and drift_gap_frac is None) else not (
        exceeds_noise and exceeds_magnitude)
    pattern, rule = _pattern(neg_stable, resolved, th)
    row["pattern"] = pattern
    row["pattern_rule"] = rule
    row["pattern_reading"] = PATTERN_READINGS[pattern]
    row["pattern_reading_is_a_lookup"] = True
    return row


def _pattern(neg_stable: Optional[bool], pos_resolved: Optional[bool], th) -> tuple[str, str]:
    """Assign the discriminator pattern and return it with the rule that produced it."""
    rule = (
        f"negative SHIFTED = drift > {th.drift_se_multiple} x SE AND drift > "
        f"{th.drift_negative_sd_max} x the negative population's own SD (both required); "
        "positive CHANGED = |change| exceeds the combined Wilson counting half-widths of the "
        "two timepoints (self-calibrating, no invented threshold)"
    )
    if neg_stable is None or pos_resolved is None:
        return "UNDETERMINED", rule
    if neg_stable and pos_resolved:
        return "STABLE_NEGATIVE_SHIFTED_POSITIVE", rule
    if neg_stable and not pos_resolved:
        return "STABLE_NEGATIVE_STABLE_POSITIVE", rule
    if not neg_stable and pos_resolved:
        return "SHIFTED_NEGATIVE_SHIFTED_POSITIVE", rule
    return "SHIFTED_NEGATIVE_STABLE_POSITIVE", rule


# ── 2-D gate geometry ─────────────────────────────────────────────────────────
def _mode_2d(x: np.ndarray, y: np.ndarray, bins: int = 64, smooth: float = 1.5):
    """Location of the 2-D density peak of a cloud, via a smoothed 2-D histogram.

    Separable Gaussian smoothing implemented with numpy only (the pipeline's own
    ``heat_density_2d`` uses scipy, which the estimator core deliberately avoids).
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return None, None
    xs, ys = x[ok], y[ok]
    x_lo, x_hi = np.percentile(xs, [0.5, 99.5])
    y_lo, y_hi = np.percentile(ys, [0.5, 99.5])
    if not (np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo):
        return None, None
    if not (np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo):
        return None, None
    H, xe, ye = np.histogram2d(
        xs, ys, bins=bins,
        range=[[float(x_lo), float(x_hi)], [float(y_lo), float(y_hi)]],
    )
    half = max(1, int(math.ceil(3.0 * smooth)))
    offs = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (offs / smooth) ** 2)
    k /= k.sum()
    # Separable convolution along each axis.
    Hs = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, H)
    Hs = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, Hs)
    if Hs.max() <= 0:
        return None, None
    ix, iy = np.unravel_index(int(np.argmax(Hs)), Hs.shape)
    xc = 0.5 * (xe[ix] + xe[ix + 1])
    yc = 0.5 * (ye[iy] + ye[iy + 1])
    return float(xc), float(yc)


def _nk_quadrant(ctx: DiagnosticsContext, s: SampleState, rec: MetricRecorder,
                 book: FlagBook, tp: str) -> dict[str, Any]:
    """How far the CD3 x CD56 quadrant boundaries sit from the NK cloud itself.

    NK are defined as a quadrant (CD19- CD3- CD56+), so a boundary can slice through the
    cloud even when both 1-D histograms look clean. Distances are expressed in robust SDs of
    the NK cloud along each axis, which is how a cytometrist already reads a gate.
    """
    th = ctx.thresholds
    out: dict[str, Any] = {"available": False}
    nk = s.masks.get("NK")
    cd3 = s.signals.get("cd3")
    cd56 = s.signals.get("cd56")
    t3, t56 = s.cuts.get("cd3"), s.cuts.get("cd56")
    if nk is None or cd3 is None or cd56 is None or t3 is None or t56 is None:
        out["reason"] = "NK mask or CD3/CD56 cutoffs unavailable"
        return out
    n_nk = int(np.count_nonzero(nk))
    if n_nk < th.min_parent_events:
        out["reason"] = f"only {n_nk} NK events (< MIN_PARENT {th.min_parent_events})"
        out["n_nk"] = n_nk
        return out

    x, y = cd3[nk], cd56[nk]
    sd3, sd56 = E.robust_sd(x), E.robust_sd(y)
    mx, my = _mode_2d(x, y)
    t3, t56 = float(t3), float(t56)

    # NK are CD3-NEGATIVE, so the CD3 boundary lies ABOVE the cloud;
    # NK are CD56-POSITIVE, so the CD56 boundary lies BELOW it.
    margin_cd3 = None if (sd3 is None or mx is None) else (t3 - mx) / sd3
    margin_cd56 = None if (sd56 is None or my is None) else (my - t56) / sd56

    near = None
    if sd3 and sd56:
        near_mask = (np.abs(x - t3) < sd3) | (np.abs(y - t56) < sd56)
        near = float(np.count_nonzero(near_mask)) / float(n_nk)

    # Quadrant occupancy within CD19-, for context on how the split lands overall.
    occupancy = None
    cd19n = s.masks.get("cd19n")
    if cd19n is not None and np.count_nonzero(cd19n) >= th.min_parent_events:
        a3, a56 = cd3[cd19n], cd56[cd19n]
        tot = float(len(a3))
        occupancy = {
            "cd3neg_cd56pos_NK": round(float(np.mean((a3 < t3) & (a56 > t56))), 5),
            "cd3pos_cd56neg_T": round(float(np.mean((a3 > t3) & (a56 < t56))), 5),
            "cd3neg_cd56neg": round(float(np.mean((a3 < t3) & (a56 < t56))), 5),
            "cd3pos_cd56pos": round(float(np.mean((a3 > t3) & (a56 > t56))), 5),
            "n_cd19neg": int(tot),
        }

    out.update(
        {
            "available": True,
            "n_nk": n_nk,
            "nk_mode_cd3": rounded(mx, 3),
            "nk_mode_cd56": rounded(my, 3),
            "nk_robust_sd_cd3": rounded(sd3, 3),
            "nk_robust_sd_cd56": rounded(sd56, 3),
            "cd3_boundary_margin_in_sd": rounded(margin_cd3, 3),
            "cd56_boundary_margin_in_sd": rounded(margin_cd56, 3),
            "fraction_within_one_sd_of_a_boundary": rounded(near, 4),
            "quadrant_occupancy": occupancy,
        }
    )
    for name, val in (("cd3", margin_cd3), ("cd56", margin_cd56)):
        rec.add(
            tier=2, subject_type="gate", marker=name, timepoint=tp,
            metric=f"nk_quadrant_margin_{name}", value=rounded(val, 3), unit=U_ROBUST_SD,
            note="distance from the NK 2-D density mode to this quadrant boundary",
        )
        book.raise_if(
            val is not None and val < th.quadrant_margin_min,
            code=F_QUADRANT_MARGIN, tier=2, subject=f"NK {name} boundary @ {tp}",
            rule=(f"NK 2-D density mode sits < {th.quadrant_margin_min} robust SD from the "
                  f"{name.upper()} quadrant boundary"),
            measured={
                "margin_in_sd": rounded(val, 3), "n_nk": n_nk,
                "nk_mode_cd3": rounded(mx, 3), "nk_mode_cd56": rounded(my, 3),
                "cd3_cut": rounded(t3, 3), "cd56_cut": rounded(t56, 3),
            },
            resolution_hint=(
                "The boundary is inside the NK cloud, so the NK count is set by the boundary "
                "rather than by a gap. Both 1-D histograms can look clean while this is true."
            ),
            value=val, threshold=th.quadrant_margin_min, higher_is_worse=False,
        )
    rec.add(
        tier=2, subject_type="gate", marker="nk", timepoint=tp,
        metric="nk_boundary_event_mass", value=rounded(near, 4), unit=U_FRACTION,
        note="fraction of NK events within one robust SD of a quadrant boundary",
    )
    book.raise_if(
        near is not None and near > th.boundary_mass_max,
        code=F_BOUNDARY_MASS, tier=2, subject=f"NK cloud @ {tp}",
        rule=f"> {th.boundary_mass_max} of NK events lie within one robust SD of a boundary",
        measured={"fraction_near_boundary": rounded(near, 4), "n_nk": n_nk},
        resolution_hint=(
            "Many events sit where a small cutoff move reassigns them. Read tier 3's "
            "cutoff-attributable range for %NK at this timepoint to see what it costs."
        ),
        value=near, threshold=th.boundary_mass_max,
    )
    return out


def _donor_car_quadrant(ctx: DiagnosticsContext, s: SampleState, rec: MetricRecorder,
                        book: FlagBook, tp: str) -> dict[str, Any]:
    """How far the HLA x CAR quadrant boundaries sit from the clouds they divide.

    This is the panel that defines both remaining headline populations — Donor NK by the HLA
    boundary and CAR+ Donor NK by the CAR boundary — and until now it had no numeric surrogate
    at all. ``_nk_quadrant`` covered CD3 x CD56 and ``_scatter_containment`` covered FSC/SSC,
    so of the three 2-D gates in the QC report the one carrying ``%Donor NK`` and ``%CAR+``
    was the one a text-only reader could say nothing about.

    Three margins, because three different clouds can be sliced:

      * ``hla_boundary_margin_host_in_sd`` — the HOST NK cloud (HLA-negative) against the HLA
        boundary. This is the one that matters pre-infusion and early, when host NK dominate:
        a boundary inside that cloud manufactures donor cells out of host ones.
      * ``hla_boundary_margin_donor_in_sd`` — the DONOR cloud against the same boundary, which
        is what matters once engraftment has happened.
      * ``car_boundary_margin_in_sd`` — the CAR-negative mode WITHIN Donor NK against the CAR
        boundary. Measured inside Donor NK because that is the parent ``%CAR+`` is a fraction
        of; measuring it across all NK would answer a question the study does not report.

    Signs are normalized so POSITIVE always means "clear of the boundary", and the HLA axis
    follows ``hla_dim`` — with a dim-donor mismatch the donor side is BELOW the cutoff and
    every margin inverts with it. Getting that backwards would report the host cloud's margin
    as the donor cloud's.
    """
    th = ctx.thresholds
    out: dict[str, Any] = {"available": False}
    nk = s.masks.get("NK")
    hla = s.signals.get("hla")
    car = s.signals.get("car")
    hla_cut, car_cut = s.cuts.get("hla"), s.cuts.get("car")
    if nk is None or hla is None or hla_cut is None:
        out["reason"] = "NK mask, HLA signal or HLA cutoff unavailable"
        return out
    n_nk = int(np.count_nonzero(nk))
    if n_nk < th.min_parent_events:
        out["reason"] = f"only {n_nk} NK events (< MIN_PARENT {th.min_parent_events})"
        out["n_nk"] = n_nk
        return out

    hla_cut = float(hla_cut)
    dim = bool(ctx.hla_dim)
    donor = s.donor_side_mask(dim)
    host = (nk & ~donor) if donor is not None else None
    n_donor = None if donor is None else int(np.count_nonzero(donor))
    n_host = None if host is None else int(np.count_nonzero(host))

    def _hla_margin(mask, donor_side: bool):
        """Signed distance (robust SDs) from a cloud's HLA mode to the HLA boundary.

        Positive = the cloud sits clear of the boundary on the side it belongs to.
        """
        if mask is None or int(np.count_nonzero(mask)) < th.min_parent_events:
            return None, None, None
        v = E.finite(hla[mask])
        if v.size < th.min_parent_events:
            return None, None, None
        mode, sd = E.half_sample_mode(v), E.robust_sd(v)
        if mode is None or not sd:
            return None, rounded(mode, 3), rounded(sd, 3)
        # Donor is above the cutoff unless the mismatch is dim-donor; host is the other side.
        above = donor_side != dim
        signed = (mode - hla_cut) if above else (hla_cut - mode)
        return signed / sd, rounded(mode, 3), rounded(sd, 3)

    m_host, host_mode, host_sd = _hla_margin(host, donor_side=False)
    m_donor, donor_mode, donor_sd = _hla_margin(donor, donor_side=True)

    near_hla = None
    v_nk = E.finite(hla[nk])
    sd_nk = E.robust_sd(v_nk)
    if sd_nk and v_nk.size:
        near_hla = float(np.count_nonzero(np.abs(v_nk - hla_cut) < sd_nk)) / float(v_nk.size)

    out.update({
        "available": True,
        "n_nk": n_nk,
        "n_donor": n_donor,
        "n_host_nk": n_host,
        "hla_cut": rounded(hla_cut, 3),
        "hla_dim_polarity": dim,
        "host_nk_hla_mode": host_mode,
        "host_nk_hla_robust_sd": host_sd,
        "donor_nk_hla_mode": donor_mode,
        "donor_nk_hla_robust_sd": donor_sd,
        "hla_boundary_margin_host_in_sd": rounded(m_host, 3),
        "hla_boundary_margin_donor_in_sd": rounded(m_donor, 3),
        "fraction_nk_within_one_sd_of_hla_boundary": rounded(near_hla, 4),
    })

    # ---- the CAR boundary, inside Donor NK ----------------------------------
    car_margin = near_car = None
    if car is None or car_cut is None:
        out["car_reason"] = "CAR signal or CAR cutoff unavailable"
    elif donor is None or (n_donor or 0) < th.min_parent_events:
        out["car_reason"] = (
            f"only {n_donor} Donor NK events (< MIN_PARENT {th.min_parent_events}) — the CAR "
            "boundary cannot be measured against a cloud that is not there. This is expected "
            "pre-infusion and is NOT evidence that the boundary is well placed."
        )
    else:
        car_cut = float(car_cut)
        cv = E.finite(car[donor])
        # The CAR-negative mode, not the whole cloud's mode: at high CAR+ fractions the
        # dominant mode IS the positive population, and a margin measured from it would read
        # a well-separated gate as a boundary sitting inside the cloud.
        neg = cv[cv < car_cut]
        car_neg_mode = car_sd = None
        if neg.size < th.min_parent_events:
            # A null with its reason, not a bare null. This is the case a fully-transduced
            # product produces: there is no CAR-negative population inside Donor NK for the
            # boundary to sit clear OF, so the margin is undefined rather than large. Reading
            # the missing value as a comfortable margin would invert the finding.
            out["car_reason"] = (
                f"only {int(neg.size)} of {int(cv.size)} Donor NK events fall below the CAR "
                f"cutoff (< MIN_PARENT {th.min_parent_events}), so there is no CAR-negative "
                "mode to measure a margin from. The margin is UNDEFINED here, not wide — "
                "judge this boundary against the tier-4 controls instead."
            )
        else:
            car_neg_mode = E.half_sample_mode(neg)
            car_sd = E.robust_sd(neg)
            if car_neg_mode is not None and car_sd:
                car_margin = (car_cut - car_neg_mode) / car_sd
            if car_sd and cv.size:
                near_car = (float(np.count_nonzero(np.abs(cv - car_cut) < car_sd))
                            / float(cv.size))
        out.update({
            "car_cut": rounded(car_cut, 3),
            "donor_nk_car_negative_events": int(neg.size),
            "donor_nk_car_negative_mode": rounded(car_neg_mode, 3),
            "donor_nk_car_negative_robust_sd": rounded(car_sd, 3),
            "car_boundary_margin_in_sd": rounded(car_margin, 3),
            "fraction_donor_within_one_sd_of_car_boundary": rounded(near_car, 4),
        })

    # Quadrant occupancy within NK, for context on how the split lands overall.
    if car is not None and car_cut is not None:
        cc = float(car_cut)
        h, c = hla[nk], car[nk]
        donor_side = (h < hla_cut) if dim else (h > hla_cut)
        out["quadrant_occupancy"] = {
            "donor_car_pos": round(float(np.mean(donor_side & (c > cc))), 5),
            "donor_car_neg": round(float(np.mean(donor_side & (c <= cc))), 5),
            "host_car_pos": round(float(np.mean(~donor_side & (c > cc))), 5),
            "host_car_neg": round(float(np.mean(~donor_side & (c <= cc))), 5),
            "n_nk": n_nk,
        }

    for name, val, n_pool in (("hla_host", m_host, n_host),
                              ("hla_donor", m_donor, n_donor),
                              ("car", car_margin, n_donor)):
        rec.add(
            tier=2, subject_type="gate", marker=name.split("_")[0], timepoint=tp,
            metric=f"donor_car_quadrant_margin_{name}", value=rounded(val, 3), unit=U_ROBUST_SD,
            note="distance from this cloud's mode to the HLA/CAR quadrant boundary it faces",
        )
        book.raise_if(
            val is not None and val < th.quadrant_margin_min,
            code=F_QUADRANT_MARGIN, tier=2, subject=f"donor/CAR {name} boundary @ {tp}",
            rule=(f"the cloud's mode sits < {th.quadrant_margin_min} robust SD from the "
                  f"{name.upper()} quadrant boundary in the HLA x CAR gate"),
            measured={
                "margin_in_sd": rounded(val, 3),
                "n_pool": n_pool,
                "hla_cut": rounded(hla_cut, 3),
                "car_cut": None if car_cut is None else rounded(float(car_cut), 3),
                "hla_dim_polarity": dim,
            },
            resolution_hint=(
                "This boundary defines a reported population directly — the HLA one sets "
                "%Donor NK, the CAR one sets %CAR+ (of Donor NK). A mode this close to it "
                "means the count is set by the boundary rather than by a gap, and the 1-D "
                "overlay for the same marker can still look clean."
            ),
            value=val, threshold=th.quadrant_margin_min, higher_is_worse=False,
        )
    for name, val, n_pool in (("NK cloud vs the HLA boundary", near_hla, n_nk),
                              ("Donor NK vs the CAR boundary", near_car, n_donor)):
        rec.add(
            tier=2, subject_type="gate", marker="donor_car", timepoint=tp,
            metric=f"boundary_event_mass|{name}", value=rounded(val, 4), unit=U_FRACTION,
            note="fraction of the parent population within one robust SD of this boundary",
        )
        book.raise_if(
            val is not None and val > th.boundary_mass_max,
            code=F_BOUNDARY_MASS, tier=2, subject=f"{name} @ {tp}",
            rule=(f"> {th.boundary_mass_max} of the parent population lies within one robust "
                  "SD of this HLA x CAR boundary"),
            measured={"fraction_near_boundary": rounded(val, 4), "n_pool": n_pool},
            resolution_hint=(
                "Many events sit where a small move of this boundary reassigns them. Tier 3's "
                "cutoff-attributable range for the metric this boundary defines says what "
                "that is worth in percentage points."
            ),
            value=val, threshold=th.boundary_mass_max,
        )
    return out


def _scatter_containment(ctx: DiagnosticsContext, s: SampleState, rec: MetricRecorder,
                         book: FlagBook, tp: str) -> dict[str, Any]:
    """Does the locked FSC/SSC gate still contain this timepoint's own density island?

    Re-estimates the island freely (``lymph_scatter_gate`` with no cap) and asks what
    fraction of it survives the bounds the pipeline actually applied. The comparison is
    REPORTED rather than applied — ``soft_lock_scatter`` already does the applying.
    """
    th = ctx.thresholds
    out: dict[str, Any] = {"available": False}
    if s.df is None:
        out["reason"] = "event frame released"
        return out
    fsca, ssca = (s.scatter[0], s.scatter[1]) if s.scatter and len(s.scatter) >= 2 else (None, None)
    if fsca is None or ssca is None:
        out["reason"] = "scatter channels unavailable"
        return out
    applied_lo = s.applied.get("fsc_lo")
    applied_hi = s.applied.get("fsc_hi")
    applied_ssc = s.applied.get("ssc_hi")
    if None in (applied_lo, applied_hi, applied_ssc):
        out["reason"] = "applied scatter bounds unavailable"
        return out
    try:
        free_mask, free_lo, free_hi, free_ssc = lymph_scatter_gate(fsca, ssca, ssc_cap=None)
    except Exception as e:  # a density-island failure must not take down the tier
        out["reason"] = f"free density island could not be estimated: {e}"
        return out

    n_free = int(np.count_nonzero(free_mask))
    if n_free < th.min_parent_events:
        out["reason"] = f"free island has only {n_free} events"
        return out

    inside = (
        free_mask
        & (fsca >= float(applied_lo))
        & (fsca <= float(applied_hi))
        & (ssca < float(applied_ssc))
    )
    containment = float(np.count_nonzero(inside)) / float(n_free)

    fx, fy = fsca[free_mask], ssca[free_mask]
    sd_f, sd_s = E.robust_sd(fx), E.robust_sd(fy)
    centre_applied_fsc = 0.5 * (float(applied_lo) + float(applied_hi))
    offset_fsc = None if not sd_f else (float(np.median(fx)) - centre_applied_fsc) / sd_f
    ssc_headroom = None if not sd_s else (float(applied_ssc) - float(np.median(fy))) / sd_s

    out.update(
        {
            "available": True,
            "applied_fsc_lo": rounded(applied_lo, 5),
            "applied_fsc_hi": rounded(applied_hi, 5),
            "applied_ssc_hi": rounded(applied_ssc, 5),
            "free_fsc_lo": rounded(free_lo, 5),
            "free_fsc_hi": rounded(free_hi, 5),
            "free_ssc_hi": rounded(free_ssc, 5),
            "free_island_events": n_free,
            "containment": rounded(containment, 4),
            "spillover": rounded(1.0 - containment, 4),
            "fsc_centroid_offset_in_sd": rounded(offset_fsc, 3),
            "ssc_headroom_in_sd": rounded(ssc_headroom, 3),
            "scatter_method_applied": s.applied.get("scatter_method"),
        }
    )
    rec.add(
        tier=2, subject_type="gate", marker="scatter", timepoint=tp,
        metric="locked_scatter_containment", value=rounded(containment, 4), unit=U_FRACTION,
        note="fraction of the freely re-estimated density island inside the applied bounds",
    )
    rec.add(
        tier=2, subject_type="gate", marker="scatter", timepoint=tp,
        metric="fsc_centroid_offset", value=rounded(offset_fsc, 3), unit=U_ROBUST_SD,
        note="island centroid vs the centre of the applied FSC window",
    )
    book.raise_if(
        containment < th.scatter_containment_min,
        code=F_SCATTER_CONTAINMENT, tier=2, subject=f"locked scatter @ {tp}",
        rule=f"locked scatter bounds contain < {th.scatter_containment_min} of the free island",
        measured={
            "containment": rounded(containment, 4),
            "free_island_events": n_free,
            "applied": [rounded(applied_lo, 5), rounded(applied_hi, 5), rounded(applied_ssc, 5)],
            "free": [rounded(free_lo, 5), rounded(free_hi, 5), rounded(free_ssc, 5)],
        },
        resolution_hint=(
            "The locked gate no longer contains the cloud it was locked around, so the "
            "lymphocyte denominator at this timepoint is cut by the gate rather than by the "
            "cloud's own edge. Every '(of lymph)' percentage inherits this."
        ),
        value=containment, threshold=th.scatter_containment_min, higher_is_worse=False,
    )
    return out
