"""Tier 3 — does any of this change the answer?

Without this tier every other number is aesthetic. A cytometrist never stops at "the cutoff
is slightly off"; they immediately ask *so what?* — and the answer is either "nothing, there
are no events there" or "%NK swings by four points".

Three flavours, in ascending order of value:

**(a) Local fragility.** Each cutoff is perturbed and every dependent percentage recomputed.
Step sizes are chosen to mean something rather than to be round: one overlay histogram bin
(the precision floor of anything a human can read off the PNG) and 10% of the reference
negative-to-positive gap (a technical/biological scale).

**(b) The locked-versus-per-sample counterfactual.** ``calibrate_patient`` already computes
each file's own valley before the anchoring discards it in favour of the locked reference
value. Replaying with those per-file valleys answers the question the anchored method is
built around: what would this timepoint have reported under its own cutoff? The spread of the
per-sample numbers IS the run-to-run drift the unified cutoff exists to remove, so this
quantifies the variance the method removes *and* the bias it may introduce at a drifted
timepoint — in one table. Only the fluorescence cutoffs are swapped; the scatter gate and the
donor/CAR cutoffs are held fixed so the comparison isolates one variable.

**(c) A cutoff-attributable range on every headline number**, so a reported percentage can
carry its own uncertainty:

    NK = 18.3% of lymph (cutoff-attributable 16.1-19.4; counting +/-1.2)

Two separately labelled error sources, because they have different remedies: more events
fixes counting noise, and only a better cutoff fixes the other.

Two honesty constraints are enforced. Perturbations are one-at-a-time, so the range is a
marginal sensitivity and NOT a joint confidence region — the emitted note says so. And the
tier carries a wall-clock budget; if it runs out, the markers that were dropped are named in
the output rather than quietly omitted.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ..gates import gate_with, percentages
from .context import DiagnosticsContext, SampleState
from .flags import F_COUNTERFACTUAL_GAP, F_FRAGILE_METRIC, FlagBook
from .markers import HEADLINE_METRICS, MarkerSpec
from .record import U_PP, U_RATIO, MetricRecorder, rounded, safe_ratio

# Fluorescence cutoffs the counterfactual swaps. Scatter and donor/CAR are deliberately held
# fixed so the comparison isolates the fluorescence anchoring.
_COUNTERFACTUAL_KEYS = ("ld", "cd45", "cd14", "cd19", "cd3", "cd56", "cd4")

# Default event cap for the perturbation sweep. Sensitivity is a derivative — a relative
# change — so a subsample resolves it fine, and the SAME subsample is reused across every
# perturbation of a file, which makes each delta exact for that subsample rather than
# confounded with resampling noise.
DEFAULT_SENSITIVITY_EVENTS = 40_000
DEFAULT_TIME_BUDGET_S = 420.0


def run(
    ctx: DiagnosticsContext,
    rec: MetricRecorder,
    book: FlagBook,
    tier1: dict[str, Any],
    tier0: dict[str, Any],
    max_events: Optional[int] = None,
    time_budget_s: Optional[float] = None,
) -> dict[str, Any]:
    """Compute tier 3. Returns a JSON-ready dict including the uncertainty envelope."""
    th = ctx.thresholds
    cap = int(max_events if max_events is not None
              else ctx.meta.get("diagnostics_sensitivity_events", DEFAULT_SENSITIVITY_EVENTS))
    budget = float(time_budget_s if time_budget_s is not None
                   else ctx.meta.get("diagnostics_sensitivity_seconds", DEFAULT_TIME_BUDGET_S))
    started = time.time()

    ref_markers = (tier1 or {}).get("markers", {})
    offsets = _offset_plan(ctx, ref_markers)

    out: dict[str, Any] = {
        "available": True,
        "event_cap_per_file": cap,
        "time_budget_s": budget,
        "offset_plan": offsets,
        "perturbations": [],
        "counterfactual": [],
        "envelope": [],
        "policy_variance": {},
        "coverage": {"completed": [], "dropped": [], "reason": None},
        "notes": [
            "Perturbations are ONE-AT-A-TIME. The reported range is a marginal sensitivity to "
            "each cutoff separately, NOT a joint confidence region over all cutoffs.",
            "Deltas are computed against the same event subsample as the perturbed run, so "
            "they isolate the cutoff change from any resampling difference.",
            "The counterfactual swaps only the fluorescence cutoffs; the scatter gate and the "
            "donor/CAR cutoffs are held at the anchored values.",
            "Every tier-3 value is re-gated on a CAPPED subsample, so columns ending "
            "'_at_subsample' are NOT the pipeline's reported numbers and will differ from "
            "multilineage.csv by a fraction of a point. Each row carries headline_value_pct "
            "alongside; quote THAT as the number and the tier-3 span as its uncertainty. "
            "Differences and ranges are unaffected: both sides of every comparison come from "
            "the same subsample.",
        ],
    }

    per_metric_extremes: dict[tuple[str, str], dict[str, Any]] = {}

    for s in ctx.samples:
        tp = s.timepoint or s.filename
        if s.df is None:
            out["coverage"]["dropped"].append({"timepoint": tp, "reason": "event frame released"})
            continue
        sub = _subsample(s, cap, th.random_seed)
        _base_masks, base_pct = _gate(ctx, sub, s.cuts)
        if base_pct is None:
            out["coverage"]["dropped"].append(
                {"timepoint": tp, "reason": "baseline re-gate failed on the subsample"})
            continue

        # How much the subsample alone shifts each headline number. Reported so a reader can
        # see that the deltas below are larger than the subsampling artefact.
        for metric in HEADLINE_METRICS:
            full_v, sub_v = s.pct.get(metric), base_pct.get(metric)
            if full_v is None or sub_v is None:
                continue
            rec.add(
                tier=3, subject_type="metric", timepoint=tp,
                metric=f"subsample_shift|{metric}",
                value=rounded(float(sub_v) - float(full_v), 4), unit=U_PP,
                note=(f"effect of the {len(sub)}-event sensitivity subsample on this metric "
                      "(not a cutoff effect; the floor below which deltas are not meaningful)"),
            )

        exhausted = False
        for spec in ctx.marker_specs():
            if time.time() - started > budget:
                exhausted = True
                out["coverage"]["dropped"].append(
                    {"timepoint": tp, "marker": spec.key, "reason": "time budget exhausted"})
                continue
            plan = offsets.get(spec.key)
            if not plan or not plan.get("offsets"):
                out["coverage"]["dropped"].append(
                    {"timepoint": tp, "marker": spec.key,
                     "reason": "no offset scale available (missing reference structure)"})
                continue
            rows = _sweep_marker(ctx, s, tp, spec, sub, base_pct, plan, rec)
            out["perturbations"].extend(rows)
            out["coverage"]["completed"].append({"timepoint": tp, "marker": spec.key})
            for r in rows:
                _accumulate_extremes(per_metric_extremes, r)
        if exhausted:
            out["coverage"]["reason"] = (
                f"wall-clock budget of {budget}s exhausted; the dropped list above names every "
                "marker x timepoint that was NOT swept. Raise "
                "metadata.json -> diagnostics_sensitivity_seconds to cover them."
            )

        cf = _counterfactual(ctx, s, tp, sub, base_pct, rec, book)
        if cf:
            out["counterfactual"].extend(cf)

    out["envelope"] = _envelope(ctx, tier0, per_metric_extremes, rec, book)
    out["policy_variance"] = _policy_variance(out["counterfactual"], rec)
    out["elapsed_s"] = round(time.time() - started, 2)
    return out


# ── perturbation plan ─────────────────────────────────────────────────────────
def _offset_plan(ctx: DiagnosticsContext, ref_markers: dict[str, Any]) -> dict[str, Any]:
    """Per-marker perturbation step sizes, each with the reason it was chosen."""
    plan: dict[str, Any] = {}
    for spec in ctx.marker_specs():
        m = ref_markers.get(spec.key) or {}
        cut = ctx.cut_value(spec.key)
        if cut is None:
            continue
        steps: list[dict[str, Any]] = []
        axis = m.get("overlay_axis") or {}
        bw = axis.get("bin_width")
        if bw:
            steps.append({"name": "one_overlay_bin", "size": float(bw),
                          "why": "one histogram bin of the overlay PNG — the precision floor "
                                 "of anything readable off the figure"})
        gap = m.get("mode_gap")
        if gap:
            steps.append({"name": "ten_percent_of_gap", "size": 0.10 * float(gap),
                          "why": "10% of the reference negative-to-positive gap — a "
                                 "technical/biological scale"})
        if not steps:
            sd = m.get("negative_sd_used")
            if sd:
                steps.append({"name": "quarter_negative_sd", "size": 0.25 * float(sd),
                              "why": "fallback: a quarter of the negative-population SD "
                                     "(no overlay axis or mode gap was available)"})
        offsets: list[dict[str, Any]] = []
        for st in steps:
            if st["size"] <= 0:
                continue
            for sign in (-1.0, 1.0):
                offsets.append({
                    "name": f"{'minus' if sign < 0 else 'plus'}_{st['name']}",
                    "delta": sign * st["size"],
                    "why": st["why"],
                })
        plan[spec.key] = {"cutoff": rounded(cut, 3), "offsets": offsets,
                          "scales": steps}
    return plan


def _subsample(s: SampleState, cap: int, seed: int):
    """A deterministic event subsample, reused across every perturbation of this file.

    Reusing ONE subsample per file is what makes each delta exact: the perturbed and
    unperturbed runs see identical events, so the difference is purely the cutoff move.
    """
    df = s.df
    n = len(df)
    if n <= cap:
        return df
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, cap, replace=False))
    return df.iloc[idx].reset_index(drop=True)


def _gate(ctx: DiagnosticsContext, df, cuts: dict[str, Any]):
    """Re-gate ``df`` with ``cuts``; returns (masks, percentages) or (None, None)."""
    try:
        masks, _g, _scat, _applied = gate_with(df, cuts, ctx.channels, ssc_cap=None)
        return masks, percentages(masks)
    except Exception as e:
        print(f"  [diagnostics] re-gate failed: {e}", flush=True)
        return None, None


# ── (a) local fragility ───────────────────────────────────────────────────────
def _sweep_marker(
    ctx: DiagnosticsContext, s: SampleState, tp: str, spec: MarkerSpec,
    sub, base_pct: dict[str, Any], plan: dict[str, Any], rec: MetricRecorder,
) -> list[dict[str, Any]]:
    """Perturb one cutoff through its offset plan and record every dependent metric's move."""
    rows: list[dict[str, Any]] = []
    base_cut = s.cuts.get(spec.key)
    if base_cut is None:
        return rows
    targets = [m for m in HEADLINE_METRICS if m in spec.downstream]
    if not targets:
        return rows

    for off in plan["offsets"]:
        trial = dict(s.cuts)
        trial[spec.key] = float(base_cut) + float(off["delta"])
        _masks, pct = _gate(ctx, sub, trial)
        if pct is None:
            continue
        for metric in targets:
            b, v = base_pct.get(metric), pct.get(metric)
            if b is None or v is None:
                continue
            delta = float(v) - float(b)
            rows.append(
                {
                    "timepoint": tp,
                    "marker": spec.key,
                    "offset_name": off["name"],
                    "offset_delta": rounded(off["delta"], 4),
                    "offset_rationale": off["why"],
                    "cutoff_base": rounded(base_cut, 3),
                    "cutoff_trial": rounded(trial[spec.key], 3),
                    "metric": metric,
                    "value_base_pct": rounded(b, 4),
                    "value_trial_pct": rounded(v, 4),
                    "delta_pp": rounded(delta, 4),
                    "delta_pp_per_unit_cutoff": rounded(
                        safe_ratio(delta, off["delta"]), 6),
                    "relative_delta": rounded(safe_ratio(delta, b), 5),
                }
            )
            rec.add(
                tier=3, subject_type="metric", marker=spec.key, timepoint=tp,
                metric=f"sensitivity_{off['name']}|{metric}",
                value=rounded(delta, 4), unit=U_PP,
                normalized_value=rounded(safe_ratio(delta, b), 5), normalized_unit=U_RATIO,
                note=f"{metric} moves this much when the {spec.key} cutoff moves {off['name']}",
            )
    return rows


def _accumulate_extremes(store: dict, row: dict[str, Any]) -> None:
    """Track the min/max excursion of each (timepoint, metric) across all perturbations."""
    key = (row["timepoint"], row["metric"])
    cur = store.setdefault(
        key,
        {
            "timepoint": row["timepoint"],
            "metric": row["metric"],
            "value_base_pct": row["value_base_pct"],
            "low_pct": row["value_base_pct"],
            "high_pct": row["value_base_pct"],
            "driver_low": None,
            "driver_high": None,
        },
    )
    v = row["value_trial_pct"]
    if v is None:
        return
    if cur["low_pct"] is None or v < cur["low_pct"]:
        cur["low_pct"] = v
        cur["driver_low"] = f"{row['marker']}:{row['offset_name']}"
    if cur["high_pct"] is None or v > cur["high_pct"]:
        cur["high_pct"] = v
        cur["driver_high"] = f"{row['marker']}:{row['offset_name']}"


# ── (b) locked vs per-sample counterfactual ───────────────────────────────────
def _counterfactual(
    ctx: DiagnosticsContext, s: SampleState, tp: str, sub,
    base_pct: dict[str, Any], rec: MetricRecorder, book: FlagBook,
) -> list[dict[str, Any]]:
    """What this timepoint would have reported under its OWN valleys instead of the locked ones."""
    th = ctx.thresholds
    trial = dict(s.cuts)
    swapped: dict[str, Any] = {}
    unavailable: list[str] = []
    for k in _COUNTERFACTUAL_KEYS:
        pf = s.perfile_cuts.get(k)
        if pf is None:
            if s.cuts.get(k) is not None:
                unavailable.append(k)
            continue
        if s.cuts.get(k) is None:
            continue
        if float(pf) == float(s.cuts[k]):
            continue  # already per-file (typically the reference timepoint itself)
        trial[k] = float(pf)
        swapped[k] = {"locked": rounded(s.cuts[k], 3), "per_sample": rounded(pf, 3)}

    if not swapped:
        return [
            {
                "timepoint": tp,
                "metric": None,
                "note": ("no fluorescence cutoff differs from its per-file valley here "
                         "(expected at the reference timepoint, where the two coincide)"),
                "swapped_cutoffs": {},
                "per_sample_unavailable": unavailable,
            }
        ]

    _masks, pct = _gate(ctx, sub, trial)
    if pct is None:
        return []

    rows: list[dict[str, Any]] = []
    for metric in HEADLINE_METRICS:
        b, v = base_pct.get(metric), pct.get(metric)
        if b is None or v is None:
            continue
        delta = float(v) - float(b)
        rows.append(
            {
                "timepoint": tp,
                "metric": metric,
                "headline_value_pct": rounded(s.pct.get(metric), 4),
                "locked_value_pct_at_subsample": rounded(b, 4),
                "per_sample_value_pct_at_subsample": rounded(v, 4),
                "delta_pp": rounded(delta, 4),
                "relative_delta": rounded(safe_ratio(delta, b), 5),
                "swapped_cutoffs": swapped,
                "per_sample_unavailable": unavailable,
            }
        )
        rec.add(
            tier=3, subject_type="metric", timepoint=tp,
            metric=f"locked_minus_per_sample|{metric}",
            value=rounded(-delta, 4), unit=U_PP,
            normalized_value=rounded(safe_ratio(delta, b), 5), normalized_unit=U_RATIO,
            note=("locked value minus the per-sample-cutoff value; the spread of the "
                  "per-sample numbers across timepoints is the drift anchoring removes"),
        )
        book.raise_if(
            abs(delta) > th.counterfactual_pp_max,
            code=F_COUNTERFACTUAL_GAP, tier=3, subject=f"{metric} @ {tp}",
            rule=(f"|locked - per-sample| > {th.counterfactual_pp_max} pp for this metric"),
            measured={
                "locked_pct": rounded(b, 4),
                "per_sample_pct": rounded(v, 4),
                "delta_pp": rounded(delta, 4),
                "swapped_cutoffs": swapped,
            },
            resolution_hint=(
                "Anchoring is materially changing this number. That is the method working as "
                "designed IF the negative population is stable here (tier 2 pattern "
                "STABLE_NEGATIVE_*), and a possible transfer bias if it is not. Read the two "
                "rows together before deciding."
            ),
            value=abs(delta), threshold=th.counterfactual_pp_max,
        )
    return rows


def _policy_variance(counterfactual: list[dict[str, Any]], rec: MetricRecorder) -> dict[str, Any]:
    """Cross-timepoint spread of each metric under the locked vs per-sample policy.

    The headline justification for anchoring: if the per-sample SD is materially larger than
    the locked SD, the locked policy is removing exactly that much run-to-run variance.
    """
    by_metric: dict[str, dict[str, list[float]]] = {}
    for r in counterfactual:
        m = r.get("metric")
        lk, ps = r.get("locked_value_pct_at_subsample"), r.get("per_sample_value_pct_at_subsample")
        if not m or lk is None or ps is None:
            continue
        d = by_metric.setdefault(m, {"locked": [], "per_sample": []})
        d["locked"].append(float(lk))
        d["per_sample"].append(float(ps))

    out: dict[str, Any] = {}
    for metric, d in by_metric.items():
        if len(d["locked"]) < 3:
            continue
        sd_locked = float(np.std(np.asarray(d["locked"]), ddof=1))
        sd_persample = float(np.std(np.asarray(d["per_sample"]), ddof=1))
        out[metric] = {
            "n_timepoints": len(d["locked"]),
            "sd_locked_pp": rounded(sd_locked, 4),
            "sd_per_sample_pp": rounded(sd_persample, 4),
            "variance_removed_pp": rounded(sd_persample - sd_locked, 4),
            "sd_ratio_per_sample_over_locked": rounded(safe_ratio(sd_persample, sd_locked), 4),
            "caveat": (
                "Cross-timepoint SD mixes real longitudinal biology with cutoff drift, so this "
                "is not a pure precision estimate. It is comparable BETWEEN the two policies "
                "because the biology is identical in both."
            ),
        }
        rec.add(
            tier=3, subject_type="metric", metric=f"policy_sd_reduction|{metric}",
            value=rounded(sd_persample - sd_locked, 4), unit=U_PP,
            normalized_value=rounded(safe_ratio(sd_persample, sd_locked), 4),
            normalized_unit=U_RATIO,
            note="cross-timepoint SD under per-sample cutoffs minus SD under the locked cutoff",
        )
    return out


# ── (c) uncertainty envelope ──────────────────────────────────────────────────
def _envelope(
    ctx: DiagnosticsContext, tier0: dict[str, Any],
    extremes: dict[tuple[str, str], dict[str, Any]], rec: MetricRecorder, book: FlagBook,
) -> list[dict[str, Any]]:
    """Join the cutoff-attributable range with the counting interval for each headline number."""
    th = ctx.thresholds
    counting_index = {
        (r["timepoint"], r["metric"]): r for r in (tier0 or {}).get("counting", [])
    }
    rows: list[dict[str, Any]] = []
    for (tp, metric), e in sorted(extremes.items()):
        base = e["value_base_pct"]
        lo, hi = e["low_pct"], e["high_pct"]
        span = None if (lo is None or hi is None) else float(hi) - float(lo)
        c = counting_index.get((tp, metric), {})
        # Tier 3 re-gates a capped subsample, so its base value is NOT the headline number the
        # pipeline reported. Both travel in the row, plus their difference, because a column
        # called "value_pct" that silently disagrees with multilineage.csv by a percentage
        # point invites a reader to quote the wrong one.
        headline = c.get("value_pct")
        rows.append(
            {
                "timepoint": tp,
                "metric": metric,
                "headline_value_pct": headline,
                "value_pct_at_sensitivity_subsample": base,
                "headline_minus_subsample_pp": rounded(
                    None if (headline is None or base is None)
                    else float(headline) - float(base), 4),
                "cutoff_low_pct": lo,
                "cutoff_high_pct": hi,
                "cutoff_range_pp": rounded(span, 4),
                "cutoff_range_driver_low": e["driver_low"],
                "cutoff_range_driver_high": e["driver_high"],
                "counting_ci_low_pct": c.get("counting_ci_low_pct"),
                "counting_ci_high_pct": c.get("counting_ci_high_pct"),
                "counting_ci_halfwidth_pp": c.get("counting_ci_halfwidth_pp"),
                "n_parent": c.get("n_parent"),
                "n_numerator": c.get("n_numerator"),
                "dominant_uncertainty": _dominant(span, c.get("counting_ci_halfwidth_pp")),
            }
        )
        rec.add(
            tier=3, subject_type="metric", timepoint=tp, metric=f"cutoff_range|{metric}",
            value=rounded(span, 4), unit=U_PP,
            normalized_value=rounded(safe_ratio(span, base), 5), normalized_unit=U_RATIO,
            note=("full one-at-a-time cutoff-attributable range for this reported percentage; "
                  "the range is measured on the tier-3 subsample, so apply it to the headline "
                  "value as a WIDTH rather than reading its endpoints as absolute bounds"),
        )
        book.raise_if(
            span is not None and span > th.sensitivity_pp_max,
            code=F_FRAGILE_METRIC, tier=3, subject=f"{metric} @ {tp}",
            rule=f"cutoff-attributable range > {th.sensitivity_pp_max} pp",
            measured={
                "headline_value_pct": c.get("value_pct"),
                "value_pct_at_sensitivity_subsample": base,
                "cutoff_low_pct": lo, "cutoff_high_pct": hi,
                "range_pp": rounded(span, 4),
                "driver_low": e["driver_low"], "driver_high": e["driver_high"],
            },
            resolution_hint=(
                "Quote this number with its cutoff-attributable range rather than as a point "
                "estimate. The named driver identifies WHICH cutoff to iterate on if a "
                "tighter number is needed."
            ),
            value=span, threshold=th.sensitivity_pp_max,
        )
    return rows


def _dominant(cutoff_span_pp: Optional[float], counting_half_pp: Optional[float]) -> Optional[str]:
    """Which error source is larger — the one that says what to do about it."""
    if cutoff_span_pp is None and counting_half_pp is None:
        return None
    if counting_half_pp is None:
        return "cutoff"
    if cutoff_span_pp is None:
        return "counting"
    counting_span = 2.0 * float(counting_half_pp)
    if abs(float(cutoff_span_pp) - counting_span) < 1e-9:
        return "comparable"
    return "cutoff" if float(cutoff_span_pp) > counting_span else "counting"
