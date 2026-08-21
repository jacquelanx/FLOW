"""Tier 0 — is there enough good data to say anything?

A failure here makes every higher tier meaningless, which is why it runs first and why its
output is meant to be read first. Three things get measured:

1. **The retention chain.** Already computed by the pipeline; recomputed here so it sits
   beside the counting statistics rather than in a separate table.

2. **Counting uncertainty on every reported percentage.** The interesting denominators in
   this study are small and get smaller exactly where the biology is most interesting —
   ``%CAR+ (of Donor NK)`` late after infusion can rest on a few dozen donor NK events. A
   Wilson interval travels with each percentage so a longitudinal claim cannot be built on
   counting noise. The pipeline's own ``MIN_PARENT`` / ``MIN_POS`` floors are reused.

3. **The time axis.** Every other tier asks how a cutoff was DERIVED or TRANSFERRED. None of
   them looks at the resulting time series, so a value that leaps between adjacent timepoints
   and back — the signature of a sample that differs technically from its neighbours rather
   than biologically — passes every one of them. ``_discontinuity`` measures that directly, and
   ``_per_timepoint`` gives the digest a per-SAMPLE view to sit beside the per-RULE flag list.
   Without it a bad timepoint is only visible as one entry inside a rule's tally.

4. **Within-file acquisition stability.** The FCS Time channel survives into the
   transform-space frame, so median fluorescence and event rate can be tracked across the
   run. This catches clogs, pressure instability and mid-run staining drift — and it is the
   one place where the numeric approach sees something *none of the current figures show*.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import estimators as E
from .context import DiagnosticsContext, SampleState, parent_counts, reliability
from .flags import (
    F_ACQ_DRIFT,
    F_ACQ_RATE,
    F_LOW_PARENT,
    F_LOW_POSITIVE,
    F_NO_TIME_CHANNEL,
    F_TEMPORAL_DISCONTINUITY,
    F_WIDE_COUNTING_CI,
    FlagBook,
)
from .markers import HEADLINE_METRICS, METRIC_NUMERATORS, METRIC_PARENTS
from .record import (
    U_COUNT,
    U_FRACTION,
    U_PCT,
    U_PP,
    U_RATIO,
    U_ROBUST_SD,
    MetricRecorder,
    rounded,
)

# Markers tracked for within-file drift. The lineage markers that define populations; the
# donor/CAR channels are included because a mid-run shift there moves the headline numbers.
_ACQ_MARKERS = ("cd45", "cd14", "cd19", "cd3", "cd56", "hla", "car", "ld")
_ACQ_BINS = 10


def run(ctx: DiagnosticsContext, rec: MetricRecorder, book: FlagBook) -> dict[str, Any]:
    """Compute tier 0 for every sample. Returns a JSON-ready dict."""
    th = ctx.thresholds
    samples_out: list[dict[str, Any]] = []
    counting_out: list[dict[str, Any]] = []
    per_timepoint_out: list[dict[str, Any]] = []

    for s in ctx.samples:
        tp = s.timepoint or s.filename
        n_analyzed = int(len(s.df)) if s.df is not None else None

        retention = _retention(s)
        for name, val in retention.items():
            rec.add(
                tier=0,
                subject_type="sample",
                timepoint=tp,
                metric=f"retention_{name}",
                value=rounded(val, 3),
                unit=U_PCT,
                note="fraction of analyzed events surviving this gate step",
            )
        rec.add(
            tier=0, subject_type="sample", timepoint=tp, metric="n_total_events",
            value=s.n_total_events, unit=U_COUNT,
            note="instrument event count before subsampling",
        )
        rec.add(
            tier=0, subject_type="sample", timepoint=tp, metric="n_events_analyzed",
            value=n_analyzed, unit=U_COUNT,
            normalized_value=rounded(
                None if not s.n_total_events else (n_analyzed or 0) / s.n_total_events, 4),
            normalized_unit=U_FRACTION,
            note="events actually gated (pipeline subsample)",
        )

        acq = _acquisition(ctx, s, rec, book, tp)
        counting = _counting(s, rec, book, tp, th)
        counting_out.extend(counting)
        per_timepoint_out.append(
            {
                "timepoint": tp,
                "filename": s.filename,
                "n_total_events": s.n_total_events,
                "n_events_analyzed": n_analyzed,
                "n_lymphocytes": _mask_count(s, "lympho"),
                "n_cd45p": _mask_count(s, "cd45p"),
                "n_live": _mask_count(s, "live"),
                "retention_live_pct": rounded(retention.get("live"), 3),
                "retention_cd45p_pct": rounded(retention.get("cd45p"), 3),
                "retention_lymphocytes_pct": rounded(retention.get("lymphocytes"), 3),
                "lineage_purity_pct": rounded(retention.get("lineage_purity"), 3),
                "acquisition_signal_drift_sd": acq.get("max_median_drift_in_sd"),
                "acquisition_rate_cv": acq.get("rate_cv"),
            }
        )
        samples_out.append(
            {
                "timepoint": tp,
                "filename": s.filename,
                "n_total_events": s.n_total_events,
                "n_events_analyzed": n_analyzed,
                "retention_pct": {k: rounded(v, 3) for k, v in retention.items()},
                "acquisition": acq,
            }
        )

    discontinuity_out = _discontinuity(counting_out, ctx, rec, book, th)

    return {
        "samples": samples_out,
        "per_timepoint": per_timepoint_out,
        "counting": counting_out,
        "discontinuity": discontinuity_out,
        "min_parent_events": th.min_parent_events,
        "min_positive_events": th.min_positive_events,
        "note": (
            "Counting intervals are Wilson score intervals on the gated event counts. They "
            "describe sampling noise ONLY — they say nothing about whether the cutoff that "
            "defined the numerator is in the right place (tiers 1-3)."
        ),
    }


def _mask_count(s: SampleState, key: str) -> Optional[int]:
    m = s.masks.get(key)
    return None if m is None else int(np.count_nonzero(m))


def _discontinuity(
    counting: list[dict[str, Any]],
    ctx: DiagnosticsContext,
    rec: MetricRecorder,
    book: FlagBook,
    th,
) -> list[dict[str, Any]]:
    """Does each timepoint's value sit inside the envelope of its two temporal neighbours?

    **Why an envelope and not a difference.** A real biological change is a STEP: the value
    moves and the neighbours differ from each other in the same direction. A sample that
    differs technically from its neighbours is a SPIKE: it lands far outside both, while they
    remain consistent with each other. Testing "outside the neighbours' envelope" separates the
    two without asserting any rate of change — a monotone rise puts every interior value
    BETWEEN its neighbours, so a trend cannot trigger this no matter how steep it is.

    **The tolerance is the data's own noise.** The envelope is widened by the combined Wilson
    counting intervals of the values being compared, so the rule fires only when the excursion
    is larger than the sampling noise of the numbers themselves.

    Endpoints have one neighbour and are reported as not-assessable rather than omitted. The
    check is also honest about its own weakness: one bad timepoint makes its GOOD neighbours
    look like spikes, so the resolution hint says to read a run of flags together.
    """
    by_metric: dict[str, list[dict[str, Any]]] = {}
    order = {tp: i for i, tp in enumerate(ctx.timepoint_labels())}
    for row in counting:
        by_metric.setdefault(row["metric"], []).append(row)

    out: list[dict[str, Any]] = []
    k = float(th.discontinuity_ci_multiple)
    for metric, rows in by_metric.items():
        series = sorted(rows, key=lambda r: order.get(r["timepoint"], 999))
        for i, cur in enumerate(series):
            tp = cur["timepoint"]
            prev = series[i - 1] if i > 0 else None
            nxt = series[i + 1] if i + 1 < len(series) else None
            base = {"timepoint": tp, "metric": metric, "value_pct": cur["value_pct"],
                    "n_parent": cur["n_parent"]}
            if prev is None or nxt is None:
                out.append({**base, "assessable": False,
                            "reason": ("first timepoint in the series — no earlier neighbour"
                                       if prev is None else
                                       "last timepoint in the series — no later neighbour")})
                continue
            v, vp, vn = cur["value_pct"], prev["value_pct"], nxt["value_pct"]
            hw, hwp, hwn = (cur["counting_ci_halfwidth_pp"], prev["counting_ci_halfwidth_pp"],
                            nxt["counting_ci_halfwidth_pp"])
            if None in (v, vp, vn) or None in (hw, hwp, hwn):
                out.append({**base, "assessable": False,
                            "reason": "a value or its counting interval could not be measured "
                                      "at this timepoint or a neighbour"})
                continue
            lo_n, hi_n = min(vp, vn), max(vp, vn)
            tol_lo = k * (float(hw) + float(hwp if vp <= vn else hwn))
            tol_hi = k * (float(hw) + float(hwn if vn >= vp else hwp))
            below = lo_n - float(v) - tol_lo
            above = float(v) - hi_n - tol_hi
            excursion_pp = max(below, above)
            outside = excursion_pp > 0
            tol = tol_lo if below >= above else tol_hi
            row = {
                **base,
                "assessable": True,
                "prev_timepoint": prev["timepoint"], "prev_value_pct": vp,
                "next_timepoint": nxt["timepoint"], "next_value_pct": vn,
                "neighbour_envelope_low_pct": rounded(lo_n, 3),
                "neighbour_envelope_high_pct": rounded(hi_n, 3),
                "tolerance_pp": rounded(tol, 3),
                "excursion_beyond_tolerance_pp": rounded(excursion_pp if outside else 0.0, 3),
                "excursion_in_tolerances": rounded(
                    (excursion_pp / tol + 1.0) if (outside and tol) else None, 3),
                "direction": ("below both neighbours" if below >= above
                              else "above both neighbours") if outside else "within envelope",
                "discontinuous": bool(outside),
            }
            out.append(row)
            rec.add(
                tier=0, subject_type="metric", timepoint=tp,
                metric=f"temporal_excursion|{metric}",
                value=rounded(excursion_pp if outside else 0.0, 3), unit=U_PP,
                normalized_value=row["excursion_in_tolerances"], normalized_unit=U_RATIO,
                note=("distance outside the two neighbours' envelope; normalized = in multiples "
                      "of the combined counting intervals"),
            )
            book.raise_if(
                outside,
                code=F_TEMPORAL_DISCONTINUITY, tier=0, subject=f"{metric} @ {tp}",
                rule=(f"value lies outside the envelope of BOTH temporal neighbours by more "
                      f"than {th.discontinuity_ci_multiple}x their combined Wilson counting "
                      f"intervals (a spike, not a step: a monotone trend cannot trigger this)"),
                measured={"metric": metric, "value_pct": v, "prev": {prev["timepoint"]: vp},
                          "next": {nxt["timepoint"]: vn}, "tolerance_pp": rounded(tol, 3),
                          "excursion_pp": rounded(excursion_pp, 3),
                          "n_parent": cur["n_parent"], "direction": row["direction"]},
                resolution_hint=(
                    "Sampling noise does not explain this value, so something about the sample "
                    "differs from its neighbours: check whether this file carries the same "
                    "panel, tube and specimen type before using it in a longitudinal claim, and "
                    "say explicitly whether you are including or excluding it. Read a RUN of "
                    "these flags together — one anomalous timepoint makes its otherwise sound "
                    "neighbours look like spikes too."
                ),
                value=excursion_pp + (tol or 0.0), threshold=(tol or None),
            )
    return out


def _retention(s: SampleState) -> dict[str, Optional[float]]:
    """Gate-step survival as a percentage of analyzed events (mirrors the pipeline's strip)."""
    if s.df is None:
        return {}
    n = float(len(s.df))
    if n <= 0:
        return {}
    out: dict[str, Optional[float]] = {}
    for name, key in (
        ("scatter", "lymph_scatter"),
        ("singlets", "sing"),
        ("live", "live"),
        ("cd45p", "cd45p"),
        ("lymphocytes", "lympho"),
    ):
        m = s.masks.get(key)
        out[name] = None if m is None else 100.0 * float(np.count_nonzero(m)) / n
    purity = s.pct.get("lineage_purity")
    out["lineage_purity"] = None if purity is None else float(purity)
    return out


def _counting(
    s: SampleState,
    rec: MetricRecorder,
    book: FlagBook,
    tp: str,
    th,
) -> list[dict[str, Any]]:
    """Wilson intervals + countability for every headline metric at one timepoint."""
    rows: list[dict[str, Any]] = []
    for metric in HEADLINE_METRICS:
        num_key = METRIC_NUMERATORS.get(metric)
        den_key = METRIC_PARENTS.get(metric)
        if not num_key or not den_key:
            continue
        n_pos, n_parent = parent_counts(s, num_key, den_key)
        rel = reliability(n_parent, n_pos)
        value = s.pct.get(metric)
        lo = hi = half = None
        if n_parent:
            lo_f, hi_f = E.wilson_interval(n_pos or 0, n_parent)
            if lo_f is not None and hi_f is not None:
                lo, hi = 100.0 * lo_f, 100.0 * hi_f
                half = 0.5 * (hi - lo)

        row = {
            "timepoint": tp,
            "metric": metric,
            "value_pct": rounded(value, 3),
            "n_numerator": n_pos,
            "n_parent": n_parent,
            "numerator_mask": num_key,
            "parent_mask": den_key,
            "counting_ci_low_pct": rounded(lo, 3),
            "counting_ci_high_pct": rounded(hi, 3),
            "counting_ci_halfwidth_pp": rounded(half, 3),
            "parent_ok": rel["parent_ok"],
            "positive_ok": rel["positive_ok"],
        }
        rows.append(row)

        rec.add(
            tier=0, subject_type="metric", timepoint=tp, metric=f"counting_ci_halfwidth|{metric}",
            value=rounded(half, 3), unit=U_PP,
            normalized_value=rounded(
                None if not value else (half / float(value) if half is not None and float(value) else None), 4),
            normalized_unit=U_RATIO,
            note=f"Wilson half-width on {n_pos}/{n_parent} events; normalized = relative to the value itself",
        )

        book.raise_if(
            n_parent is not None and n_parent < th.min_parent_events,
            code=F_LOW_PARENT, tier=0, subject=f"{metric} @ {tp}",
            rule=(f"parent events < MIN_PARENT ({th.min_parent_events}); pipeline constant "
                  "from calibrate.py"),
            measured={"n_parent": n_parent, "metric": metric, "parent_mask": den_key},
            resolution_hint=(
                "Treat this timepoint's value as uncountable rather than small. Acquiring more "
                "events is the only fix; no cutoff change recovers a missing denominator."
            ),
            value=n_parent, threshold=th.min_parent_events, higher_is_worse=False,
        )
        book.raise_if(
            n_parent is not None and n_parent >= th.min_parent_events
            and n_pos is not None and n_pos < th.min_positive_events,
            code=F_LOW_POSITIVE, tier=0, subject=f"{metric} @ {tp}",
            rule=f"numerator events < MIN_POS ({th.min_positive_events}); pipeline constant",
            measured={"n_positive": n_pos, "n_parent": n_parent, "metric": metric},
            resolution_hint=(
                "The percentage is a ratio of very few events; quote it with the Wilson "
                "interval or as 'below detection' rather than as a point estimate."
            ),
            value=n_pos, threshold=th.min_positive_events, higher_is_worse=False,
        )
        book.raise_if(
            half is not None and half > th.counting_ci_pp_max,
            code=F_WIDE_COUNTING_CI, tier=0, subject=f"{metric} @ {tp}",
            rule=f"Wilson counting half-width > {th.counting_ci_pp_max} pp",
            measured={
                "halfwidth_pp": rounded(half, 3), "value_pct": rounded(value, 3),
                "n_positive": n_pos, "n_parent": n_parent,
            },
            resolution_hint=(
                "Counting noise alone spans more than the flagged width. Any longitudinal "
                "change smaller than this interval is not resolvable at this event count."
            ),
            value=half, threshold=th.counting_ci_pp_max,
        )
    return rows


def _acquisition(
    ctx: DiagnosticsContext,
    s: SampleState,
    rec: MetricRecorder,
    book: FlagBook,
    tp: str,
) -> dict[str, Any]:
    """Within-file stability across acquisition time — rate steadiness and signal drift."""
    th = ctx.thresholds
    out: dict[str, Any] = {"time_channel": None, "event_rate_cv": None, "markers": {}}
    if s.df is None:
        return out
    tcol = ctx.time_channel(s.df)
    if tcol is None:
        book.add(
            code=F_NO_TIME_CHANNEL, tier=0, subject=f"{tp}",
            rule="no acquisition Time channel present in the FCS frame",
            measured={"file": s.filename},
            resolution_hint=(
                "Acquisition stability cannot be assessed for this file. Every other tier is "
                "unaffected; only the within-run drift check is unavailable."
            ),
        )
        return out

    tvals = np.asarray(s.df[tcol].values, dtype=float)
    ok = np.isfinite(tvals)
    if ok.sum() < _ACQ_BINS * 20 or np.nanmax(tvals[ok]) <= np.nanmin(tvals[ok]):
        out["time_channel"] = tcol
        out["note"] = "Time channel present but unusable (constant or too few events)"
        return out
    out["time_channel"] = tcol

    # Event rate: equal-WIDTH time bins, so counts are proportional to instantaneous rate.
    # (Equal-count bins would be flat by construction and measure nothing.)
    t_lo, t_hi = float(np.nanmin(tvals[ok])), float(np.nanmax(tvals[ok]))
    counts, _ = np.histogram(tvals[ok], bins=np.linspace(t_lo, t_hi, _ACQ_BINS + 1))
    rate_cv = E.coefficient_of_variation(counts)
    out["event_rate_cv"] = rounded(rate_cv, 4)
    out["acquisition_span"] = rounded(t_hi - t_lo, 3)
    rec.add(
        tier=0, subject_type="sample", timepoint=tp, metric="event_rate_cv",
        value=rounded(rate_cv, 4), unit=U_RATIO,
        note=f"CV of event counts across {_ACQ_BINS} equal-width bins of '{tcol}'",
    )
    book.raise_if(
        rate_cv is not None and rate_cv > th.acquisition_rate_cv_max,
        code=F_ACQ_RATE, tier=0, subject=f"{tp}",
        rule=f"event-rate CV across acquisition > {th.acquisition_rate_cv_max}",
        measured={"event_rate_cv": rounded(rate_cv, 4), "n_bins": _ACQ_BINS,
                  "counts": [int(c) for c in counts]},
        resolution_hint=(
            "Uneven acquisition (clog, pressure change, or a paused run). Check whether the "
            "unsteady stretch coincides with a shift in this file's marker medians below."
        ),
        value=rate_cv, threshold=th.acquisition_rate_cv_max,
    )

    # Signal drift measured on a stable, abundant reference population rather than all
    # events: a change in population MIX across the run would otherwise masquerade as drift.
    pool_key = "cd45p" if s.masks.get("cd45p") is not None and np.count_nonzero(s.masks["cd45p"]) >= 500 else "sing"
    pool = s.masks.get(pool_key)
    if pool is None:
        return out
    out["drift_pool"] = pool_key

    for key in _ACQ_MARKERS:
        vals = s.signals.get(key)
        if vals is None:
            continue
        v = vals[pool]
        t = tvals[pool]
        binned = E.binned_medians(v, t, n_bins=_ACQ_BINS)
        if binned is None:
            continue
        _centres, meds, _counts = binned
        span = float(np.max(meds) - np.min(meds))
        net = float(meds[-1] - meds[0])
        sd = E.robust_sd(v)
        span_sd = None if not sd else span / sd
        net_sd = None if not sd else net / sd
        out["markers"][key] = {
            "median_span": rounded(span, 3),
            "median_net_change": rounded(net, 3),
            "median_span_in_robust_sd": rounded(span_sd, 4),
            "median_net_change_in_robust_sd": rounded(net_sd, 4),
            "robust_sd": rounded(sd, 3),
            "n_bins": int(len(meds)),
        }
        rec.add(
            tier=0, subject_type="marker", marker=key, timepoint=tp,
            metric="acquisition_median_span",
            value=rounded(span, 3), unit="biex transform units",
            normalized_value=rounded(span_sd, 4), normalized_unit=U_ROBUST_SD,
            note=f"max-min of {key} medians across {_ACQ_BINS} time bins within '{pool_key}'",
        )
        book.raise_if(
            span_sd is not None and span_sd > th.acquisition_drift_sd_max,
            code=F_ACQ_DRIFT, tier=0, subject=f"{key} @ {tp}",
            rule=(f"within-file median span across acquisition time > "
                  f"{th.acquisition_drift_sd_max} robust SD of the marker"),
            measured={
                "median_span_in_robust_sd": rounded(span_sd, 4),
                "median_net_change_in_robust_sd": rounded(net_sd, 4),
                "pool": pool_key,
                "medians": [rounded(m, 3) for m in meds.tolist()],
            },
            resolution_hint=(
                "The marker's central tendency moved during acquisition, so a single cutoff "
                "cannot be optimal across the whole file. Compare with the event-rate CV: "
                "co-occurring instability points to an instrument cause rather than biology."
            ),
            value=span_sd, threshold=th.acquisition_drift_sd_max,
        )
    return out
