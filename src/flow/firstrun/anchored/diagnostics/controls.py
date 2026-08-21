"""Tier 4 — internal controls: falsifiable checks against populations with known truth.

The strongest numbers in the whole diagnostics set, because they need no threshold
philosophy. Some populations in this experiment have a KNOWN answer, so the metric is simply
"did the locked cutoff get it right?". This dataset supplies four for free:

===========================  ==================  ==================================================
Control                      Expected            What it validates
===========================  ==================  ==================================================
Host (non-donor) NK, CAR      ~0% CAR+           CAR cutoff false-positive rate, measured IN the
                                                 same tube as the real measurement
Pre-infusion donor NK         ~0% donor          Donor-HLA cutoff specificity — no donor cells
                                                 exist before infusion, and this is measured at
                                                 the very timepoint the anchor was derived from
NT-NK control tube            ~0% CAR+           CAR cutoff specificity in a designed negative
CAR product tube              high, known        CAR cutoff SENSITIVITY — the other error direction
===========================  ==================  ==================================================

The host-NK check is the elegant one: within every post-infusion tube, host NK cells are
CAR-negative by construction, so their apparent CAR+ rate is a direct, per-timepoint,
assumption-free empirical false-positive rate. If it reads 4% at D28 then a measured 6% CAR+
among donor NK is mostly artefact — and no amount of inspecting ``overlay_car.png`` would
reveal that.

Polarity is load-bearing here. ``hla_dim`` means donor cells are the DIM side, so every donor
/host split reads through ``SampleState.donor_side_mask`` rather than an inline comparison.
Getting it backwards would invert exactly the checks this tier exists to provide.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..gates import gate_with, percentages
from . import estimators as E
from .context import DiagnosticsContext, SampleState
from .flags import (
    F_CONTROL_FALSE_POSITIVE,
    F_CONTROL_MISSING,
    F_POSITIVE_CONTROL_LOW,
    FlagBook,
)
from .record import U_FRACTION, MetricRecorder, rounded

# Control tube kinds as classified by ``calibrate.classify_file``.
CTRL_NTNK = "ntnk"
CTRL_CAR = "car"
CTRL_CBMC = "cbmc"


def run(ctx: DiagnosticsContext, rec: MetricRecorder, book: FlagBook) -> dict[str, Any]:
    """Compute tier 4. Returns a JSON-ready dict."""
    out: dict[str, Any] = {
        "available": True,
        "hla_dim_polarity": ctx.hla_dim,
        "hla_specificity": ctx.hla_specificity,
        "car_cutoff": rounded(ctx.car_cut, 3),
        "hla_cutoff": rounded(ctx.hla_cut, 3),
        "host_nk_car": [],
        "pre_infusion_donor": [],
        "control_tubes": {},
        "note": (
            "These are the only checks here with a known correct answer. A locked cutoff that "
            "fails one of them is failing on a population whose truth does not depend on any "
            "threshold choice."
        ),
    }
    out["host_nk_car"] = _host_nk_car(ctx, rec, book)
    out["pre_infusion_donor"] = _pre_infusion_donor(ctx, rec, book)
    out["control_tubes"] = _control_tubes(ctx, rec, book)
    return out


# ── host NK as an in-tube CAR-negative control ────────────────────────────────
def _host_nk_car(ctx: DiagnosticsContext, rec: MetricRecorder, book: FlagBook) -> list[dict[str, Any]]:
    """CAR+ rate among host (non-donor) NK — an empirical per-timepoint false-positive rate."""
    th = ctx.thresholds
    rows: list[dict[str, Any]] = []
    if ctx.car_cut is None or ctx.hla_cut is None:
        book.add(
            code=F_CONTROL_MISSING, tier=4, subject="host NK CAR check",
            rule="requires both an HLA (donor) cutoff and a CAR cutoff",
            measured={"hla_cutoff": ctx.hla_cut, "car_cutoff": ctx.car_cut},
            resolution_hint=(
                "Without both cutoffs the host/donor split cannot be made, so the strongest "
                "available CAR specificity check is unavailable. Every CAR+ number in this run "
                "therefore has no in-tube false-positive estimate."
            ),
        )
        return rows

    for s in ctx.samples:
        tp = s.timepoint or s.filename
        donor = s.donor_side_mask(ctx.hla_dim)
        nk = s.masks.get("NK")
        car_vals = s.signals.get("car")
        if donor is None or nk is None or car_vals is None:
            continue
        host = nk & ~donor
        n_host = int(np.count_nonzero(host))
        row: dict[str, Any] = {
            "timepoint": tp,
            "is_pre_infusion": s.is_pre_infusion,
            "n_host_nk": n_host,
            "n_donor_nk": int(np.count_nonzero(donor)),
        }
        if n_host < th.min_parent_events:
            row["available"] = False
            row["reason"] = f"only {n_host} host NK events (< MIN_PARENT {th.min_parent_events})"
            rows.append(row)
            continue

        car_pos = int(np.count_nonzero(car_vals[host] > float(ctx.car_cut)))
        rate = car_pos / n_host
        lo, hi = E.wilson_interval(car_pos, n_host)
        row.update(
            {
                "available": True,
                "n_host_nk_car_positive": car_pos,
                "false_positive_rate": rounded(rate, 5),
                "false_positive_ci": [rounded(lo, 5), rounded(hi, 5)],
                "reported_car_of_donor_pct": rounded(s.pct.get("%CAR+ (of Donor NK)"), 4),
            }
        )
        # The comparison that makes the number actionable: how much of the reported CAR+
        # signal could be explained by this measured false-positive rate.
        reported = s.pct.get("%CAR+ (of Donor NK)")
        if reported is not None and float(reported) > 0:
            row["fpr_share_of_reported_car"] = rounded(
                (rate * 100.0) / float(reported), 4)
        rows.append(row)

        rec.add(
            tier=4, subject_type="control", marker="car", timepoint=tp,
            metric="host_nk_car_false_positive_rate",
            value=rounded(rate, 5), unit=U_FRACTION,
            normalized_value=row.get("fpr_share_of_reported_car"),
            normalized_unit="fraction of the reported %CAR+ (of Donor NK) this rate could explain",
            note=("host NK are CAR-negative by construction, so this is an assumption-free "
                  "in-tube false-positive rate for the locked CAR cutoff"),
        )
        book.raise_if(
            rate > th.control_false_positive_max,
            code=F_CONTROL_FALSE_POSITIVE, tier=4, subject=f"host NK CAR+ @ {tp}",
            rule=(f"a population that is CAR-negative by construction reads > "
                  f"{th.control_false_positive_max} positive"),
            measured={
                "false_positive_rate": rounded(rate, 5),
                "ci": [rounded(lo, 5), rounded(hi, 5)],
                "n_host_nk": n_host,
                "n_car_positive": car_pos,
                "reported_car_of_donor_pct": rounded(reported, 4),
                "fpr_share_of_reported_car": row.get("fpr_share_of_reported_car"),
            },
            resolution_hint=(
                "Subtract this rate from the reported CAR+ before interpreting it, or raise the "
                "CAR cutoff until this control reads clean and re-check the CAR product tube "
                "so specificity is not bought at the cost of sensitivity."
            ),
            value=rate, threshold=th.control_false_positive_max,
        )
    return rows


# ── pre-infusion donor NK as a specificity check on the HLA cutoff ────────────
def _pre_infusion_donor(ctx: DiagnosticsContext, rec: MetricRecorder,
                        book: FlagBook) -> list[dict[str, Any]]:
    """Donor NK% before infusion — should be ~0, and the reference timepoint is one of these."""
    th = ctx.thresholds
    rows: list[dict[str, Any]] = []
    if ctx.hla_cut is None:
        return rows
    pre = [s for s in ctx.samples if s.is_pre_infusion]
    if not pre:
        book.add(
            code=F_CONTROL_MISSING, tier=4, subject="pre-infusion donor check",
            rule="requires at least one pre-infusion (Baseline / Screen / Pre) sample",
            measured={"timepoints": ctx.timepoint_labels()},
            resolution_hint=(
                "Without a pre-infusion sample the donor-HLA cutoff has no in-study negative "
                "control, so its specificity rests entirely on the control tubes."
            ),
        )
        return rows

    for s in pre:
        tp = s.timepoint or s.filename
        donor = s.donor_side_mask(ctx.hla_dim)
        nk = s.masks.get("NK")
        if donor is None or nk is None:
            continue
        n_nk = int(np.count_nonzero(nk))
        n_donor = int(np.count_nonzero(donor))
        row: dict[str, Any] = {
            "timepoint": tp,
            "is_reference_timepoint": s.is_reference,
            "n_nk": n_nk,
            "n_donor_nk": n_donor,
            "donor_of_nk_pct": rounded(s.pct.get("%Donor NK (of NK)"), 4),
            "donor_of_lymph_pct": rounded(s.pct.get("%Donor NK (of lymph)"), 4),
        }
        if n_nk < th.min_parent_events:
            row["available"] = False
            row["reason"] = f"only {n_nk} NK events (< MIN_PARENT {th.min_parent_events})"
            rows.append(row)
            continue
        rate = n_donor / n_nk
        lo, hi = E.wilson_interval(n_donor, n_nk)
        row.update({"available": True, "donor_rate_of_nk": rounded(rate, 5),
                    "donor_rate_ci": [rounded(lo, 5), rounded(hi, 5)]})
        rows.append(row)

        rec.add(
            tier=4, subject_type="control", marker="hla", timepoint=tp,
            metric="pre_infusion_donor_rate_of_nk",
            value=rounded(rate, 5), unit=U_FRACTION,
            note=("no donor cells exist before infusion, so this is a direct specificity "
                  "measurement for the locked donor-HLA cutoff"),
        )
        book.raise_if(
            rate > th.control_false_positive_max,
            code=F_CONTROL_FALSE_POSITIVE, tier=4,
            subject=f"pre-infusion donor NK @ {tp}",
            rule=(f"a pre-infusion sample (no donor cells possible) reads > "
                  f"{th.control_false_positive_max} donor-HLA positive"),
            measured={
                "donor_rate_of_nk": rounded(rate, 5),
                "ci": [rounded(lo, 5), rounded(hi, 5)],
                "n_nk": n_nk, "n_donor_nk": n_donor,
                "is_reference_timepoint": s.is_reference,
                "hla_dim_polarity": ctx.hla_dim,
            },
            resolution_hint=(
                "This is a false-positive floor for every donor NK number in the study. If the "
                "flagged timepoint is also the reference, the anchor was derived on a sample "
                "the donor cutoff already mis-reads."
            ),
            value=rate, threshold=th.control_false_positive_max,
        )
    return rows


# ── designed control tubes ────────────────────────────────────────────────────
def _control_cuts(ctx: DiagnosticsContext) -> dict[str, Any]:
    """The unified cutoffs, applied to a control tube.

    The question a control tube answers is "what does the LOCKED cutoff say about a tube whose
    answer we already know", so the control is gated with the anchored cutoffs rather than
    with its own valleys.
    """
    C: dict[str, Any] = dict(ctx.anchor.get("reference_cuts") or {})
    C.update(ctx.anchor.get("locked") or {})
    C["hla"] = ctx.hla_cut
    C["car"] = ctx.car_cut
    C["hla_dim"] = ctx.hla_dim
    # CD8 is CD4-negative in this panel; mirror the anchor's transfer policy.
    C["cd8"] = None
    C["cd8_from_cd4_neg"] = True
    C["_anchor_scatter"] = "locked"
    return C


def _control_tubes(ctx: DiagnosticsContext, rec: MetricRecorder,
                   book: FlagBook) -> dict[str, Any]:
    """NT-NK (known CAR-negative), CAR product (known CAR-positive), CBMC (descriptive)."""
    th = ctx.thresholds
    out: dict[str, Any] = {}
    C = _control_cuts(ctx)

    expectations = {
        CTRL_NTNK: {
            "expected": "approximately 0% CAR+ (non-transduced NK)",
            "direction": "negative",
        },
        CTRL_CAR: {
            "expected": "high CAR+ (the transduced product)",
            "direction": "positive",
        },
        CTRL_CBMC: {
            "expected": "no fixed expectation — reported descriptively",
            "direction": "descriptive",
        },
    }

    for kind, exp in expectations.items():
        if kind not in ctx.controls:
            out[kind] = {"available": False, "reason": "control tube not present in the dataset"}
            book.add(
                code=F_CONTROL_MISSING, tier=4, subject=f"{kind} control tube",
                rule=f"a '{kind}' control tube was not found in the dataset",
                measured={"present_controls": sorted(ctx.controls.keys())},
                resolution_hint=(
                    f"The {kind} check is unavailable. For 'ntnk' and 'car' this removes one "
                    "side of the CAR cutoff's validation; the in-tube host-NK check still "
                    "covers specificity if it ran."
                ),
            )
            continue

        fname, df = ctx.controls[kind]
        res: dict[str, Any] = {
            "available": False, "file": fname,
            "expected": exp["expected"], "direction": exp["direction"],
        }
        try:
            masks, g, _scat, _applied = gate_with(df, C, ctx.channels, ssc_cap=None)
        except Exception as e:
            res["reason"] = f"gating the control tube failed: {e}"
            out[kind] = res
            continue

        pct = percentages(masks)
        nk = masks.get("NK")
        n_nk = 0 if nk is None else int(np.count_nonzero(nk))
        res.update({"n_nk": n_nk, "n_live": int(np.count_nonzero(masks.get("live", [])))})

        # Donor-side fraction, for context on what the tube actually is.
        donor_frac = None
        if ctx.hla_cut is not None and g.get("hla") is not None and nk is not None and n_nk:
            hla = g["hla"][nk]
            side = (hla < float(ctx.hla_cut)) if ctx.hla_dim else (hla > float(ctx.hla_cut))
            donor_frac = float(np.mean(side))
        res["donor_side_fraction_of_nk"] = rounded(donor_frac, 5)
        res["nk_of_lymph_pct"] = rounded(pct.get("%NK (of lymph)"), 4)

        if ctx.car_cut is None or g.get("car") is None or not n_nk:
            res["reason"] = "no CAR cutoff or no NK events in the control tube"
            out[kind] = res
            continue
        if n_nk < th.min_parent_events:
            res["reason"] = f"only {n_nk} NK events (< MIN_PARENT {th.min_parent_events})"
            out[kind] = res
            continue

        car_vals = g["car"][nk]
        n_pos = int(np.count_nonzero(car_vals > float(ctx.car_cut)))
        rate = n_pos / n_nk
        lo, hi = E.wilson_interval(n_pos, n_nk)
        res.update(
            {
                "available": True,
                "n_car_positive": n_pos,
                "car_positive_rate_of_nk": rounded(rate, 5),
                "car_positive_ci": [rounded(lo, 5), rounded(hi, 5)],
            }
        )
        out[kind] = res

        rec.add(
            tier=4, subject_type="control", marker="car", timepoint=f"control:{kind}",
            metric="control_car_positive_rate", value=rounded(rate, 5), unit=U_FRACTION,
            note=f"{fname}: expected {exp['expected']}",
        )

        if exp["direction"] == "negative":
            book.raise_if(
                rate > th.control_false_positive_max,
                code=F_CONTROL_FALSE_POSITIVE, tier=4, subject=f"{kind} control tube",
                rule=(f"a designed CAR-negative control reads > "
                      f"{th.control_false_positive_max} CAR+"),
                measured={
                    "file": fname, "car_positive_rate": rounded(rate, 5),
                    "ci": [rounded(lo, 5), rounded(hi, 5)],
                    "n_nk": n_nk, "n_car_positive": n_pos,
                },
                resolution_hint=(
                    "The locked CAR cutoff calls positives in a tube that has none. Note this "
                    "tube may also be the tube the cutoff was DERIVED from (NT-NK p99.9 in "
                    "calibrate.py), in which case a non-trivial rate here means the derivation "
                    "percentile is too permissive."
                ),
                value=rate, threshold=th.control_false_positive_max,
            )
        elif exp["direction"] == "positive":
            book.raise_if(
                rate < th.positive_control_min,
                code=F_POSITIVE_CONTROL_LOW, tier=4, subject=f"{kind} control tube",
                rule=(f"a designed CAR-positive control reads < {th.positive_control_min} CAR+"),
                measured={
                    "file": fname, "car_positive_rate": rounded(rate, 5),
                    "ci": [rounded(lo, 5), rounded(hi, 5)],
                    "n_nk": n_nk, "n_car_positive": n_pos,
                },
                resolution_hint=(
                    "Guards the error direction the specificity checks cannot see: a cutoff so "
                    "high that real CAR+ cells are not counted. If both this and a host-NK "
                    "false-positive flag are present, the CAR channel has poor separation "
                    "rather than a mis-placed cutoff."
                ),
                value=rate, threshold=th.positive_control_min, higher_is_worse=False,
            )
    return out
