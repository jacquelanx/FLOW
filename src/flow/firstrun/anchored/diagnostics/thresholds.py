"""Flagging rules, their defaults, and — importantly — where each default comes from.

An invented threshold is hardcoded analysis wearing a lab coat. Every rule here therefore
carries a ``provenance``:

  * ``self-calibrating`` — the threshold is derived from the data at run time (e.g. drift is
    compared against the bootstrap standard error of the estimator that measured it). These
    need no scientific buy-in because they only ever assert "bigger than this measurement's
    own noise".
  * ``pipeline-constant`` — the value already exists in the anchored pipeline (``gates.py``
    fraction bands, ``calibrate.MIN_PARENT`` / ``MIN_POS``). Reusing them imports the
    pipeline's stated position instead of adding a new one.
  * ``settable-default`` — a genuine scientific choice with no self-calibrating form. These
    are the ones to review; each states its rationale and can be overridden per dataset via
    ``metadata.json`` without touching code:

        {"diagnostics_thresholds": {"drift_gap_fraction": 0.10}}

Nothing here decides whether the analysis is right. These rules decide only what gets
SURFACED, and every emitted flag carries the rule text that produced it so it can be argued
with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

# provenance -> human explanation, echoed into the emitted JSON so the audit trail is
# self-describing rather than requiring this module to be read.
PROVENANCE_NOTES = {
    "self-calibrating": (
        "Derived from the data at run time (compared against the measurement's own "
        "resampling noise). No scientific constant is asserted."
    ),
    "pipeline-constant": (
        "Already encoded in the anchored pipeline (gates.py plausible-fraction bands, "
        "calibrate.MIN_PARENT / MIN_POS). Reused rather than re-invented."
    ),
    "settable-default": (
        "A scientific choice with a documented default. Override per dataset via "
        "metadata.json -> diagnostics_thresholds."
    ),
}

# field name -> (provenance, rationale)
THRESHOLD_PROVENANCE: dict[str, tuple[str, str]] = {
    "drift_se_multiple": (
        "self-calibrating",
        "Negative-mode drift counts only once it exceeds this multiple of the bootstrap SE "
        "of the mode estimate itself — i.e. once it is distinguishable from estimator noise.",
    ),
    "drift_negative_sd_max": (
        "settable-default",
        "The magnitude drift rule: how far the negative population moved, in multiples of its "
        "OWN standard deviation. Scale-free, and unlike the mode gap it exists for a unimodal "
        "pool too. Both this AND the noise rule must be exceeded to flag, because at 30k+ "
        "events the noise rule alone fires on drifts far too small to matter. The 0.5 default "
        "is empirically separated: on a validation run with a 300-unit drift injected at one "
        "timepoint, the injected defect measured 1.17 SD while every uninjected marker x "
        "timepoint measured <= 0.31 SD.",
    ),
    "drift_gap_fraction": (
        "settable-default",
        "Reported normalization only (NOT a flagging rule): drift as a fraction of the "
        "reference negative-to-positive gap. Useful for reading a drift against the distance "
        "to the positive population, but undefined for a unimodal pool.",
    ),
    "drift_pvalue_max": (
        "settable-default",
        "Permutation p-value below which the negative distributions at two timepoints are "
        "called distinguishable. Conventional 0.05.",
    ),
    "placement_z_min": (
        "settable-default",
        "A cutoff should sit at least this many robust SDs above the negative mode. 2.0 "
        "mirrors the conventional flow reading of a gate being 'clear of the negative'.",
    ),
    "placement_loss_sd_max": (
        "settable-default",
        "Transfer rule: flag when a timepoint's placement margin LOSES more than this many "
        "robust SDs versus the reference. A signed difference rather than a ratio — placement "
        "is signed, so a ratio misreads a margin going from -0.27 to -1.43 (a 1.16 SD loss) as "
        "a 5.3x improvement.",
    ),
    "trough_depth_max": (
        "settable-default",
        "Density at the cutoff, as a fraction of the geometric mean of the two modal "
        "densities. Above this the cutoff is sitting on a shoulder rather than in a valley.",
    ),
    "gap_position_low": (
        "settable-default",
        "Lower bound on the cutoff's normalized position between the two modes. Below this "
        "the cutoff is inside the negative population.",
    ),
    "gap_position_high": (
        "settable-default",
        "Upper bound on the same. Above this the cutoff is inside the positive population.",
    ),
    "unimodal_p_min": (
        "settable-default",
        "Both unimodality tests (Silverman bootstrap, mixture LRT) must reject unimodality "
        "below this p-value for a valley-derived cutoff to be considered valley-supported. "
        "Conventional 0.05.",
    ),
    "mixture_separation_min": (
        "settable-default",
        "Second route to valley evidence: standardized separation of the two fitted mixture "
        "components, in pooled SDs. 2.0 is the point at which two Gaussians are visibly "
        "resolved. Needed because Silverman's bootstrap loses power on unequal mixtures — a "
        "real 10%-weight population can be called unimodal by it.",
    ),
    "mixture_min_component_weight": (
        "settable-default",
        "Minimum weight of the smaller fitted component for the mixture route to count. Stops "
        "a two-component fit chasing an outlier tail and calling it a population. 0.02 matches "
        "the loosest positive-fraction band gates.py itself accepts (CD45's 0.02).",
    ),
    "overton_disagreement_max": (
        "settable-default",
        "How far the cutoff-free (Overton subtraction) positive fraction may differ from the "
        "cutoff-based fraction before it is surfaced. Two estimates of the same quantity, only "
        "one of which depends on the cutoff, so a gap localizes the cutoff as the cause. 0.05 "
        "is five percentage points of the parent population.",
    ),
    "discontinuity_ci_multiple": (
        "self-calibrating",
        "A timepoint's value counts as discontinuous with its neighbours only once it sits "
        "outside their envelope by this multiple of the COMBINED Wilson counting intervals of "
        "the values being compared — i.e. once the excursion is bigger than the sampling noise "
        "of the numbers themselves. No biological rate of change is asserted.",
    ),
    "counting_ci_pp_max": (
        "settable-default",
        "Half-width of the Wilson counting interval, in percentage points, above which a "
        "reported percentage is too event-limited to carry a longitudinal claim.",
    ),
    "control_false_positive_max": (
        "settable-default",
        "A population that is negative by construction (host NK for CAR, pre-infusion for "
        "donor HLA) should read at most this positive fraction. 0.02 is a strict but "
        "achievable specificity bar for a locked cutoff.",
    ),
    "positive_control_min": (
        "settable-default",
        "A known-positive control (the CAR product tube) should read at least this positive "
        "fraction. Guards the OTHER error direction — a cutoff so high it reports nothing.",
    ),
    "sensitivity_pp_max": (
        "settable-default",
        "Cutoff-attributable range on a headline percentage, in percentage points, above "
        "which the number is fragile to where the line was drawn.",
    ),
    "counterfactual_pp_max": (
        "settable-default",
        "Locked-versus-per-sample difference, in percentage points, above which anchoring is "
        "materially changing that timepoint's reported number (not necessarily wrongly — "
        "removing per-sample drift is the method's purpose — but it should be visible).",
    ),
    "acquisition_drift_sd_max": (
        "settable-default",
        "Within-file median fluorescence drift across acquisition time, in robust SDs of "
        "that marker. Catches clogs and pressure instability, which no current figure shows.",
    ),
    "acquisition_rate_cv_max": (
        "settable-default",
        "CV of event rate across equal-time bins of the acquisition, above which the run was "
        "not steady.",
    ),
    "boundary_mass_max": (
        "settable-default",
        "Fraction of a population's events lying within one robust SD of a gate boundary. "
        "High values mean small cutoff moves reassign many events.",
    ),
    "quadrant_margin_min": (
        "settable-default",
        "Distance from the 2-D NK density mode to each quadrant boundary, in robust SDs of "
        "the NK cloud along that axis. Below this the boundary is cutting into the cloud.",
    ),
    "scatter_containment_min": (
        "settable-default",
        "Fraction of the freshly-estimated density island that falls inside the locked "
        "scatter bounds. Below this the locked gate no longer contains the cloud it was "
        "locked around.",
    ),
    "manual_divergence_pp": (
        "pipeline-constant",
        "compare.summarize already reports within_5pp / within_10pp; the 10 pp band is reused "
        "here to flag individual auto-vs-manual divergences rather than introducing a new one.",
    ),
    "min_parent_events": (
        "pipeline-constant",
        "calibrate.MIN_PARENT — the pipeline's own floor for a countable denominator.",
    ),
    "min_positive_events": (
        "pipeline-constant",
        "calibrate.MIN_POS — the pipeline's own floor for a countable numerator.",
    ),
    "fallback_audit_min_events": (
        "settable-default",
        "When the reference pool cannot carry a marker's tier-1 audit, the audit is taken at "
        "the EARLIEST timepoint whose pool reaches this size, and only at the largest "
        "available pool if none do. MIN_PARENT (50) is a COUNTING floor — enough events to "
        "report a percentage — and a density estimate needs more than that: a mode, a trough "
        "depth and a bootstrap SE measured on 51 events would carry noise into every drift "
        "number computed against them. 10x MIN_PARENT is the stated multiple; earliest-that-"
        "qualifies rather than largest, because the fallback also becomes tier 2's drift "
        "baseline and the least biology should have intervened between it and the reference.",
    ),
}


@dataclass
class Thresholds:
    """Flagging thresholds. See ``THRESHOLD_PROVENANCE`` for where each one comes from."""

    # Tier 2 — transfer / drift
    drift_se_multiple: float = 1.0
    drift_negative_sd_max: float = 0.5
    drift_gap_fraction: float = 0.15   # reported normalization only, not a flagging rule
    drift_pvalue_max: float = 0.05
    overton_disagreement_max: float = 0.05
    placement_loss_sd_max: float = 0.5

    # Tier 1 — cutoff foundation
    placement_z_min: float = 2.0
    trough_depth_max: float = 0.60
    gap_position_low: float = 0.15
    gap_position_high: float = 0.85
    unimodal_p_min: float = 0.05
    mixture_separation_min: float = 2.0
    mixture_min_component_weight: float = 0.02

    # Tier 0 — data adequacy
    counting_ci_pp_max: float = 5.0
    discontinuity_ci_multiple: float = 3.0
    acquisition_drift_sd_max: float = 0.5
    acquisition_rate_cv_max: float = 0.25
    min_parent_events: int = 50   # calibrate.MIN_PARENT
    min_positive_events: int = 10  # calibrate.MIN_POS

    # Tier 1 — where a fallback audit may be taken
    fallback_audit_min_events: int = 500

    # Tier 3 — sensitivity
    sensitivity_pp_max: float = 2.0
    counterfactual_pp_max: float = 5.0

    # Tier 4 — internal controls
    control_false_positive_max: float = 0.02
    positive_control_min: float = 0.20

    # Tier 5 — operator concordance (band reused from compare.summarize)
    manual_divergence_pp: float = 10.0

    # 2-D gate integrity
    boundary_mass_max: float = 0.10
    quadrant_margin_min: float = 1.5
    scatter_containment_min: float = 0.90

    # Resampling budgets. Not scientific thresholds — cost knobs. Raising them narrows
    # confidence intervals and refines p-value resolution at linear runtime cost.
    n_bootstrap: int = 200
    n_permutation: int = 200
    n_unimodality_boot: int = 80
    random_seed: int = 0

    # Overrides actually applied, recorded so the emitted audit trail shows what was changed.
    overrides_applied: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, meta: dict | None) -> "Thresholds":
        """Build from ``metadata.json``'s optional ``diagnostics_thresholds`` block.

        Unknown keys are reported and ignored rather than silently dropped — a typo in a
        threshold name would otherwise look like a threshold that had been set.
        """
        t = cls()
        block = ((meta or {}).get("diagnostics_thresholds") or {})
        if not isinstance(block, dict):
            return t
        known = {f.name for f in fields(cls)} - {"overrides_applied"}
        applied: dict[str, Any] = {}
        for k, v in block.items():
            if k not in known:
                print(f"  [diagnostics] ignoring unknown threshold {k!r}", flush=True)
                continue
            try:
                current = getattr(t, k)
                cast = int(v) if isinstance(current, int) and not isinstance(current, bool) else float(v)
            except (TypeError, ValueError):
                print(f"  [diagnostics] ignoring non-numeric threshold {k!r}={v!r}", flush=True)
                continue
            setattr(t, k, cast)
            applied[k] = cast
        t.overrides_applied = applied
        if applied:
            print(f"  [diagnostics] threshold overrides from metadata.json: {applied}", flush=True)
        return t

    def describe(self) -> list[dict[str, Any]]:
        """Every threshold with its value, provenance and rationale, for the audit trail."""
        out = []
        for f in fields(self):
            if f.name == "overrides_applied":
                continue
            prov, why = THRESHOLD_PROVENANCE.get(
                f.name, ("cost-knob", "Resampling budget; affects precision and runtime only.")
            )
            out.append(
                {
                    "threshold": f.name,
                    "value": getattr(self, f.name),
                    "default": f.default,
                    "overridden": f.name in self.overrides_applied,
                    "provenance": prov,
                    "rationale": why,
                }
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
