"""Flags: measurements that crossed a stated rule — never conclusions.

The discipline this module enforces is the whole reason the diagnostics can exist without
becoming hardcoded analysis:

    A flag may say   "cd56 @ D28: negative-mode drift 0.38 (9% of the reference gap;
                      1.4x the bootstrap SE); rule = drift > 1.0 x SE"
    A flag may NOT say "the CD56 cutoff is wrong at D28"

The first is a measurement plus the rule that surfaced it, and the agent (or the biologist)
is free to disagree with the rule. The second is the analysis, and if this module makes that
call then the science has moved out of the agent and into a constant.

Every flag therefore carries:
  * ``rule`` — the exact rule text, including its threshold, so it can be argued with.
  * ``measured`` — the numbers, so the rule can be re-evaluated by hand.
  * ``exceedance`` — measured/threshold, used ONLY for ordering the output. Ordering by how
    far a rule was exceeded is not the same as ranking scientific importance, and the emitted
    text says so.
  * ``resolution_hint`` — what measurement or action would settle it. This is what makes the
    output usable by an iterative review pass rather than a static report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Flag codes. Grouped by tier so the emitted summary can be read top-down.
# Tier 0 — data adequacy
F_LOW_PARENT = "LOW_PARENT_EVENTS"
F_LOW_POSITIVE = "LOW_POSITIVE_EVENTS"
F_WIDE_COUNTING_CI = "WIDE_COUNTING_INTERVAL"
F_ACQ_DRIFT = "ACQUISITION_SIGNAL_DRIFT"
F_ACQ_RATE = "ACQUISITION_RATE_UNSTEADY"
F_NO_TIME_CHANNEL = "NO_TIME_CHANNEL"
F_TEMPORAL_DISCONTINUITY = "TEMPORAL_DISCONTINUITY"

# Tier 1 — cutoff foundation at the reference
F_NO_VALLEY_EVIDENCE = "NO_VALLEY_EVIDENCE"
# The pipeline never claimed a valley for this cutoff (HLA, CAR: control-bracketed or fitted)
# AND none exists in the pool it is applied to. Distinct from NO_VALLEY_EVIDENCE, which is a
# contradiction between the pipeline's stated derivation and the data; this is a cutoff that
# ASSERTS a positive fraction by construction, so its placement metrics are descriptive and
# the only external checks on it are tiers 4 and 5. Without this code such a cutoff is silent
# in tier 1 — the audit table records valley_supported=false and nothing surfaces it.
F_ASSERTED_CUTOFF = "ASSERTED_CUTOFF_NO_VALLEY_EVIDENCE"
# The reference timepoint's pool could not carry this marker's audit at all (too few parent
# events, or no estimable density). Reported because "no tier-1 row" otherwise reads exactly
# like "nothing wrong at tier 1", and for a cutoff the reference cannot audit there is no
# ruler-quality measurement anywhere unless one is taken elsewhere.
F_REFERENCE_AUDIT_UNAVAILABLE = "REFERENCE_AUDIT_UNAVAILABLE"
F_STRUCTURE_ONLY_IN_MIXTURE_FIT = "STRUCTURE_ONLY_IN_MIXTURE_FIT"
F_SHALLOW_TROUGH = "SHALLOW_TROUGH_AT_CUTOFF"
F_CUTOFF_INSIDE_POPULATION = "CUTOFF_INSIDE_POPULATION"
F_LOW_PLACEMENT_MARGIN = "LOW_PLACEMENT_MARGIN"
F_PERCENTILE_FALLBACK = "PERCENTILE_FALLBACK_SUSPECTED"
F_FRACTION_OUT_OF_BAND = "POSITIVE_FRACTION_OUT_OF_PIPELINE_BAND"
F_OVERTON_DISAGREES = "OVERTON_DISAGREES_WITH_CUTOFF"
F_MODE_ESTIMATORS_DISAGREE = "MODE_ESTIMATORS_DISAGREE"

# Tier 2 — transfer validity
F_NEG_MODE_DRIFT = "NEGATIVE_MODE_DRIFT"
F_NEG_WIDTH_CHANGE = "NEGATIVE_WIDTH_CHANGE"
F_NEG_DISTRIBUTION_SHIFT = "NEGATIVE_DISTRIBUTION_SHIFT"
F_PLACEMENT_MARGIN_LOST = "PLACEMENT_MARGIN_LOST_ON_TRANSFER"
F_NEGATIVE_UNDEFINED = "NEGATIVE_POPULATION_UNDEFINED"
F_QUADRANT_MARGIN = "QUADRANT_BOUNDARY_IN_CLOUD"
F_BOUNDARY_MASS = "HIGH_BOUNDARY_EVENT_MASS"
F_SCATTER_CONTAINMENT = "LOCKED_SCATTER_CONTAINMENT_LOW"

# Tier 3 — sensitivity
F_FRAGILE_METRIC = "CUTOFF_FRAGILE_METRIC"
F_COUNTERFACTUAL_GAP = "LOCKED_VS_PERSAMPLE_GAP"

# Tier 4 — internal controls
F_CONTROL_FALSE_POSITIVE = "KNOWN_NEGATIVE_READS_POSITIVE"
F_POSITIVE_CONTROL_LOW = "KNOWN_POSITIVE_READS_LOW"
F_CONTROL_MISSING = "CONTROL_UNAVAILABLE"

# Tier 5 — operator concordance
F_MANUAL_DIVERGENCE = "MANUAL_CONCORDANCE_DIVERGENCE"
F_MANUAL_COVERAGE = "MANUAL_COVERAGE_PARTIAL"

# Cross-cutting
F_REPRODUCTION_MISMATCH = "REPRODUCTION_MISMATCH"
F_TIER_FAILED = "TIER_COMPUTATION_FAILED"


@dataclass
class Flag:
    """One rule crossing. See the module docstring for what a flag may and may not assert."""

    code: str
    tier: int
    subject: str
    rule: str
    measured: dict[str, Any]
    resolution_hint: str
    exceedance: Optional[float] = None

    def to_row(self) -> dict[str, Any]:
        """Flat dict for the CSV writer (measurements JSON-encoded into one column)."""
        return {
            "code": self.code,
            "tier": self.tier,
            "subject": self.subject,
            "rule": self.rule,
            "exceedance": None if self.exceedance is None else round(self.exceedance, 4),
            "measured": json.dumps(self.measured, sort_keys=True, default=_json_default),
            "resolution_hint": self.resolution_hint,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(o: Any):
    """Make numpy scalars/arrays JSON-serializable without importing numpy at module load."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


@dataclass
class FlagBook:
    """Collects flags and keeps them ordered for output."""

    flags: list[Flag] = field(default_factory=list)

    def add(
        self,
        code: str,
        tier: int,
        subject: str,
        rule: str,
        measured: dict[str, Any],
        resolution_hint: str,
        exceedance: Optional[float] = None,
    ) -> Flag:
        f = Flag(
            code=code,
            tier=tier,
            subject=subject,
            rule=rule,
            measured=measured,
            resolution_hint=resolution_hint,
            exceedance=exceedance,
        )
        self.flags.append(f)
        return f

    def raise_if(
        self,
        condition: bool,
        *,
        code: str,
        tier: int,
        subject: str,
        rule: str,
        measured: dict[str, Any],
        resolution_hint: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        higher_is_worse: bool = True,
    ) -> Optional[Flag]:
        """Add a flag when ``condition`` holds, computing ``exceedance`` from value/threshold.

        ``higher_is_worse=False`` inverts the ratio so that a metric which fails by being
        too SMALL (containment, positive-control signal, placement margin) still orders with
        the worst offender first.
        """
        if not condition:
            return None
        exc = None
        if value is not None and threshold is not None:
            try:
                if higher_is_worse and threshold != 0:
                    exc = float(value) / float(threshold)
                elif not higher_is_worse and value not in (0, 0.0):
                    exc = float(threshold) / float(value)
                elif not higher_is_worse:
                    exc = float("inf")
            except (TypeError, ValueError, ZeroDivisionError):
                exc = None
        return self.add(
            code=code,
            tier=tier,
            subject=subject,
            rule=rule,
            measured=measured,
            resolution_hint=resolution_hint,
            exceedance=exc,
        )

    def ordered(self) -> list[Flag]:
        """Flags sorted by tier, then by descending exceedance (worst offender first).

        Ordering is a presentation aid, not a severity judgement — a tier-0 note can matter
        more than a tier-3 exceedance, which is why the tier is always printed alongside.
        """
        def key(f: Flag):
            exc = f.exceedance
            if exc is None or exc != exc:  # None / NaN
                rank = 0.0
            elif exc == float("inf"):
                rank = float("1e18")
            else:
                rank = float(exc)
            return (f.tier, -rank, f.code, f.subject)

        return sorted(self.flags, key=key)

    def by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.flags:
            counts[f.code] = counts.get(f.code, 0) + 1
        return counts

    def rows(self) -> list[dict[str, Any]]:
        return [f.to_row() for f in self.ordered()]

    def __len__(self) -> int:
        return len(self.flags)
