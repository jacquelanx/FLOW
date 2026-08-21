"""The tidy metric record every tier writes into.

One presentational rule is enforced here, because it is the difference between numbers a
reader can judge and numbers they cannot:

    **Every value is paired with a dimensionless twin.**

``drift = 0.38`` alone is unjudgeable — 0.38 of what, and is that a lot? The same row also
carries ``normalized_value = 0.09`` with ``normalized_unit = "fraction of reference
negative-to-positive gap"``. The raw value is what you check against the overlay PNG; the
normalized value is what tells you whether to care. Rows that genuinely have no meaningful
normalization (counts, p-values, fractions that are already dimensionless) leave it null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Unit vocabulary, kept small and reused so the emitted table can be grouped by unit.
U_TRANSFORM = "biex transform units"
U_PCT = "percent"
U_PP = "percentage points"
U_FRACTION = "fraction (0-1)"
U_COUNT = "events"
U_ROBUST_SD = "robust SD of the negative population"
U_GAP_FRACTION = "fraction of reference negative-to-positive gap"
U_SE_MULTIPLE = "multiples of the estimator's bootstrap SE"
U_PVALUE = "p-value"
U_RATIO = "ratio"
U_INDEX = "index (dimensionless)"


@dataclass
class MetricRecorder:
    """Accumulates tidy long-format metric rows."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        tier: int,
        metric: str,
        value: Any,
        unit: str = "",
        marker: str = "",
        timepoint: str = "",
        subject_type: str = "",
        normalized_value: Any = None,
        normalized_unit: str = "",
        note: str = "",
    ) -> None:
        """Record one measurement. ``None`` values are kept — honest missingness is data."""
        self.rows.append(
            {
                "tier": tier,
                "subject_type": subject_type,
                "marker": marker,
                "timepoint": timepoint,
                "metric": metric,
                "value": _clean(value),
                "unit": unit,
                "normalized_value": _clean(normalized_value),
                "normalized_unit": normalized_unit,
                "note": note,
            }
        )

    def extend(self, other: "MetricRecorder") -> None:
        self.rows.extend(other.rows)

    def __len__(self) -> int:
        return len(self.rows)


def _clean(v: Any) -> Any:
    """Coerce numpy scalars to plain Python; leave None as None (never 0.0)."""
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    if isinstance(v, float):
        # NaN is not a measurement; report it as missing so downstream readers can trust
        # "value is not null" to mean "this was measured".
        return None if v != v else v
    return v


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """``numerator / denominator``, or None if either is missing or the divisor is ~0."""
    if numerator is None or denominator is None:
        return None
    try:
        n, d = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return None
    if not (n == n) or not (d == d) or abs(d) < 1e-15:
        return None
    return n / d


def rounded(v: Optional[float], places: int = 4) -> Optional[float]:
    """Round for display without turning a missing value into a number."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(f, places)
