"""Tier 5 — does it agree with the operator?

No new science here: ``compare.py`` already computes auto-vs-manual deltas and ``summarize``
already reduces them to MAE / within-5pp / within-10pp. What was missing was granularity and
placement — a patient-level MAE tells you disagreement exists but not WHERE, and the coverage
of that MAE (how many timepoints the manual CSV actually matched) belongs next to the number
rather than in the run log.

So this tier surfaces:
  * per-timepoint and per-metric mean absolute deviation, so the worst offender is named;
  * coverage — which timepoints the manual file covers and which it does not;
  * the pipeline's own 5pp / 10pp bands reused as the flagging rule, so no new threshold
    enters through this door.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..compare import summarize
from .context import DiagnosticsContext
from .flags import F_MANUAL_COVERAGE, F_MANUAL_DIVERGENCE, FlagBook
from .record import U_PP, MetricRecorder, rounded


def _find_compare(out_dir: Path, pid: str) -> Optional[Path]:
    for cand in (out_dir / "compare_manual.csv", out_dir / f"Compare_manual_{pid}.csv"):
        if cand.exists():
            return cand
    hits = sorted(out_dir.glob("Compare_manual_*.csv"))
    return hits[0] if hits else None


def run(ctx: DiagnosticsContext, rec: MetricRecorder, book: FlagBook) -> dict[str, Any]:
    """Compute tier 5. Returns a JSON-ready dict."""
    th = ctx.thresholds
    path = _find_compare(ctx.out_dir, ctx.patient_id)
    if path is None:
        return {
            "available": False,
            "reason": (
                "no manual-gating comparison table was produced (no manual_gating.csv in the "
                "dataset). Operator concordance is unavailable, so the internal controls in "
                "tier 4 carry the whole burden of external validation."
            ),
        }
    try:
        cmp = pd.read_csv(path)
    except Exception as e:
        return {"available": False, "reason": f"could not read {path.name}: {e}"}
    if cmp.empty or not {"timepoint", "metric", "abs_delta"} <= set(cmp.columns):
        return {"available": False, "reason": f"{path.name} is empty or has unexpected columns"}

    out: dict[str, Any] = {
        "available": True,
        "source": path.name,
        "overall": summarize(cmp),
        "by_timepoint": [],
        "by_metric": [],
        "coverage": {},
        "worst_rows": [],
        "note": (
            "Flagging reuses the 10 pp band already used by compare.summarize, so this tier "
            "introduces no new threshold. Manual gating is a human reference, not ground "
            "truth — a divergence localizes disagreement, it does not assign fault."
        ),
    }

    # ---- coverage ------------------------------------------------------------
    covered = sorted({str(t) for t in cmp["timepoint"].dropna().unique()})
    all_tps = [s.timepoint or s.filename for s in ctx.samples]
    missing = [t for t in all_tps if t not in covered]
    out["coverage"] = {
        "timepoints_compared": covered,
        "timepoints_in_run": all_tps,
        "timepoints_not_compared": missing,
        "fraction_covered": rounded(
            (len(covered) / len(all_tps)) if all_tps else None, 4),
        "n_comparisons": int(len(cmp)),
    }
    rec.add(
        tier=5, subject_type="run", metric="manual_coverage_fraction",
        value=out["coverage"]["fraction_covered"], unit="fraction of run timepoints",
        note="an MAE over 1 of 12 timepoints reads exactly like an MAE over all 12",
    )
    book.raise_if(
        bool(missing),
        code=F_MANUAL_COVERAGE, tier=5, subject="manual concordance coverage",
        rule="at least one timepoint in the run has no manual-gating counterpart",
        measured={
            "timepoints_not_compared": missing,
            "n_covered": len(covered),
            "n_in_run": len(all_tps),
        },
        resolution_hint=(
            "The overall MAE describes only the covered timepoints. Do not read it as "
            "validation of the timepoints listed here as uncompared."
        ),
        value=float(len(missing)), threshold=1.0,
    )

    # ---- per-timepoint -------------------------------------------------------
    for tp, grp in cmp.groupby("timepoint"):
        mae = float(grp["abs_delta"].mean())
        worst = grp.loc[grp["abs_delta"].idxmax()]
        row = {
            "timepoint": str(tp),
            "n_metrics": int(len(grp)),
            "mae_pp": rounded(mae, 4),
            "max_abs_delta_pp": rounded(float(grp["abs_delta"].max()), 4),
            "worst_metric": str(worst.get("metric")),
            "within_5pp_pct": rounded(100.0 * float((grp["abs_delta"] <= 5).mean()), 2),
            "within_10pp_pct": rounded(100.0 * float((grp["abs_delta"] <= 10).mean()), 2),
        }
        out["by_timepoint"].append(row)
        rec.add(
            tier=5, subject_type="metric", timepoint=str(tp), metric="manual_mae",
            value=rounded(mae, 4), unit=U_PP,
            note=f"mean |auto - manual| across {len(grp)} metrics at this timepoint",
        )

    # ---- per-metric ----------------------------------------------------------
    for metric, grp in cmp.groupby("metric"):
        mae = float(grp["abs_delta"].mean())
        # Signed mean shows whether the pipeline systematically over- or under-calls, which a
        # mean ABSOLUTE deviation hides entirely.
        bias = float(grp["delta"].mean()) if "delta" in grp.columns else None
        out["by_metric"].append(
            {
                "metric": str(metric),
                "n_timepoints": int(len(grp)),
                "mae_pp": rounded(mae, 4),
                "signed_bias_pp": rounded(bias, 4),
                "max_abs_delta_pp": rounded(float(grp["abs_delta"].max()), 4),
            }
        )
        rec.add(
            tier=5, subject_type="metric", metric=f"manual_mae|{metric}",
            value=rounded(mae, 4), unit=U_PP,
            normalized_value=rounded(bias, 4),
            normalized_unit="signed mean deviation (auto - manual), pp",
            note="MAE hides direction; the signed bias shows systematic over/under-calling",
        )

    # ---- individual divergences ---------------------------------------------
    band = th.manual_divergence_pp
    bad = cmp[cmp["abs_delta"] > band].sort_values("abs_delta", ascending=False)
    out["worst_rows"] = [
        {k: (rounded(v, 4) if isinstance(v, (int, float)) else str(v)) for k, v in r.items()}
        for r in bad.head(20).to_dict(orient="records")
    ]
    for r in bad.head(20).to_dict(orient="records"):
        book.raise_if(
            True,
            code=F_MANUAL_DIVERGENCE, tier=5,
            subject=f"{r.get('metric')} @ {r.get('timepoint')}",
            rule=(f"|auto - manual| > {band} pp (the band already used by "
                  "compare.summarize's within_10pp statistic)"),
            measured={
                "auto": rounded(r.get("auto"), 4),
                "manual": rounded(r.get("manual"), 4),
                "delta_pp": rounded(r.get("delta"), 4),
                "abs_delta_pp": rounded(r.get("abs_delta"), 4),
            },
            resolution_hint=(
                "Cross-reference this timepoint's tier 2 pattern and tier 3 cutoff range: a "
                "divergence at a timepoint that is also flagged for drift points at the "
                "transfer; a divergence at a stable timepoint points at the cutoff itself or "
                "at a genuine gating-strategy difference from the operator."
            ),
            value=float(r.get("abs_delta") or 0.0), threshold=band,
        )
    if len(bad) > 20:
        out["truncated"] = {
            "n_divergences": int(len(bad)),
            "n_flagged": 20,
            "note": (
                f"{len(bad)} rows exceeded {band} pp; only the 20 largest were flagged "
                "individually. The full set is in the comparison table."
            ),
        }
    return out
