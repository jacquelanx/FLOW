"""Writers — four co-existing representations, because they serve different readers.

  * **Tidy long CSV** (``diagnostics_metrics.csv``) — for the biologist, the mentor, and any
    downstream reanalysis. One row per measurement, with units and a normalized twin.
  * **Flat per-subject CSVs** — one row per marker (tier 1), per marker x timepoint (tier 2),
    per reported number (the uncertainty envelope), and so on. These are what a person
    actually reads, and every one of them is auto-loaded into the agent's notebook by the
    FLOW first-run harness.
  * **A deterministic text summary** — flagged items only, plus the headline tables. A flat
    table of several thousand measurements buries the handful that matter; this is the part
    written to be READ.
  * **A worklist** (``diagnostics_worklist.txt`` + ``diagnostics_obligations.json``) — the
    handful of items a reader is obliged to open, each naming the rows behind it. Counting
    honestly is not the same as being finishable: see ``obligations.py``.

Computing well and presenting badly wastes the effort, so the text summary opens with how to
read it, and every flag prints the rule that produced it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import obligations as obl
from .flags import FlagBook
from .record import MetricRecorder

# Output filenames. Every ``.csv`` here is picked up automatically by the first-run harness
# and exposed in the agent's notebook as a variable named after the file.
METRICS_CSV = "diagnostics_metrics.csv"
FLAGS_CSV = "diagnostics_flags.csv"
CUTOFF_AUDIT_CSV = "diagnostics_cutoff_audit.csv"
TRANSFER_CSV = "diagnostics_transfer.csv"
GATE_GEOMETRY_CSV = "diagnostics_gate_geometry.csv"
UNCERTAINTY_CSV = "diagnostics_uncertainty.csv"
COUNTERFACTUAL_CSV = "diagnostics_counterfactual.csv"
CONTROLS_CSV = "diagnostics_controls.csv"
CONCORDANCE_CSV = "diagnostics_concordance.csv"
THRESHOLDS_CSV = "diagnostics_thresholds.csv"
TIMEPOINTS_CSV = "diagnostics_timepoints.csv"
DISCONTINUITY_CSV = "diagnostics_discontinuity.csv"
JSON_FILE = "cutoff_diagnostics.json"
SUMMARY_TXT = "cutoff_diagnostics_summary.txt"
DIGEST_TXT = "cutoff_diagnostics_digest.txt"
# The worklist, rendered. Written as text for the same reason the digest is: the harness
# prints a file rather than parsing a manifest, so what reaches the reader is decided here.
WORKLIST_TXT = "diagnostics_worklist.txt"

# The digest is the representation that has to survive a bounded observation window (the
# agent's notebook truncates cell output), so its size is a construction constraint rather
# than an outcome. See ``render_digest``.
DIGEST_MAX_CHARS = 11500


# ── generic csv helper ────────────────────────────────────────────────────────
def _write_csv(path: Path, rows: Sequence[dict[str, Any]],
               preferred: Iterable[str] = ()) -> Optional[Path]:
    """Write ``rows`` with the union of their keys as columns; ``preferred`` keys come first.

    Rows here are heterogeneous by design — an unavailable measurement carries a ``reason``
    field its available siblings do not — so the column set is the union rather than the
    first row's keys, which would silently drop those explanations.
    """
    if not rows:
        return None
    seen: list[str] = []
    for k in preferred:
        if any(k in r for r in rows):
            seen.append(k)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seen, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in seen})
    return path


def _cell(v: Any) -> Any:
    """Render a value for CSV: containers become compact JSON, None becomes empty."""
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str, sort_keys=True)
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


# ── flatteners ────────────────────────────────────────────────────────────────
def cutoff_audit_rows(tier1: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per marker: is the ruler well made at the reference timepoint?"""
    rows: list[dict[str, Any]] = []
    for key, m in sorted((tier1 or {}).get("markers", {}).items()):
        base = {
            "marker": key,
            "label": m.get("label"),
            "channel": m.get("channel"),
            "cutoff": m.get("cutoff"),
            "available": m.get("available"),
            # WHERE this row was measured. Normally the reference timepoint; when the
            # reference pool could not carry the audit it is another timepoint, and then this
            # row is placement in THAT distribution rather than a check on the derivation. A
            # reader who cannot see which has been handed the wrong claim, so both columns
            # travel on every row — including the unavailable ones below.
            "audited_at_timepoint": m.get("audited_at_timepoint"),
            "audited_at_reference": m.get("audited_at_reference"),
            "reference_audit_unavailable_reason": m.get("reference_audit_unavailable_reason"),
            "audit_scope_note": m.get("audit_scope_note"),
        }
        if not m.get("available"):
            base["reason"] = m.get("reason")
            rows.append(base)
            continue
        silv = m.get("unimodality_silverman") or {}
        lrt = m.get("unimodality_mixture_lrt") or {}
        base.update(
            {
                "pool": m.get("pool"),
                "pool_events": m.get("pool_events"),
                "positive_side": m.get("positive_side"),
                "valley_derived_by_pipeline": m.get("valley_derived_by_pipeline"),
                "n_modes": m.get("n_modes"),
                "negative_mode": m.get("negative_mode"),
                "positive_mode": m.get("positive_mode"),
                "mode_gap": m.get("mode_gap"),
                "trough_location": m.get("trough_location"),
                "trough_depth_ratio": m.get("trough_depth_ratio"),
                "density_at_cutoff_ratio": m.get("density_at_cutoff_ratio"),
                "placement_in_negative_sd": m.get("placement_in_negative_sd"),
                "normalized_gap_position": m.get("normalized_gap_position"),
                "cutoff_to_trough_distance_gap_fraction":
                    m.get("cutoff_to_trough_distance_gap_fraction"),
                "negative_sd_used": m.get("negative_sd_used"),
                "negative_sd_source": m.get("negative_sd_source"),
                "mode_estimator_disagreement_in_sd": m.get("mode_estimator_disagreement_in_sd"),
                "stain_index": m.get("stain_index"),
                "p_unimodal_silverman": silv.get("p_value_unimodal"),
                "p_unimodal_mixture_lrt": lrt.get("p_value_unimodal"),
                "delta_bic_favouring_two": lrt.get("delta_bic_favouring_two"),
                "standardized_separation": lrt.get("standardized_separation"),
                "valley_supported": m.get("valley_supported"),
                "valley_evidence_decided_by": (m.get("valley_evidence") or {}).get("decided_by"),
                "smallest_component_weight":
                    (m.get("valley_evidence") or {}).get("smallest_component_weight"),
                "positive_fraction": m.get("positive_fraction"),
                "pipeline_accepted_fraction": m.get("pipeline_accepted_fraction"),
                "position_within_accepted_band": m.get("position_within_accepted_band"),
                "cutoff_percentile_rank_in_pool": m.get("cutoff_percentile_rank_in_pool"),
                "suspected_percentile_fallback": m.get("suspected_percentile_fallback"),
                "overlay_bin_width": (m.get("overlay_axis") or {}).get("bin_width"),
            }
        )
        rows.append(base)
    return rows


def transfer_rows(tier2: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per marker x timepoint: does the locked cutoff still hold there?"""
    rows: list[dict[str, Any]] = []
    for key, m in sorted((tier2 or {}).get("markers", {}).items()):
        if not m.get("available"):
            rows.append({"marker": key, "available": False, "reason": m.get("reason")})
            continue
        for tp, r in m.get("timepoints", {}).items():
            row = {
                "marker": key,
                "label": m.get("label"),
                "timepoint": tp,
                "is_reference": r.get("is_reference"),
                "available": r.get("available"),
                "cutoff": m.get("cutoff"),
            }
            row.update(
                {k: v for k, v in r.items()
                 if k not in ("is_reference", "available", "pattern_reading",
                              "pattern_reading_is_a_lookup")}
            )
            row["pattern_reading"] = r.get("pattern_reading")
            rows.append(row)
    return rows


def gate_geometry_rows(tier2: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per timepoint: every 2-D gate's margins, plus locked-scatter containment.

    Three gate families, one row, prefixed by gate: ``nk_`` (CD3 x CD56), ``donor_car_``
    (HLA x CAR — the gate that defines %Donor NK and %CAR+) and ``scatter_`` (the locked
    FSC/SSC island). One wide row per timepoint rather than three tables, so the harness loads
    one more DataFrame and not three, and so a reader comparing gates at one timepoint reads
    across instead of joining.
    """
    rows: list[dict[str, Any]] = []
    for tp, g in (tier2 or {}).get("gate_geometry", {}).items():
        row: dict[str, Any] = {"timepoint": tp}
        for prefix, block in (("nk", g.get("nk_quadrant") or {}),
                              ("donor_car", g.get("donor_car_quadrant") or {}),
                              ("scatter", g.get("locked_scatter") or {})):
            for k, v in block.items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)
    return rows


def controls_rows(tier4: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per control check, all three kinds in one table with a ``check`` column."""
    rows: list[dict[str, Any]] = []
    for r in (tier4 or {}).get("host_nk_car", []):
        rows.append({"check": "host_nk_car_false_positive",
                     "expected": "~0 (host NK are CAR-negative by construction)", **r})
    for r in (tier4 or {}).get("pre_infusion_donor", []):
        rows.append({"check": "pre_infusion_donor_specificity",
                     "expected": "~0 (no donor cells exist before infusion)", **r})
    for kind, r in ((tier4 or {}).get("control_tubes") or {}).items():
        rows.append({"check": f"control_tube_{kind}", "timepoint": f"control:{kind}", **r})
    return rows


def concordance_rows(tier5: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-timepoint and per-metric operator agreement in one table."""
    if not (tier5 or {}).get("available"):
        return []
    rows: list[dict[str, Any]] = []
    for r in tier5.get("by_timepoint", []):
        rows.append({"scope": "timepoint", **r})
    for r in tier5.get("by_metric", []):
        rows.append({"scope": "metric", **r})
    return rows


# ── top-level write ───────────────────────────────────────────────────────────
def write_all(
    out_dir: Path,
    report: dict[str, Any],
    rec: MetricRecorder,
    book: FlagBook,
) -> dict[str, str]:
    """Write every artifact. Returns ``{label: path}`` for the ones actually written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    def _note(label: str, p: Optional[Path], rows: Sequence[Any] = ()) -> None:
        if p is not None:
            written[label] = str(p)
            row_counts[label] = len(rows)

    t0 = report.get("tier0_data_adequacy") or {}
    t1 = report.get("tier1_cutoff_foundation") or {}
    t2 = report.get("tier2_transfer_validity") or {}
    t3 = report.get("tier3_sensitivity") or {}
    t4 = report.get("tier4_internal_controls") or {}
    t5 = report.get("tier5_operator_concordance") or {}

    # Table-driven so every writer is declared once and the digest's drill-down map can
    # report each table's row count without re-reading the files it just wrote.
    tables: list[tuple[str, str, Sequence[dict[str, Any]], tuple[str, ...]]] = [
        ("metrics", METRICS_CSV, rec.rows,
         ("tier", "subject_type", "marker", "timepoint", "metric", "value", "unit",
          "normalized_value", "normalized_unit", "note")),
        ("flags", FLAGS_CSV, book.rows(),
         ("tier", "code", "subject", "exceedance", "rule", "measured", "resolution_hint")),
        ("cutoff_audit", CUTOFF_AUDIT_CSV, cutoff_audit_rows(t1),
         ("marker", "label", "cutoff", "audited_at_timepoint", "audited_at_reference",
          "placement_in_negative_sd", "normalized_gap_position",
          "density_at_cutoff_ratio", "valley_supported")),
        ("transfer", TRANSFER_CSV, transfer_rows(t2),
         ("marker", "timepoint", "baseline_timepoint", "baseline_is_reference",
          "negative_mode_drift", "negative_mode_drift_in_negative_sd",
          "negative_mode_drift_in_se", "negative_mode_drift_gap_fraction",
          "placement_in_negative_sd", "placement_change_in_negative_sd", "pattern")),
        ("gate_geometry", GATE_GEOMETRY_CSV, gate_geometry_rows(t2), ("timepoint",)),
        ("uncertainty", UNCERTAINTY_CSV, t3.get("envelope", []),
         ("timepoint", "metric", "headline_value_pct", "value_pct_at_sensitivity_subsample",
          "headline_minus_subsample_pp", "cutoff_low_pct", "cutoff_high_pct",
          "cutoff_range_pp", "counting_ci_halfwidth_pp", "dominant_uncertainty")),
        ("counterfactual", COUNTERFACTUAL_CSV, t3.get("counterfactual", []),
         ("timepoint", "metric", "headline_value_pct", "locked_value_pct_at_subsample",
          "per_sample_value_pct_at_subsample", "delta_pp")),
        ("controls", CONTROLS_CSV, controls_rows(t4),
         ("check", "timepoint", "expected", "available")),
        ("concordance", CONCORDANCE_CSV, concordance_rows(t5), ("scope",)),
        ("thresholds", THRESHOLDS_CSV, report.get("thresholds", []),
         ("threshold", "value", "default", "overridden", "provenance", "rationale")),
        ("timepoints", TIMEPOINTS_CSV, t0.get("per_timepoint", []),
         ("timepoint", "filename", "n_total_events", "n_events_analyzed", "n_lymphocytes",
          "lineage_purity_pct", "retention_cd45p_pct", "retention_live_pct")),
        ("discontinuity", DISCONTINUITY_CSV, t0.get("discontinuity", []),
         ("timepoint", "metric", "value_pct", "discontinuous", "direction",
          "prev_timepoint", "prev_value_pct", "next_timepoint", "next_value_pct",
          "tolerance_pp", "excursion_beyond_tolerance_pp", "excursion_in_tolerances",
          "assessable", "reason")),
    ]
    for label, filename, rows, preferred in tables:
        _note(label, _write_csv(out_dir / filename, rows, preferred=preferred), rows)

    # The worklist, built here rather than by the caller because it needs the FINAL table
    # list: an obligation may only point a reader at a table that was actually written.
    # ``report`` is mutated for the same reason the digest is rendered last — the manifest
    # belongs in the nested JSON, which the caller re-dumps once the file list is known.
    manifest = obl.build(report, book, written=written)
    report["obligations"] = manifest
    op = out_dir / obl.OBLIGATIONS_JSON
    op.write_text(json.dumps(manifest, indent=2, default=str))
    written["obligations"] = str(op)
    row_counts["obligations"] = len(manifest.get("items") or [])

    wp = out_dir / WORKLIST_TXT
    wp.write_text(obl.render_worklist(manifest))
    written["worklist"] = str(wp)

    jp = out_dir / JSON_FILE
    jp.write_text(json.dumps(report, indent=2, default=str))
    written["json"] = str(jp)

    sp = out_dir / SUMMARY_TXT
    sp.write_text(render_summary(report, book))
    written["summary"] = str(sp)

    # Written last: the digest indexes everything above, so it needs the final file list.
    dp = out_dir / DIGEST_TXT
    dp.write_text(render_digest(report, book, row_counts=row_counts, written=written,
                                obligations=manifest))
    written["digest"] = str(dp)
    return written


# ── text summary ──────────────────────────────────────────────────────────────
def _fmt(v: Any, places: int = 3) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:
            return "—"
        return f"{v:.{places}f}"
    return str(v)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], indent: str = "  ") -> str:
    """Fixed-width text table; keeps the summary readable in a terminal and in a notebook."""
    if not rows:
        return f"{indent}(none)\n"
    cols = [[str(h)] + [_fmt(r[i]) if i < len(r) else "" for r in rows]
            for i, h in enumerate(headers)]
    widths = [max(len(c) for c in col) for col in cols]
    out = []
    out.append(indent + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    out.append(indent + "  ".join("-" * w for w in widths))
    for r in rows:
        cells = [_fmt(r[i]) if i < len(r) else "" for i in range(len(headers))]
        out.append(indent + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(out) + "\n"


HOW_TO_READ = """\
HOW TO READ THIS FILE
  * Everything here is a MEASUREMENT, not a conclusion. A "flag" is a measurement that
    crossed a STATED RULE; the rule is printed with it, so you can disagree with the rule.
    Nothing in this file decides whether the analysis is right — that judgement is yours.
  * Values are paired with a dimensionless twin. "drift = 0.38" is unjudgeable; the same row
    gives it as a fraction of the reference negative-to-positive gap and as a multiple of the
    measurement's own bootstrap noise.
  * Missing is not zero. A null means "could not be measured", and the reason is given. In
    particular, absent drift for an NK-bright timepoint means the negative population does not
    exist there — NOT that nothing moved.
  * The tiers form a chain of trust; a failure low down makes higher tiers meaningless:
      tier 0  data adequacy        — is there enough good data to say anything?
      tier 1  cutoff foundation    — is the ruler well made where it was derived?
      tier 2  transfer validity    — does it still fit at the other timepoints?
      tier 3  sensitivity          — does any of this change the answer?
      tier 4  internal controls    — do populations with a KNOWN answer come out right?
      tier 5  operator concordance — does it agree with the manual gating?
  * Flags are ordered by tier, then by how far the rule was exceeded. That ordering is a
    presentation aid, NOT a ranking of scientific importance — a tier-0 note can matter more
    than a large tier-3 exceedance.
"""


def render_summary(report: dict[str, Any], book: FlagBook) -> str:
    """The deterministic, human- and agent-readable rendering. Flagged items first."""
    L: list[str] = []
    pid = report.get("patient_id", "?")
    L.append("=" * 80)
    L.append(f"CUTOFF DIAGNOSTICS — {pid}")
    L.append("=" * 80)
    L.append("")
    L.append(HOW_TO_READ)

    # ---- reproduction ------------------------------------------------------
    rep = report.get("reproduction") or {}
    L.append("REPRODUCTION CHECK  (does this audit describe the run that produced the tables?)")
    if not rep.get("available"):
        L.append("  Could not verify — no multilineage table was found to compare against.")
    elif rep.get("faithful"):
        L.append(f"  OK — {rep.get('checked')} reported values reproduced exactly from "
                 f"{rep.get('source')} (max |delta| = {rep.get('max_abs_delta_pp')} pp).")
    else:
        L.append(f"  MISMATCH — {len(rep.get('mismatches') or [])} of {rep.get('checked')} "
                 f"values differ from {rep.get('source')}.")
        L.append("  Everything below may describe a different gating than the headline tables.")
        for m in (rep.get("mismatches") or [])[:8]:
            L.append(f"    {m}")
    L.append("")

    # ---- configuration ----------------------------------------------------
    cfg = report.get("configuration") or {}
    L.append("RUN CONFIGURATION")
    L.append(f"  reference timepoint : {cfg.get('reference_timepoint')}")
    L.append(f"  anchor derivation   : {cfg.get('anchor_derivation')}")
    L.append(f"  timepoints          : {', '.join(cfg.get('timepoints') or []) or '—'}")
    L.append(f"  control tubes       : {', '.join(cfg.get('control_tubes') or []) or 'none'}")
    L.append(f"  donor HLA cutoff    : {_fmt(cfg.get('hla_cutoff'))}"
             f"   (dim-donor polarity: {_fmt(cfg.get('hla_dim'))})")
    L.append(f"  CAR cutoff          : {_fmt(cfg.get('car_cutoff'))}")
    L.append(f"  subsample per file  : {cfg.get('subsample')}")
    if cfg.get("threshold_overrides"):
        L.append(f"  threshold overrides : {cfg['threshold_overrides']}")
    for n in report.get("notes") or []:
        L.append(f"  note: {n}")
    for sk in report.get("skipped") or []:
        L.append(f"  skipped {sk.get('file')}: {sk.get('reason')}")
    L.append("")

    # ---- flags -------------------------------------------------------------
    ordered = book.ordered()
    L.append(f"FLAGGED ITEMS ({len(ordered)})")
    if not ordered:
        L.append("  No rule was crossed. That is not the same as 'the cutoffs are correct' — "
                 "it means no rule in this set fired.")
    for f in ordered:
        exc = "" if f.exceedance is None else f"  [x{f.exceedance:.2f} the rule]"
        L.append(f"  [tier {f.tier}] {f.code}  —  {f.subject}{exc}")
        L.append(f"      rule     : {f.rule}")
        L.append(f"      measured : {json.dumps(f.measured, default=str, sort_keys=True)}")
        L.append(f"      next     : {f.resolution_hint}")
    L.append("")

    # ---- headline numbers with uncertainty ---------------------------------
    t3 = report.get("tier3_sensitivity") or {}
    env = t3.get("envelope") or []
    L.append("HEADLINE NUMBERS WITH TWO LABELLED UNCERTAINTY SOURCES  (tier 3c)")
    L.append("  'cutoff' = attributable to where the line was drawn (one-at-a-time, not joint).")
    L.append("  'counting' = Wilson interval from the event counts. Different remedies:")
    L.append("  more events fixes counting; only a better cutoff fixes the other.")
    L.append("  'value%' is the pipeline's reported number; the cutoff bounds are re-gated on "
             "tier 3's capped subsample, so read 'range_pp' as a WIDTH on the value rather "
             "than treating cutoff_lo/hi as absolute bounds.")
    L.append(_table(
        ["timepoint", "metric", "value%", "cutoff_lo", "cutoff_hi", "range_pp",
         "count_+-pp", "dominant", "driver"],
        [[e.get("timepoint"), e.get("metric"), e.get("headline_value_pct"),
          e.get("cutoff_low_pct"),
          e.get("cutoff_high_pct"), e.get("cutoff_range_pp"),
          e.get("counting_ci_halfwidth_pp"), e.get("dominant_uncertainty"),
          e.get("cutoff_range_driver_high") or e.get("cutoff_range_driver_low")]
         for e in env]))
    cov = t3.get("coverage") or {}
    if cov.get("dropped"):
        L.append(f"  COVERAGE GAP — {obl.describe_coverage_gap(cov)}")
        if cov.get("reason"):
            L.append(f"    {cov['reason']}")
        for d in cov["dropped"][:10]:
            L.append(f"    dropped: {d}")
    L.append("")

    # ---- tier 1 ------------------------------------------------------------
    t1 = report.get("tier1_cutoff_foundation") or {}
    L.append("CUTOFF FOUNDATION AT THE REFERENCE  (tier 1)")
    if not t1.get("available"):
        L.append(f"  unavailable: {t1.get('reason')}")
    else:
        L.append(_table(
            ["marker", "cutoff", "place_SD", "gap_pos", "dens@cut", "p_silv", "p_lrt",
             "frac+", "valley?", "rank", "fallback?"],
            [[m.get("label") or k, m.get("cutoff"), m.get("placement_in_negative_sd"),
              m.get("normalized_gap_position"), m.get("density_at_cutoff_ratio"),
              (m.get("unimodality_silverman") or {}).get("p_value_unimodal"),
              (m.get("unimodality_mixture_lrt") or {}).get("p_value_unimodal"),
              m.get("positive_fraction"), m.get("valley_supported"),
              m.get("cutoff_percentile_rank_in_pool"),
              m.get("suspected_percentile_fallback")]
             for k, m in sorted(t1.get("markers", {}).items()) if m.get("available")]))
        for k, m in sorted(t1.get("markers", {}).items()):
            if not m.get("available"):
                L.append(f"  {k}: unavailable — {m.get('reason')}")
            elif not m.get("audited_at_reference", True):
                L.append(f"  {k}: AUDITED AT {m.get('audited_at_timepoint')}, not at the "
                         f"reference — {m.get('reference_audit_unavailable_reason')}")
                L.append(f"      {m.get('audit_scope_note')}")
    L.append("")

    # ---- tier 2 ------------------------------------------------------------
    t2 = report.get("tier2_transfer_validity") or {}
    L.append("TRANSFER VALIDITY ACROSS TIMEPOINTS  (tier 2)")
    L.append("  Drift is measured on the NEGATIVE population only — the internal reference")
    L.append("  biology should not move. 'pattern' is a lookup against the rule in each row.")
    if not t2.get("available"):
        L.append(f"  unavailable: {t2.get('reason')}")
    else:
        rows = []
        for k, m in sorted(t2.get("markers", {}).items()):
            if not m.get("available"):
                continue
            for tp, r in m.get("timepoints", {}).items():
                if r.get("is_reference") or not r.get("available"):
                    continue
                rows.append([k, tp, r.get("negative_mode_drift"),
                             r.get("negative_mode_drift_in_negative_sd"),
                             r.get("negative_mode_drift_in_se"),
                             r.get("placement_in_negative_sd"),
                             r.get("placement_change_in_negative_sd"),
                             r.get("emd_negative_permutation_p"),
                             r.get("pattern")])
        L.append(_table(
            ["marker", "timepoint", "drift", "drift/negSD", "drift/SE", "place_SD",
             "place_change", "emd_p", "pattern"], rows))
        undef = [(k, tp) for k, m in (t2.get("markers") or {}).items()
                 if m.get("available")
                 for tp, r in m.get("timepoints", {}).items()
                 if r.get("negative_population_defined") is False]
        if undef:
            L.append("  Negative population UNDEFINED (drift is null, not zero) at: "
                     + ", ".join(f"{k}@{tp}" for k, tp in undef))
    L.append("")

    # ---- gate geometry -----------------------------------------------------
    geo = (t2 or {}).get("gate_geometry") or {}
    if geo:
        L.append("GATE GEOMETRY  (tier 2, 2-D)")
        L.append(_table(
            ["timepoint", "NK_n", "cd3_margin_SD", "cd56_margin_SD", "near_boundary",
             "scatter_containment", "fsc_offset_SD"],
            [[tp, (g.get("nk_quadrant") or {}).get("n_nk"),
              (g.get("nk_quadrant") or {}).get("cd3_boundary_margin_in_sd"),
              (g.get("nk_quadrant") or {}).get("cd56_boundary_margin_in_sd"),
              (g.get("nk_quadrant") or {}).get("fraction_within_one_sd_of_a_boundary"),
              (g.get("locked_scatter") or {}).get("containment"),
              (g.get("locked_scatter") or {}).get("fsc_centroid_offset_in_sd")]
             for tp, g in geo.items()]))
        L.append("")

    # ---- tier 3b -----------------------------------------------------------
    cf = t3.get("counterfactual") or []
    pv = t3.get("policy_variance") or {}
    L.append("LOCKED vs PER-SAMPLE CUTOFFS  (tier 3b — what anchoring is actually doing)")
    L.append("  The per-sample column is what each timepoint WOULD have reported under its own")
    L.append("  valley. Their spread across timepoints is the run-to-run drift anchoring removes.")
    L.append(_table(
        ["timepoint", "metric", "locked%", "per_sample%", "delta_pp"],
        [[r.get("timepoint"), r.get("metric"), r.get("locked_value_pct_at_subsample"),
          r.get("per_sample_value_pct_at_subsample"), r.get("delta_pp")]
         for r in cf if r.get("metric")]))
    if pv:
        L.append("  Cross-timepoint spread under each policy:")
        L.append(_table(
            ["metric", "SD_locked_pp", "SD_per_sample_pp", "variance_removed_pp", "ratio"],
            [[m, d.get("sd_locked_pp"), d.get("sd_per_sample_pp"),
              d.get("variance_removed_pp"), d.get("sd_ratio_per_sample_over_locked")]
             for m, d in sorted(pv.items())], indent="    "))
        L.append("    caveat: cross-timepoint SD mixes real biology with cutoff drift. It is")
        L.append("    comparable BETWEEN policies because the biology is identical in both.")
    L.append("")

    # ---- tier 4 ------------------------------------------------------------
    t4 = report.get("tier4_internal_controls") or {}
    L.append("INTERNAL CONTROLS  (tier 4 — the only checks with a known correct answer)")
    hn = t4.get("host_nk_car") or []
    if hn:
        L.append("  Host (non-donor) NK CAR+ rate — CAR-negative by construction, expect ~0:")
        L.append(_table(
            ["timepoint", "n_host_NK", "false_pos_rate", "CI_low", "CI_high",
             "reported_CAR%", "share_explained"],
            [[r.get("timepoint"), r.get("n_host_nk"), r.get("false_positive_rate"),
              (r.get("false_positive_ci") or [None, None])[0],
              (r.get("false_positive_ci") or [None, None])[1],
              r.get("reported_car_of_donor_pct"), r.get("fpr_share_of_reported_car")]
             for r in hn], indent="    "))
    pre = t4.get("pre_infusion_donor") or []
    if pre:
        L.append("  Pre-infusion donor NK — no donor cells possible, expect ~0:")
        L.append(_table(
            ["timepoint", "is_reference", "n_NK", "donor_rate", "donor_of_NK%"],
            [[r.get("timepoint"), r.get("is_reference_timepoint"), r.get("n_nk"),
              r.get("donor_rate_of_nk"), r.get("donor_of_nk_pct")] for r in pre],
            indent="    "))
    tubes = t4.get("control_tubes") or {}
    if tubes:
        L.append("  Designed control tubes:")
        L.append(_table(
            ["tube", "available", "expected", "n_NK", "CAR+_rate", "donor_frac", "reason"],
            [[k, v.get("available"), v.get("expected"), v.get("n_nk"),
              v.get("car_positive_rate_of_nk"), v.get("donor_side_fraction_of_nk"),
              v.get("reason")] for k, v in sorted(tubes.items())], indent="    "))
    L.append("")

    # ---- tier 0 ------------------------------------------------------------
    t0 = report.get("tier0_data_adequacy") or {}
    L.append("DATA ADEQUACY  (tier 0)")
    L.append(_table(
        ["timepoint", "n_total", "n_analyzed", "live%", "CD45+%", "lymph%", "purity%",
         "rate_CV"],
        [[s.get("timepoint"), s.get("n_total_events"), s.get("n_events_analyzed"),
          (s.get("retention_pct") or {}).get("live"),
          (s.get("retention_pct") or {}).get("cd45p"),
          (s.get("retention_pct") or {}).get("lymphocytes"),
          (s.get("retention_pct") or {}).get("lineage_purity"),
          (s.get("acquisition") or {}).get("event_rate_cv")]
         for s in t0.get("samples", [])]))
    L.append("")

    # ---- tier 5 ------------------------------------------------------------
    t5 = report.get("tier5_operator_concordance") or {}
    L.append("OPERATOR CONCORDANCE  (tier 5)")
    if not t5.get("available"):
        L.append(f"  unavailable: {t5.get('reason')}")
    else:
        ov = t5.get("overall") or {}
        L.append(f"  overall: MAE={_fmt(ov.get('mae'))} pp   <=5pp={_fmt(ov.get('within_5pp'))}%"
                 f"   <=10pp={_fmt(ov.get('within_10pp'))}%   n={ov.get('n')}")
        c = t5.get("coverage") or {}
        L.append(f"  coverage: {len(c.get('timepoints_compared') or [])} of "
                 f"{len(c.get('timepoints_in_run') or [])} timepoints"
                 + (f"; NOT compared: {', '.join(c.get('timepoints_not_compared') or [])}"
                    if c.get("timepoints_not_compared") else ""))
        L.append(_table(
            ["timepoint", "MAE_pp", "worst_metric", "max_abs_pp", "<=5pp%"],
            [[r.get("timepoint"), r.get("mae_pp"), r.get("worst_metric"),
              r.get("max_abs_delta_pp"), r.get("within_5pp_pct")]
             for r in t5.get("by_timepoint", [])]))
        L.append(_table(
            ["metric", "MAE_pp", "signed_bias_pp", "max_abs_pp"],
            [[r.get("metric"), r.get("mae_pp"), r.get("signed_bias_pp"),
              r.get("max_abs_delta_pp")] for r in t5.get("by_metric", [])]))
    L.append("")

    # ---- failures ----------------------------------------------------------
    failures = report.get("tier_failures") or []
    if failures:
        L.append("TIERS THAT FAILED TO COMPUTE")
        for f in failures:
            L.append(f"  tier {f.get('tier')} ({f.get('name')}): {f.get('error')}")
        L.append("  Other tiers are unaffected — each is computed in isolation.")
        L.append("")

    L.append("WHAT THIS CANNOT CATCH")
    for lim in report.get("limitations") or []:
        L.append(f"  * {lim}")
    L.append("")
    L.append(f"Files written: {', '.join(sorted((report.get('files') or {}).values()))}")
    return "\n".join(L) + "\n"


# ── digest ────────────────────────────────────────────────────────────────────
# What each table answers, for the digest's drill-down map. Keyed by the label used in
# ``write_all``; a label with no entry here still gets listed, just without the gloss.
TABLE_PURPOSE = {
    "flags": "every flag in full: its rule, its numbers, and what would settle it",
    "cutoff_audit": "per marker at the reference: is the ruler well made",
    "transfer": "per marker x timepoint: does the locked cutoff still hold",
    "uncertainty": ("every headline % with cutoff- and counting-uncertainty, separately "
                    "(quote headline_value_pct; '_at_subsample' columns are tier-3 re-gates)"),
    "counterfactual": "locked vs per-sample cutoffs: what the anchoring is doing",
    "controls": "populations with a KNOWN answer: do they come out right",
    "concordance": "agreement with the operator's manual gating",
    "gate_geometry": "2-D quadrant margins and locked-scatter containment",
    "metrics": "every measurement, tidy long, with units",
    "thresholds": "every rule, its value, and where the value came from",
    "timepoints": "per SAMPLE: events, retention and lineage purity at each timepoint",
    "discontinuity": "per timepoint x metric: does the value fit between its neighbours",
    "summary": "the full prose report (large; read the specific section you need)",
    "json": "the whole report, nested",
    "obligations": "the worklist as a manifest: ids, discharge conditions, machine-readable",
    "worklist": "THE WORKLIST: what this run leaves you to read, in priority order",
}

DIGEST_HEADER = """\
CUTOFF DIAGNOSTICS — DIGEST  (a bounded INDEX, not the report)
  Read this first. It names EVERY group of flags with its full count, then shows the single
  worst case in each group. It is deliberately short enough to survive this notebook's output
  limit; the per-row detail lives in the CSVs named at the bottom, which are already loaded
  as DataFrames in this notebook.
  Same discipline as the full report: every entry is a MEASUREMENT that crossed a STATED
  RULE, printed with the rule so you can disagree with it. Nothing here concludes that a
  cutoff is wrong. A null means "could not be measured", never zero.\
"""


def _split_subject(subject: str) -> tuple[str, str]:
    """Split a flag subject into (item, scope) on the ' @ ' the flag emitters use.

    Kept as a pure string split rather than a lookup against a marker/timepoint list: the
    digest must summarize whatever subjects the tiers produced, including ones added later.
    """
    s = str(subject or "").strip()
    if " @ " in s:
        item, scope = s.split(" @ ", 1)
        return item.strip(), scope.strip()
    return s, ""


def _top_counts(values: Sequence[str], k: int = 4) -> str:
    """Render a value->count tally, most frequent first, naming what was not shown."""
    tally: dict[str, int] = {}
    for v in values:
        if v:
            tally[v] = tally.get(v, 0) + 1
    if not tally:
        return ""
    ordered = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = " ".join(f"{name}({n})" for name, n in ordered[:k])
    if len(ordered) > k:
        shown += f" +{len(ordered) - k} more"
    return shown


def _exc(value: Optional[float]) -> str:
    """Render an exceedance ratio, naming the two ways it can be absent rather than hiding
    them behind a number: an unbounded ratio (the rule's threshold is zero, e.g. "no events
    at all") and a rule that is not a numeric threshold at all."""
    if value is None or value != value:
        return "n/a (not a numeric rule)"
    if value == float("inf"):
        return "unbounded"
    return f"{value:.2f}x"


def _clip(text: str, limit: int) -> str:
    """One-line clip. Used on rule text, which is prose of unbounded length."""
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _flag_groups(book: FlagBook) -> list[tuple[int, str, list]]:
    """Group flags by (tier, code), preserving ``book.ordered()``.

    Because ``ordered()`` sorts by tier then descending exceedance, the first flag in each
    group is that group's worst offender.
    """
    groups: dict[tuple[int, str], list] = {}
    for f in book.ordered():
        groups.setdefault((f.tier, f.code), []).append(f)
    return [(tier, code, fs) for (tier, code), fs in groups.items()]


def _group_block(tier: int, code: str, fs: Sequence[Any], detail: int) -> list[str]:
    """Render one flag group at a given detail level (2 = full, 1 = no tally, 0 = head only).

    Degrading detail rather than dropping groups is deliberate: an omitted group would read
    as "nothing fired there", which is exactly the false negative the diagnostics exist to
    prevent. Counts are never degraded.
    """
    worst = fs[0]
    lines = [f"  [t{tier}] {code}  x{len(fs)}   worst exceedance {_exc(worst.exceedance)}"]
    if detail >= 1:
        items = [_split_subject(f.subject)[0] for f in fs]
        scopes = [_split_subject(f.subject)[1] for f in fs]
        tally_items = _top_counts(items)
        tally_scopes = _top_counts(scopes)
        if tally_items:
            lines.append(f"        items : {_clip(tally_items, 96)}")
        if tally_scopes:
            lines.append(f"        at    : {_clip(tally_scopes, 96)}")
    if detail >= 2:
        lines.append(f"        rule  : {_clip(worst.rule, 104)}")
        lines.append(f"        worst : {_clip(worst.subject, 60)}")
    return lines


def _flags_by_timepoint(book: FlagBook) -> dict[str, int]:
    """How many flags landed on each timepoint, across every tier."""
    counts: dict[str, int] = {}
    for f in book.flags:
        scope = _split_subject(f.subject)[1]
        if scope:
            counts[scope] = counts.get(scope, 0) + 1
    return counts


def _sample_axis_lines(report: dict[str, Any], book: FlagBook) -> list[str]:
    """The per-SAMPLE view of the run.

    The flag list is indexed by RULE, so a single bad timepoint shows up only as one entry
    inside a rule's tally — easy to skim past, and a real run did exactly that: it dropped the
    two timepoints that contradicted its conclusion without ever naming them. This section
    gives the same information on the axis a reader actually reasons about.
    """
    t0 = report.get("tier0_data_adequacy") or {}
    rows = t0.get("per_timepoint") or []
    disc = [d for d in (t0.get("discontinuity") or []) if d.get("discontinuous")]
    if not rows:
        return []

    per_tp_flags = _flags_by_timepoint(book)
    disc_by_tp: dict[str, list[dict[str, Any]]] = {}
    for d in disc:
        disc_by_tp.setdefault(str(d.get("timepoint")), []).append(d)

    L: list[str] = ["", "PER-TIMEPOINT (the sample axis — the flag list below is indexed by "
                        "RULE, so one bad sample hides inside a tally)"]
    L.append(_table(
        ["timepoint", "analyzed", "lymph", "purity%", "flags", "off-trend"],
        [[r.get("timepoint"), r.get("n_events_analyzed"), r.get("n_lymphocytes"),
          r.get("lineage_purity_pct"), per_tp_flags.get(str(r.get("timepoint")), 0),
          len(disc_by_tp.get(str(r.get("timepoint")), []))]
         for r in rows]))

    if disc:
        L.append(f"  OFF-TREND VALUES ({len(disc)}) — a value outside the envelope of BOTH its "
                 "temporal neighbours by more than their combined counting intervals. This is a "
                 "spike, not a step: a monotone trend cannot produce it, so sampling noise does "
                 "not explain the value and something about the sample differs from its "
                 "neighbours. Decide explicitly whether to include each one, and SAY SO.")
        for d in sorted(disc, key=lambda x: -(x.get("excursion_in_tolerances") or 0))[:10]:
            L.append(
                f"    {_fmt(d.get('value_pct'))}% {d.get('metric')} @ {d.get('timepoint')}"
                f"  ({_fmt(d.get('excursion_in_tolerances'))}x tolerance, "
                f"{d.get('direction')}: {d.get('prev_timepoint')}="
                f"{_fmt(d.get('prev_value_pct'))}%, {d.get('next_timepoint')}="
                f"{_fmt(d.get('next_value_pct'))}%)")
        if len(disc) > 10:
            L.append(f"    [+{len(disc) - 10} more — all of them in "
                     f"{DISCONTINUITY_CSV}]")
        L.append("    NOTE: one anomalous timepoint makes its otherwise sound neighbours look "
                 "off-trend too, so read a RUN of these together rather than one at a time.")
    else:
        n_assessable = sum(1 for d in (t0.get("discontinuity") or []) if d.get("assessable"))
        n_skipped = len(t0.get("discontinuity") or []) - n_assessable
        L.append(f"  OFF-TREND VALUES (0) — every value assessable on this axis "
                 f"({n_assessable} of {n_assessable + n_skipped} timepoint x metric pairs) sits "
                 "inside its neighbours' envelope. The rest are series endpoints or had no "
                 "measurable counting interval; they were not assessed, which is not the same "
                 f"as passing. See {DISCONTINUITY_CSV}.")
    return L


def render_digest(
    report: dict[str, Any],
    book: FlagBook,
    *,
    row_counts: Optional[dict[str, int]] = None,
    written: Optional[dict[str, str]] = None,
    obligations: Optional[dict[str, Any]] = None,
    max_chars: int = DIGEST_MAX_CHARS,
) -> str:
    """A bounded index of the whole report, sized to survive a truncated output window.

    The full summary runs to hundreds of kilobytes on a real study, and the agent's notebook
    keeps only the head and tail of a cell's output — so printing the summary deletes its
    middle, which is where the flags are. This renders the same information as an index:
    every flag group with its full count and worst case, plus where to read the rest.

    Bounded BY CONSTRUCTION, not by a trailing cut. If the fixed sections plus the flag
    groups would exceed ``max_chars``, group detail is degraded in stages (tally lines, then
    rule lines) and, only as a last resort, the tail of the group list is replaced by a line
    that states how many groups and flags it stands for. Counts always survive.
    """
    row_counts = row_counts or {}
    written = written or {}
    ordered = book.ordered()
    groups = _flag_groups(book)

    head: list[str] = [DIGEST_HEADER, ""]

    rep = report.get("reproduction") or {}
    if not rep.get("available"):
        head.append("REPRODUCTION: could not verify — no multilineage table to compare against. "
                    "Everything below may not describe the run that produced the tables.")
    elif rep.get("faithful"):
        head.append(f"REPRODUCTION: OK — {rep.get('checked')} reported values reproduced exactly "
                    f"from {rep.get('source')}.")
    else:
        head.append(f"REPRODUCTION: MISMATCH — {len(rep.get('mismatches') or [])} of "
                    f"{rep.get('checked')} values differ from {rep.get('source')}. Treat every "
                    "diagnostic below as describing a possibly different gating.")

    cfg = report.get("configuration") or {}
    tps = cfg.get("timepoints") or []
    head.append(
        f"RUN: reference={cfg.get('reference_timepoint')} | {len(tps)} timepoints "
        f"({_clip(', '.join(tps), 88)}) | controls="
        f"{', '.join(cfg.get('control_tubes') or []) or 'none'}"
    )
    head.append(
        f"     anchor={_clip(cfg.get('anchor_derivation') or '—', 60)} | "
        f"HLA={_fmt(cfg.get('hla_cutoff'))} (dim: {_fmt(cfg.get('hla_dim'))}) | "
        f"CAR={_fmt(cfg.get('car_cutoff'))} | subsample={cfg.get('subsample')}"
    )
    if cfg.get("threshold_overrides"):
        head.append(f"     threshold overrides in effect: {cfg['threshold_overrides']}")
    for n in (report.get("notes") or [])[:4]:
        head.append(f"     note: {_clip(n, 110)}")

    fails = report.get("tier_failures") or []
    if fails:
        head.append("")
        head.append(f"TIERS THAT FAILED TO COMPUTE ({len(fails)}) — not 'nothing found', "
                    "NOT PERFORMED:")
        for f in fails[:6]:
            head.append(f"  tier {f.get('tier')} {f.get('name')}: {_clip(f.get('error'), 90)}")

    head.extend(_sample_axis_lines(report, book))

    head.append("")
    if not ordered:
        head.append("FLAGGED ITEMS (0) — no rule in this set fired. That is not the same as "
                    "'the cutoffs are correct'.")
    else:
        head.append(f"FLAGGED ITEMS: {len(ordered)} flag(s) in {len(groups)} group(s), "
                    "ordered by tier then by how far the rule was exceeded. Ordering is a "
                    "presentation aid, NOT a ranking of importance.")

    # A counted group is not yet work. The worklist turns the tallies above into a finite
    # list, so it is announced here — next to the counts it exists to make actionable —
    # rather than buried in the file inventory at the bottom.
    if obligations is not None:
        n_items = len(obligations.get("items") or [])
        n_more = int(obligations.get("n_not_promoted") or 0)
        more = (f" A further {n_more} candidate(s) were NOT promoted and are therefore NOT "
                f"addressed by that list.") if n_more else ""
        head.append("")
        head.append(f"WORKLIST: {n_items} obligation(s) — the reading this run leaves you, in "
                    f"priority order, each naming the exact rows behind it. Full list: "
                    f"{obl.OBLIGATIONS_JSON}.{more}")

    tail: list[str] = []
    t3 = report.get("tier3_sensitivity") or {}
    env = t3.get("envelope") or []
    if env:
        dom = _top_counts([str(e.get("dominant_uncertainty") or "") for e in env], k=3)
        tail.append("")
        tail.append(f"HEADLINE NUMBERS: {len(env)} carry a two-source uncertainty; dominant "
                    f"source: {dom or '—'}. More events fixes counting; only a better cutoff "
                    "fixes the other. Per-number detail: diagnostics_uncertainty.csv.")
    cov = t3.get("coverage") or {}
    if cov.get("dropped"):
        # Names the MARKERS, not just the count: a bare count is what five trajectories
        # discharged without any of them learning the gap was entirely on the CAR cutoff.
        tail.append("COVERAGE GAP: " + _clip(obl.describe_coverage_gap(cov), 420)
                    + " The pairs are listed in cutoff_diagnostics.json.")

    lims = report.get("limitations") or []
    if lims:
        labels = [_clip(str(x).split(".")[0], 64) for x in lims]
        tail.append("")
        tail.append("KNOWN BLIND SPOTS of this numeric pass (the figure still wins here): "
                    + "; ".join(labels) + ". Full text: cutoff_diagnostics_summary.txt.")

    tail.append("")
    tail.append("FULL DETAIL — every .csv below is already a DataFrame in this notebook, keyed "
                "by its filename stem:")
    for label, path in written.items():
        name = Path(path).name
        n = row_counts.get(label)
        size = f"{n} rows" if n is not None else ""
        gloss = TABLE_PURPOSE.get(label, "")
        tail.append(f"  {name:<38} {size:<10} {gloss}")

    def assemble(detail: int, keep_groups: Optional[int]) -> str:
        body = []
        shown = groups if keep_groups is None else groups[:keep_groups]
        for tier, code, fs in shown:
            body.extend(_group_block(tier, code, fs, detail))
        hidden = groups[len(shown):]
        if hidden:
            n_flags = sum(len(fs) for _, _, fs in hidden)
            body.append(f"  [+{len(hidden)} more group(s) covering {n_flags} flag(s), not shown "
                        f"here to stay inside this window — all of them are in "
                        f"{Path(written.get('flags', 'diagnostics_flags.csv')).name}]")
        acct = (
            f"THIS DIGEST accounts for all {len(ordered)} flag(s) by group and shows "
            f"{len(shown)} worst-case example(s); the remaining "
            f"{max(0, len(ordered) - len(shown))} flag row(s) are in "
            f"{Path(written.get('flags', 'diagnostics_flags.csv')).name}. Nothing was dropped "
            "without being counted here."
        )
        return "\n".join(head + body + tail + ["", acct]) + "\n"

    # Degrade in stages; never drop a group's existence or its count before its detail.
    for detail in (2, 1, 0):
        text = assemble(detail, None)
        if len(text) <= max_chars:
            return text
    # Still over budget at the leanest detail: trim the tail of the group list, which the
    # replacement line accounts for by group and flag count. One group always survives.
    for keep in range(len(groups) - 1, 0, -1):
        if len(assemble(0, keep)) <= max_chars:
            return assemble(0, keep)
    return assemble(0, 1)
