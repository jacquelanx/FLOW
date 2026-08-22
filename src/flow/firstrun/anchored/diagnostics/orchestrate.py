"""Run every tier, in isolation, and write the artifacts.

Two structural choices matter here:

**Per-tier isolation.** Each tier is wrapped so that one failing computation cannot take down
the whole pass. A failed tier is recorded as a flag AND in ``tier_failures``, and the rest of
the report is still produced — a diagnostics pass that dies on a scatter-gate edge case would
otherwise silently remove the audit from a run that most needs it.

**Tier order is a dependency order, not a preference.** Tier 1 needs the reference structure
before tier 2 can compare against it; tier 3's envelope needs tier 0's counting intervals to
put the two uncertainty sources side by side. Tiers 0-2 also need the event frames, which are
released only once every tier has run.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from . import adequacy, concordance, controls, emit, foundation, sensitivity, transfer
from .context import build_context
from .flags import F_REPRODUCTION_MISMATCH, F_TIER_FAILED, FlagBook
from .record import MetricRecorder
from .thresholds import PROVENANCE_NOTES, Thresholds

ANCHOR_FILENAME = "operator_anchor.json"

# Stated plainly because a fixed metric list that pretends to completeness is worse than one
# that names its blind spots. These are the cases where looking at the figure still wins.
LIMITATIONS = (
    "Unanticipated population structure. A genuine third mode or an unexpected intermediate "
    "population is only partly covered: n_modes and the two unimodality tests will flag that "
    "the shape is not two clean peaks, but no fixed metric list recognizes what it was not "
    "designed to look for.",
    "Compensation and spillover artefacts. Diagonal smear in a 2-D plot is instantly obvious "
    "to a trained eye. Only partially covered here, via NK-cloud spread and boundary margins.",
    "Aesthetic wrongness. An experienced reader sometimes sees that a plot is wrong for "
    "reasons they would struggle to articulate, and is right. No metric captures that.",
    "Novel instrument failure modes. Anything outside the enumerated checks passes silently.",
    "Joint cutoff effects. The sensitivity sweep is one-at-a-time, so it does not capture the "
    "interaction of two cutoffs moving together.",
    "The overlay PNGs and the QC PDF still exist and are still worth reading. This pass makes "
    "the audit rigorous and auditable within a defined scope; it does not remove the human.",
)


def _resolve_patient_dir(data_dir: Path, out_dir: Path) -> Path:
    """Find the patient dir the pipeline actually consumed.

    ``anchored_nk_panel`` normalizes odd upload layouts into ``<out>/_dataset/fcs``; when the
    dataset was already ``<data>/fcs`` it is used directly. Checking the normalized view FIRST
    matters — auditing the raw upload would gate a different (or differently named) file set
    than the run being audited.
    """
    normalized = out_dir / "_dataset"
    if (normalized / "fcs").is_dir():
        return normalized
    if (data_dir / "fcs").is_dir():
        return data_dir
    raise FileNotFoundError(
        f"diagnostics: no fcs/ folder found under {normalized} or {data_dir}. The anchored "
        "first-run must have produced one of these before diagnostics can replay it."
    )


def _load_anchor(out_dir: Path, explicit: Optional[str | Path]) -> dict[str, Any]:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"diagnostics: anchor file not found: {p}")
        return json.loads(p.read_text())
    candidates = (
        out_dir / "outputs" / "provenance" / ANCHOR_FILENAME,
        out_dir / ANCHOR_FILENAME,
    )
    p = next((candidate for candidate in candidates if candidate.exists()), None)
    if p is None:
        hits = sorted(out_dir.glob("*operator_anchor*.json"))
        hits.extend(sorted((out_dir / "outputs" / "provenance").glob(
            "*operator_anchor*.json"
        )))
        if not hits:
            raise FileNotFoundError(
                f"diagnostics: no {ANCHOR_FILENAME} in {out_dir} or its governed "
                "outputs/provenance directory. The anchored first-run writes it; without "
                "the anchor there are no locked cutoffs to audit."
            )
        p = hits[0]
    return json.loads(p.read_text())


def _guard(name: str, tier: int, fn, book: FlagBook, failures: list[dict[str, Any]],
           verbose: bool = True) -> dict[str, Any]:
    """Run one tier, converting any exception into a flag plus a recorded failure."""
    t0 = time.time()
    try:
        res = fn()
        if verbose:
            print(f"  [diagnostics] tier {tier} ({name}) done in "
                  f"{time.time() - t0:.1f}s", flush=True)
        return res
    except Exception as e:  # a tier must never take down the pass
        tb = traceback.format_exc(limit=6)
        failures.append({"tier": tier, "name": name, "error": f"{type(e).__name__}: {e}",
                         "traceback": tb})
        book.add(
            code=F_TIER_FAILED, tier=tier, subject=f"tier {tier}: {name}",
            rule="the tier raised an exception and was skipped",
            measured={"error": f"{type(e).__name__}: {e}"},
            resolution_hint=(
                "Every other tier is unaffected — they are computed in isolation. Treat this "
                "tier's checks as NOT PERFORMED rather than as passed."
            ),
        )
        if verbose:
            print(f"  [diagnostics] tier {tier} ({name}) FAILED: {type(e).__name__}: {e}",
                  flush=True)
        return {"available": False, "reason": f"tier failed: {type(e).__name__}: {e}"}


def run_diagnostics(
    data_dir: str | Path,
    out_dir: str | Path,
    plots_dir: str | Path | None = None,
    anchor_path: str | Path | None = None,
    patient_dir: str | Path | None = None,
    subsample: Optional[int] = None,
    sensitivity_events: Optional[int] = None,
    sensitivity_seconds: Optional[float] = None,
    run_sensitivity: bool = True,
    verbose: bool = True,
    full_summary: bool = False,
) -> dict[str, Any]:
    """Replay the anchored run, compute all six tiers, write the artifacts.

    Returns the full report dict (also written as ``cutoff_diagnostics.json``).

    ``verbose`` prints the bounded DIGEST to stdout; ``full_summary=True`` prints the whole
    prose report instead, which is for a human at a terminal — never for the FLOW harness,
    whose observation window would keep only the head and tail of it.
    """
    started = time.time()
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    pdir = Path(patient_dir).resolve() if patient_dir else _resolve_patient_dir(data_dir, out_dir)
    anchor = _load_anchor(out_dir, anchor_path)

    meta_path = pdir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    th = Thresholds.from_metadata(meta)

    if verbose:
        print(f"[diagnostics] auditing {pdir} against {ANCHOR_FILENAME}", flush=True)

    ctx = build_context(
        patient_dir=pdir, out_dir=out_dir, anchor=anchor, plots_dir=plots_dir,
        thresholds=th, subsample=subsample, verbose=verbose,
    )

    rec = MetricRecorder()
    book = FlagBook()
    failures: list[dict[str, Any]] = []

    if ctx.reproduction.available and not ctx.reproduction.faithful:
        book.add(
            code=F_REPRODUCTION_MISMATCH, tier=0, subject="diagnostics replay fidelity",
            rule=("every reported percentage must reproduce exactly from the pipeline's own "
                  "multilineage table (the pipeline is deterministic, so it should)"),
            measured={
                "checked": ctx.reproduction.checked,
                "n_mismatches": len(ctx.reproduction.mismatches),
                "max_abs_delta_pp": ctx.reproduction.max_abs_delta_pp,
                "source": ctx.reproduction.source,
                "examples": ctx.reproduction.mismatches[:5],
            },
            resolution_hint=(
                "This replay is not auditing the run that produced the headline tables, so "
                "every diagnostic below is suspect. Check that the diagnostics subsample "
                "matches the pipeline's and that the same anchor file was used."
            ),
        )

    # Dependency order: 1 feeds 2 and 3; 0 feeds 3's envelope.
    t0 = _guard("data adequacy", 0, lambda: adequacy.run(ctx, rec, book), book, failures, verbose)
    t1 = _guard("cutoff foundation", 1, lambda: foundation.run(ctx, rec, book),
                book, failures, verbose)
    t2 = _guard("transfer validity", 2, lambda: transfer.run(ctx, rec, book, t1),
                book, failures, verbose)
    if run_sensitivity:
        t3 = _guard(
            "sensitivity", 3,
            lambda: sensitivity.run(ctx, rec, book, t1, t0,
                                    max_events=sensitivity_events,
                                    time_budget_s=sensitivity_seconds),
            book, failures, verbose,
        )
    else:
        t3 = {"available": False,
              "reason": "sensitivity sweep disabled by request (--no-sensitivity)"}
    t4 = _guard("internal controls", 4, lambda: controls.run(ctx, rec, book),
                book, failures, verbose)
    t5 = _guard("operator concordance", 5, lambda: concordance.run(ctx, rec, book),
                book, failures, verbose)

    report: dict[str, Any] = {
        "schema": "flow_cutoff_diagnostics/v1",
        "patient_id": ctx.patient_id,
        "generated_by": "anchored.diagnostics",
        "configuration": {
            "patient_dir": str(pdir),
            "out_dir": str(out_dir),
            "reference_timepoint": ctx.reference_timepoint,
            "anchor_derivation": (t1 or {}).get("anchor_derivation"),
            "anchor_schema": anchor.get("schema"),
            "anchor_method": anchor.get("method"),
            "timepoints": ctx.timepoint_labels(),
            "control_tubes": sorted(ctx.controls.keys()),
            "channels": ctx.channels,
            "hla_cutoff": ctx.hla_cut,
            "car_cutoff": ctx.car_cut,
            "hla_dim": ctx.hla_dim,
            "subsample": ctx.subsample,
            "threshold_overrides": th.overrides_applied,
        },
        "reproduction": {
            "available": ctx.reproduction.available,
            "faithful": ctx.reproduction.faithful,
            "checked": ctx.reproduction.checked,
            "max_abs_delta_pp": ctx.reproduction.max_abs_delta_pp,
            "source": ctx.reproduction.source,
            "mismatches": ctx.reproduction.mismatches,
        },
        "notes": ctx.notes,
        "skipped": ctx.skipped,
        "thresholds": th.describe(),
        "provenance_notes": PROVENANCE_NOTES,
        "tier0_data_adequacy": t0,
        "tier1_cutoff_foundation": t1,
        "tier2_transfer_validity": t2,
        "tier3_sensitivity": t3,
        "tier4_internal_controls": t4,
        "tier5_operator_concordance": t5,
        "tier_failures": failures,
        "flags": [f.to_dict() for f in book.ordered()],
        "flag_counts": book.by_code(),
        "limitations": list(LIMITATIONS),
        "discipline": (
            "Every entry is a measurement. A flag is a measurement that crossed a stated rule, "
            "and the rule travels with it. This module does not conclude whether a cutoff is "
            "correct — that judgement belongs to the reader."
        ),
    }

    ctx.release_events()
    files = emit.write_all(out_dir, report, rec, book)
    report["files"] = files
    # Re-render the summary now that the file list is known, so the text names its own outputs.
    (out_dir / emit.SUMMARY_TXT).write_text(emit.render_summary(report, book))
    (out_dir / emit.JSON_FILE).write_text(json.dumps(report, indent=2, default=str))
    report["elapsed_s"] = round(time.time() - started, 2)

    if verbose:
        print("", flush=True)
        # The DIGEST, not the summary. This stdout is captured by the FLOW first-run harness
        # and lands in a truncated observation window, so printing the full summary (hundreds
        # of KB on a real study) would delete its middle — where the flags are. The digest is
        # bounded by construction and names the files holding the rest.
        print((out_dir / (emit.SUMMARY_TXT if full_summary else emit.DIGEST_TXT)).read_text(),
              flush=True)
        print(f"[diagnostics] {len(rec)} measurements, {len(book)} flag(s), "
              f"{len(failures)} tier failure(s) in {report['elapsed_s']}s", flush=True)
        print(f"[diagnostics] wrote: {', '.join(sorted(Path(p).name for p in files.values()))}",
              flush=True)
        if not full_summary:
            print(f"[diagnostics] full prose report: {out_dir / emit.SUMMARY_TXT}", flush=True)
    return report
