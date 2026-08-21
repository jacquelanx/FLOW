"""ANCHORED first-run DIAGNOSTICS — numeric audit of the locked cutoffs.

Runs AFTER ``first_run.py`` and reads only what that run already wrote plus the same event
data. It does not modify, re-run, or override any part of the first-run: the composition
numbers in ``multilineage.csv`` remain the authoritative headline.

Why this exists. The analysis specification asks the agent to judge whether each unified
cutoff "is derived reasonably and holds across timepoints" by opening the ``overlay_*.png``
figures. A text-only model cannot see a PNG — it receives a filename — so that half of the
instruction can only be answered by fabrication. This pass computes, from the event arrays
that generated those figures, the quantities a cytometrist reads off them.

FLOW first-run contract (same as the first-run script, so the harness needs no special case):
    python first_run_diagnostics.py --data <DATA_DIR> --out <OUT_DIR> --plots <PLOTS_DIR>

Outputs written to --out (every CSV is auto-loaded into the agent's notebook):
  * ``cutoff_diagnostics_digest.txt``   — the bounded INDEX printed to stdout: every flag group
                                          with its full count and worst case, plus where the
                                          per-row detail lives. Sized to survive the agent's
                                          truncated observation window; the summary is not.
  * ``diagnostics_worklist.txt``        — the WORKLIST: the short, capped list of items this
                                          run leaves the reader to open, each naming the exact
                                          rows behind it. Printed last by the FLOW harness.
                                          Counting 375 findings honestly is not the same as
                                          leaving a finishable amount of work; see
                                          ``anchored/diagnostics/obligations.py``.
  * ``diagnostics_obligations.json``    — the same worklist as a manifest: ids, statements and
                                          the conditions that discharge each one.
  * ``cutoff_diagnostics_summary.txt``  — the readable report: flagged items + headline tables.
  * ``diagnostics_flags.csv``           — measurements that crossed a stated rule, with the rule.
  * ``diagnostics_cutoff_audit.csv``    — per marker at the reference: is the ruler well made?
  * ``diagnostics_transfer.csv``        — per marker x timepoint: does the locked cutoff hold?
  * ``diagnostics_uncertainty.csv``     — every headline % with cutoff-attributable AND counting
                                          uncertainty, separately labelled. Quote
                                          ``headline_value_pct``: the ``*_at_subsample`` columns
                                          are tier-3 re-gates on a capped event subsample and
                                          differ slightly from ``multilineage.csv`` by design.
  * ``diagnostics_counterfactual.csv``  — locked vs per-sample cutoffs: what anchoring is doing.
  * ``diagnostics_controls.csv``        — checks against populations with a KNOWN answer.
  * ``diagnostics_concordance.csv``     — per-timepoint / per-metric manual agreement.
  * ``diagnostics_gate_geometry.csv``   — 2-D NK quadrant margins, locked-scatter containment.
  * ``diagnostics_metrics.csv``         — every measurement, tidy long, with units.
  * ``diagnostics_thresholds.csv``      — every rule, its value, and where the value came from.
  * ``cutoff_diagnostics.json``         — the whole report, nested.

Configuration (all from the dataset's ``metadata.json`` — no FLOW code change needed):
  * ``diagnostics_thresholds``           — override any flagging rule, e.g.
                                           {"drift_gap_fraction": 0.10}
  * ``diagnostics_sensitivity_events``   — event cap for the tier-3 sweep (default 40000)
  * ``diagnostics_sensitivity_seconds``  — wall-clock budget for tier 3 (default 420)

Exits 0 even when the audit cannot run (e.g. no anchor present), because a missing audit must
never be reported as an analysis failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the vendored ``anchored`` package importable whether this runs from the FLOW source
# tree or as a standalone script placed beside the ``anchored/`` folder in a project dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from anchored.diagnostics import run_diagnostics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset dir (as given to the first-run).")
    ap.add_argument("--out", required=True, help="The first-run's output dir.")
    ap.add_argument("--plots", default="", help="Where the first-run wrote its plots.")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="Skip the tier-3 perturbation sweep (the slowest tier).")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    if not out_dir.is_dir():
        print(f"[diagnostics] skipped: output dir {out_dir} does not exist — the first-run "
              "must produce its tables before they can be audited.", flush=True)
        return 0

    try:
        report = run_diagnostics(
            data_dir=args.data,
            out_dir=out_dir,
            plots_dir=args.plots or None,
            run_sensitivity=not args.no_sensitivity,
            verbose=True,
        )
    except FileNotFoundError as e:
        # No anchor / no fcs: this project did not run the anchored first-run. Not an error.
        print(f"[diagnostics] skipped: {e}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 - never let the audit break the run
        print(f"[diagnostics] FAILED (the first-run tables are unaffected): "
              f"{type(e).__name__}: {e}", flush=True)
        return 0

    n_flags = len(report.get("flags") or [])
    rep = report.get("reproduction") or {}
    print(f"\nDIAGNOSTICS_OK {n_flags} flag(s); replay reproduced the pipeline's numbers: "
          f"{rep.get('faithful')}", flush=True)
    print("The DIGEST above is a bounded index: it names every flag group with its full count "
          "and worst case. Read the per-row detail from diagnostics_flags.csv (already a "
          "DataFrame in the notebook), and cutoff_diagnostics_summary.txt for the prose "
          "report. Flags are MEASUREMENTS, not conclusions.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
