"""CLI: ``python -m anchored.diagnostics --data <DATA> --out <OUT> [--plots <PLOTS>]``.

Deliberately mirrors the FLOW first-run contract (``--data`` / ``--out`` / ``--plots``) so the
harness can invoke it exactly like a first-run script, with no special-casing.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="anchored.diagnostics",
        description=(
            "Audit the anchored pipeline's locked cutoffs numerically: are they well founded "
            "at the reference, do they still hold at every timepoint, and does any of it "
            "change the reported numbers."
        ),
    )
    ap.add_argument("--data", required=True, help="Dataset dir (as given to the first-run).")
    ap.add_argument("--out", required=True,
                    help="The first-run's output dir (holds operator_anchor.json).")
    ap.add_argument("--plots", default="", help="Plots dir, for locating the overlay figures.")
    ap.add_argument("--patient-dir", default=None,
                    help="Override the replayed patient dir (default: <out>/_dataset, else <data>).")
    ap.add_argument("--anchor", default=None, help="Override the anchor JSON path.")
    ap.add_argument("--subsample", type=int, default=None,
                    help="Events per FCS. MUST match the pipeline's or the reproduction "
                         "check will fail by construction (default: metadata.json's value).")
    ap.add_argument("--sensitivity-events", type=int, default=None,
                    help="Event cap for the tier-3 perturbation sweep.")
    ap.add_argument("--sensitivity-seconds", type=float, default=None,
                    help="Wall-clock budget for the tier-3 sweep; dropped work is named in "
                         "the output rather than silently omitted.")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="Skip tier 3 (the slowest tier; it re-gates per perturbation).")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--full-summary", action="store_true",
                    help="Print the whole prose report instead of the bounded digest. For a "
                         "human at a terminal; the FLOW harness always wants the digest.")
    args = ap.parse_args(argv)

    from .orchestrate import run_diagnostics

    try:
        run_diagnostics(
            data_dir=args.data,
            out_dir=args.out,
            plots_dir=args.plots or None,
            anchor_path=args.anchor,
            patient_dir=args.patient_dir,
            subsample=args.subsample,
            sensitivity_events=args.sensitivity_events,
            sensitivity_seconds=args.sensitivity_seconds,
            run_sensitivity=not args.no_sensitivity,
            verbose=not args.quiet,
            full_summary=args.full_summary,
        )
    except Exception as e:
        # A diagnostics failure must never look like an analysis failure.
        print(f"\nDIAGNOSTICS_FAILED {type(e).__name__}: {e}", flush=True)
        return 1
    print("\nDIAGNOSTICS_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
