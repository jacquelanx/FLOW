#!/usr/bin/env python3
"""Build a Nature-methods-style operator anchor from Baseline + manual %.

Usage
-----
  python -m pipeline.build_reference \\
      --patient-dir patients/UPN27 \\
      --manual-ref reference/UPN27_manual_gating.csv
"""
from __future__ import annotations

import argparse
from .reference_anchor import build_operator_anchor


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Calibrate Baseline FSA×SSA to manual reference (operator anchor)")
    ap.add_argument("--patient-dir", required=True)
    ap.add_argument("--manual-ref", required=True,
                    help="CSV with Baseline row (b_p,t_p,nk_p,cd4_p,cd8_p)")
    ap.add_argument("--reference-tp", default="Baseline")
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument("--subsample", type=int, default=200_000)
    a = ap.parse_args(argv)
    build_operator_anchor(
        a.patient_dir, a.manual_ref,
        reference_tp=a.reference_tp,
        out_path=a.out,
        subsample=a.subsample,
    )


if __name__ == "__main__":
    main()
