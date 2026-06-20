#!/usr/bin/env python3
"""Generate a small SYNTHETIC demo project for FLOW (no real/PHI data).

Creates ``examples/demo/`` with the exact file shapes lab members upload:
  metadata.json, flow.csv, alc.csv, events.csv (combined per-event table),
  config.yaml, true_lab_results.csv.

The data is random noise with generic detector names. We deliberately encode NO analysis
logic and NO specific real panel — metadata.json states only neutral FACTS (which detector
carries which synthetic stain, a couple of dates) so the agent has something to interpret.
The agent must still derive the entire analysis itself.

Usage:
    python scripts/generate_synthetic_demo.py [--out examples/demo] [--events 4000] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

# Generic, made-up detector + label names — NOT a real assay panel.
DETECTORS = ["DET-A", "DET-B", "DET-C", "DET-D", "DET-E"]
TIMEPOINTS = [
    ("Baseline", "6/25/2025"),
    ("Day14", "7/9/2025"),
    ("Day28", "7/23/2025"),
    ("Day90", "9/23/2025"),
]


def _gauss_positive(rng: random.Random, mu: float, sigma: float) -> float:
    return max(0.0, rng.gauss(mu, sigma))


def generate(out: Path, n_events: int, seed: int) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)

    # --- metadata.json (FACTS only; generic synthetic stains, no gating instructions) ---
    metadata = {
        "patient_study": "SYNTHETIC-DEMO (no real or PHI data)",
        "infusion_date": "2025-06-18",
        "hla_specificity": "SYN-HLA-DEMO",
        "hla_channel": "DET-D",
        "hla_polarity": "donor",
        "car_channel": "DET-E",
        "panel_notes": (
            "Synthetic demo. Detectors DET-A..DET-C carry generic lineage stains; "
            "DET-D carries the synthetic HLA reagent; DET-E carries the synthetic CAR "
            "reagent. These are FACTS about the data, not analysis instructions."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- flow.csv (label,date) ---
    with (out / "flow.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "date"])
        for label, date in TIMEPOINTS:
            w.writerow([label, date])

    # --- alc.csv (date,alc) absolute lymphocyte counts over time ---
    with (out / "alc.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "alc"])
        for _, date in TIMEPOINTS:
            w.writerow([date, round(_gauss_positive(rng, 1.2, 0.4), 2)])

    # --- events.csv: one row per cell, sample_id + label + one column per detector ---
    per_tp = max(1, n_events // len(TIMEPOINTS))
    with (out / "events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label"] + DETECTORS)
        for ti, (label, _date) in enumerate(TIMEPOINTS):
            sample_id = f"S{ti + 1:02d}"
            for _ in range(per_tp):
                # Two latent blobs so there IS structure to discover — but we never tell
                # the agent where it is or how to find it.
                comp = rng.random() < 0.3
                row = [sample_id, label]
                for di, det in enumerate(DETECTORS):
                    base = 3.0 + 0.4 * di
                    if comp and di >= 3:
                        val = _gauss_positive(rng, base + 1.5, 0.6)
                    else:
                        val = _gauss_positive(rng, base, 0.5)
                    row.append(round(math.expm1(val), 2))  # heavy-tailed, like real intensities
                w.writerow(row)

    # --- config.yaml (analysis-free) ---
    from flow.config import default_config_yaml  # local import; script is run from repo

    question = (
        "Using the synthetic cytometry data and the experiment metadata, characterize how "
        "the labeled cell signal on the CAR detector changes across the timepoints, and "
        "report the trend with supporting evidence."
    )
    cfg = default_config_yaml(question)
    cfg = cfg.replace("description: \"\"", 'description: "Synthetic demo project."')
    (out / "config.yaml").write_text(cfg)

    # --- true_lab_results.csv (EVALUATION ONLY; never read during analysis) ---
    with (out / "true_lab_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "metric", "value"])
        for label, _ in TIMEPOINTS:
            w.writerow([label, "synthetic_car_pct", round(rng.uniform(0.5, 12.0), 2)])

    print(f"Synthetic demo project written to: {out}")
    for p in sorted(out.iterdir()):
        print(f"  - {p.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic FLOW demo project.")
    ap.add_argument("--out", default="examples/demo")
    ap.add_argument("--events", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    generate(Path(args.out), args.events, args.seed)


if __name__ == "__main__":
    main()
