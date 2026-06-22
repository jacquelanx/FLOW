#!/usr/bin/env python3
"""Generate a small SYNTHETIC demo project for FLOW (no real/PHI data).

Creates ``examples/demo/`` with the exact file shapes lab members upload:
  metadata.json, flow.csv, alc.csv, events.csv (combined per-event table),
  config.yaml, true_lab_results.csv.

The data is random noise with generic detector names. We deliberately encode NO analysis
logic and NO specific real panel — metadata.json states only neutral FACTS (which detector
carries which synthetic stain, a couple of dates) so the agent has something to interpret.
The agent must still derive the entire analysis itself.

Two trends are intentionally encoded so we can test how well the agent detects trends:

  * OBVIOUS trend — a CAR+ subpopulation (high DET-E, co-elevated DET-D) EXPANDS over time.
    Its fraction rises ~5% -> ~15% -> ~35% -> ~55% across the four timepoints, so the mean
    DET-E (and DET-D) and the % positive climb dramatically. Any reasonable analysis of the
    CAR detector should catch this.

  * SUBTLE trend — a lineage marker (DET-B) gradually DECLINES (~25%) across the same
    timepoints. It is unrelated to the CAR population and lives in a channel the question
    does not highlight, so only an agent that profiles ALL channels over time will find it.

There are 3 replicate samples per timepoint (with small per-sample jitter), so the trends
have genuine replication and significance can be assessed. ALC is left as noise (no trend).

Usage:
    python scripts/generate_synthetic_demo.py [--out examples/demo] [--events 6000] [--seed 7]
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
SAMPLES_PER_TP = 3  # replicates per timepoint

# --- Encoded ground truth (the agent never sees these) -----------------------------------
# OBVIOUS: the CAR+ subpopulation fraction expands across timepoints.
CAR_FRAC = [0.05, 0.15, 0.35, 0.55]
# SUBTLE: the DET-B lineage marker's log-mean drifts down across timepoints.
DETB_LOGBASE = [3.40, 3.31, 3.22, 3.13]
# Per-detector baseline log-means for the "negative"/background distribution.
BG_LOGBASE = {"DET-A": 3.0, "DET-B": 3.40, "DET-C": 3.8, "DET-D": 4.0, "DET-E": 4.4}
# CAR+ cells are brighter on the HLA (DET-D) and CAR (DET-E) detectors.
CAR_BOOST = {"DET-D": 1.4, "DET-E": 1.8}


def _gauss_positive(rng: random.Random, mu: float, sigma: float) -> float:
    return max(0.0, rng.gauss(mu, sigma))


def _intensity(rng: random.Random, log_mu: float, sigma: float = 0.5) -> float:
    # Heavy-tailed (log-normal-like) intensity, like real cytometry channels.
    return round(math.expm1(_gauss_positive(rng, log_mu, sigma)), 2)


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
            "Synthetic demo. Detectors DET-A, DET-B, DET-C carry generic lineage stains; "
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

    # --- alc.csv (date,alc) absolute lymphocyte counts — left as noise (no encoded trend) ---
    with (out / "alc.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "alc"])
        for _, date in TIMEPOINTS:
            w.writerow([date, round(_gauss_positive(rng, 1.2, 0.35), 2)])

    # --- events.csv: one row per cell; sample_id + label + one column per detector ---
    per_sample = max(1, n_events // (len(TIMEPOINTS) * SAMPLES_PER_TP))
    sample_counter = 0
    with (out / "events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label"] + DETECTORS)
        for ti, (label, _date) in enumerate(TIMEPOINTS):
            for _rep in range(SAMPLES_PER_TP):
                sample_counter += 1
                sample_id = f"S{sample_counter:02d}"
                # Small per-sample (replicate) jitter so the trends have real variation.
                car_frac = min(0.95, max(0.0, CAR_FRAC[ti] + rng.gauss(0.0, 0.03)))
                detb_base = DETB_LOGBASE[ti] + rng.gauss(0.0, 0.02)
                for _ in range(per_sample):
                    car_pos = rng.random() < car_frac
                    row = [sample_id, label]
                    for det in DETECTORS:
                        if det == "DET-B":
                            log_mu = detb_base  # subtle lineage decline over time
                        else:
                            log_mu = BG_LOGBASE[det]
                            if car_pos and det in CAR_BOOST:
                                log_mu += CAR_BOOST[det]  # CAR+ cells brighter on DET-D/DET-E
                        row.append(_intensity(rng, log_mu))
                    w.writerow(row)

    # --- config.yaml (analysis-free) ---
    from flow.config import default_config_yaml  # local import; script is run from repo

    question = (
        "Using the synthetic cytometry data and the experiment metadata, characterize how the "
        "detector signals and cell populations change across the timepoints following infusion. "
        "Report every trend you find — both the most prominent change and any smaller, secondary "
        "trends — with quantitative supporting evidence."
    )
    cfg = default_config_yaml(question)
    cfg = cfg.replace('description: ""', 'description: "Synthetic demo project (two encoded trends)."')
    (out / "config.yaml").write_text(cfg)

    # --- true_lab_results.csv (EVALUATION ONLY; never read during analysis) ---
    # The designed ground truth, for grading the agent after the fact.
    with (out / "true_lab_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "metric", "value"])
        for ti, (label, _date) in enumerate(TIMEPOINTS):
            w.writerow([label, "true_car_pos_fraction", round(CAR_FRAC[ti], 3)])
            w.writerow([label, "detB_lineage_logmean", round(DETB_LOGBASE[ti], 3)])

    print(f"Synthetic demo project written to: {out}")
    for p in sorted(out.iterdir()):
        print(f"  - {p.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic FLOW demo project.")
    ap.add_argument("--out", default="examples/demo")
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    generate(Path(args.out), args.events, args.seed)


if __name__ == "__main__":
    main()
