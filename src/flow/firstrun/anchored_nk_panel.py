"""ANCHORED first-run template — NK cell-therapy panel with reference anchoring.

What it does:
  1. Pick a REFERENCE timepoint (pre-infusion / Baseline).
  2. Derive a reference anchor and explicit transfer policy, calibrating to the operator's
     manual gating when an eligible manual CSV is present.
  3. Apply the declared reference, soft-lock, and per-file refinement rules once. The exact
     applied threshold and provenance are emitted for every file and gate; the agent
     interprets the deterministic run and does not recompute it.

FLOW first-run contract:
    python first_run.py --data <DATA_DIR> --out <OUT_DIR> --plots <PLOTS_DIR>

Outputs written below ``--out/outputs``:
  * ``tables/multilineage.csv`` — composition % per timepoint on the unified cutoff.
  * ``unified_cutoffs.csv``    — compatibility summary of reference values/transfer policy.
  * ``gate_parameters.csv``    — exact applied cutoff, operator, channel and provenance per file.
  * ``composition_shift.csv``  — how each population moves across timepoints.
  * ``temporal_cell_type_summary.csv`` — percentage and gated-event-count changes with
                                         explicit denominators and abundance limitations.
  * ``compare_manual.csv``     — auto vs. manual Δ per metric (only if a manual CSV exists).
  * ``reports/QC_Report_<study>.pdf`` — full gate sequence; bivariate panels use deterministic
                                  heat-scatter fields with 20/40/65% density contours;
                                  paired scatter panels distinguish the all-event viable gate
                                  from Live-CD45 lymph-density geometry.
  * ``reports/Interactive_Longitudinal_QC_<study>.html`` — interactive percentage dashboard;
                                      denominators are explicit and it is not an HTML rendering
                                      of the composite QC PDF.
  * ``tables/lymph_density_geometry.csv`` — fitted Live-CD45 FSC/SSC mode centers, low-SSC mode,
                                      and applied-gate retention (diagnostic only).
  * ``reports/report_page_index.csv`` — composite page number, timepoint, and named companion PDF.
  * ``reports/pages/pdf/*.pdf`` — every composite page as a separately named PDF.
  * ``reports/Longitudinal_Cell_Composition_<study>.pdf`` — percentages above and exact-date
                                                     ALC-calibrated K/uL estimates below;
                                                     gated counts are the fallback.
  * ``reports/Acquisition_Cleaning_Sensitivity_<study>.pdf`` — canonical versus deterministic
                                                     Time-cleaned sensitivity for flagged files.
  * ``multilineage_time_cleaned_sensitivity.csv`` and
    ``acquisition_cleaning_population_comparison.csv`` — before/after population results using
                                                     the same gates without refitting.
  * ``membership/time_cleaning_excluded_events.csv.gz`` — exact candidate event IDs removed
                                                     only in the sensitivity analysis.
Plus, under ``outputs/plots``: ``overlay_<marker>.png`` (all timepoints, shared axis, unified cut).
Governance: ``first_run_bundle.json`` content-addresses inputs, source, configuration,
and generated artifacts.

Configuration (read from the dataset's ``metadata.json`` — changeable without touching FLOW):
  * ``reference_timepoint``  (default "Baseline")            — D2
  * ``anchor_mode``          ("auto" | "manual" | "negative", default "auto") — D1
  * ``nk_dominant_frac``     (default 0.92)                  — D3
  * ``subsample``            (default 200000 events per FCS).
The dataset dir must contain ``fcs/`` (the FCS files) and ``metadata.json``; an optional
``manual_gating.csv`` (or ``reference/manual_gating.csv``) enables manual calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the vendored ``anchored`` package importable whether this script runs from the FLOW
# source tree or as a standalone first_run.py placed beside the ``anchored/`` folder.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd  # noqa: E402

from anchored.compare import compare, load_manual, summarize  # noqa: E402
from anchored.calibrate import classify_file  # noqa: E402
from anchored.manual_check import (  # noqa: E402
    ManualGatingError, check_manual_gating, coverage_note,
)
from anchored.fcs_io import inspect_fcs_metadata, load_xform, resolve_channels  # noqa: E402
from anchored.control_compensation import (  # noqa: E402
    build_control_repository, candidate_for_acquisition,
    infer_control_fluorescence_channels,
)
from anchored.evidence import build_first_run_bundle  # noqa: E402
from anchored.flow_outputs import (  # noqa: E402
    build_negative_anchor, histogram_overlays, write_composition_shift, write_unified_cutoffs,
)
from anchored.output_layout import FirstRunOutputLayout  # noqa: E402
from anchored.reference_anchor import build_operator_anchor, load_anchor  # noqa: E402
from anchored.run import process_patient  # noqa: E402


DATA_DICTIONARY = """\
FLOW anchored first-run outputs (deterministic; the agent INTERPRETS these — it must NOT
recompute the composition numbers; it AUDITS the cutoff, deterministic diagnostic
measurements, policy summary, exact gate-parameter table, and plots. Any refinement is a
separate governed shadow rather than a mutation of the canonical run).

outputs/tables/multilineage.csv
                      — one row per timepoint: population %s on the UNIFIED cutoff.
                       Key columns: '%B (of lymph)', '%T (of lymph)', '%NK (of lymph)',
                       '%CD4 (of lymph)', '%CD8 (of lymph)', '%Donor NK (of lymph)',
                       '%CAR+ (of Donor NK)', plus event counts and QC (donor_reliable, ...).
unified_cutoffs.csv  — reference values and transfer-policy summary (legacy filename).
gate_parameters.csv  — exact file-level applied values, comparison operators, semantic roles,
                       and provenance. This is the authoritative threshold table.
composition_shift.csv— per population: value at each timepoint + Δ from the reference timepoint
                       and Δ from the previous timepoint (the cellular composition SHIFT).
temporal_cell_type_summary.csv
                      — one row per configured region and timepoint: percentage, explicit
                        denominator, gated-event count, changes from reference/previous,
                        and exact-date ALC-calibrated K/uL where available.
compare_manual.csv   — (if a manual CSV was provided) auto vs. manual %, delta, abs_delta.
outputs/plots/overlay_*.png
                      — per marker, all timepoints on a shared axis with the unified cut drawn.
                       These plots are for a HUMAN reader. The quantities a reader would take
                       from them are measured in
                       diagnostics_cutoff_audit.csv and diagnostics_transfer.csv; do not
                       claim visual observations unless image pixels were supplied.
outputs/reports/report_page_index.csv and outputs/reports/pages/pdf/*.pdf
                      — named page-level companions for the composite QC report.
lymph_density_geometry.csv
                      — deterministic Live-CD45 density-mode diagnostics plotted in panel 2;
                        never used to change gate membership or reported percentages.
multilineage_time_cleaned_sensitivity.csv
                      — separately labeled Time-cleaned sensitivity using the canonical gate
                        parameters without refitting; never replaces multilineage.csv.
acquisition_cleaning_population_comparison.csv
                      — exact canonical, cleaned, and percentage-point values per metric.
"""


def _find_manual(data_dir: Path) -> Path | None:
    for cand in (data_dir / "manual_gating.csv",
                 data_dir / "reference" / "manual_gating.csv"):
        if cand.exists():
            return cand
    return None


def _load_meta(data_dir: Path) -> dict:
    mp = data_dir / "metadata.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            return {}
    return {}


def _collect_fcs(data_dir: Path) -> list[Path]:
    """Every ``.fcs`` under the dataset, regardless of how it was uploaded.

    FLOW lays FCS out differently per upload path: uploaded one-by-one they land flat in the
    project root; uploaded as a ``.zip`` they go into a subfolder named after the zip; a
    zip literally named ``fcs.zip`` yields the canonical ``fcs/``. A mixed upload splits them
    across both. Collecting recursively covers all of these.
    """
    return sorted(data_dir.rglob("*.fcs"))


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _unique_name(f: Path, data_dir: Path, seen: dict[str, Path]) -> str | None:
    """Flat filename for ``f`` that collides with nothing already in ``seen``.

    A folder-per-timepoint upload repeats basenames (``Specimen_001_Car.fcs`` in each
    acquisition folder). Keeping only the first would silently drop most of the dataset, so
    the source folders are folded into the name as a ``<folder>__`` prefix — the same
    convention the app's zip extractor uses, and the one the classifier looks past. Returns
    None when the exact same file is reached twice (mixed zip + loose upload).
    """
    if f.name not in seen:
        return f.name
    if seen[f.name].resolve() == f.resolve():
        return None
    try:
        folders = list(f.relative_to(data_dir).parent.parts)
    except ValueError:
        folders = [f.parent.name]
    name = f"{'-'.join(folders)}__{f.name}" if folders else f.name
    if name in seen:
        if seen[name].resolve() == f.resolve():
            return None
        stem, suffix = name[: -len(f.suffix)], f.suffix
        n = 2
        while f"{stem}-{n}{suffix}" in seen:
            n += 1
        name = f"{stem}-{n}{suffix}"
    return name


def _normalize_dataset(data_dir: Path, work_root: Path) -> Path:
    """Return a patient dir the vendored pipeline can consume: ``<dir>/fcs/`` + sidecars.

    The pipeline (``process_patient`` / ``build_operator_anchor`` /
    ``build_negative_anchor``) hardcodes ``patient_dir/'fcs'``. If the dataset is already in
    that exact layout we use it directly; otherwise we build a normalized view under the
    writable work area — an ``fcs/`` folder of symlinks (copies as a fallback) gathering
    EVERY ``.fcs`` found anywhere in the project, plus the sidecar CSVs and ``metadata.json``.
    ``/data`` is read-only, so nothing is written back into the dataset.
    """
    all_fcs = _collect_fcs(data_dir)
    if not all_fcs:
        raise SystemExit(f"anchored_nk_panel: no .fcs files found under {data_dir}")

    # Fast path: already exactly data_dir/fcs/*.fcs and nowhere else.
    fcs_sub = data_dir / "fcs"
    if all(p.parent == fcs_sub for p in all_fcs):
        return data_dir

    norm = work_root / "_dataset"
    (norm / "fcs").mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    for f in all_fcs:
        name = _unique_name(f, data_dir, seen)
        if name is None:  # same file reached by two upload paths (zip + loose)
            print(f"  [normalize] duplicate FCS ignored: {f}", flush=True)
            continue
        seen[name] = f
        _link_or_copy(f, norm / "fcs" / name)
    # Sidecar inputs the pipeline reads from the patient-dir root (loose or from a zip, both
    # land at the project root); manual gating may also sit under reference/.
    for name in ("metadata.json", "flow.csv", "alc.csv", "cbc.csv", "manual_gating.csv"):
        src = data_dir / name
        if src.exists():
            _link_or_copy(src, norm / name)
    ref_manual = data_dir / "reference" / "manual_gating.csv"
    if ref_manual.exists():
        _link_or_copy(ref_manual, norm / "manual_gating.csv")

    srcs = sorted({str(p.parent) for p in all_fcs})
    print(f"[anchored] normalized dataset: {len(seen)} FCS from {len(srcs)} folder(s) "
          f"{srcs if len(srcs) <= 4 else srcs[:4] + ['...']} -> '{norm / 'fcs'}'", flush=True)
    return norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset dir (fcs/, metadata.json, ...).")
    ap.add_argument("--out", required=True, help="Where to write result CSVs.")
    ap.add_argument("--plots", default="", help="Where to write plots.")
    ap.add_argument(
        "--compensation-controls", default="",
        help="Optional project-local directory containing single-stain/unstained control FCS.",
    )
    ap.add_argument(
        "--analysis-contract", default="",
        help="Optional frozen machine-readable analysis contract to hash into the bundle.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = FirstRunOutputLayout.create(out_dir)
    plots_dir = Path(args.plots).resolve() if args.plots else layout.plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    compensation_control_root = (
        Path(args.compensation_controls).resolve() if args.compensation_controls else None
    )
    if compensation_control_root is not None and not compensation_control_root.is_dir():
        raise SystemExit(
            f"anchored_nk_panel: compensation-control directory not found: "
            f"{compensation_control_root}"
        )
    compensation_control_files = (
        sorted(
            path for path in compensation_control_root.rglob("*.fcs")
            if classify_file(path.name) == "comp"
        )
        if compensation_control_root is not None else []
    )
    analysis_contract = Path(args.analysis_contract).resolve() if args.analysis_contract else None
    if analysis_contract is not None and not analysis_contract.is_file():
        raise SystemExit(f"anchored_nk_panel: analysis contract not found: {analysis_contract}")

    meta = _load_meta(data_dir)
    # D1/D2/D3 knobs — defaults here for now; changing them
    # is a metadata.json edit, no FLOW code change.
    reference_tp = str(meta.get("reference_timepoint", "Baseline"))
    anchor_mode = str(meta.get("anchor_mode", "auto")).lower()
    subsample = int(meta.get("subsample", 200_000))
    ch = resolve_channels(meta)

    # Normalize the dataset layout so the vendored pipeline finds the FCS files wherever the
    # app put them (flat in the project root, in fcs/, or nested). Everything downstream uses
    # this patient_dir instead of the raw --data dir.
    patient_dir = _normalize_dataset(data_dir, out_dir)

    # Prepare the same compensation fallback before anchor construction. Otherwise a
    # run-matched control-derived matrix could rescue the canonical run but the earlier
    # reference-anchor load would fail first when embedded compensation is absent/invalid.
    patient_fcs = sorted((patient_dir / "fcs").glob("*.fcs"))
    anchor_preflight = {}
    anchor_expected_channels = []
    for path in patient_fcs:
        if classify_file(path.name) == "comp":
            continue
        try:
            audit = inspect_fcs_metadata(path)
        except Exception:
            continue
        anchor_preflight[str(path.resolve())] = audit
        if not anchor_expected_channels and audit.get("embedded_matrix_channels"):
            anchor_expected_channels = list(audit["embedded_matrix_channels"])
    if not anchor_expected_channels and compensation_control_files:
        anchor_expected_channels = infer_control_fluorescence_channels(
            compensation_control_files
        )
    anchor_control_repository = (
        build_control_repository(
            compensation_control_files, classify_file, anchor_expected_channels
        )
        if compensation_control_files and anchor_expected_channels else {"control_sets": []}
    )

    def control_candidate(path: Path):
        audit = anchor_preflight.get(str(path.resolve()))
        if audit is None:
            try:
                audit = inspect_fcs_metadata(path)
            except Exception:
                return None
        return candidate_for_acquisition(
            anchor_control_repository,
            str(audit.get("acquisition_id") or "UNASSIGNED"),
            specimen_voltages=audit.get("detector_voltages"),
            require_voltage_match=True,
        )

    manual_ref = _find_manual(patient_dir)
    use_manual = manual_ref is not None and anchor_mode in ("auto", "manual")
    if anchor_mode == "manual" and manual_ref is None:
        print("anchored_nk_panel: anchor_mode='manual' but no manual_gating.csv found — "
              "falling back to negative-population anchor.", flush=True)
        use_manual = False

    anchor_path = layout.provenance / "operator_anchor.json"

    # ── 0. Validate the manual CSV BEFORE it steers anything ──────────
    # A label the parser can't match is dropped silently downstream, which would report a
    # confident MAE over whatever happened to match. Say so here instead.
    check = None
    if use_manual:
        try:
            check = check_manual_gating(manual_ref, patient_dir / "fcs", reference_tp)
        except ManualGatingError as e:
            raise SystemExit(f"\nanchored_nk_panel: manual gating file is unusable.\n\n{e}\n")
        print(f"[anchored] {check.report()}", flush=True)

    # ── 1. Build the anchor (the unified cutoff) ──────────────────────
    if use_manual:
        print(f"[anchored] calibrating reference '{reference_tp}' to manual gating "
              f"({manual_ref})", flush=True)
        build_operator_anchor(patient_dir, manual_ref, reference_tp=reference_tp,
                              out_path=anchor_path, subsample=subsample,
                              control_repository=anchor_control_repository)
    else:
        print(f"[anchored] no manual calibration — negative-population anchor at "
              f"'{reference_tp}'", flush=True)
        build_negative_anchor(
            patient_dir, ch, reference_tp, anchor_path, subsample=subsample,
            control_repository=anchor_control_repository,
        )

    # ── 2. Run pipeline once (deterministic) with the anchor ───────────────
    process_patient(patient_dir, out_dir=out_dir,
                    manual_ref=(manual_ref if use_manual else None),
                    subsample=subsample, verbose=True, anchor_path=anchor_path,
                    compensation_control_paths=compensation_control_files)

    meta_pid = meta.get("patient_study") or data_dir.name

    # ── 3. Adapt outputs to FLOW's standard names + add the new tables ─────
    her_multi = layout.tables / "multilineage.csv"
    if not her_multi.is_file():
        raise SystemExit("anchored_nk_panel: no multilineage output produced")
    multi_df = pd.read_csv(her_multi)
    # Preserve the established canonical CSV serialization after retiring the old
    # Multilineage_SSA_<study>.csv alias. This is formatting normalization only.
    multi_df.to_csv(her_multi, index=False)
    anchor = load_anchor(anchor_path)
    applied_policy_path = layout.qc / "applied_gate_policy.json"
    applied_policy = (
        json.loads(applied_policy_path.read_text()) if applied_policy_path.exists() else {}
    )
    write_unified_cutoffs(anchor, layout.tables, applied_policy=applied_policy)
    write_composition_shift(multi_df, layout.tables, reference_tp=reference_tp)

    manual_note = ""
    if use_manual:
        man = load_manual(manual_ref)
        cmp = compare(multi_df, man)
        cmp.to_csv(layout.tables / "compare_manual.csv", index=False)
        summary = summarize(cmp)
        # Always qualify the MAE with its coverage — an MAE over one of twelve timepoints
        # otherwise reads exactly like an MAE over all twelve.
        manual_note = coverage_note(cmp, check)
        print(f"[anchored] vs manual: MAE={summary.get('mae')} pp  "
              f"<=5pp={summary.get('within_5pp')}%  <=10pp={summary.get('within_10pp')}%  "
              f"({manual_note})", flush=True)

    # ── 4. Cross-timepoint overlay plots (shared axis + unified cut) ───────
    order = None
    flow_csv = patient_dir / "flow.csv"
    if flow_csv.exists():
        try:
            order = pd.read_csv(flow_csv)["label"].astype(str).tolist()
        except Exception:
            order = None
    pairs = []
    fcs_dir = patient_dir / "fcs"
    if fcs_dir.is_dir():
        for f in sorted(fcs_dir.glob("*.fcs")):
            try:
                df, _ = load_xform(
                    f, subsample=subsample,
                    control_derived_candidate=control_candidate(f),
                )
                pairs.append((f.name, df))
            except Exception as e:
                print(f"  overlay: could not load {f.name}: {e}", flush=True)
    try:
        histogram_overlays(pairs, ch, anchor, plots_dir, order=order,
                           hla_cut=anchor.get("donor_cut"), car_cut=anchor.get("car_cut"))
    except Exception as e:
        print(f"  overlay plots skipped: {e}", flush=True)

    with open(layout.reports / "first_run_summary.txt", "w") as f:
        f.write(DATA_DICTIONARY)
        f.write(f"\nReference timepoint: {reference_tp} | "
                f"anchor: {'manual-calibrated' if use_manual else 'negative-population'} | "
                f"samples: {len(multi_df)}\n")
        # The agent reads this file; the MAE's coverage belongs next to the MAE, not only
        # in the run log.
        if check is not None:
            f.write("\n" + check.report() + "\n")
            if manual_note:
                f.write(f"compare_manual.csv covers: {manual_note}\n")
        f.write(
            f"\nCompensation control evidence: "
            f"{len(compensation_control_files)} FCS from "
            f"{compensation_control_root or 'not supplied'}\n"
        )
        f.write(
            "Time-channel exclusions are emitted only as a separately labeled sensitivity "
            "analysis; canonical membership remains unchanged.\n"
        )

    # ── 5. Publish the typed human output package, then freeze the bundle ─────
    layout.write_catalog(str(meta_pid))
    source_files = [Path(__file__), *sorted((_HERE / "anchored").glob("*.py")),
                    *sorted((_HERE / "anchored").glob("*.json"))]
    config_files = [path for path in (
        data_dir / "metadata.json", data_dir / "flow.csv", data_dir / "alc.csv",
        data_dir / "cbc.csv", manual_ref, anchor_path,
        analysis_contract,
    ) if path is not None and Path(path).exists()]
    bundle = build_first_run_bundle(
        out_dir=out_dir,
        input_files=[*_collect_fcs(data_dir), *compensation_control_files],
        source_files=source_files,
        config_files=config_files,
    )
    print("\nFIRST_RUN_OK wrote", len(multi_df), "timepoints.")
    print("Outputs: outputs/reports/, outputs/tables/, outputs/qc/, outputs/membership/, "
          "outputs/plots/, outputs/provenance/"
          + (", compare_manual.csv" if use_manual else "")
          + ", output_index.csv, first_run_bundle.json")
    print("FirstRunBundle SHA256:", bundle["first_run_bundle_sha256"])


if __name__ == "__main__":
    main()
