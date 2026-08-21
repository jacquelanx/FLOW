#!/usr/bin/env python3
"""
Agent-ready CAR-NK flow cytometry pipeline (FSA × SSA first).

Usage
-----
  python -m pipeline.run --patient-dir patients/UPN27
  python -m pipeline.run --patient-dir patients/UPN27 --manual-ref reference/UPN27_manual_gating.csv
  python -m pipeline.run --manifest manifest.csv --out-dir output
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .fcs_io import CH_OPT, load_xform, inspect_fcs_metadata, resolve_channels, has_full_panel
from .acquisition_qc import acquisition_cleaning_keep_mask, acquisition_exclusion_reason
from .acquisition_cleaning import write_acquisition_cleaning_report
from .compensation import (
    CompensationError, summarize_compensation, verify_control_settings,
)
from .control_compensation import (
    build_control_repository, candidate_for_acquisition,
    compare_specimens_to_control_repository, infer_control_fluorescence_channels,
)
from .gates import calibrate_patient, gate_with, percentages
from .calibrate import (
    classify_file, car_cut_from_controls,
    donor_cut, adaptive_car_from_donor, nk_hla_car, MIN_PARENT, MIN_POS,
)
from .qc_plots import write_compact_qc_report, write_longitudinal_composition_report
from .flow_outputs import load_exact_alc_matches, write_temporal_cell_type_summary
from .output_layout import FirstRunOutputLayout
from .compare import (
    compare, summarize, timepoint_from_file, load_manual, is_baseline, is_pre_infusion,
    timepoint_sort_key,
)
from .report import write_report
from .reference_anchor import (
    find_anchor, load_anchor, apply_anchor_cuts, build_operator_anchor,
    transfer_lineage_for_nk_dominant, refine_timepoint_b_cd14,
)

ROOT = Path(__file__).resolve().parents[1]


_GATE_PARAMETER_META = {
    "fsc_lo": ("FSC-A", ">=", "lymph-report lower FSC boundary"),
    "fsc_hi": ("FSC-A", "<=", "lymph-report upper FSC boundary"),
    "ssc_hi": ("SSC-A", "<", "lymph-report upper SSC boundary"),
    "viable_fsc_lo": ("FSC-A", ">=", "viable density-island display extent"),
    "viable_fsc_hi": ("FSC-A", "<=", "viable density-island display extent"),
    "viable_ssc_hi": ("SSC-A", "<=", "viable density-island display extent"),
    "ld": ("ld", "<", "live"),
    "cd45": ("cd45", ">", "CD45 positive"),
    "cd14": ("cd14", ">=", "CD14 positive branch"),
    "cd19": ("cd19", ">=", "B-cell region"),
    "cd3": ("cd3", ">", "T-cell dimension"),
    "cd56": ("cd56", ">", "NK-cell dimension"),
    "cd4": ("cd4", ">", "CD4 region within T"),
    "cd8": ("cd8", ">", "direct CD8 region within T"),
    "hla": ("hla", "polarity-dependent", "configured donor-HLA region"),
    "car": ("car", ">", "CAR-positive region within configured donor NK"),
}


def _gate_parameter_rows(
    *, fname, timepoint, cuts, channels, anchor, cut_context,
):
    """Return the exact applied parameters and their file-level provenance."""
    reference_tp = anchor.get("reference_timepoint") if anchor else None
    is_reference = bool(anchor and timepoint == reference_tp)
    rows = []
    for parameter, (channel_key, operator, role) in _GATE_PARAMETER_META.items():
        value = cuts.get(parameter)
        if parameter == "cd8" and cuts.get("cd8_from_cd4_neg"):
            source = "derived_cd4_negative"
            detail = "No direct CD8 threshold; CD8 := T and CD4-negative"
            operator_value = "derived"
        elif parameter in {"viable_fsc_lo", "viable_fsc_hi", "viable_ssc_hi"}:
            source = "per_file_density_island"
            detail = "Measured independently on this file"
            operator_value = operator
        elif parameter in {"fsc_lo", "fsc_hi", "ssc_hi"}:
            source = "soft_locked_per_file_density" if anchor else "per_file_density"
            detail = (
                "Per-file density geometry constrained by the reference scatter policy"
                if anchor else "Per-file density geometry"
            )
            operator_value = operator
        elif parameter == "hla":
            source = cut_context["hla_source"]
            detail = cut_context["hla_detail"]
            operator_value = "<" if cuts.get("hla_dim") else ">"
        elif parameter == "car":
            source = cut_context["car_source"]
            detail = cut_context["car_detail"]
            operator_value = operator
        elif parameter == "cd4" and anchor and (anchor.get("reference_cuts") or {}).get("cd4") is not None:
            source = "reference_anchor_transfer"
            detail = "Reference CD4 threshold applied to this file"
            operator_value = operator
        elif parameter in {"cd3", "cd56"} and cut_context.get("lineage_transfer") == "reference_nk_dominant":
            source = "reference_anchor_nk_dominant"
            detail = "Reference CD3/CD56 thresholds applied under the predeclared NK-dominant rule"
            operator_value = operator
        elif parameter == "cd19" and cut_context.get("b_fit"):
            source = cut_context["b_fit"]
            detail = "Applied CD19 threshold from the declared timepoint refinement policy"
            operator_value = operator
        elif parameter == "cd14" and cut_context.get("cd14_fit"):
            source = cut_context["cd14_fit"]
            detail = "Applied CD14 threshold from the declared timepoint refinement policy"
            operator_value = operator
        elif is_reference and anchor:
            source = "reference_anchor"
            detail = "Reference-timepoint threshold"
            operator_value = operator
        else:
            source = "per_file_estimate"
            detail = "Per-file deterministic fluorescence estimate with reference fallback if unavailable"
            operator_value = operator
        channel = (
            channel_key if channel_key in {"FSC-A", "SSC-A"}
            else channels.get(channel_key, CH_OPT.get(channel_key, ""))
        )
        rows.append({
            "file": fname,
            "timepoint": timepoint,
            "parameter": parameter,
            "channel": channel,
            "cutoff": value,
            "operator": operator_value,
            "semantic_role": role,
            "source": source,
            "source_detail": detail,
            "reference_timepoint": reference_tp,
            "is_reference_timepoint": is_reference,
            "cd8_is_cd4_negative_derived": bool(
                parameter == "cd8" and cuts.get("cd8_from_cd4_neg")
            ),
        })
    return rows


def _append_membership(path: Path, table: pd.DataFrame, *, header: bool) -> None:
    """Append a deterministic gzip member (mtime=0) without retaining all events in RAM."""
    with path.open("wb" if header else "ab") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                table.to_csv(text_handle, index=False, header=header)


def process_patient(
    patient_dir: str | Path,
    out_dir: str | Path = "output",
    manual_ref: str | Path | None = None,
    subsample: int = 200_000,
    verbose: bool = True,
    anchor_path: str | Path | None = None,
    build_anchor: bool = False,
    compensation_control_paths: list[str | Path] | None = None,
):
    patient_dir = Path(patient_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = FirstRunOutputLayout.create(out_dir)

    meta = {}
    mp = patient_dir / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    pid = meta.get("patient_study") or patient_dir.name
    patient_name = meta.get("name") or pid
    ch = resolve_channels(meta)
    polarity = (meta.get("hla_polarity") or "donor").strip().lower()
    hla_spec = meta.get("hla_specificity") or ""

    fcs_dir = patient_dir / "fcs"
    if not fcs_dir.is_dir():
        raise FileNotFoundError(f"No fcs/ folder in {patient_dir}")

    # Resolve manual ref early (needed to build anchor)
    if manual_ref is None:
        cand = ROOT / "reference" / f"{pid}_manual_gating.csv"
        if cand.exists():
            manual_ref = cand

    anchor = None

    controls = {}
    control_src = {}
    loaded = []  # (fname, df, ntot, audit)
    load_audits = []
    biological_control_audits = []
    n_comp = 0
    all_fcs = sorted(f for f in fcs_dir.iterdir() if f.suffix.lower() == ".fcs")
    internal_comp_paths = [path for path in all_fcs if classify_file(path.name) == "comp"]
    external_comp_paths = [
        Path(path) for path in (compensation_control_paths or [])
        if Path(path).is_file() and Path(path).suffix.lower() == ".fcs"
        and classify_file(Path(path).name) == "comp"
    ]
    all_comp_paths = sorted(
        {path.resolve() for path in [*internal_comp_paths, *external_comp_paths]},
        key=lambda path: str(path).lower(),
    )

    # Read specimen metadata before applying any matrix so run-matched control candidates can
    # serve as a fallback only when embedded compensation is absent or invalid.
    preflight = {}
    expected_matrix_channels = []
    for path in all_fcs:
        if classify_file(path.name) == "comp":
            continue
        try:
            metadata_audit = inspect_fcs_metadata(path)
        except Exception:
            continue
        preflight[str(path.resolve())] = metadata_audit
        if not expected_matrix_channels and metadata_audit.get("embedded_matrix_channels"):
            expected_matrix_channels = list(metadata_audit["embedded_matrix_channels"])
    if not expected_matrix_channels and all_comp_paths:
        expected_matrix_channels = infer_control_fluorescence_channels(all_comp_paths)
    control_repository = build_control_repository(
        all_comp_paths, classify_file, expected_matrix_channels
    ) if all_comp_paths and expected_matrix_channels else {
        "schema_version": "flow.compensation_control_repository.v1",
        "state": "BLOCKED",
        "expected_channels": expected_matrix_channels,
        "controls": [],
        "control_sets": [],
    }
    specimen_preflight = [
        preflight.get(str(path.resolve()), {}) for path in all_fcs
        if classify_file(path.name) == "timepoint"
    ]
    for candidate in control_repository.get("control_sets", []):
        eligible_files = []
        for audit in specimen_preflight:
            match = candidate_for_acquisition(
                control_repository,
                str(audit.get("acquisition_id") or "UNASSIGNED"),
                specimen_voltages=audit.get("detector_voltages"),
                require_voltage_match=True,
            )
            if match and match.get("matrix_sha256") == candidate.get("matrix_sha256"):
                eligible_files.append(str(audit.get("file") or "unknown"))
        candidate["matching_supplied_specimen_count"] = len(eligible_files)
        candidate["eligible_supplied_specimen_files"] = sorted(eligible_files)
        candidate["eligible_for_any_supplied_specimen_fallback"] = bool(eligible_files)

    # Nature-style operator anchor. Build it only after compensation candidates exist so a
    # missing/invalid embedded matrix follows the same fallback policy as the main run.
    if build_anchor:
        if not manual_ref or not Path(manual_ref).exists():
            raise FileNotFoundError("--build-anchor requires --manual-ref")
        build_operator_anchor(
            patient_dir, manual_ref, subsample=subsample,
            control_repository=control_repository,
        )
    apath = find_anchor(patient_dir, anchor_path)
    if apath is not None:
        anchor = load_anchor(apath)
        if verbose:
            print(f"[{pid}] operator anchor: {apath.name}  "
                  f"(SSC_hi={anchor['locked']['ssc_hi']:.3f}, "
                  f"cal MAE={anchor.get('calibration_mae_pp')} pp)", flush=True)
    comp_control_audits = list(control_repository.get("controls", []))
    comp_control_rows = comp_control_audits
    for f in all_fcs:
        kind = classify_file(f.name)
        if kind == "comp":
            # Instrument compensation control: full fluorescence panel but not a specimen.
            # Metadata/settings are audited, but events never reach gates or median cuts.
            n_comp += 1
            continue
        metadata_audit = preflight.get(str(f.resolve()), {})
        control_candidate = candidate_for_acquisition(
            control_repository,
            str(metadata_audit.get("acquisition_id") or "UNASSIGNED"),
            specimen_voltages=metadata_audit.get("detector_voltages"),
            require_voltage_match=True,
        )
        try:
            df, ntot, load_audit = load_xform(
                f, subsample=subsample, return_audit=True,
                control_derived_candidate=control_candidate,
            )
        except CompensationError:
            # Compensation failures invalidate fluorescence identity and therefore the
            # pipeline. Never downgrade them to an ordinary skipped-file warning.
            raise
        except Exception as e:
            if verbose:
                print(f"  !!  {f.name}: load failed: {e}", flush=True)
            continue
        if not has_full_panel(df, ch):
            if verbose:
                print(f"  --  {f.name}: skipped (incomplete panel)", flush=True)
            continue
        if kind != "timepoint":
            biological_control_audits.append(load_audit)
            # One control of each kind is used for the study-wide donor/CAR cut. When a
            # dataset ships one per acquisition folder, say which one won.
            if verbose:
                prev = control_src.get(kind)
                note = f" (replaces {prev})" if prev else ""
                print(f"  [ctrl] {f.name} → {kind}{note}", flush=True)
            controls[kind] = df
            control_src[kind] = f.name
        else:
            load_audits.append(load_audit)
            loaded.append((f.name, df, ntot, load_audit))

    if verbose and n_comp:
        print(f"[{pid}] skipped {n_comp} compensation control file(s)", flush=True)
    if not loaded:
        raise RuntimeError(f"No timepoint FCS files found in {fcs_dir}")

    compensation_summary = summarize_compensation(load_audits)
    compensation_selection = compare_specimens_to_control_repository(
        load_audits, control_repository
    )
    control_verification = verify_control_settings(
        load_audits, comp_control_audits, comp_control_rows)
    compensation_summary["control_verification"] = control_verification
    compensation_summary["selection"] = compensation_selection
    if control_verification["state"] == "BLOCKED":
        compensation_summary["state"] = "BLOCKED"
    specimen_comp_rows = [{
        "file": audit["file"],
        "acquisition_id": audit["acquisition_id"],
        "acquisition_date": audit.get("acquisition_date"),
        "cytometer_serial": audit.get("cytometer_serial"),
        "event_count": audit["event_count"],
        "compensation_source": audit["compensation_source"],
        "compensation_applied": audit["compensation_applied"],
        "compensation_state": audit["compensation_state"],
        "compensation_reason": audit["compensation_reason"],
        "matrix_sha256": audit["matrix_sha256"],
        "matrix_dimension": audit["matrix_dimension"],
        "matrix_condition_number": audit["matrix_condition_number"],
        "matrix_validation_state": audit["matrix_validation_state"],
        "control_derived_candidate_sha256": audit.get("control_derived_candidate_sha256"),
        "control_derived_candidate_state": audit.get("control_derived_candidate_state"),
    } for audit in load_audits]
    pd.DataFrame(specimen_comp_rows).to_csv(
        layout.qc / "specimen_compensation_map.csv", index=False)
    inventory_columns = [
        "file", "source_path", "acquisition_id", "acquisition_date", "cytometer",
        "cytometer_serial", "role", "control_target", "event_identity", "event_count",
        "metadata_state", "used_for_canonical_compensation",
    ]
    pd.DataFrame(comp_control_rows).reindex(columns=inventory_columns).to_csv(
        layout.qc / "compensation_control_inventory.csv", index=False
    )
    selection_rows = pd.DataFrame(compensation_selection["specimens"])
    selection_rows.to_csv(layout.qc / "compensation_selection.csv", index=False)
    residual_columns = [
        "file", "matrix_source", "matrix_sha256", "primary_channel",
        "secondary_channel", "residual_ratio", "absolute_residual_ratio",
    ]
    pd.DataFrame(compensation_selection["embedded_residual_details"]).reindex(
        columns=residual_columns
    ).to_csv(layout.qc / "compensation_residuals.csv", index=False)
    (layout.qc / "control_derived_compensation_candidates.json").write_text(json.dumps({
        "schema_version": control_repository["schema_version"],
        "state": control_repository["state"],
        "expected_channels": control_repository["expected_channels"],
        "control_sets": control_repository["control_sets"],
    }, indent=2, sort_keys=True))
    (layout.qc / "compensation_diagnostics.json").write_text(json.dumps({
        "summary": compensation_summary,
        "specimens": specimen_comp_rows,
        "controls": comp_control_rows,
        "control_sets": control_repository.get("control_sets", []),
        "biological_analysis_controls": biological_control_audits,
        "selection": compensation_selection,
    }, indent=2, sort_keys=True))

    acquisition_rows = []
    voltage_rows = []
    for audit in load_audits:
        qc = audit["acquisition_qc"]
        acquisition_rows.append({
            "file": audit["file"],
            "acquisition_id": audit["acquisition_id"],
            **{k: v for k, v in qc.items() if k != "candidate_time_intervals"},
            "candidate_interval_count": len(qc.get("candidate_time_intervals", [])),
            "candidate_event_percent": (
                None if qc.get("candidate_event_fraction") is None
                else 100.0 * qc["candidate_event_fraction"]
            ),
        })
        for voltage in audit["detector_voltages"]:
            voltage_rows.append({
                "file": audit["file"],
                "acquisition_id": audit["acquisition_id"],
                **voltage,
            })
    for audit in comp_control_audits:
        for voltage in audit.get("detector_voltages", []):
            voltage_rows.append({
                "file": audit["file"],
                "acquisition_id": audit["acquisition_id"],
                "file_role": "compensation_control",
                **voltage,
            })
    pd.DataFrame(acquisition_rows).to_csv(layout.qc / "acquisition_qc.csv", index=False)
    pd.DataFrame(voltage_rows).to_csv(layout.qc / "detector_voltages.csv", index=False)
    (layout.qc / "acquisition_qc_details.json").write_text(json.dumps({
        "policy": {
            "canonical_event_exclusion": False,
            "secondary_time_cleaned_sensitivity": True,
            "sensitivity_reuses_canonical_gate_parameters_without_refitting": True,
            "signal_instability_is_a_proxy_for_voltage_spiking": True,
            "within_file_voltage_change_directly_measured": False,
            "human_review_required_before_promoting_event_exclusions": True,
        },
        "files": [{
            "file": audit["file"],
            "acquisition_id": audit["acquisition_id"],
            "acquisition_qc": audit["acquisition_qc"],
        } for audit in load_audits],
    }, indent=2, sort_keys=True))

    pairs = [(fn, df) for fn, df, _, _ in loaded]

    locked_scatter = anchor["locked"] if anchor else None
    base_ssc = locked_scatter["ssc_hi"] if locked_scatter else None
    if base_ssc is None:
        for fn, df in pairs:
            if is_baseline(fn):
                from .gates import file_cuts
                bc = file_cuts(df, ch)
                base_ssc = bc.get("ssc_hi")
                if verbose and base_ssc is not None:
                    print(f"[{pid}] Baseline SSC_hi anchor = {base_ssc:.3f}", flush=True)
                break

    # Re-estimate fluorescence inside locked scatter when anchor present
    perfile, med = calibrate_patient(
        pairs, ch, ssc_cap=None, locked_scatter=locked_scatter)
    if verbose:
        print(f"[{pid}] lineage cuts (median): " +
              ", ".join(f"{k}={('%.0f'%v) if v is not None and k not in ('fsc_lo','fsc_hi','ssc_hi') else (('%.3f'%v) if v is not None else 'NA')}"
                        for k, v in med.items() if k in ("ld", "cd45", "cd14", "cd19", "cd3", "cd56", "ssc_hi")),
              flush=True)

    # Donor / CAR cuts — use anchored cuts for NK extraction when available
    def _cuts_for_nk(fn, df):
        pf = perfile[fn]
        C = {k: (pf[k] if pf.get(k) is not None else med.get(k)) for k in med}
        if anchor:
            C = apply_anchor_cuts(C, anchor, fn)
        return C

    pre_pool, post_pool, base_pool = [], [], []
    for fn, df in pairs:
        Cnk = _cuts_for_nk(fn, df)
        hla, _ = nk_hla_car(df, Cnk, ch)
        if is_pre_infusion(fn):
            pre_pool.append(hla)
            if is_baseline(fn):
                base_pool.append(hla)
        else:
            post_pool.append(hla)
    pre_hla = np.concatenate(pre_pool) if pre_pool else None
    post_hla = np.concatenate(post_pool) if post_pool else None
    base_hla = np.concatenate(base_pool) if base_pool else None

    ctrl_meds = []
    for key in ("ntnk", "cbmc", "car"):
        if key in controls:
            # controls: use median lineage + locked scatter
            Cctrl = dict(med)
            if locked_scatter:
                Cctrl.update(locked_scatter)
            hla, _ = nk_hla_car(controls[key], Cctrl, ch)
            if len(hla) >= 100:
                ctrl_meds.append(float(np.median(hla)))

    hla_dim = False
    if not str(hla_spec).strip():
        hla_cut = None
        dnotes = {"donor_cut": "no donor HLA marker assigned → donor not gated"}
    else:
        # Prefer Nature-style fitted donor cut from anchor when available
        if anchor and anchor.get("donor_cut") is not None:
            hla_cut = float(anchor["donor_cut"])
            dnotes = {"donor_cut": f"anchor-fitted HLA cut={hla_cut:.0f} "
                      f"(MAE {anchor.get('donor_cut_mae_pp')} pp vs manual)"}
        else:
            hla_cut, dnotes = donor_cut(pre_hla, post_hla, ctrl_meds, polarity, base_hla)
            hla_dim = bool(dnotes.pop("hla_dim", False))

    car_control_derived = False
    if anchor and anchor.get("car_cut") is not None:
        car_cut = float(anchor["car_cut"])
        cnotes = {"car_source": f"anchor-fitted CAR cut={car_cut:.0f} "
                  f"(MAE {anchor.get('car_cut_mae_pp')} pp vs manual)"}
    else:
        car_cut, cnotes = car_cut_from_controls(controls, med, ch)
        car_control_derived = car_cut is not None
        if car_cut is None:
            car_cut = adaptive_car_from_donor(pairs, med, ch, hla_cut, hla_dim)
            cnotes["car_source"] = (f"adaptive GMM={car_cut:.0f}" if car_cut else "none")

    hla_source = "reference_anchor" if anchor and anchor.get("donor_cut") is not None else "control_or_adaptive"
    car_source = "reference_anchor" if anchor and anchor.get("car_cut") is not None else (
        "control_derived" if car_control_derived
        else "adaptive"
    )
    applied_gate_policy = {
        "schema_version": "flow.applied_gate_policy.v1",
        "patient_study": pid,
        "reference_timepoint": anchor.get("reference_timepoint") if anchor else None,
        "operator_anchor_file": apath.name if apath else None,
        "operator_anchor_role": (
            "reference and transfer policy; not a claim that all per-file thresholds are identical"
            if anchor else None
        ),
        "scatter_policy": "soft_locked_per_file_density" if anchor else "per_file_density",
        "lineage_policy": (anchor.get("transfer_policy") if anchor else {
            "lineage": "per_file_estimate",
        }),
        "hla_cut": hla_cut,
        "hla_dim": hla_dim,
        "hla_source": hla_source,
        "hla_detail": dnotes,
        "car_cut": car_cut,
        "car_source": car_source,
        "car_detail": cnotes,
        "actual_parameters_artifact": "gate_parameters.csv",
        "cd8_semantics": "CD8 is derived as CD4-negative within T when cd8_from_cd4_neg is true",
    }
    (layout.qc / "applied_gate_policy.json").write_text(
        json.dumps(applied_gate_policy, indent=2, sort_keys=True)
    )

    if verbose:
        print(f"[{pid}] donor={hla_spec or 'none'} cut={('%.0f'%hla_cut) if hla_cut else 'NA'}  "
              f"CAR cut={('%.0f'%car_cut) if car_cut else 'NA'}", flush=True)
        for k, v in {**dnotes, **cnotes}.items():
            print(f"        {k}: {v}", flush=True)

    out_csv = layout.tables / "multilineage.csv"
    out_pdf = layout.reports / f"QC_Report_{pid}.pdf"
    # This is an interactive longitudinal dashboard, not an HTML rendering of the
    # multi-page composite PDF. A distinct stem prevents two unlike artifacts from
    # masquerading as format variants of one report.
    out_html = layout.reports / f"Interactive_Longitudinal_QC_{pid}.html"

    rows = []
    cleaned_rows = []
    cleaning_summary_rows = []
    cleaning_comparison_rows = []
    cleaning_interval_rows = []
    excluded_event_frames = []
    retention_rows = []
    gate_parameter_rows = []
    membership_path = layout.membership / "canonical_membership.csv.gz"
    membership_coverage = []
    membership_first = True
    scatter_policy = "soft_lock" if anchor else "density"
    gate_cache = []  # defer per-TP pages until after cover/retention
    for fn, df, ntot, load_audit in loaded:
        pf = perfile[fn]
        C = {k: (pf[k] if pf.get(k) is not None else med.get(k)) for k in med}
        C["cd8_from_cd4_neg"] = bool(pf.get("cd8_from_cd4_neg") or False)
        if anchor:
            C = apply_anchor_cuts(C, anchor, fn)
            C = transfer_lineage_for_nk_dominant(df, C, anchor, ch)
            C["_tp"] = timepoint_from_file(fn)
            C = refine_timepoint_b_cd14(df, C, anchor, ch)
            ssc_cap = None  # soft-lock already applied
        else:
            ssc_cap = base_ssc
        C["hla"] = hla_cut
        C["car"] = car_cut
        C["hla_dim"] = hla_dim
        if C.get("cd3") is None or C.get("cd56") is None:
            if verbose:
                print(f"  !!  {fn}: no CD3/CD56 cut — skipped", flush=True)
            continue
        m, g, scat, cuts = gate_with(df, C, ch, ssc_cap=ssc_cap)
        p = percentages(m)
        tp = timepoint_from_file(fn)
        acquisition_qc = load_audit["acquisition_qc"]
        cleaning_keep = acquisition_cleaning_keep_mask(
            df["__time_raw"].to_numpy(dtype=float), acquisition_qc
        )
        cleaned_masks = {
            node: np.asarray(mask, dtype=bool)[cleaning_keep] for node, mask in m.items()
        }
        cleaned_percentages = percentages(cleaned_masks)
        excluded_count = int((~cleaning_keep).sum())
        analyzed_count = int(len(cleaning_keep))
        full_candidate_count = int(acquisition_qc.get("candidate_event_count") or 0)
        full_candidate_fraction = acquisition_qc.get("candidate_event_fraction")
        sensitivity_result = (
            "EXCLUDED_CANDIDATE_INTERVALS" if excluded_count
            else "NOT_EVALUABLE_NO_EXCLUSION"
            if acquisition_qc.get("state") == "NOT_EVALUABLE"
            else "NO_CANDIDATE_EVENTS"
        )
        cleaning_summary_rows.append({
            "file": fn,
            "timepoint": tp,
            "acquisition_id": load_audit["acquisition_id"],
            "acquisition_qc_state": acquisition_qc.get("state"),
            "cleaning_policy_id": acquisition_qc.get("cleaning_policy_id"),
            "source_event_count": int(ntot),
            "analyzed_event_count": analyzed_count,
            "cleaned_analyzed_event_count": int(cleaning_keep.sum()),
            "excluded_analyzed_event_count": excluded_count,
            "excluded_analyzed_event_percent": (
                100.0 * excluded_count / analyzed_count if analyzed_count else None
            ),
            "full_file_candidate_event_count": full_candidate_count,
            "full_file_candidate_event_percent": (
                None if full_candidate_fraction is None else 100.0 * full_candidate_fraction
            ),
            "candidate_interval_count": len(
                acquisition_qc.get("candidate_time_intervals", [])
            ),
            "canonical_event_exclusion_applied": False,
            "sensitivity_event_exclusion_applied": bool(excluded_count),
            "canonical_gate_parameters_refit": False,
            "sensitivity_result": sensitivity_result,
            "voltage_spike_claim": "NOT_MADE",
        })
        for interval in acquisition_qc.get("candidate_time_intervals", []):
            cleaning_interval_rows.append({
                "file": fn,
                "timepoint": tp,
                "acquisition_id": load_audit["acquisition_id"],
                **interval,
            })
        gate_parameter_rows.extend(_gate_parameter_rows(
            fname=fn,
            timepoint=tp,
            cuts=cuts,
            channels=ch,
            anchor=anchor,
            cut_context={
                "hla_source": hla_source,
                "hla_detail": json.dumps(dnotes, sort_keys=True),
                "car_source": car_source,
                "car_detail": json.dumps(cnotes, sort_keys=True),
                "lineage_transfer": C.get("_lineage_transfer"),
                "b_fit": C.get("_b_fit"),
                "cd14_fit": C.get("_cd14_fit"),
            },
        ))
        membership = pd.DataFrame({
            "patient_id": pid,
            "sample_id": fn,
            "event_id": df["__event_id"].astype(np.int64),
            "time_raw": df["__time_raw"],
            **{node: np.asarray(mask, dtype=np.uint8) for node, mask in m.items()},
        })
        if excluded_count:
            excluded = membership.loc[
                ~cleaning_keep, ["patient_id", "sample_id", "event_id", "time_raw"]
            ].copy()
            excluded["exclusion_reason"] = [
                acquisition_exclusion_reason(value, acquisition_qc)
                for value in excluded["time_raw"].to_numpy(dtype=float)
            ]
            excluded["cleaning_policy_id"] = acquisition_qc.get("cleaning_policy_id")
            excluded_event_frames.append(excluded)
        _append_membership(membership_path, membership, header=membership_first)
        membership_first = False
        membership_coverage.append({
            "patient_id": pid,
            "sample_id": fn,
            "source_event_count": int(ntot),
            "membership_event_count": int(len(membership)),
            "complete_event_coverage": bool(len(membership) == int(ntot)),
        })
        gate_cache.append((fn, tp, m, g, scat, cuts, p))

        n_tot_loaded = len(df)
        retention_rows.append({
            "timepoint": tp or fn,
            "n_total": ntot,
            "lymph_pct": round(100.0 * m["lymph_scatter"].sum() / n_tot_loaded, 3),
            "sing_pct": round(100.0 * m["sing"].sum() / n_tot_loaded, 3),
            "live_pct": round(100.0 * m["live"].sum() / n_tot_loaded, 3),
            "cd45_pct": round(100.0 * m["cd45p"].sum() / n_tot_loaded, 3),
            "lympho_pct": round(100.0 * m["lympho"].sum() / n_tot_loaded, 3),
            "purity": p.get("lineage_purity", 0),
        })

        n_nk = int(m["NK"].sum())
        n_donor = int(m["Donor"].sum())
        n_car = int(m["CAR"].sum())
        metrics = {k: v for k, v in p.items() if not k.endswith("_reliable")}
        row = {
            "file": fn,
            "timepoint": tp,
            "n_events": ntot,
            "analysis_variant": "canonical_all_analyzed_events",
            "analyzed_event_count": analyzed_count,
            **metrics,
            "donor_marker": hla_spec,
            "donor_reliable": p.get("donor_reliable", n_nk >= MIN_PARENT and n_donor >= MIN_POS),
            "car_reliable": p.get("car_reliable", n_donor >= MIN_PARENT and n_car >= MIN_POS),
            "n_scatter": int(m["lymph_scatter"].sum()),
            "n_live": int(m["live"].sum()),
            "n_lympho": int(m["lympho"].sum()),
            "n_B": int(m["B"].sum()),
            "n_T": int(m["T"].sum()),
            "n_NK": n_nk,
            "n_Donor": n_donor,
            "n_CAR": n_car,
            "ssc_hi": cuts.get("ssc_hi"),
            "fsc_lo": cuts.get("fsc_lo"),
            "fsc_hi": cuts.get("fsc_hi"),
            "scatter_method": cuts.get("scatter_method"),
            "compensation_state": load_audit["compensation_state"],
            "compensation_matrix_sha256": load_audit["matrix_sha256"],
            "acquisition_id": load_audit["acquisition_id"],
            "acquisition_qc_state": load_audit["acquisition_qc"]["state"],
            "event_rate_cv": load_audit["acquisition_qc"].get("event_rate_cv"),
            "signal_spike_proxy_event_pct": (
                None if load_audit["acquisition_qc"].get("candidate_event_fraction") is None
                else 100.0 * load_audit["acquisition_qc"]["candidate_event_fraction"]
            ),
            "canonical_event_exclusion_applied": False,
        }
        if "CD4" in m:
            row["n_CD4"] = int(m["CD4"].sum())
            row["n_CD8"] = int(m["CD8"].sum())
        rows.append(row)

        cleaned_metrics = {
            key: value for key, value in cleaned_percentages.items()
            if not key.endswith("_reliable")
        }
        cleaned_row = dict(row)
        cleaned_row.update(cleaned_metrics)
        cleaned_row.update({
            "analysis_variant": "time_cleaned_sensitivity",
            "cleaned_analyzed_event_count": int(cleaning_keep.sum()),
            "excluded_analyzed_event_count": excluded_count,
            "excluded_analyzed_event_percent": (
                100.0 * excluded_count / analyzed_count if analyzed_count else None
            ),
            "sensitivity_event_exclusion_applied": bool(excluded_count),
            "canonical_gate_parameters_refit": False,
            "n_scatter": int(cleaned_masks["lymph_scatter"].sum()),
            "n_live": int(cleaned_masks["live"].sum()),
            "n_lympho": int(cleaned_masks["lympho"].sum()),
            "n_B": int(cleaned_masks["B"].sum()),
            "n_T": int(cleaned_masks["T"].sum()),
            "n_NK": int(cleaned_masks["NK"].sum()),
            "n_Donor": int(cleaned_masks["Donor"].sum()),
            "n_CAR": int(cleaned_masks["CAR"].sum()),
            "donor_reliable": cleaned_percentages.get(
                "donor_reliable",
                int(cleaned_masks["NK"].sum()) >= MIN_PARENT
                and int(cleaned_masks["Donor"].sum()) >= MIN_POS,
            ),
            "car_reliable": cleaned_percentages.get(
                "car_reliable",
                int(cleaned_masks["Donor"].sum()) >= MIN_PARENT
                and int(cleaned_masks["CAR"].sum()) >= MIN_POS,
            ),
        })
        if "CD4" in cleaned_masks:
            cleaned_row["n_CD4"] = int(cleaned_masks["CD4"].sum())
            cleaned_row["n_CD8"] = int(cleaned_masks["CD8"].sum())
        cleaned_rows.append(cleaned_row)
        for metric in sorted(set(p) | set(cleaned_percentages)):
            if not str(metric).startswith("%"):
                continue
            canonical_value = p.get(metric)
            cleaned_value = cleaned_percentages.get(metric)
            delta = None
            try:
                if canonical_value is not None and cleaned_value is not None:
                    delta = float(cleaned_value) - float(canonical_value)
            except (TypeError, ValueError):
                delta = None
            cleaning_comparison_rows.append({
                "file": fn,
                "timepoint": tp,
                "metric": metric,
                "canonical_percent": canonical_value,
                "cleaned_percent": cleaned_value,
                "delta_percentage_points": delta,
            })
        if verbose:
            print(
                f"  OK  {fn}: B {p['%B (of lymph)']}%  T {p['%T (of lymph)']}%  "
                f"NK {p['%NK (of lymph)']}%  "
                f"CD14+ {p.get('%Monocytes (of live)', p.get('%CD14+ (of CD45+)', '—'))}%  "
                f"purity {p.get('lineage_purity', '—')}%  "
                f"Donor {p.get('%Donor NK (of lymph)', p.get('%Donor NK (of NK)'))}%  "
                f"CAR {p['%CAR+ (of Donor NK)']}%  "
                f"scat={p.get('%Lymph_scatter (of total)')}%",
                flush=True,
            )

    auto_df = pd.DataFrame(rows)
    # stable timepoint order from flow.csv if present
    flow_csv = patient_dir / "flow.csv"
    if flow_csv.exists() and "timepoint" in auto_df.columns:
        order = pd.read_csv(flow_csv)["label"].astype(str).tolist()
        auto_df["_ord"] = auto_df["timepoint"].apply(
            lambda x: order.index(x) if x in order else 999)
        auto_df = auto_df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
        # keep retention / gate_cache in same order
        order_map = {tp: i for i, tp in enumerate(order)}
        retention_rows.sort(key=lambda r: order_map.get(str(r["timepoint"]), 999))
        gate_cache.sort(key=lambda t: order_map.get(str(t[1]), 999))
    elif "timepoint" in auto_df.columns:
        # No flow.csv: fall back to chronological order parsed from the labels, so
        # "Δ from previous timepoint" downstream means what it says. Files with no
        # timepoint keep their filename order at the end.
        auto_df["_ord"] = auto_df["timepoint"].apply(timepoint_sort_key)
        auto_df = auto_df.sort_values("_ord", kind="stable").drop(
            columns="_ord").reset_index(drop=True)
        retention_rows.sort(key=lambda r: timepoint_sort_key(r["timepoint"]))
        gate_cache.sort(key=lambda t: timepoint_sort_key(t[1]))

    auto_df.to_csv(out_csv, index=False)
    cleaned_df = pd.DataFrame(cleaned_rows)
    if not cleaned_df.empty and "timepoint" in cleaned_df.columns:
        if flow_csv.exists():
            cleaned_order = pd.read_csv(flow_csv)["label"].astype(str).tolist()
            cleaned_df["_ord"] = cleaned_df["timepoint"].apply(
                lambda value: cleaned_order.index(value) if value in cleaned_order else 999
            )
        else:
            cleaned_df["_ord"] = cleaned_df["timepoint"].apply(timepoint_sort_key)
        cleaned_df = cleaned_df.sort_values("_ord", kind="stable").drop(
            columns="_ord"
        ).reset_index(drop=True)
    cleaned_multilineage_path = layout.tables / "multilineage_time_cleaned_sensitivity.csv"
    cleaned_df.to_csv(cleaned_multilineage_path, index=False)

    cleaning_summary_df = pd.DataFrame(cleaning_summary_rows)
    cleaning_comparison_df = pd.DataFrame(cleaning_comparison_rows)
    cleaning_intervals_df = pd.DataFrame(cleaning_interval_rows)
    if not cleaning_summary_df.empty:
        cleaning_summary_df["_ord"] = cleaning_summary_df["timepoint"].apply(
            lambda value: timepoint_sort_key(value) if value else (99_999, "")
        )
        cleaning_summary_df = cleaning_summary_df.sort_values(
            "_ord", kind="stable"
        ).drop(columns="_ord").reset_index(drop=True)
    cleaning_summary_path = layout.qc / "acquisition_cleaning_summary.csv"
    cleaning_comparison_path = layout.tables / "acquisition_cleaning_population_comparison.csv"
    cleaning_intervals_path = layout.qc / "acquisition_cleaning_intervals.csv"
    cleaning_summary_df.to_csv(cleaning_summary_path, index=False)
    cleaning_comparison_df.to_csv(cleaning_comparison_path, index=False)
    cleaning_intervals_df.to_csv(cleaning_intervals_path, index=False)
    excluded_event_path = layout.membership / "time_cleaning_excluded_events.csv.gz"
    excluded_columns = [
        "patient_id", "sample_id", "event_id", "time_raw", "exclusion_reason",
        "cleaning_policy_id",
    ]
    excluded_events = (
        pd.concat(excluded_event_frames, ignore_index=True)
        if excluded_event_frames else pd.DataFrame(columns=excluded_columns)
    ).reindex(columns=excluded_columns)
    _append_membership(excluded_event_path, excluded_events, header=True)
    cleaning_report_path = layout.reports / f"Acquisition_Cleaning_Sensitivity_{pid}.pdf"
    write_acquisition_cleaning_report(
        cleaning_summary_df,
        cleaning_comparison_df,
        cleaning_intervals_df,
        cleaning_report_path,
        patient_id=pid,
        compensation_state=compensation_summary["state"],
    )
    temporal_summary_path = write_temporal_cell_type_summary(
        auto_df,
        layout.tables,
        patient_id=pid,
        reference_tp=(anchor.get("reference_timepoint", "Baseline") if anchor else "Baseline"),
        identity_validated=False,
        alc_matches=load_exact_alc_matches(patient_dir),
    )
    gate_params_df = pd.DataFrame(gate_parameter_rows)
    if not gate_params_df.empty:
        gate_params_df["_timepoint_order"] = gate_params_df["timepoint"].apply(
            lambda value: timepoint_sort_key(value) if value else (99_999, "")
        )
        gate_params_df = gate_params_df.sort_values(
            ["_timepoint_order", "file", "parameter"], kind="stable"
        ).drop(columns="_timepoint_order")
    gate_params_df.to_csv(layout.tables / "gate_parameters.csv", index=False)
    (layout.membership / "canonical_membership_manifest.json").write_text(json.dumps({
        "schema_version": "flow.canonical_membership.v1",
        "primary_key": ["patient_id", "sample_id", "event_id"],
        "membership_file": membership_path.name,
        "coverage": membership_coverage,
        "complete_event_coverage": bool(
            membership_coverage and all(row["complete_event_coverage"]
                                        for row in membership_coverage)
        ),
        "manual_truth_scoring_authorized": False,
        "manual_truth_scoring_requirements": (
            "full event coverage, two reviewers, adjudication, compatible hierarchy, and "
            "patient-grouped development/validation/locked split"
        ),
    }, indent=2, sort_keys=True))

    qc_states = set(auto_df.get("acquisition_qc_state", pd.Series(dtype=str)).dropna())
    if any(str(audit["compensation_state"]).startswith("BLOCKED")
           for audit in load_audits) or compensation_summary["state"] == "BLOCKED":
        technical_state = "BLOCKED"
    elif compensation_summary["state"] != "PASS" or qc_states & {"REVIEW", "FAIL", "NOT_EVALUABLE"}:
        technical_state = "REVIEW"
    else:
        technical_state = "TECHNICAL_PASS"
    (layout.qc / "release_assessment.json").write_text(json.dumps({
        "technical_state": technical_state,
        "biological_release_authorized": False,
        "human_flow_analyst_signoff_required": True,
        "compensation_state": compensation_summary["state"],
        "acquisition_qc_states": sorted(qc_states),
        "time_cleaning_sensitivity_state": "AVAILABLE_FOR_HUMAN_REVIEW",
        "time_cleaning_promoted_to_canonical": False,
        "time_cleaning_excluded_analyzed_events": int(
            cleaning_summary_df.get(
                "excluded_analyzed_event_count", pd.Series(dtype=int)
            ).sum()
        ),
        "manual_truth_state": "NOT_EVALUATED",
        "gate_parameter_provenance_state": "PASS",
        "population_identity_state": "BLOCKED",
        "assay_semantics_state": "REVIEW",
        "release_reason_codes": [
            "HUMAN_SIGNOFF_REQUIRED",
            *(["COMPENSATION_NOT_VERIFIED"] if compensation_summary["state"] != "PASS" else []),
            *(["ACQUISITION_QC_NOT_PASS"] if qc_states & {"REVIEW", "FAIL", "NOT_EVALUABLE"} else []),
            "MANUAL_TRUTH_NOT_EVALUATED",
        ],
    }, indent=2, sort_keys=True))

    cmp_summary = None
    cmp = None
    cmp_path = layout.tables / f"compare_manual_{pid}.csv"
    if manual_ref is None:
        cand = ROOT / "reference" / f"{pid}_manual_gating.csv"
        if cand.exists():
            manual_ref = cand
    if manual_ref and Path(manual_ref).exists():
        man = load_manual(manual_ref)
        cmp = compare(auto_df, man)
        cmp.to_csv(cmp_path, index=False)
        cmp_summary = summarize(cmp)
        if verbose:
            print(f"\n[{pid}] vs manual: MAE={cmp_summary['mae']} pp  "
                  f"≤5pp={cmp_summary['within_5pp']}%  ≤10pp={cmp_summary['within_10pp']}%",
                  flush=True)
            if cmp_summary.get("worst"):
                w = cmp_summary["worst"]
                print(f"        worst: {w.get('timepoint')} {w.get('metric')} "
                      f"Δ={w.get('delta')}", flush=True)

    longitudinal_pdf = layout.reports / f"Longitudinal_Cell_Composition_{pid}.pdf"
    write_longitudinal_composition_report(
        auto_df,
        longitudinal_pdf,
        patient_id=pid,
        technical_state=technical_state,
        compensation_state=compensation_summary["state"],
        population_identity_state="BLOCKED",
        temporal_summary_path=temporal_summary_path,
    )

    with PdfPages(out_pdf, metadata={
        "Creator": "FLOW governed first run",
        "Producer": "FLOW",
        "CreationDate": None,
        "ModDate": None,
    }) as pdf:
        write_compact_qc_report(
            pdf, pid, gate_cache,
            cmp_summary=cmp_summary, compare_df=cmp,
            scatter_policy=scatter_policy,
            render_dir=layout.report_pages_png,
            page_pdf_dir=layout.report_pages_pdf,
            page_index_path=layout.reports / "report_page_index.csv",
            geometry_path=layout.tables / "lymph_density_geometry.csv",
            composite_pdf_name=out_pdf.name,
            acquisition_rows=acquisition_rows,
            technical_state=technical_state,
            compensation_state=compensation_summary["state"],
            population_identity_state="BLOCKED",
        )

    write_report(
        auto_df,
        out_html,
        patient_name,
        pid,
        cmp_summary,
        technical_state=technical_state,
        compensation_state=compensation_summary["state"],
    )

    if verbose:
        print(f"\n[{pid}] CSV  → {out_csv}")
        print(f"[{pid}] QC   → {out_pdf}")
        print(f"[{pid}] HTML → {out_html}")
        if cmp_summary:
            print(f"[{pid}] CMP  → {cmp_path}")

    return {
        "csv": str(out_csv),
        "pdf": str(out_pdf),
        "html": str(out_html),
        "compare": str(cmp_path) if cmp_summary else None,
        "summary": cmp_summary,
        "n_timepoints": len(auto_df),
        "technical_state": technical_state,
        "compensation": str(layout.qc / "compensation_diagnostics.json"),
        "acquisition_qc": str(layout.qc / "acquisition_qc.csv"),
        "acquisition_cleaning_summary": str(cleaning_summary_path),
        "acquisition_cleaning_comparison": str(cleaning_comparison_path),
        "acquisition_cleaning_report": str(cleaning_report_path),
        "time_cleaning_excluded_events": str(excluded_event_path),
        "time_cleaned_multilineage": str(cleaned_multilineage_path),
        "temporal_cell_type_summary": str(temporal_summary_path),
        "longitudinal_pdf": str(longitudinal_pdf),
        "report_page_index": str(layout.reports / "report_page_index.csv"),
        "lymph_density_geometry": str(layout.tables / "lymph_density_geometry.csv"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="CAR-NK flow pipeline — FSA×SSA first, reference-anchored")
    ap.add_argument("--patient-dir", help="patients/UPN## folder")
    ap.add_argument("--manifest", help="CSV with patient_dir column")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--manual-ref", default=None,
                    help="Manual gating CSV for validation / anchor build")
    ap.add_argument("--anchor", default=None,
                    help="Operator anchor JSON (Nature-style locked scatter)")
    ap.add_argument("--build-anchor", action="store_true",
                    help="Calibrate Baseline to manual ref, write operator anchor, then run")
    ap.add_argument("--subsample", type=int, default=200_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not args.patient_dir and not args.manifest:
        ap.error("Provide --patient-dir or --manifest")

    dirs = []
    if args.manifest:
        man = pd.read_csv(args.manifest)
        dirs = man["patient_dir"].tolist()
    if args.patient_dir:
        dirs = [args.patient_dir]

    results = []
    for d in dirs:
        print(f"\n{'='*60}\nProcessing {d}\n{'='*60}")
        results.append(process_patient(
            d, out_dir=args.out_dir, manual_ref=args.manual_ref,
            subsample=args.subsample, verbose=not args.quiet,
            anchor_path=args.anchor, build_anchor=args.build_anchor,
        ))
    return results


if __name__ == "__main__":
    main()
