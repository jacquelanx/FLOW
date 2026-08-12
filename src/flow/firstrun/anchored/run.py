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
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .fcs_io import load_xform, resolve_channels, has_full_panel
from .gates import calibrate_patient, gate_with, percentages
from .calibrate import (
    classify_file, car_cut_from_controls,
    donor_cut, adaptive_car_from_donor, nk_hla_car, MIN_PARENT, MIN_POS,
)
from .qc_plots import write_compact_qc_report
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


def process_patient(
    patient_dir: str | Path,
    out_dir: str | Path = "output",
    manual_ref: str | Path | None = None,
    subsample: int = 200_000,
    verbose: bool = True,
    anchor_path: str | Path | None = None,
    build_anchor: bool = False,
):
    patient_dir = Path(patient_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Nature-style operator anchor
    if build_anchor:
        if not manual_ref or not Path(manual_ref).exists():
            raise FileNotFoundError("--build-anchor requires --manual-ref")
        build_operator_anchor(patient_dir, manual_ref, subsample=subsample)
    anchor = None
    apath = find_anchor(patient_dir, anchor_path)
    if apath is not None:
        anchor = load_anchor(apath)
        if verbose:
            print(f"[{pid}] operator anchor: {apath.name}  "
                  f"(SSC_hi={anchor['locked']['ssc_hi']:.3f}, "
                  f"cal MAE={anchor.get('calibration_mae_pp')} pp)", flush=True)

    controls = {}
    control_src = {}
    loaded = []  # (fname, df, ntot)
    n_comp = 0
    for f in sorted(fcs_dir.iterdir()):
        if not f.suffix.lower() == ".fcs":
            continue
        kind = classify_file(f.name)
        if kind == "comp":
            # Instrument compensation control: full fluorescence panel but not a specimen.
            # Skipped before loading so it never reaches the gates or the median cuts.
            n_comp += 1
            continue
        try:
            df, ntot = load_xform(f, subsample=subsample)
        except Exception as e:
            if verbose:
                print(f"  !!  {f.name}: load failed: {e}", flush=True)
            continue
        if not has_full_panel(df, ch):
            if verbose:
                print(f"  --  {f.name}: skipped (incomplete panel)", flush=True)
            continue
        if kind != "timepoint":
            # One control of each kind is used for the study-wide donor/CAR cut. When a
            # dataset ships one per acquisition folder, say which one won.
            if verbose:
                prev = control_src.get(kind)
                note = f" (replaces {prev})" if prev else ""
                print(f"  [ctrl] {f.name} → {kind}{note}", flush=True)
            controls[kind] = df
            control_src[kind] = f.name
        else:
            loaded.append((f.name, df, ntot))

    if verbose and n_comp:
        print(f"[{pid}] skipped {n_comp} compensation control file(s)", flush=True)
    if not loaded:
        raise RuntimeError(f"No timepoint FCS files found in {fcs_dir}")

    pairs = [(fn, df) for fn, df, _ in loaded]

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

    if anchor and anchor.get("car_cut") is not None:
        car_cut = float(anchor["car_cut"])
        cnotes = {"car_source": f"anchor-fitted CAR cut={car_cut:.0f} "
                  f"(MAE {anchor.get('car_cut_mae_pp')} pp vs manual)"}
    else:
        car_cut, cnotes = car_cut_from_controls(controls, med, ch)
        if car_cut is None:
            car_cut = adaptive_car_from_donor(pairs, med, ch, hla_cut, hla_dim)
            cnotes["car_source"] = (f"adaptive GMM={car_cut:.0f}" if car_cut else "none")

    if verbose:
        print(f"[{pid}] donor={hla_spec or 'none'} cut={('%.0f'%hla_cut) if hla_cut else 'NA'}  "
              f"CAR cut={('%.0f'%car_cut) if car_cut else 'NA'}", flush=True)
        for k, v in {**dnotes, **cnotes}.items():
            print(f"        {k}: {v}", flush=True)

    out_csv = out_dir / f"Multilineage_SSA_{pid}.csv"
    out_pdf = out_dir / f"QC_Report_{pid}.pdf"
    out_pdf_legacy = out_dir / f"QC_Report_{pid}_SSA.pdf"
    out_html = out_dir / f"QC_Report_{pid}.html"
    out_html_legacy = out_dir / f"QC_Report_{pid}_SSA.html"

    rows = []
    retention_rows = []
    scatter_policy = "soft_lock" if anchor else "density"
    gate_cache = []  # defer per-TP pages until after cover/retention
    for fn, df, ntot in loaded:
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
        }
        if "CD4" in m:
            row["n_CD4"] = int(m["CD4"].sum())
            row["n_CD8"] = int(m["CD8"].sum())
        rows.append(row)
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

    cmp_summary = None
    cmp = None
    cmp_path = out_dir / f"Compare_manual_{pid}.csv"
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

    with PdfPages(out_pdf) as pdf:
        write_compact_qc_report(
            pdf, pid, gate_cache,
            cmp_summary=cmp_summary, compare_df=cmp,
            scatter_policy=scatter_policy,
        )

    # Keep legacy filenames as copies for downstream scripts
    try:
        import shutil
        shutil.copy2(out_pdf, out_pdf_legacy)
    except Exception:
        pass

    write_report(auto_df, out_html, patient_name, pid, cmp_summary)
    try:
        import shutil
        shutil.copy2(out_html, out_html_legacy)
    except Exception:
        pass

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
