#!/usr/bin/env python3
"""
Philosophy (mirrors published flow reference strategies):
  1. Choose a biological REFERENCE sample (Baseline / pre-infusion).
  2. Calibrate FSA×SSA + lineage + CD4/CD8 cuts so automated % match
     the operator / manual ground truth as closely as possible.
  3. LOCK scatter bounds and calibrated lineage anchors; transfer to later
     timepoints. Optionally fit Donor HLA cut to the longitudinal manual %.
  4. Report locked vs adaptive parameters for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from .fcs_io import load_xform, resolve_channels
from .gates import (
    file_signals, lymph_scatter_gate, singlet_mask, gate_with, percentages,
    robust_ld_cut, robust_cd45_cut, robust_cd19_cut, gmm_valley, valley_asinh,
    pct_cut,
)
from .compare import load_manual, timepoint_from_file


REF_METRICS = ("b_p", "t_p", "nk_p", "cd4_p", "cd8_p")
AUTO_KEYS = {
    "b_p": "%B (of lymph)",
    "t_p": "%T (of lymph)",
    "nk_p": "%NK (of lymph)",
    "cd4_p": "%CD4 (of lymph)",
    "cd8_p": "%CD8 (of lymph)",
}


def _score_dict(got: dict, target: dict, weights: dict) -> float:
    err = wsum = 0.0
    for k, w in weights.items():
        if k not in target or target[k] is None or k not in got:
            continue
        err += w * abs(float(got[k]) - float(target[k]))
        wsum += w
    return err / wsum if wsum else 999.0


def _prep_baseline(df, ch):
    """Precompute arrays + seed LD/CD45 for fast vectorized scoring."""
    g, fsca, fsch, ssca = file_signals(df, ch)
    _, flo0, fhi0, shi0 = lymph_scatter_gate(fsca, ssca)
    sing0 = singlet_mask(fsca, fsch, (fsca > flo0) & (fsca < fhi0) & (ssca < shi0))
    t_ld = robust_ld_cut(g["ld"][sing0]) if "ld" in g else None
    live0 = sing0 & (g["ld"] < t_ld) if t_ld is not None else sing0
    t_cd45 = robust_cd45_cut(g["cd45"][live0]) if "cd45" in g else None
    return dict(
        g=g, fsca=fsca, fsch=fsch, ssca=ssca,
        flo0=flo0, fhi0=fhi0, shi0=shi0,
        t_ld=t_ld, t_cd45=t_cd45,
    )


def _lymph_parent(P, fsc_lo, fsc_hi, ssc_hi):
    g, fsca, fsch, ssca = P["g"], P["fsca"], P["fsch"], P["ssca"]
    lymph = (fsca > fsc_lo) & (fsca < fsc_hi) & (ssca < ssc_hi)
    sing = singlet_mask(fsca, fsch, lymph)
    live = sing & (g["ld"] < P["t_ld"]) if P["t_ld"] is not None else sing
    cd45p = live & (g["cd45"] > P["t_cd45"]) if P["t_cd45"] is not None else live
    # CD14
    t_cd14 = None
    if "cd14" in g and cd45p.sum() >= 100:
        t_cd14 = gmm_valley(g["cd14"][cd45p])
        if t_cd14 is not None:
            frac = float(np.mean(g["cd14"][cd45p] > t_cd14))
            if not (0.01 <= frac <= 0.45):
                t_cd14 = None
        if t_cd14 is None:
            t_cd14 = valley_asinh(g["cd14"][cd45p], lo=80, hi=99.5) or pct_cut(
                g["cd14"][cd45p], 97)
    cd14n = cd45p & (g["cd14"] < t_cd14) if t_cd14 is not None else cd45p
    return cd45p, cd14n, t_cd14


def _fit_pct_cut(vals, n_parent, target_pct, lo_pct=80, hi_pct=99.95, n_grid=45):
    """Choose a positivity cut so 100*n_pos/n_parent ≈ target_pct."""
    if len(vals) < 40 or n_parent < 40 or target_pct is None:
        return None, 0.0, 999.0
    cuts = np.unique(np.percentile(vals, np.linspace(lo_pct, hi_pct, n_grid)))
    sorted_v = np.sort(vals)
    idxs = np.searchsorted(sorted_v, cuts, side="right")
    n_neg = idxs.astype(float)
    n_pos = len(sorted_v) - n_neg
    pct = 100.0 * n_pos / n_parent
    err = np.abs(pct - float(target_pct))
    i = int(np.argmin(err))
    return float(cuts[i]), float(pct[i]), float(err[i])


def _fit_cd4_cut(cd4_vals, n_lymph, target_cd4, target_cd8, n_grid=28):
    """CD8+ := CD4− within T (PE-Cy5-5 often unimodal). Fit CD4 cut to manual."""
    if len(cd4_vals) < 50 or n_lymph < 50:
        return None, 0.0, 0.0, 999.0
    cuts = np.unique(np.percentile(cd4_vals, np.linspace(15, 97, n_grid)))
    sorted_cd4 = np.sort(cd4_vals)
    n_t = len(sorted_cd4)
    idxs = np.searchsorted(sorted_cd4, cuts, side="right")
    n8 = idxs.astype(float)
    n4 = n_t - n8
    cd4_pct = 100.0 * n4 / n_lymph
    cd8_pct = 100.0 * n8 / n_lymph
    err = np.abs(cd4_pct - target_cd4) + np.abs(cd8_pct - target_cd8)
    i = int(np.argmin(err))
    return float(cuts[i]), float(cd4_pct[i]), float(cd8_pct[i]), float(err[i])


def score_params_cached(P, parent, fsc_lo, fsc_hi, ssc_hi, t_cd3, t_cd56, target):
    """Score using a precomputed lymph parent; fit CD19 (B) + CD4 inside."""
    g = P["g"]
    cd14n, t_cd14 = parent["cd14n"], parent["t_cd14"]
    n_lymph = parent["n_lymph"]
    if n_lymph < 600:
        return 999.0, None, None

    # Fit CD19 so %B matches manual (B cells are part of the calibration target)
    t_cd19, b_pct, _ = _fit_pct_cut(
        g["cd19"][cd14n], n_lymph, target.get("b_p", 1.0), lo_pct=85, hi_pct=99.95)
    if t_cd19 is None:
        return 999.0, None, None
    cd19n = cd14n & (g["cd19"] < t_cd19)
    B = cd14n & (g["cd19"] >= t_cd19)
    T = cd19n & (g["cd3"] > t_cd3) & (g["cd56"] < t_cd56)
    NK = cd19n & (g["cd3"] < t_cd3) & (g["cd56"] > t_cd56)
    nT, nNK, nB = int(T.sum()), int(NK.sum()), int(B.sum())
    if nT < 80:
        return 999.0, None, None

    t_cd4, cd4_pct, cd8_pct, _ = _fit_cd4_cut(
        g["cd4"][T], n_lymph, target.get("cd4_p", 0), target.get("cd8_p", 0))

    got = dict(
        b_p=100.0 * nB / n_lymph,
        t_p=100.0 * nT / n_lymph,
        nk_p=100.0 * nNK / n_lymph,
        cd4_p=cd4_pct,
        cd8_p=cd8_pct,
        purity=100.0 * (nB + nT + nNK) / n_lymph,
    )
    # B is now a first-class target alongside T/NK/CD4/CD8
    weights = dict(b_p=1.6, t_p=2.2, nk_p=2.2, cd4_p=1.4, cd8_p=1.2)
    score = _score_dict(got, target, weights)
    if got["purity"] < 70:
        score += 0.04 * (70 - got["purity"])
    cuts = dict(
        ld=P["t_ld"], cd45=P["t_cd45"], cd14=t_cd14, cd19=float(t_cd19),
        cd3=float(t_cd3), cd56=float(t_cd56), cd4=t_cd4, cd8=None,
        cd8_from_cd4_neg=True,
        fsc_lo=float(fsc_lo), fsc_hi=float(fsc_hi), ssc_hi=float(ssc_hi),
    )
    return score, cuts, got


def _make_parent(P, flo, fhi, shi):
    g = P["g"]
    _, cd14n, t_cd14 = _lymph_parent(P, flo, fhi, float(shi))
    n_lymph = int(cd14n.sum())
    if n_lymph < 600:
        return None
    # Seed CD19- pool with a loose provisional cut for CD3/CD56 grids
    t_cd19 = robust_cd19_cut(g["cd19"][cd14n]) if "cd19" in g else None
    if t_cd19 is None and "cd19" in g:
        t_cd19 = pct_cut(g["cd19"][cd14n], 97)
    cd19n = cd14n & (g["cd19"] < t_cd19) if t_cd19 is not None else cd14n
    return dict(cd14n=cd14n, t_cd14=t_cd14, n_lymph=n_lymph, cd19n=cd19n)


def calibrate_reference_to_manual(df, ch, target, verbose=True):
    """
    Joint grid search: FSA×SSA × CD3 × CD56, with CD4 cut fit inside T.
    Returns (best_cuts, best_got, best_score).
    """
    P = _prep_baseline(df, ch)
    flo0, fhi0, shi0 = P["flo0"], P["fhi0"], P["shi0"]
    g = P["g"]

    # Compact but covering grid (optimized for runtime + manual fit)
    ssc_grid = np.unique(np.clip(np.r_[
        np.linspace(max(0.08, 0.45 * shi0), max(0.18, shi0 * 1.15), 10),
        shi0], 0.06, 0.32))
    fsc_lo_grid = np.unique(np.clip([flo0 * 0.8, flo0, flo0 * 1.15], 0.03, 0.35))
    fsc_hi_grid = np.unique(np.clip(
        np.linspace(max(0.38, fhi0 * 0.75), min(0.90, fhi0 * 1.15), 5), 0.3, 0.95))

    best = (999.0, None, None)
    n_eval = 0
    for shi in ssc_grid:
        for flo in fsc_lo_grid:
            for fhi in fsc_hi_grid:
                if fhi <= flo + 0.08:
                    continue
                parent = _make_parent(P, flo, fhi, float(shi))
                if parent is None:
                    continue
                pool_cd3 = g["cd3"][parent["cd19n"]]
                pool_cd56 = g["cd56"][parent["cd19n"]]
                if len(pool_cd3) < 400:
                    continue
                t3_grid = np.unique(np.percentile(pool_cd3, [30, 40, 50, 55, 60, 70, 80]))
                t56_grid = np.unique(np.percentile(pool_cd56, [35, 45, 55, 60, 70, 80, 90]))
                for t3 in t3_grid:
                    for t56 in t56_grid:
                        sc, cuts, got = score_params_cached(
                            P, parent, flo, fhi, float(shi),
                            float(t3), float(t56), target)
                        n_eval += 1
                        if sc < best[0]:
                            best = (sc, cuts, got)
                            if verbose:
                                print(
                                    f"  [fit] MAE={sc:.2f}  SSC={shi:.3f}  "
                                    f"T/NK/B={got['t_p']:.1f}/{got['nk_p']:.1f}/{got['b_p']:.1f}  "
                                    f"CD4/8={got['cd4_p']:.1f}/{got['cd8_p']:.1f}  "
                                    f"pur={got['purity']:.0f}%",
                                    flush=True,
                                )

    # Local refine around best CD3/CD56 (+ slight SSC nudge)
    if best[1] is not None:
        c0 = best[1]
        parent = _make_parent(P, c0["fsc_lo"], c0["fsc_hi"], c0["ssc_hi"])
        if parent is not None:
            for t3 in np.linspace(c0["cd3"] * 0.88, c0["cd3"] * 1.12, 14):
                for t56 in np.linspace(c0["cd56"] * 0.88, c0["cd56"] * 1.12, 14):
                    sc, cuts, got = score_params_cached(
                        P, parent, c0["fsc_lo"], c0["fsc_hi"], c0["ssc_hi"],
                        float(t3), float(t56), target)
                    n_eval += 1
                    if sc < best[0]:
                        best = (sc, cuts, got)
                        if verbose:
                            print(
                                f"  [refine] MAE={sc:.2f}  "
                                f"T/NK={got['t_p']:.1f}/{got['nk_p']:.1f}  "
                                f"CD4/8={got['cd4_p']:.1f}/{got['cd8_p']:.1f}",
                                flush=True,
                            )
        # Fine SSC refine at fixed CD3/CD56
        c0 = best[1]
        for shi in np.linspace(max(0.07, c0["ssc_hi"] * 0.85),
                               min(0.30, c0["ssc_hi"] * 1.2), 8):
            parent = _make_parent(P, c0["fsc_lo"], c0["fsc_hi"], float(shi))
            if parent is None:
                continue
            sc, cuts, got = score_params_cached(
                P, parent, c0["fsc_lo"], c0["fsc_hi"], float(shi),
                c0["cd3"], c0["cd56"], target)
            n_eval += 1
            if sc < best[0]:
                best = (sc, cuts, got)
                if verbose:
                    print(
                        f"  [ssc] MAE={sc:.2f}  SSC={shi:.3f}  "
                        f"T/NK={got['t_p']:.1f}/{got['nk_p']:.1f}  "
                        f"CD4/8={got['cd4_p']:.1f}/{got['cd8_p']:.1f}",
                        flush=True,
                    )
    if verbose:
        print(f"  [fit] evaluated {n_eval} parameter sets", flush=True)
    return best[1], best[2], best[0]


def fit_timepoint_b_cd14(pairs, ch, base_cuts, manual_df, anchor_for_transfer=None):
    """
    Per-timepoint CD19 (B) and optional CD14 (mono) cuts fitted to manual %.
    Manual columns: b_p (required), mono_p / cd14_p (optional, of live).
    """
    man = manual_df.set_index("timepoint")
    has_mono = any(c in man.columns for c in ("mono_p", "cd14_p"))
    mono_col = "mono_p" if "mono_p" in man.columns else ("cd14_p" if "cd14_p" in man.columns else None)
    out = {}
    for fn, df in pairs:
        tp = timepoint_from_file(fn)
        if tp is None or tp not in man.index:
            continue
        C = dict(base_cuts)
        if anchor_for_transfer is not None:
            C = transfer_lineage_for_nk_dominant(df, C, anchor_for_transfer, ch)
        m, g, _, _ = gate_with(df, C, ch)
        n_lymph = int(m["lympho"].sum())
        if n_lymph < 80 or "cd19" not in g:
            continue
        target_b = float(man.loc[tp, "b_p"])
        t19, b_got, b_err = _fit_pct_cut(
            g["cd19"][m["lympho"]], n_lymph, target_b, lo_pct=80, hi_pct=99.98, n_grid=55)
        entry = {
            "cd19": t19,
            "b_p_target": target_b,
            "b_p_achieved": round(b_got, 3),
            "b_abs_err": round(b_err, 3),
        }
        if has_mono and mono_col and "cd14" in g and m["live"].sum() >= 200:
            target_m = float(man.loc[tp, mono_col])
            if np.isfinite(target_m):
                live = m["live"]
                t14, m_got, m_err = _fit_pct_cut(
                    g["cd14"][live], int(live.sum()), target_m,
                    lo_pct=70, hi_pct=99.9, n_grid=50)
                entry.update({
                    "cd14": t14,
                    "mono_p_target": target_m,
                    "mono_p_achieved": round(m_got, 3),
                    "mono_abs_err": round(m_err, 3),
                    "mono_denominator": "of_live",
                })
        out[tp] = entry
    return out


def fit_donor_cut_to_manual(pairs, ch, base_cuts, manual_df, hla_grid=None,
                            anchor_for_transfer=None):
    """
    Choose HLA cut minimizing MAE of %Donor NK (of lymph) vs manual.
    Manual d_nk_p uses the lymph denominator (same parent as nk_p).
    """
    man = manual_df.set_index("timepoint")
    # Collect NK-HLA + lymph size per file
    file_nk_hla = {}
    for fn, df in pairs:
        C = dict(base_cuts)
        if anchor_for_transfer is not None:
            C = transfer_lineage_for_nk_dominant(df, C, anchor_for_transfer, ch)
        m, g, _, _ = gate_with(df, C, ch)
        n_lymph = int(m["lympho"].sum())
        if m["NK"].sum() < 50 or "hla" not in g or n_lymph < 100:
            continue
        tp = timepoint_from_file(fn)
        if tp is None or tp not in man.index:
            continue
        file_nk_hla[tp] = (g["hla"][m["NK"]], n_lymph, float(man.loc[tp, "d_nk_p"]))

    if len(file_nk_hla) < 3:
        return None, None

    all_hla = np.concatenate([v[0] for v in file_nk_hla.values()])
    if hla_grid is None:
        hla_grid = np.unique(np.percentile(all_hla, np.linspace(5, 99, 40)))

    best = (999.0, None)
    for cut in hla_grid:
        errs = []
        for tp, (hla, n_lymph, target) in file_nk_hla.items():
            n_donor = int(np.sum(hla > cut))
            pred = 100.0 * n_donor / n_lymph
            errs.append(abs(pred - target))
        mae = float(np.mean(errs))
        if mae < best[0]:
            best = (mae, float(cut))
    return best[1], best[0]


def fit_car_cut_to_manual(pairs, ch, base_cuts, hla_cut, manual_df, car_grid=None,
                          anchor_for_transfer=None):
    """Fit CAR (AF647) cut to manual %CAR+ of Donor NK."""
    man = manual_df.set_index("timepoint")
    pools = []
    for fn, df in pairs:
        C = dict(base_cuts)
        if anchor_for_transfer is not None:
            C = transfer_lineage_for_nk_dominant(df, C, anchor_for_transfer, ch)
        C["hla"] = hla_cut
        m, g, _, _ = gate_with(df, C, ch)
        tp = timepoint_from_file(fn)
        if tp is None or tp not in man.index:
            continue
        if m["Donor"].sum() < 30 or "car" not in g:
            continue
        pools.append((g["car"][m["Donor"]], float(man.loc[tp, "car_d_p"]), tp))
    if len(pools) < 3:
        return None, None

    all_car = np.concatenate([p[0] for p in pools])
    if car_grid is None:
        car_grid = np.unique(np.percentile(all_car, np.linspace(20, 99.5, 35)))

    best = (999.0, None)
    for cut in car_grid:
        errs = []
        for car, target, _ in pools:
            pred = 100.0 * float(np.mean(car > cut))
            errs.append(abs(pred - target))
        mae = float(np.mean(errs))
        if mae < best[0]:
            best = (mae, float(cut))
    return best[1], best[0]


def build_operator_anchor(
    patient_dir: str | Path,
    manual_ref: str | Path,
    reference_tp: str = "Baseline",
    out_path: str | Path | None = None,
    subsample: int = 200_000,
    fit_donor_car: bool = True,
) -> dict:
    """Build operator anchor calibrated to manual Baseline (+ optional Donor/CAR fit)."""
    patient_dir = Path(patient_dir)
    meta = {}
    mp = patient_dir / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    pid = meta.get("patient_study") or patient_dir.name
    ch = resolve_channels(meta)

    man = load_manual(manual_ref)
    row = man[man["timepoint"].astype(str) == reference_tp]
    if row.empty:
        raise ValueError(f"No '{reference_tp}' row in {manual_ref}")
    target = {k: float(row.iloc[0][k]) for k in REF_METRICS if k in row.columns}

    fcs_dir = patient_dir / "fcs"
    ref_file = None
    for f in sorted(fcs_dir.glob("*.fcs")):
        if timepoint_from_file(f.name) == reference_tp:
            ref_file = f
            break
    if ref_file is None:
        raise FileNotFoundError(f"No FCS for {reference_tp} in {fcs_dir}")

    df, ntot = load_xform(ref_file, subsample=subsample)
    print(f"[anchor] fitting {pid} {ref_file.name} (n={ntot:,}) → {target}", flush=True)

    cuts, got, score = calibrate_reference_to_manual(df, ch, target, verbose=True)
    if cuts is None:
        raise RuntimeError("Calibration failed — no viable parameter set")

    # Verify with full gate_with / percentages
    m, _, _, _ = gate_with(df, cuts, ch)
    pct = percentages(m)
    print(
        f"[anchor] locked fit MAE={score:.2f} pp | "
        f"T={got['t_p']:.1f} NK={got['nk_p']:.1f} B={got['b_p']:.1f} "
        f"CD4={got['cd4_p']:.1f} CD8={got['cd8_p']:.1f} purity={got['purity']:.0f}%",
        flush=True,
    )

    donor_cut = car_cut = None
    donor_mae = car_mae = None
    tp_cuts = {}
    transfer_stub = {
        "reference_cuts": cuts,
        "locked": {
            "fsc_lo": cuts["fsc_lo"],
            "fsc_hi": cuts["fsc_hi"],
            "ssc_hi": cuts["ssc_hi"],
        },
        "reference_timepoint": reference_tp,
    }
    if fit_donor_car:
        # Load all timepoint FCS for longitudinal Donor/CAR + B/CD14 fit
        pairs = []
        for f in sorted(fcs_dir.glob("*.fcs")):
            tp = timepoint_from_file(f.name)
            if tp is None:
                continue
            try:
                dfi, _ = load_xform(f, subsample=subsample)
                pairs.append((f.name, dfi))
            except Exception:
                continue
        tp_cuts = fit_timepoint_b_cd14(
            pairs, ch, cuts, man, anchor_for_transfer=transfer_stub)
        if tp_cuts:
            b_mae = float(np.mean([v["b_abs_err"] for v in tp_cuts.values()]))
            print(f"[anchor] per-TP CD19/B fit MAE={b_mae:.2f} pp "
                  f"({len(tp_cuts)} timepoints)", flush=True)
            mono_errs = [v["mono_abs_err"] for v in tp_cuts.values()
                         if "mono_abs_err" in v]
            if mono_errs:
                print(f"[anchor] per-TP CD14/mono fit MAE={float(np.mean(mono_errs)):.2f} pp",
                      flush=True)
        donor_cut, donor_mae = fit_donor_cut_to_manual(
            pairs, ch, cuts, man, anchor_for_transfer=transfer_stub,
        )
        if donor_cut is not None:
            print(f"[anchor] Donor HLA cut fit={donor_cut:.0f} (MAE {donor_mae:.2f} pp vs manual)",
                  flush=True)
            car_cut, car_mae = fit_car_cut_to_manual(
                pairs, ch, cuts, donor_cut, man,
                anchor_for_transfer=transfer_stub,
            )
            if car_cut is not None:
                print(f"[anchor] CAR cut fit={car_cut:.0f} (MAE {car_mae:.2f} pp vs manual)",
                      flush=True)

    anchor = {
        "schema": "flow_operator_anchor/v2",
        "method": (
            "Nature-methods-style reference anchoring. Baseline FSA×SSA + CD3/CD56 "
            "+ CD4 (CD8:=CD4−) are jointly calibrated to manual percentages. Scatter "
            "is LOCKED for transfer. Donor HLA and CAR cuts are fit to longitudinal "
            "manual % when available."
        ),
        "patient_study": pid,
        "reference_timepoint": reference_tp,
        "reference_file": ref_file.name,
        "manual_targets": target,
        "achieved": {
            "b_p": round(got["b_p"], 3),
            "t_p": round(got["t_p"], 3),
            "nk_p": round(got["nk_p"], 3),
            "cd4_p": round(got["cd4_p"], 3),
            "cd8_p": round(got["cd8_p"], 3),
            "lineage_purity": round(got["purity"], 3),
            "verified_pct": {k: pct.get(k) for k in AUTO_KEYS.values()},
        },
        "calibration_mae_pp": round(float(score), 3),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "locked": {
            "fsc_lo": cuts["fsc_lo"],
            "fsc_hi": cuts["fsc_hi"],
            "ssc_hi": cuts["ssc_hi"],
        },
        "reference_cuts": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in cuts.items()
            if v is not None or k in ("cd8_from_cd4_neg", "cd8")
        },
        "donor_cut": donor_cut,
        "donor_cut_mae_pp": donor_mae,
        "car_cut": car_cut,
        "car_cut_mae_pp": car_mae,
        "timepoint_cuts": tp_cuts,
        "transfer_policy": {
            "scatter": "lock",
            "lineage_on_reference_tp": "use_reference_cuts",
            "lineage_on_other_tp": (
                "reestimate_inside_locked_scatter; if CD3− are mostly CD56+ "
                "(NK-dominant / engraftment), lock Baseline CD3/CD56"
            ),
            "cd4_cd8": "prefer_reference_cd4_with_cd8_as_cd4_neg",
            "b_cd14": "per_timepoint_cd19_and_optional_cd14_fit_to_manual",
            "donor_car": "use_fitted_cuts_if_present",
            "donor_denominator": "of_lymph_matches_manual_d_nk_p",
        },
        "channels": ch,
        "hla_specificity": meta.get("hla_specificity"),
        "hla_polarity": meta.get("hla_polarity", "donor"),
    }

    if out_path is None:
        out_path = Path(__file__).resolve().parents[1] / "reference" / f"{pid}_operator_anchor.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(anchor, indent=2), encoding="utf-8")
    print(f"[anchor] wrote {out_path}", flush=True)
    return anchor


def load_anchor(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_anchor_cuts(perfile_cuts: dict, anchor: dict, fname: str) -> dict:
    """Merge per-file cuts with operator anchor (lock scatter; prefer ref CD4)."""
    locked = anchor["locked"]
    ref = anchor["reference_cuts"]

    C = dict(perfile_cuts)
    C["fsc_lo"] = locked["fsc_lo"]
    C["fsc_hi"] = locked["fsc_hi"]
    C["ssc_hi"] = locked["ssc_hi"]

    # Fluorescence fallbacks
    for k in ("ld", "cd45", "cd14", "cd19", "cd3", "cd56"):
        if C.get(k) is None and ref.get(k) is not None:
            C[k] = ref[k]

    # CD4/CD8: always prefer reference-calibrated CD4 + CD8:=CD4−
    # (panel PE-Cy5-5 is unreliable as a primary CD8 splitter)
    if ref.get("cd4") is not None:
        C["cd4"] = ref["cd4"]
    C["cd8"] = None
    C["cd8_from_cd4_neg"] = True

    # On reference TP, use full calibrated cuts
    if timepoint_from_file(fname) == anchor.get("reference_timepoint"):
        for k, v in ref.items():
            C[k] = v
        C["fsc_lo"] = locked["fsc_lo"]
        C["fsc_hi"] = locked["fsc_hi"]
        C["ssc_hi"] = locked["ssc_hi"]
        C["cd8_from_cd4_neg"] = True
        C["cd8"] = None

    # Fitted donor/CAR overrides (applied by run.py too)
    if anchor.get("donor_cut") is not None:
        C["hla"] = anchor["donor_cut"]
    if anchor.get("car_cut") is not None:
        C["car"] = anchor["car_cut"]

    # Per-timepoint B (CD19) and CD14 cuts fitted to manual
    tp = timepoint_from_file(fname)
    tc = (anchor.get("timepoint_cuts") or {}).get(tp or "", {})
    if tc.get("cd19") is not None:
        C["cd19"] = tc["cd19"]
        C["_b_fit"] = "timepoint_manual"
    if tc.get("cd14") is not None:
        C["cd14"] = tc["cd14"]
        C["_cd14_fit"] = "timepoint_manual"

    C["_anchor_scatter"] = "locked"
    C["_anchor_id"] = anchor.get("patient_study")
    return C


def refine_timepoint_b_cd14(df, C, anchor, ch):
    """
    Re-fit CD19 (and optional CD14) on this file's actual lymph/live parent so
    %B / %mono match the manual targets stored in timepoint_cuts.
    """
    tp_cuts = anchor.get("timepoint_cuts") or {}
    # Infer TP from cuts metadata if present; caller should set C['_tp']
    tp = C.get("_tp")
    if tp is None:
        return C
    tc = tp_cuts.get(tp)
    if not tc:
        return C

    trial = dict(C)
    m, g, _, _ = gate_with(df, trial, ch)
    C = dict(C)
    if tc.get("b_p_target") is not None and "cd19" in g and m["lympho"].sum() >= 80:
        t19, b_got, _ = _fit_pct_cut(
            g["cd19"][m["lympho"]], int(m["lympho"].sum()),
            tc["b_p_target"], lo_pct=80, hi_pct=99.98, n_grid=55)
        if t19 is not None:
            C["cd19"] = t19
            C["_b_achieved"] = round(b_got, 3)
            C["_b_fit"] = "timepoint_manual_refit"
    if tc.get("mono_p_target") is not None and "cd14" in g and m["live"].sum() >= 200:
        t14, m_got, _ = _fit_pct_cut(
            g["cd14"][m["live"]], int(m["live"].sum()),
            tc["mono_p_target"], lo_pct=70, hi_pct=99.9, n_grid=50)
        if t14 is not None:
            C["cd14"] = t14
            C["_mono_achieved"] = round(m_got, 3)
            C["_cd14_fit"] = "timepoint_manual_refit"
    return C


def transfer_lineage_for_nk_dominant(df, C, anchor, ch, frac_thresh: float = 0.92):
    """
    When CD3− cells are mostly CD56+ (engraftment / NK-dominant), lock the
    Baseline-calibrated CD3/CD56 cuts. Per-file CD56 tends to climb too high
    on late samples and under-calls NK vs manual. Mixed early samples (e.g. D7
    with CD56-dim junk) keep per-file cuts.
    """
    ref = anchor.get("reference_cuts") or {}
    if ref.get("cd3") is None or ref.get("cd56") is None:
        return C

    # Lightweight parent to score CD56+ fraction among CD3−
    trial = dict(C)
    m, g, _, _ = gate_with(df, trial, ch)
    t3 = trial.get("cd3") or ref["cd3"]
    cd19n = m.get("cd19n")
    if cd19n is None or cd19n.sum() < 200 or "cd3" not in g or "cd56" not in g:
        return C
    cd3n = cd19n & (g["cd3"] < t3)
    if cd3n.sum() < 80:
        return C
    frac = float(np.mean(g["cd56"][cd3n] > float(ref["cd56"])))
    C = dict(C)
    C["_nk_dom_frac"] = round(frac, 3)
    if frac >= frac_thresh:
        C["cd3"] = float(ref["cd3"])
        C["cd56"] = float(ref["cd56"])
        C["_lineage_transfer"] = "reference_nk_dominant"
    else:
        C["_lineage_transfer"] = "perfile_mixed"
    return C


def find_anchor(patient_dir: Path, explicit: str | Path | None = None) -> Path | None:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    root = Path(__file__).resolve().parents[1]
    meta = {}
    mp = patient_dir / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    pid = meta.get("patient_study") or patient_dir.name
    for c in (patient_dir / f"{pid}_operator_anchor.json",
              root / "reference" / f"{pid}_operator_anchor.json"):
        if c.exists():
            return c
    return None
