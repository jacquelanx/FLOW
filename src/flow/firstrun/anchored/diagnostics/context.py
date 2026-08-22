"""Reconstruct the anchored pipeline's state so it can be audited — without touching it.

The diagnostics need the same objects ``run.process_patient`` builds internally: the per-file
cutoff dict actually applied, the gate masks, the event-level signals, and the per-file
valleys the anchoring discarded. None of that is written to disk, so this module rebuilds it
by replaying the exact same calls in the same order.

Replaying rather than refactoring ``run.py`` is a deliberate trade. The cost is a second FCS
load and a second gating pass. The benefit is that the first-run script stays byte-identical,
which is a hard requirement here.

Because a replay can silently drift from the thing it replays, ``ReproductionCheck``
re-derives every reported percentage and compares it against ``multilineage.csv``. The whole
pipeline is deterministic — ``load_xform`` subsamples with ``random_state=0``, the GMM uses
``random_state=0``, the density gate uses ``seed=0`` — so agreement should be EXACT to the
3 decimals the CSV carries. Any mismatch means this replay is not auditing what actually ran,
and that is reported as a flag rather than papered over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..calibrate import (
    MIN_PARENT,
    MIN_POS,
    adaptive_car_from_donor,
    car_cut_from_controls,
    classify_file,
    donor_cut,
    nk_hla_car,
)
from ..compare import is_baseline, is_pre_infusion, timepoint_from_file, timepoint_sort_key
from ..fcs_io import has_full_panel, load_xform, resolve_channels
from ..gates import calibrate_patient, gate_with, percentages
from ..reference_anchor import (
    apply_anchor_cuts,
    refine_timepoint_b_cd14,
    transfer_lineage_for_nk_dominant,
)
from .markers import MARKERS, POOL_CD19N_CD3NEG, MarkerSpec, resolve_polarity
from .thresholds import Thresholds

# Percentage columns compared against multilineage.csv by the reproduction check. Anything
# the pipeline reports and the diagnostics recompute belongs here.
_REPRO_METRICS = (
    "%Lymph_scatter (of total)",
    "%Live (of singlets)",
    "%CD45+ (of live)",
    "%Lymphocytes (of CD45+)",
    "%B (of lymph)",
    "%T (of lymph)",
    "%NK (of lymph)",
    "%Donor NK (of lymph)",
    "%CAR+ (of Donor NK)",
    "lineage_purity",
)
# The CSV stores percentages rounded to 3 decimals; anything above this is real divergence.
_REPRO_TOL_PP = 0.01


@dataclass
class SampleState:
    """Everything needed to audit one timepoint file."""

    filename: str
    timepoint: Optional[str]
    n_total_events: int          # event count reported by the instrument, pre-subsample
    df: Optional[pd.DataFrame]   # transform-space events (released after the tiers run)
    perfile_cuts: dict[str, Any]  # per-file valleys BEFORE anchoring (the counterfactual)
    cuts: dict[str, Any]          # the cutoff dict actually applied by the pipeline
    applied: dict[str, Any]       # cuts as returned by gate_with (resolved scatter bounds)
    masks: dict[str, np.ndarray]
    signals: dict[str, np.ndarray]
    scatter: tuple
    pct: dict[str, Any]
    is_reference: bool
    is_pre_infusion: bool

    def pool_mask(self, pool: str) -> Optional[np.ndarray]:
        """Boolean mask for a marker pool identifier, including the derived CD56 pool."""
        if pool == POOL_CD19N_CD3NEG:
            base = self.masks.get("cd19n")
            cd3 = self.signals.get("cd3")
            t3 = self.cuts.get("cd3")
            if base is None or cd3 is None or t3 is None:
                return None
            return base & (cd3 < float(t3))
        return self.masks.get(pool)

    def marker_values(self, spec: MarkerSpec, applied: bool = False) -> Optional[np.ndarray]:
        """Marker intensities inside the derivation pool (or the application pool)."""
        vals = self.signals.get(spec.key)
        if vals is None:
            return None
        mask = self.pool_mask(spec.applied_pool if applied else spec.pool)
        if mask is None:
            return None
        return vals[mask]

    def donor_side_mask(self, hla_dim: bool) -> Optional[np.ndarray]:
        """Mask of NK events on the DONOR side of the HLA cutoff.

        Mirrors ``gate_with``: donor is below the cutoff when the mismatch is dim-donor. Held
        here (rather than recomputed ad hoc) because getting the polarity backwards would
        invert the host-NK control check, which is the most load-bearing number in tier 4.
        """
        nk = self.masks.get("NK")
        hla = self.signals.get("hla")
        cut = self.cuts.get("hla")
        if nk is None or hla is None or cut is None:
            return None
        side = (hla < float(cut)) if hla_dim else (hla > float(cut))
        return nk & side

    def release(self) -> None:
        """Drop the event frame. Masks/signals are views into it, so they go too."""
        self.df = None


@dataclass
class ReproductionCheck:
    """Whether this replay reproduces the pipeline's own reported numbers."""

    checked: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    max_abs_delta_pp: Optional[float] = None
    source: Optional[str] = None
    available: bool = False

    @property
    def faithful(self) -> bool:
        return self.available and not self.mismatches


@dataclass
class DiagnosticsContext:
    """The replayed pipeline state, plus the config that produced it."""

    patient_dir: Path
    out_dir: Path
    plots_dir: Optional[Path]
    meta: dict[str, Any]
    patient_id: str
    channels: dict[str, str]
    anchor: dict[str, Any]
    reference_timepoint: str
    subsample: int
    thresholds: Thresholds
    hla_cut: Optional[float]
    car_cut: Optional[float]
    hla_dim: bool
    hla_specificity: str
    samples: list[SampleState] = field(default_factory=list)
    controls: dict[str, tuple[str, pd.DataFrame]] = field(default_factory=dict)
    reproduction: ReproductionCheck = field(default_factory=ReproductionCheck)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    # ---- lookups -------------------------------------------------------------
    @property
    def reference(self) -> Optional[SampleState]:
        for s in self.samples:
            if s.is_reference:
                return s
        return None

    def marker_specs(self) -> list[MarkerSpec]:
        """Marker specs with HLA polarity resolved, limited to cutoffs that actually exist."""
        out = []
        for spec in MARKERS:
            spec = resolve_polarity(spec, self.hla_dim)
            if self.cut_value(spec.key) is None:
                continue
            out.append(spec)
        return out

    def cut_value(self, key: str) -> Optional[float]:
        """The locked/unified cutoff for a key, as recorded by the anchor + run policy."""
        if key == "hla":
            return self.hla_cut
        if key == "car":
            return self.car_cut
        v = (self.anchor.get("reference_cuts") or {}).get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def timepoint_labels(self) -> list[str]:
        return [s.timepoint or s.filename for s in self.samples]

    def sample_for_timepoint(self, label: Optional[str]) -> Optional[SampleState]:
        """The replayed sample carrying ``label``, or None.

        Needed because tier 1 may audit a marker away from the reference (see
        ``foundation._audit_with_fallback``), and tier 2 then has to measure drift against the
        sample that audit was actually taken at rather than against a reference pool that was
        too small to carry it.
        """
        if label is None:
            return None
        for s in self.samples:
            if (s.timepoint or s.filename) == label:
                return s
        return None

    def release_events(self) -> None:
        for s in self.samples:
            s.release()
        self.controls = {}

    def time_channel(self, df: pd.DataFrame) -> Optional[str]:
        """Name of the acquisition Time column, if the instrument wrote one."""
        for c in df.columns:
            if str(c).strip().lower() == "time":
                return str(c)
        return None


# ── loading ────────────────────────────────────────────────────────────────────
def _load_meta(patient_dir: Path) -> dict:
    mp = patient_dir / "metadata.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            return {}
    return {}


def _order_key(meta_order: Optional[list[str]]):
    """Sort key reproducing run.py's timepoint ordering (flow.csv order, else chronological)."""
    if meta_order:
        idx = {tp: i for i, tp in enumerate(meta_order)}
        return lambda s: (idx.get(str(s.timepoint), 999), timepoint_sort_key(s.timepoint))
    return lambda s: timepoint_sort_key(s.timepoint)


def build_context(
    patient_dir: str | Path,
    out_dir: str | Path,
    anchor: dict[str, Any],
    plots_dir: str | Path | None = None,
    thresholds: Optional[Thresholds] = None,
    subsample: Optional[int] = None,
    verbose: bool = True,
) -> DiagnosticsContext:
    """Replay the anchored pipeline's per-file state for ``patient_dir``.

    ``anchor`` is the already-loaded ``operator_anchor.json``. The replay follows
    ``run.process_patient`` step for step: classify files, calibrate per-file valleys inside
    the locked scatter, resolve the donor/CAR cutoffs, then build each file's cutoff dict via
    ``apply_anchor_cuts`` -> ``transfer_lineage_for_nk_dominant`` -> ``refine_timepoint_b_cd14``.
    """
    patient_dir = Path(patient_dir)
    out_dir = Path(out_dir)
    meta = _load_meta(patient_dir)
    th = thresholds or Thresholds.from_metadata(meta)
    pid = meta.get("patient_study") or patient_dir.name
    ch = resolve_channels(meta)
    polarity = (meta.get("hla_polarity") or "donor").strip().lower()
    hla_spec = meta.get("hla_specificity") or ""
    reference_tp = str(anchor.get("reference_timepoint") or meta.get("reference_timepoint", "Baseline"))
    # Must match the pipeline's subsample or the replay stops reproducing it.
    sub = int(subsample if subsample is not None else meta.get("subsample", 200_000))

    fcs_dir = patient_dir / "fcs"
    if not fcs_dir.is_dir():
        raise FileNotFoundError(f"diagnostics: no fcs/ folder in {patient_dir}")

    ctx = DiagnosticsContext(
        patient_dir=patient_dir,
        out_dir=out_dir,
        plots_dir=Path(plots_dir) if plots_dir else None,
        meta=meta,
        patient_id=pid,
        channels=ch,
        anchor=anchor,
        reference_timepoint=reference_tp,
        subsample=sub,
        thresholds=th,
        hla_cut=None,
        car_cut=None,
        hla_dim=False,
        hla_specificity=hla_spec,
    )

    # ── 1. classify + load, exactly as run.py does ────────────────────────────
    loaded: list[tuple[str, pd.DataFrame, int]] = []
    controls: dict[str, tuple[str, pd.DataFrame]] = {}
    for f in sorted(fcs_dir.iterdir()):
        if f.suffix.lower() != ".fcs":
            continue
        kind = classify_file(f.name)
        if kind == "comp":
            continue  # instrument compensation control, not a specimen
        try:
            df, ntot = load_xform(f, subsample=sub)
        except Exception as e:
            ctx.skipped.append({"file": f.name, "reason": f"load failed: {e}"})
            continue
        if not has_full_panel(df, ch):
            ctx.skipped.append({"file": f.name, "reason": "incomplete panel"})
            continue
        if kind != "timepoint":
            controls[kind] = (f.name, df)
        else:
            loaded.append((f.name, df, ntot))

    if not loaded:
        raise RuntimeError(f"diagnostics: no timepoint FCS files in {fcs_dir}")
    ctx.controls = controls
    pairs = [(fn, df) for fn, df, _ in loaded]

    # ── 2. per-file valleys inside the locked scatter (run.py step) ───────────
    locked_scatter = anchor.get("locked")
    perfile, med = calibrate_patient(pairs, ch, ssc_cap=None, locked_scatter=locked_scatter)

    def _cuts_for_nk(fn: str) -> dict:
        pf = perfile[fn]
        C = {k: (pf[k] if pf.get(k) is not None else med.get(k)) for k in med}
        return apply_anchor_cuts(C, anchor, fn)

    # ── 3. donor HLA + CAR cutoffs (run.py step) ─────────────────────────────
    pre_pool, post_pool, base_pool = [], [], []
    for fn, df in pairs:
        Cnk = _cuts_for_nk(fn)
        hla_vals, _ = nk_hla_car(df, Cnk, ch)
        if is_pre_infusion(fn):
            pre_pool.append(hla_vals)
            if is_baseline(fn):
                base_pool.append(hla_vals)
        else:
            post_pool.append(hla_vals)
    pre_hla = np.concatenate(pre_pool) if pre_pool else None
    post_hla = np.concatenate(post_pool) if post_pool else None
    base_hla = np.concatenate(base_pool) if base_pool else None

    ctrl_meds: list[float] = []
    for key in ("ntnk", "cbmc", "car"):
        if key in controls:
            Cctrl = dict(med)
            if locked_scatter:
                Cctrl.update(locked_scatter)
            hla_vals, _ = nk_hla_car(controls[key][1], Cctrl, ch)
            if len(hla_vals) >= 100:
                ctrl_meds.append(float(np.median(hla_vals)))

    hla_dim = False
    if not str(hla_spec).strip():
        hla_cut = None
        ctx.notes.append("No donor HLA marker assigned in metadata — donor/CAR tiers limited.")
    elif anchor.get("donor_cut") is not None:
        hla_cut = float(anchor["donor_cut"])
    else:
        hla_cut, dnotes = donor_cut(pre_hla, post_hla, ctrl_meds, polarity, base_hla)
        hla_dim = bool(dnotes.pop("hla_dim", False))
        for v in dnotes.values():
            ctx.notes.append(f"donor cutoff: {v}")

    if anchor.get("car_cut") is not None:
        car_cut = float(anchor["car_cut"])
    else:
        car_cut, cnotes = car_cut_from_controls(controls_as_frames(controls), med, ch)
        if car_cut is None:
            car_cut = adaptive_car_from_donor(pairs, med, ch, hla_cut, hla_dim)
        for v in cnotes.values():
            ctx.notes.append(f"CAR cutoff: {v}")

    ctx.hla_cut = float(hla_cut) if hla_cut is not None else None
    ctx.car_cut = float(car_cut) if car_cut is not None else None
    ctx.hla_dim = bool(hla_dim)

    # ── 4. per-file cutoff dict + gating (run.py step) ────────────────────────
    for fn, df, ntot in loaded:
        pf = perfile[fn]
        C = {k: (pf[k] if pf.get(k) is not None else med.get(k)) for k in med}
        C["cd8_from_cd4_neg"] = bool(pf.get("cd8_from_cd4_neg") or False)
        C = apply_anchor_cuts(C, anchor, fn)
        C = transfer_lineage_for_nk_dominant(df, C, anchor, ch)
        C["_tp"] = timepoint_from_file(fn)
        C = refine_timepoint_b_cd14(df, C, anchor, ch)
        C["hla"] = ctx.hla_cut
        C["car"] = ctx.car_cut
        C["hla_dim"] = ctx.hla_dim
        if C.get("cd3") is None or C.get("cd56") is None:
            ctx.skipped.append({"file": fn, "reason": "no CD3/CD56 cutoff"})
            continue
        masks, g, scat, applied = gate_with(df, C, ch, ssc_cap=None)
        pct = percentages(masks)
        tp = timepoint_from_file(fn)
        ctx.samples.append(
            SampleState(
                filename=fn,
                timepoint=tp,
                n_total_events=int(ntot),
                df=df,
                perfile_cuts=dict(pf),
                cuts=C,
                applied=applied,
                masks=masks,
                signals=g,
                scatter=scat,
                pct=pct,
                is_reference=(tp == reference_tp),
                is_pre_infusion=bool(is_pre_infusion(fn)),
            )
        )

    # ── 5. order to match the pipeline's output ordering ─────────────────────
    order = None
    flow_csv = patient_dir / "flow.csv"
    if flow_csv.exists():
        try:
            order = pd.read_csv(flow_csv)["label"].astype(str).tolist()
        except Exception:
            order = None
    ctx.samples.sort(key=_order_key(order))

    if ctx.reference is None:
        ctx.notes.append(
            f"No sample matches the reference timepoint {reference_tp!r} — tier 1 and the "
            "transfer comparison in tier 2 cannot be computed."
        )

    # ── 6. verify the replay against the pipeline's own numbers ──────────────
    ctx.reproduction = _check_reproduction(ctx, verbose=verbose)
    if verbose:
        print(
            f"  [diagnostics] replayed {len(ctx.samples)} timepoint(s), "
            f"{len(ctx.controls)} control(s); reference={reference_tp!r}; "
            f"hla_cut={ctx.hla_cut}; car_cut={ctx.car_cut}; hla_dim={ctx.hla_dim}",
            flush=True,
        )
    return ctx


def controls_as_frames(controls: dict[str, tuple[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Drop the filenames — ``calibrate`` helpers expect ``{kind: dataframe}``."""
    return {k: v[1] for k, v in controls.items()}


def _find_multilineage(out_dir: Path, pid: str) -> Optional[Path]:
    for cand in (
        out_dir / "outputs" / "tables" / "multilineage.csv",
        out_dir / "multilineage.csv",
        out_dir / f"Multilineage_SSA_{pid}.csv",
    ):
        if cand.exists():
            return cand
    hits = sorted(out_dir.glob("Multilineage_SSA_*.csv"))
    return hits[0] if hits else None


def _check_reproduction(ctx: DiagnosticsContext, verbose: bool = True) -> ReproductionCheck:
    """Compare replayed percentages against the pipeline's ``multilineage.csv``.

    A mismatch does not mean the pipeline is wrong — it means THIS REPLAY is not auditing
    what actually ran, which invalidates every downstream diagnostic. Reported loudly.
    """
    chk = ReproductionCheck()
    path = _find_multilineage(ctx.out_dir, ctx.patient_id)
    if path is None:
        return chk
    try:
        ref = pd.read_csv(path)
    except Exception:
        return chk
    chk.available = True
    chk.source = path.name
    if "file" not in ref.columns:
        chk.available = False
        return chk

    by_file = {str(r["file"]): r for _, r in ref.iterrows()}
    worst = 0.0
    for s in ctx.samples:
        row = by_file.get(s.filename)
        if row is None:
            chk.mismatches.append(
                {"file": s.filename, "metric": "(row)", "reason": "absent from multilineage.csv"}
            )
            continue
        for metric in _REPRO_METRICS:
            if metric not in row or metric not in s.pct:
                continue
            expected, got = row[metric], s.pct[metric]
            if pd.isna(expected) or got is None:
                continue
            delta = abs(float(expected) - float(got))
            chk.checked += 1
            worst = max(worst, delta)
            if delta > _REPRO_TOL_PP:
                chk.mismatches.append(
                    {
                        "file": s.filename,
                        "metric": metric,
                        "pipeline": round(float(expected), 4),
                        "replay": round(float(got), 4),
                        "delta_pp": round(delta, 4),
                    }
                )
    chk.max_abs_delta_pp = round(worst, 5) if chk.checked else None
    if verbose:
        if chk.faithful:
            print(
                f"  [diagnostics] reproduction check OK — {chk.checked} values match "
                f"{chk.source} (max |delta| = {chk.max_abs_delta_pp} pp)",
                flush=True,
            )
        elif chk.available:
            print(
                f"  [diagnostics] REPRODUCTION MISMATCH — {len(chk.mismatches)} of "
                f"{chk.checked} values differ from {chk.source}; diagnostics below may not "
                "describe the run that produced the headline tables.",
                flush=True,
            )
    return chk


def parent_counts(sample: SampleState, numerator: str, parent: str) -> tuple[Optional[int], Optional[int]]:
    """(numerator events, parent events) for a metric, or (None, None) if either is absent."""
    num = sample.masks.get(numerator)
    den = sample.masks.get(parent)
    if num is None or den is None:
        return None, None
    return int(np.count_nonzero(num)), int(np.count_nonzero(den))


def reliability(n_parent: Optional[int], n_pos: Optional[int]) -> dict[str, Any]:
    """Reuse the pipeline's own countability floors rather than inventing new ones."""
    return {
        "n_parent": n_parent,
        "n_positive": n_pos,
        "parent_ok": None if n_parent is None else bool(n_parent >= MIN_PARENT),
        "positive_ok": None if n_pos is None else bool(n_pos >= MIN_POS),
        "min_parent": MIN_PARENT,
        "min_positive": MIN_POS,
    }


