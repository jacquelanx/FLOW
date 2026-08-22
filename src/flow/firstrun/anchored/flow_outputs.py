#!/usr/bin/env python3
"""FLOW additions on top of pipeline.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Only flowkit-free imports at module top, so the pure-python helpers (unified_cutoffs,
# composition_shift) are importable in environments without flowkit. Anything that touches
# FCS I/O or the gate math is imported lazily inside the functions that need it.
from .compare import timepoint_from_file, timepoint_sort_key

# CD8 detector in the lab panel — kept as a local constant to avoid a
# top-level flowkit import just to label one row.
_CD8_CHANNEL = "PE-Cy5-5-A"
_CD4_CHANNEL = "Alexa Fluor 700-A"


# ── D1 fallback: anchor on the negative population, no manual gating needed ──────
def build_negative_anchor(
    patient_dir, ch, reference_tp, out_path, subsample=200_000,
    control_repository=None,
):
    """Build an anchor from the reference sample's negative-population valley cuts.

    Derive one cutoff per marker on the reference (pre-infusion) sample using valley finders
    (``file_cuts`` → gmm_valley / valley_asinh: leftmost negative peak, first valley), LOCK
    the scatter + lineage cuts, and transfer them to every timepoint — with no manual CSV to
    calibrate against. The resulting JSON has the SAME schema that ``build_operator_anchor``
    writes, so ``apply_anchor_cuts`` / ``process_patient`` consume it unchanged.
    """
    patient_dir = Path(patient_dir)
    meta = {}
    mp = patient_dir / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    pid = meta.get("patient_study") or patient_dir.name

    fcs_dir = patient_dir / "fcs"
    ref_file = None
    for f in sorted(fcs_dir.glob("*.fcs")):
        if timepoint_from_file(f.name) == reference_tp:
            ref_file = f
            break
    if ref_file is None:
        raise FileNotFoundError(
            f"No FCS for reference timepoint {reference_tp!r} in {fcs_dir}")

    from .fcs_io import inspect_fcs_metadata, load_xform  # lazy — pulls in flowkit
    from .control_compensation import candidate_for_acquisition
    from .gates import file_cuts

    metadata_audit = inspect_fcs_metadata(ref_file)
    control_candidate = candidate_for_acquisition(
        control_repository or {},
        str(metadata_audit.get("acquisition_id") or "UNASSIGNED"),
        specimen_voltages=metadata_audit.get("detector_voltages"),
        require_voltage_match=True,
    )
    df, ntot = load_xform(
        ref_file, subsample=subsample, control_derived_candidate=control_candidate
    )
    print(f"[neg-anchor] reference {ref_file.name} (n={ntot:,}) — deriving negative-population "
          f"valley cuts", flush=True)
    cuts = file_cuts(df, ch)  # negative-peak → first-valley per marker (her finders)

    def _f(v):
        return float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v

    anchor = {
        "schema": "flow_operator_anchor/v2",
        "method": (
            "Negative-population anchor (no manual gating). The reference timepoint's "
            "leftmost (negative) peak and first valley define one cutoff per marker; scatter "
            "and lineage cuts are LOCKED and transferred to all timepoints."
        ),
        "patient_study": pid,
        "reference_timepoint": reference_tp,
        "reference_file": ref_file.name,
        "manual_targets": {},
        "achieved": {},
        "calibration_mae_pp": None,
        "created_utc": "NOT_RECORDED_DETERMINISTIC_RUN",
        "locked": {
            "fsc_lo": _f(cuts["fsc_lo"]),
            "fsc_hi": _f(cuts["fsc_hi"]),
            "ssc_hi": _f(cuts["ssc_hi"]),
        },
        "reference_cuts": {
            k: _f(v) for k, v in cuts.items()
            if v is not None or k in ("cd8_from_cd4_neg", "cd8")
        },
        "donor_cut": None,
        "donor_cut_mae_pp": None,
        "car_cut": None,
        "car_cut_mae_pp": None,
        "timepoint_cuts": {},
        "transfer_policy": {
            "scatter": "lock",
            "lineage_on_reference_tp": "use_reference_cuts",
            "lineage_on_other_tp": (
                "reestimate_inside_locked_scatter; if CD3- are mostly CD56+ "
                "(NK-dominant / engraftment), lock reference CD3/CD56"
            ),
            "cd4_cd8": "prefer_reference_cd4_with_cd8_as_cd4_neg",
            "b_cd14": "per_file_valley (no manual to fit)",
            "donor_car": "control_or_adaptive (no manual to fit)",
        },
        "channels": ch,
        "hla_specificity": meta.get("hla_specificity"),
        "hla_polarity": meta.get("hla_polarity", "donor"),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(anchor, indent=2), encoding="utf-8")
    print(f"[neg-anchor] wrote {out_path}", flush=True)
    return anchor


# ── unified_cutoffs.csv — the auditable "one cutoff per marker" table ───────────
_LINEAGE_MARKERS = [
    ("ld", "Viability(L/D)"), ("cd45", "CD45"), ("cd14", "CD14"), ("cd19", "CD19"),
    ("cd3", "CD3"), ("cd56", "CD56"), ("cd4", "CD4"),
]


def write_unified_cutoffs(anchor, out_dir, applied_policy=None):
    """Write the reference/transfer-policy summary retained for compatibility.

    The exact file-specific values are emitted separately in ``gate_parameters.csv``.
    This table must therefore never claim that adaptive values were locked globally.
    """
    ch = anchor.get("channels", {})
    ref = anchor.get("reference_cuts", {})
    locked = anchor.get("locked", {})
    ref_tp = anchor.get("reference_timepoint", "")
    manual = bool(anchor.get("manual_targets"))
    derivation = ("calibrated to manual gating at reference timepoint"
                  if manual else "negative-population valley at reference timepoint")

    rows = []
    for key, marker in _LINEAGE_MARKERS:
        v = ref.get(key)
        if v is None:
            continue
        rows.append({
            "marker": marker,
            "channel": ch.get(key, _CD4_CHANNEL if key == "cd4" else ""),
            "cutoff": round(float(v), 2),
            "cutoff_role": "reference anchor value",
            "scope": "reference/transfer policy; actual per-file value in gate_parameters.csv",
            "reference_timepoint": ref_tp,
            "derivation": derivation,
        })
    # CD8 is defined as CD4-negative in her panel (PE-Cy5-5 unreliable as a primary splitter).
    if anchor.get("reference_cuts", {}).get("cd8_from_cd4_neg"):
        rows.append({"marker": "CD8", "channel": _CD8_CHANNEL, "cutoff": "",
                     "cutoff_role": "derived identity rule",
                     "scope": "derived (CD4-negative within T)", "reference_timepoint": ref_tp,
                     "derivation": "not a direct cut; CD8 := CD4-negative"})
    # Scatter (locked FSC/SSC lymphocyte gate).
    for k, label in [("fsc_lo", "FSC-A low"), ("fsc_hi", "FSC-A high"), ("ssc_hi", "SSC-A high")]:
        if locked.get(k) is not None:
            rows.append({"marker": f"Scatter {label}", "channel": k.upper(),
                         "cutoff": round(float(locked[k]), 4),
                         "cutoff_role": "reference scatter constraint",
                         "scope": "soft-lock policy; actual per-file geometry in gate_parameters.csv",
                         "reference_timepoint": ref_tp, "derivation": derivation})
    # Donor HLA + CAR (fitted longitudinally when manual present; else control/adaptive).
    policy = applied_policy or {}
    for k, policy_key, marker, chan in [
        ("donor_cut", "hla_cut", "HLA (Donor)", ch.get("hla", "")),
        ("car_cut", "car_cut", "CAR", ch.get("car", "")),
    ]:
        value = policy.get(policy_key, anchor.get(k))
        if value is not None:
            rows.append({"marker": marker, "channel": chan,
                         "cutoff": round(float(value), 2),
                         "cutoff_role": "applied study-level configured technical threshold",
                         "scope": "applied to files; exact per-file row in gate_parameters.csv",
                         "reference_timepoint": ref_tp,
                         "derivation": str(policy.get(
                             "hla_source" if policy_key == "hla_cut" else "car_source",
                             "fitted to manual longitudinal %" if manual else "control / adaptive",
                         ))})

    path = os.path.join(out_dir, "unified_cutoffs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["marker", "channel", "cutoff", "cutoff_role",
                                          "scope", "reference_timepoint", "derivation"])
        w.writeheader()
        w.writerows(rows)
    return path


# ── composition_shift.csv — how the populations move across timepoints ──────────
_SHIFT_METRICS = [
    "%B (of lymph)", "%T (of lymph)", "%NK (of lymph)",
    "%CD4 (of lymph)", "%CD8 (of lymph)",
    "%Donor NK (of lymph)", "%CAR+ (of Donor NK)",
]


def write_composition_shift(multi_df, out_dir, reference_tp="Baseline"):
    """Tidy table: per population, value at each timepoint + Δ vs reference and vs previous.

    Everything is computed on the UNIFIED (anchored) cutoff, so a change here is a real
    biological shift, not a moving ruler. Row order follows ``multi_df`` (already ordered
    by flow.csv, or chronologically, upstream).

    Rows are addressed positionally, so a sample whose timepoint could not be parsed
    (labelled by filename here) and two samples sharing a label both survive instead of
    collapsing the index.
    """
    if "timepoint" not in multi_df.columns:
        return None
    # Label each row: its timepoint, or its filename when the timepoint is unknown, so
    # every row is still reported and every label is distinct.
    labels = []
    seen = {}
    for pos, row in enumerate(multi_df.to_dict("records")):
        tp = row.get("timepoint")
        label = str(tp) if tp is not None and tp == tp else str(row.get("file", f"row{pos}"))
        n = seen.get(label, 0)
        seen[label] = n + 1
        labels.append(label if n == 0 else f"{label} #{n + 1}")

    ref_pos = labels.index(reference_tp) if reference_tp in labels else (0 if labels else None)
    ref = labels[ref_pos] if ref_pos is not None else None

    rows = []
    for metric in _SHIFT_METRICS:
        if metric not in multi_df.columns:
            continue
        col = list(multi_df[metric])
        prev = None
        base_val = None
        if ref_pos is not None:
            b = col[ref_pos]
            base_val = float(b) if b == b and b is not None else None
        for label, v in zip(labels, col):
            v = float(v) if (v is not None and v == v) else None  # NaN-safe
            d_ref = (round(v - base_val, 3) if (v is not None and base_val is not None) else "")
            d_prev = (round(v - prev, 3) if (v is not None and prev is not None) else "")
            rows.append({
                "population": metric,
                "timepoint": label,
                "value": round(v, 3) if v is not None else "",
                "delta_from_reference": d_ref,
                "delta_from_previous": d_prev,
                "reference_timepoint": ref,
            })
            if v is not None:
                prev = v

    path = os.path.join(out_dir, "composition_shift.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["population", "timepoint", "value",
                                          "delta_from_reference", "delta_from_previous",
                                          "reference_timepoint"])
        w.writeheader()
        w.writerows(rows)
    return path


# -- temporal_cell_type_summary.csv -------------------------------------------

_TEMPORAL_CELL_TYPES = [
    # label, percentage column, denominator label, count column, denominator count column,
    # optional low-support flag
    (
        "Lymphocytes", "%Lymphocytes (of live)", "viable live events",
        "n_lympho", "n_live", None,
    ),
    ("B", "%B (of lymph)", "CD14-negative lymph region", "n_B", "n_lympho", None),
    ("T", "%T (of lymph)", "CD14-negative lymph region", "n_T", "n_lympho", None),
    ("CD4", "%CD4 (of lymph)", "CD14-negative lymph region", "n_CD4", "n_lympho", None),
    ("CD8", "%CD8 (of lymph)", "CD14-negative lymph region", "n_CD8", "n_lympho", None),
    ("NK", "%NK (of lymph)", "CD14-negative lymph region", "n_NK", "n_lympho", None),
    (
        "Donor NK", "%Donor NK (of lymph)", "CD14-negative lymph region",
        "n_Donor", "n_lympho", "donor_reliable",
    ),
    (
        "CAR+", "%CAR+ (of Donor NK)", "configured Donor NK region",
        "n_CAR", "n_Donor", "car_reliable",
    ),
    (
        "CD14+", "%CD14+ (of live)", "viable live events",
        "n_CD14p", "n_live", None,
    ),
]


def _number_or_none(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _iso_date(value):
    text = str(value or "").strip()
    for pattern in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def load_exact_alc_matches(patient_dir):
    """Map timepoint labels to exact-date ALC values without interpolation or imputation."""
    root = Path(patient_dir)
    flow_path = root / "flow.csv"
    alc_path = root / "alc.csv"
    if not flow_path.is_file() or not alc_path.is_file():
        return {}
    date_to_values = {}
    with alc_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            date = _iso_date(row.get("date"))
            value = _number_or_none(row.get("alc"))
            if date and value is not None and value >= 0:
                date_to_values.setdefault(date, []).append(value)
    matches = {}
    with flow_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label") or "").strip()
            date = _iso_date(row.get("date"))
            values = date_to_values.get(date, []) if date else []
            unique = sorted(set(round(value, 12) for value in values))
            if label and date and len(unique) == 1:
                matches[label] = {
                    "alc_k_per_uL": float(unique[0]),
                    "source_date": date,
                    "source_file": alc_path.name,
                    "match_policy": "exact flow.csv date to alc.csv date; no imputation",
                }
    return matches


def write_temporal_cell_type_summary(
    multi_df,
    out_dir,
    *,
    patient_id="",
    reference_tp="Baseline",
    identity_validated=False,
    alc_matches=None,
):
    """Write one auditable row per timepoint and configured cell-type region.

    The percentage denominator and event-count denominator are explicit. Event counts are
    counts in the analyzed event set; they are *not* cells/uL or biological absolute
    abundance unless a separately validated bead/volumetric/clinical conversion is added.
    """
    records = list(multi_df.to_dict("records"))
    alc_matches = alc_matches or {}
    rows = []
    for population, pct_col, denominator, count_col, denom_count_col, reliability_col in (
        _TEMPORAL_CELL_TYPES
    ):
        values = [_number_or_none(record.get(pct_col)) for record in records]
        counts = [_number_or_none(record.get(count_col)) for record in records]
        ref_pos = next(
            (index for index, record in enumerate(records)
             if str(record.get("timepoint")) == str(reference_tp)),
            0 if records else None,
        )
        ref_value = values[ref_pos] if ref_pos is not None else None
        ref_count = counts[ref_pos] if ref_pos is not None else None
        previous_value = None
        previous_count = None
        for order, (record, value, count) in enumerate(zip(records, values, counts), start=1):
            reliability = "NOT_APPLICABLE"
            if reliability_col:
                reliability = (
                    "SUPPORTED" if bool(record.get(reliability_col))
                    else "REVIEW_LOW_EVENT_SUPPORT"
                )
            denominator_count = _number_or_none(record.get(denom_count_col))
            timepoint = str(record.get("timepoint") or "")
            alc_match = alc_matches.get(timepoint)
            alc_value = (
                _number_or_none(alc_match.get("alc_k_per_uL")) if alc_match else None
            )
            abundance = None
            abundance_reason = "No exact ALC match through flow.csv; no imputation"
            abundance_method = ""
            if population == "CD14+":
                abundance_reason = (
                    "ALC calibrates lymphoid regions only; no validated absolute monocyte count"
                )
            elif alc_value is not None:
                if population == "Lymphocytes":
                    abundance = alc_value
                    abundance_method = "exact-date clinical ALC"
                elif count is not None and _number_or_none(record.get("n_lympho")):
                    lymph_count = _number_or_none(record.get("n_lympho"))
                    abundance = alc_value * count / lymph_count
                    abundance_method = (
                        "exact-date ALC multiplied by configured-region events / lymph events"
                    )
                if abundance is not None:
                    abundance_reason = ""
            rows.append({
                "patient_id": patient_id,
                "timepoint_order": order,
                "timepoint": record.get("timepoint", ""),
                "source_file": record.get("file", ""),
                "population": population,
                "percentage": "" if value is None else round(value, 3),
                "percentage_denominator": denominator,
                "percentage_point_change_from_reference": (
                    "" if value is None or ref_value is None else round(value - ref_value, 3)
                ),
                "percentage_point_change_from_previous": (
                    "" if value is None or previous_value is None
                    else round(value - previous_value, 3)
                ),
                "gated_event_count": "" if count is None else int(round(count)),
                "denominator_event_count": (
                    "" if denominator_count is None else int(round(denominator_count))
                ),
                "event_count_change_from_reference": (
                    "" if count is None or ref_count is None
                    else int(round(count - ref_count))
                ),
                "event_count_change_from_previous": (
                    "" if count is None or previous_count is None
                    else int(round(count - previous_count))
                ),
                "count_unit": "gated events in analyzed event set",
                "absolute_abundance": (
                    "" if abundance is None else round(float(abundance), 6)
                ),
                "absolute_abundance_unit": "K/uL" if abundance is not None else "",
                "absolute_abundance_available": abundance is not None,
                "absolute_abundance_method": abundance_method,
                "absolute_abundance_source_date": (
                    alc_match.get("source_date", "") if alc_match else ""
                ),
                "absolute_abundance_source_value": (
                    "" if alc_value is None else round(alc_value, 6)
                ),
                "absolute_abundance_source": (
                    alc_match.get("source_file", "") if alc_match else ""
                ),
                "absolute_abundance_reason": abundance_reason,
                "support_state": reliability,
                "population_identity_state": (
                    "VALIDATED" if identity_validated else "CONFIGURED_TECHNICAL_REGION"
                ),
                "compensation_state": record.get("compensation_state", ""),
                "acquisition_qc_state": record.get("acquisition_qc_state", ""),
            })
            if value is not None:
                previous_value = value
            if count is not None:
                previous_count = count

    path = Path(out_dir) / "temporal_cell_type_summary.csv"
    fieldnames = list(rows[0]) if rows else [
        "patient_id", "timepoint_order", "timepoint", "source_file", "population",
        "percentage", "percentage_denominator", "gated_event_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


# ── histogram overlays — "same axis range across timepoints, with that cutoff" ──
_OVERLAY_MARKERS = [
    ("cd3", "CD3"), ("cd56", "CD56"), ("cd19", "CD19"), ("cd14", "CD14"),
    ("cd45", "CD45"), ("ld", "Viability(L/D)"), ("hla", "HLA (Donor)"), ("car", "CAR"),
]


def histogram_overlays(pairs, ch, anchor, plots_dir, order=None, hla_cut=None, car_cut=None):
    """One figure per marker: every timepoint's histogram on SHARED axes + the unified cut.
    ``pairs`` = list of (filename, dataframe) already in her transform space.
    """
    os.makedirs(plots_dir, exist_ok=True)
    ref_cuts = anchor.get("reference_cuts", {})

    def _tp(fn):
        return timepoint_from_file(fn) or Path(fn).stem

    # Only real timepoints — drop control files (NT-NK / CBMC / Car product) so the overlay
    # shows the longitudinal story, not the controls.
    tp_pairs = [(fn, df) for fn, df in pairs if timepoint_from_file(fn) is not None]
    ordered = tp_pairs
    if order:
        ordered = sorted(tp_pairs, key=lambda p: order.index(_tp(p[0]))
                         if _tp(p[0]) in order else len(order))
    else:
        ordered = sorted(tp_pairs, key=lambda p: timepoint_sort_key(_tp(p[0])))

    written = []
    for key, marker in _OVERLAY_MARKERS:
        detector = ch.get(key)
        if detector is None:
            continue
        cut = ref_cuts.get(key)
        if key == "hla":
            cut = hla_cut
        elif key == "car":
            cut = car_cut
        # Collect this marker's values across timepoints to fix a shared x-range.
        series = []
        for fn, df in ordered:
            if detector in df.columns:
                v = df[detector].values.astype(float)
                v = v[np.isfinite(v)]
                if len(v):
                    series.append((_tp(fn), v))
        if len(series) < 2:
            continue
        allv = np.concatenate([v for _, v in series])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        bins = np.linspace(lo, hi, 120)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        cmap = plt.get_cmap("viridis", len(series))
        for i, (tp, v) in enumerate(series):
            ax.hist(np.clip(v, lo, hi), bins=bins, histtype="step", density=True,
                    lw=1.3, color=cmap(i), label=tp)
        if cut is not None and lo <= float(cut) <= hi:
            ax.axvline(float(cut), color="red", lw=2, ls="--",
                       label=f"unified cut = {float(cut):.0f}")
        ax.set(title=f"{marker} ({detector}) — all timepoints, shared axis + unified cutoff",
               xlabel=f"{marker} intensity (transform space)", ylabel="density")
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        fig.tight_layout()
        out = os.path.join(plots_dir, f"overlay_{key}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
    return written
