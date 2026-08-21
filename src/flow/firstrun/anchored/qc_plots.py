#!/usr/bin/env python3
"""Structured multi-page QC PDF for CAR-NK hierarchical gating.

Page layout
-----------
  1. Cover — methods one-liner, full hierarchy diagram, auto vs manual table
  2. Subset summary — all timepoints × all reported subsets
  3+. One page per timepoint — full gate sequence (scatter → … → CAR+)
"""
from __future__ import annotations

import csv
import re
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (
    to_rgba, ListedColormap, LinearSegmentedColormap, LogNorm,
)
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture


# UNITO / Nat Commun–style gate fill (semi-transparent islands, not harsh lines)
GATE_FILL = "#5B9BD5"
GATE_EDGE = "#1F6AA5"
# Accepted gate regions remain visible over the density field without obscuring it.
# This is a presentation constant only; it never participates in gate derivation.
GATE_ALPHA = 0.20
GATE_EDGE_LW = 1.1
PLOT_RNG_SEED = 2701
SCATTER_INSTRUMENT_SCALE = 262144.0

LYMPH_GEOMETRY_FIELDS = [
    "patient_id", "timepoint", "sample_file", "parent", "panel",
    "parent_events", "applied_gate_events", "applied_gate_percent",
    "mode_number", "is_low_ssc_mode", "mode_weight",
    "mean_fsc_a_instrument", "mean_ssc_a_instrument",
    "cov_fsc_fsc", "cov_fsc_ssc", "cov_ssc_ssc",
    "fit_method", "diagnostic_only",
]

# Match the UPN27 review convention: a FlowJo-like heat field with explicit
# parent-density contours.  Fractions are relative to the maximum of the same
# smoothed density grid used for the heat field, so the overlay is deterministic
# and does not alter any gate or reported population.
DENSITY_CONTOUR_FRACTIONS = (0.20, 0.40, 0.65)
HEAT_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "flow_heat_density",
    (
        "#ffffff",
        "#dbeafe",
        "#60a5fa",
        "#22d3ee",
        "#fde047",
        "#f97316",
        "#b91c1c",
    ),
    N=256,
)
HEAT_DENSITY_CMAP.set_bad("white")

PDF_METADATA = {
    "Creator": "FLOW governed first run",
    "Producer": "FLOW",
    "CreationDate": None,
    "ModDate": None,
}


def _safe_filename_token(value):
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "unknown")).strip("_")
    return token or "unknown"


def _save_page_outputs(fig, *, render_path=None, page_pdf_path=None):
    """Save deterministic companion artifacts for a page in the composite report."""
    if render_path is not None:
        render_path = Path(render_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(render_path, dpi=160, bbox_inches="tight", facecolor="white")
    if page_pdf_path is not None:
        page_pdf_path = Path(page_pdf_path)
        page_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            page_pdf_path,
            format="pdf",
            facecolor="white",
            metadata=PDF_METADATA,
        )


def _add_page_footer(fig, page_number=None, total_pages=None):
    """Add an internal-review page marker without entering the plot grid."""
    if page_number is None or total_pages is None:
        return
    fig.text(
        0.985,
        0.008,
        f"INTERNAL REVIEW | {page_number}/{total_pages}",
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#667788",
    )


def _add_status_banner(fig, technical_state="REVIEW", compensation_state=None,
                       acquisition_state=None):
    parts = ["PROVISIONAL / NOT BIOLOGICALLY VALIDATED", f"RUN {technical_state}"]
    if compensation_state:
        parts.append(f"COMPENSATION {compensation_state}")
    if acquisition_state:
        parts.append(f"SAMPLE ACQUISITION {acquisition_state}")
    failing = any(str(value).upper() in {"BLOCKED", "FAIL"}
                  for value in (technical_state, compensation_state, acquisition_state))
    fig.text(
        0.5, 0.997, "  ·  ".join(parts), ha="center", va="top", fontsize=6.2,
        fontweight="bold", color="#7F1D1D" if failing else "#7C4A03",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FEE2E2" if failing else "#FFF7D6",
                  ec="#B91C1C" if failing else "#B7791F", lw=0.8),
    )


# ── citations shown on the cover page ─────────────────────────────────────────
CITATIONS = [
    (
        "Jiang et al. Nat Commun 2025 — UNITO",
        "doi:10.1038/s41467-025-56622-2",
        "Hierarchical bivariate density gating; FSC×SSC lymph islands.",
    ),
    (
        "Malek et al. Bioinformatics 2015 — flowDensity",
        "doi:10.1093/bioinformatics/btu677",
        "Sequential density thresholds mimicking expert 2D gates.",
    ),
    (
        "Finak et al. PLoS Comput Biol 2014 — OpenCyto",
        "doi:10.1371/journal.pcbi.1003806",
        "Hierarchical templates; data-driven gates per sample.",
    ),
    (
        "Finak et al. Sci Rep 2016 — HIPC",
        "doi:10.1038/srep20686",
        "Standardized FSC/SSC → singlets → live → lineage; auto ≈ central manual.",
    ),
    (
        "Rico et al. Cytometry A 2023 — doublets",
        "doi:10.1002/cyto.a.24690",
        "SSC-W×SSC-H / FSC-A×H pulse analysis for doublet exclusion.",
    ),
    (
        "Finak et al. Cytometry A 2014 — normalization",
        "doi:10.1002/cyto.a.22433",
        "Reference-sample landmark registration / gate transfer across batches.",
    ),
]


def _data_extent(x, y, full_scale=True):
    """Axis limits that do not crop the event cloud.

    Prefer transform / instrument scale [0, 1] when data live there; otherwise
    expand to the full finite min/max with a small pad (never a tight percentile
    zoom that hides events).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return (0.0, 1.0), (0.0, 1.0)
    x, y = x[ok], y[ok]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if full_scale and xmin >= -0.02 and xmax <= 1.05 and ymin >= -0.02 and ymax <= 1.05:
        return (0.0, 1.0), (0.0, 1.0)
    pad_x = 0.02 * (xmax - xmin + 1e-9)
    pad_y = 0.02 * (ymax - ymin + 1e-9)
    return (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def _shade_rect(ax, x0, x1, y0, y1, color=GATE_FILL, alpha=GATE_ALPHA,
                edge=GATE_EDGE, lw=GATE_EDGE_LW, zorder=3):
    """Semi-transparent rectangular gate region (UNITO-style)."""
    if None in (x0, x1, y0, y1):
        return
    w, h = float(x1) - float(x0), float(y1) - float(y0)
    if w <= 0 or h <= 0:
        return
    ax.add_patch(Rectangle(
        (float(x0), float(y0)), w, h,
        facecolor=to_rgba(color, alpha), edgecolor=edge, linewidth=lw,
        linestyle="-", zorder=zorder, clip_on=True,
    ))


def _gate_label(ax, text, x, y, color, *, ha="left", va="top"):
    """Directly name a shaded gate region using axes-relative coordinates."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=5.3,
        fontweight="bold",
        color=color,
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor=to_rgba("white", 0.82),
            edgecolor=to_rgba(color, 0.80),
            linewidth=0.5,
        ),
    )


def _shade_island(ax, x, y, bins=80, level_frac=0.12, color=GATE_FILL,
                  alpha=0.45, edge=GATE_EDGE, lw=GATE_EDGE_LW):
    """Filled density contour of gated events (shaded island, not a thin line)."""
    x, y = _subsample(x, y, n=60_000)
    if len(x) < 30:
        return
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = gaussian_filter(H.T.astype(float), sigma=1.4)
    if H.max() <= 0:
        return
    # Soft filled mask via pcolormesh (reliable across matplotlib versions)
    level = float(H.max() * level_frac)
    mask = np.ma.masked_where(H < level, np.ones_like(H))
    gate_cmap = ListedColormap([to_rgba(color, alpha)])
    gate_cmap.set_bad((0, 0, 0, 0))
    ax.pcolormesh(
        xe, ye, mask, cmap=gate_cmap,
        shading="auto", zorder=3, rasterized=True, vmin=0, vmax=1,
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    try:
        ax.contour(
            xc, yc, H, levels=[level], colors=[edge],
            linewidths=[lw], alpha=0.95, zorder=4,
        )
    except Exception:
        pass


def _shade_halfplane_x(ax, cut, side="below", color=GATE_FILL, alpha=GATE_ALPHA,
                       edge=GATE_EDGE):
    """Shade accepted side of a vertical 1D cut across the axes."""
    if cut is None:
        return
    x0, x1 = ax.get_xlim()
    if side == "below":
        ax.axvspan(x0, cut, facecolor=to_rgba(color, alpha),
                   edgecolor="none", zorder=2)
    else:
        ax.axvspan(cut, x1, facecolor=to_rgba(color, alpha),
                   edgecolor="none", zorder=2)
    ax.axvline(cut, color=edge, ls="-", lw=GATE_EDGE_LW, alpha=0.95, zorder=4)


def _heat_scatter_density(
    ax,
    x,
    y,
    *,
    bins=160,
    smooth=1.7,
    range_=None,
    cmap=HEAT_DENSITY_CMAP,
    alpha=0.96,
    contour_fractions=DENSITY_CONTOUR_FRACTIONS,
    contour_color="#404040",
    contour_alpha=0.70,
    contour_widths=(0.65, 0.85, 1.05),
    zorder=1,
):
    """Draw a deterministic heat-scatter field with density contours.

    The implementation mirrors the validated UPN27 QC style while remaining a
    lightweight Matplotlib primitive: finite events are deterministically
    subsampled, binned on the displayed range, Gaussian-smoothed, rendered with
    logarithmic color normalization, and contoured at fixed fractions of the
    maximum density.  It changes presentation only; gating masks and cutoffs are
    never derived here.

    Returns the rendered density grid metadata, or ``None`` when too few finite
    events are available.
    """
    x_plot, y_plot = _subsample(x, y, n=120_000, seed=0)
    if len(x_plot) < 10:
        return None

    if range_ is None:
        x_range, y_range = _data_extent(x_plot, y_plot, full_scale=True)
    else:
        x_range, y_range = range_
    if (
        not np.all(np.isfinite((*x_range, *y_range)))
        or x_range[1] <= x_range[0]
        or y_range[1] <= y_range[0]
    ):
        return None

    H, xe, ye = np.histogram2d(
        x_plot,
        y_plot,
        bins=bins,
        range=(tuple(x_range), tuple(y_range)),
    )
    H = gaussian_filter(H.T.astype(float), sigma=float(smooth))
    positive = H[H > 0]
    if not len(positive):
        return None

    vmax = float(H.max())
    vmin = max(float(np.percentile(positive, 3)), np.finfo(float).tiny)
    if vmax <= vmin:
        vmin = max(vmax * 0.1, np.finfo(float).tiny)
    H_plot = np.ma.masked_where(H <= 0, H)
    ax.pcolormesh(
        xe,
        ye,
        H_plot,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * (1.0 + 1e-6))),
        shading="auto",
        alpha=alpha,
        rasterized=True,
        zorder=zorder,
    )

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = np.asarray(contour_fractions, dtype=float) * vmax
    levels = np.unique(levels[(levels > 0) & (levels < vmax)])
    if len(levels):
        widths = list(contour_widths)
        if len(widths) != len(levels):
            widths = np.linspace(0.65, 1.05, len(levels)).tolist()
        ax.contour(
            xc,
            yc,
            H,
            levels=levels,
            colors=[contour_color],
            linewidths=widths,
            alpha=contour_alpha,
            zorder=zorder + 0.5,
        )
    return {"H": H, "xc": xc, "yc": yc, "xe": xe, "ye": ye}


def _hb(ax, x, y, xlab, ylab, mask=None, bins=160, full_axes=True,
        parent_x=None, parent_y=None):
    """Heat-scatter density with contours and optional parent-cloud context."""
    x_raw = np.asarray(x, float)
    y_raw = np.asarray(y, float)
    if parent_x is not None and parent_y is not None:
        extent_x, extent_y = _data_extent(parent_x, parent_y)
    else:
        extent_x, extent_y = _data_extent(x_raw, y_raw)
    display_range = (extent_x, extent_y)

    if parent_x is not None and parent_y is not None:
        _heat_scatter_density(
            ax,
            parent_x,
            parent_y,
            bins=bins,
            range_=display_range,
            cmap="Greys",
            alpha=0.45,
            contour_fractions=(),
            zorder=0,
        )
    if mask is not None:
        x_plot, y_plot = x_raw[mask], y_raw[mask]
    else:
        x_plot, y_plot = x_raw, y_raw
    if len(x_plot) < 10 and (parent_x is None or len(np.asarray(parent_x)) < 10):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        return
    ax.set_facecolor("white")
    if len(x_plot) >= 10:
        _heat_scatter_density(
            ax,
            x_plot,
            y_plot,
            bins=bins,
            range_=display_range,
        )
    if full_axes:
        ax.set_xlim(*extent_x)
        ax.set_ylim(*extent_y)
    ax.set_xlabel(xlab, fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)


def _vline(ax, x, color=GATE_EDGE, ls="-"):
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=GATE_EDGE_LW, alpha=0.95)


def _hline(ax, y, color=GATE_EDGE, ls="-"):
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=GATE_EDGE_LW, alpha=0.95)


def _subsample(x, y, n=120_000, seed=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) > n:
        idx = np.random.RandomState(seed).choice(len(x), n, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def _deterministic_mask_indices(mask, limit, seed):
    """Return a reproducible bounded sample of indices selected by ``mask``."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(idx) > limit:
        idx = np.random.RandomState(seed).choice(idx, limit, replace=False)
    return idx


def _pct_label(n, parent_n, total_n):
    of_p = 100.0 * n / parent_n if parent_n else 0.0
    of_t = 100.0 * n / total_n if total_n else 0.0
    return f"n={n:,}  {of_p:.1f}% of parent  ({of_t:.2f}% total)"


def _retention_row(m):
    """Counts at each hierarchy step."""
    n_tot = len(m["lymph_scatter"])
    steps = [
        ("Total", n_tot),
        ("Lymph FSC×SSC", int(m["lymph_scatter"].sum())),
        ("Singlets", int(m["sing"].sum())),
        ("Live", int(m["live"].sum())),
        ("CD45+", int(m["cd45p"].sum())),
        ("CD14− lymph", int(m["lympho"].sum())),
    ]
    return steps, n_tot


# ── cover / methods ───────────────────────────────────────────────────────────

def qc_cover_page(pdf, patient_id: str, n_timepoints: int = 0,
                  scatter_policy: str = "soft_lock"):
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.05, 0.93, patient_id, fontsize=28, fontweight="bold",
            color="#1a3a5c", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.86, "CAR-NK multilineage QC report", fontsize=14,
            color="#3d5a73", va="top", transform=ax.transAxes)
    ax.text(
        0.05, 0.81,
        f"PHI-safe study label only  ·  {n_timepoints} timepoints  ·  "
        f"scatter={scatter_policy}  ·  deterministic canonical first run",
        fontsize=8, color="#667788", va="top", transform=ax.transAxes,
    )

    # Hierarchy diagram
    ax.text(0.05, 0.74, "Gating hierarchy", fontsize=12, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes)
    stages = [
        "1 Lymph\nFSC-A×SSC-A\nheat density",
        "2 Singlets\nSSC-W×SSC-H\nW≤p99",
        "3 Live\nL/D−",
        "4 CD45+",
        "5 CD14±\nmono | lymph",
        "6 CD19−\nCD3×CD56",
        "7 Donor\nHLA · CAR",
    ]
    y0 = 0.58
    for i, lab in enumerate(stages):
        x = 0.05 + i * 0.13
        box = FancyBboxPatch(
            (x, y0), 0.115, 0.12, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#E8F1F8", edgecolor="#1a3a5c", linewidth=1.2,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(x + 0.0575, y0 + 0.06, lab, ha="center", va="center",
                fontsize=6.5, color="#1a3a5c", transform=ax.transAxes,
                linespacing=1.25)
        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x + 0.128, y0 + 0.06), xytext=(x + 0.115, y0 + 0.06),
                arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=1.2),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
            )

    ax.text(
        0.05, 0.52,
        "Method: Baseline-calibrated operator anchor (manual % match) with soft-lock "
        "scatter transfer — SSC floor from Baseline; FSC re-estimated per file via "
        "density (engrafted lymph FSC often shifts).",
        fontsize=8, color="#334455", wrap=True, transform=ax.transAxes,
        va="top",
    )

    ax.text(0.05, 0.44, "Key references (cite in methods)", fontsize=11,
            fontweight="bold", color="#1a3a5c", transform=ax.transAxes)
    y = 0.40
    for title, doi, note in CITATIONS:
        ax.text(0.06, y, f"• {title}", fontsize=7.5, fontweight="bold",
                color="#1a3a5c", transform=ax.transAxes, va="top")
        ax.text(0.08, y - 0.022, f"{doi}  —  {note}", fontsize=6.5,
                color="#556677", transform=ax.transAxes, va="top")
        y -= 0.048

    ax.text(
        0.05, 0.04,
        "Pages: cover → retention table → one page per timepoint (gate stages) → QA vs manual.",
        fontsize=7, color="#778899", transform=ax.transAxes,
    )
    pdf.savefig(fig)
    plt.close(fig)


def qc_retention_page(pdf, retention_rows: list[dict], patient_id: str):
    """Summary table of % retained at each hierarchy step."""
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id} — gate retention (% of total events)",
                 fontsize=13, fontweight="bold", color="#1a3a5c")
    ax = fig.add_subplot(111)
    ax.axis("off")
    if not retention_rows:
        ax.text(0.5, 0.5, "No retention data", ha="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    cols = ["timepoint", "n_total", "lymph%", "sing%", "live%", "cd45%", "lympho%", "purity"]
    headers = ["TP", "n events", "Lymph", "Singlet", "Live", "CD45+", "CD14−", "Purity"]
    cell = []
    for r in retention_rows:
        cell.append([
            str(r.get("timepoint", "")),
            f"{int(r.get('n_total', 0)):,}",
            f"{r.get('lymph_pct', 0):.2f}",
            f"{r.get('sing_pct', 0):.2f}",
            f"{r.get('live_pct', 0):.2f}",
            f"{r.get('cd45_pct', 0):.2f}",
            f"{r.get('lympho_pct', 0):.2f}",
            f"{r.get('purity', 0):.1f}",
        ])
    table = ax.table(
        cellText=cell, colLabels=headers, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 1.45)
    for (r, c), cell_obj in table.get_celld().items():
        if r == 0:
            cell_obj.set_facecolor("#1a3a5c")
            cell_obj.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell_obj.set_facecolor("#EEF3F8")
        cell_obj.set_edgecolor("#CCDDEE")

    ax.text(
        0.5, 0.08,
        "Values are % of total loaded events at each step (except purity = B+T+NK of CD14−). "
        "Soft-lock recovers engrafted FSC shifts; CD45 drop on debris-heavy Baseline/Pre is expected.",
        ha="center", fontsize=7.5, color="#556677", transform=ax.transAxes,
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def qc_qa_page(pdf, cmp_summary: dict | None, compare_df=None, patient_id: str = ""):
    """Auto vs manual QA summary — not crammed into panel 1."""
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id} — automated vs manual QA",
                 fontsize=13, fontweight="bold", color="#1a3a5c")
    ax = fig.add_subplot(111)
    ax.axis("off")
    if not cmp_summary:
        ax.text(0.5, 0.5, "No manual reference comparison", ha="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    mae = cmp_summary.get("mae", "—")
    w5 = cmp_summary.get("within_5pp", "—")
    w10 = cmp_summary.get("within_10pp", "—")
    ax.text(
        0.5, 0.92,
        f"MAE = {mae} pp    ≤5 pp = {w5}%    ≤10 pp = {w10}%",
        ha="center", fontsize=14, fontweight="bold", color="#1a3a5c",
        transform=ax.transAxes,
    )
    soft = cmp_summary.get("soft_target_ok")
    if soft is not None:
        ax.text(
            0.5, 0.86,
            "Soft target (≥70% cells ≤10 pp): " + ("PASS" if soft else "REVIEW"),
            ha="center", fontsize=10,
            color=("#198754" if soft else "#c0392b"),
            transform=ax.transAxes,
        )
    worst = cmp_summary.get("worst") or {}
    if worst:
        ax.text(
            0.5, 0.80,
            f"Worst cell: {worst.get('timepoint')} / {worst.get('metric')}  Δ={worst.get('delta')} pp",
            ha="center", fontsize=9, color="#556677", transform=ax.transAxes,
        )

    if compare_df is not None and len(compare_df):
        show = compare_df.copy()
        # Prefer a compact pivot if columns exist
        cols = [c for c in ("timepoint", "metric", "manual", "auto", "delta") if c in show.columns]
        if not cols:
            cols = list(show.columns)[:6]
        show = show[cols].head(40)
        cell = [[("" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v))
                 for v in row] for row in show.to_numpy()]
        table = ax.table(
            cellText=cell, colLabels=cols, loc="center", cellLoc="center",
            bbox=[0.05, 0.08, 0.90, 0.65],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for (r, c), cell_obj in table.get_celld().items():
            if r == 0:
                cell_obj.set_facecolor("#1a3a5c")
                cell_obj.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell_obj.set_facecolor("#EEF3F8")

    pdf.savefig(fig)
    plt.close(fig)


# ── per-timepoint page ────────────────────────────────────────────────────────

def _fit_live_cd45_density_modes(fsca, ssca, parent_mask, max_modes=4):
    """Fit deterministic diagnostic FSC/SSC modes inside the Live-CD45 parent.

    This is a presentation/QC fit only. It never participates in gate derivation
    or changes event membership. Scatter is converted back from the pipeline's
    0-1 linear transform to instrument units so the result can be compared with
    the historical UPN27 density-geometry view.
    """
    fsca = np.asarray(fsca, dtype=float)
    ssca = np.asarray(ssca, dtype=float)
    parent = np.asarray(parent_mask, dtype=bool)
    finite = parent & np.isfinite(fsca) & np.isfinite(ssca)
    x = fsca[finite] * SCATTER_INSTRUMENT_SCALE
    y = ssca[finite] * SCATTER_INSTRUMENT_SCALE
    parent_events = int(len(x))
    if parent_events < 40:
        return {
            "parent_events": parent_events,
            "sampled_events": parent_events,
            "fit_method": "NOT_EVALUABLE_LT40_PARENT_EVENTS",
            "modes": [],
        }

    values = np.column_stack([x, y])
    if len(values) > 30_000:
        idx = np.random.RandomState(PLOT_RNG_SEED).choice(
            len(values), 30_000, replace=False,
        )
        fit_values = values[idx]
    else:
        fit_values = values

    center = np.median(fit_values, axis=0)
    scale = np.percentile(fit_values, 90, axis=0) - np.percentile(
        fit_values, 10, axis=0,
    )
    fallback = np.std(fit_values, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(fallback > 1e-9, fallback, 1.0))
    normalized = (fit_values - center) / scale
    maximum = min(int(max_modes), max(1, len(normalized) // 50))
    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for n_components in range(1, maximum + 1):
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=PLOT_RNG_SEED,
                n_init=2,
                max_iter=300,
                reg_covar=1e-4,
            )
            model.fit(normalized)
            candidates.append((float(model.bic(normalized)), model))
    _, fitted = min(candidates, key=lambda item: item[0])

    transform = np.diag(scale)
    modes = []
    minimum_weight = max(0.02, 25.0 / len(normalized))
    for component, weight in enumerate(fitted.weights_):
        if float(weight) < minimum_weight:
            continue
        mean = fitted.means_[component] * scale + center
        covariance = transform @ fitted.covariances_[component] @ transform
        modes.append({
            "source_component": int(component),
            "weight": float(weight),
            "mean": mean,
            "covariance": covariance,
        })
    if not modes:
        component = int(np.argmax(fitted.weights_))
        mean = fitted.means_[component] * scale + center
        covariance = transform @ fitted.covariances_[component] @ transform
        modes.append({
            "source_component": component,
            "weight": float(fitted.weights_[component]),
            "mean": mean,
            "covariance": covariance,
        })
    modes.sort(key=lambda mode: (float(mode["mean"][1]), float(mode["mean"][0])))
    for number, mode in enumerate(modes, start=1):
        mode["mode_number"] = int(number)
        mode["is_low_ssc_mode"] = number == 1
    return {
        "parent_events": parent_events,
        "sampled_events": int(len(fit_values)),
        "fit_method": f"deterministic_GMM_BIC_k{fitted.n_components}",
        "modes": modes,
    }


def _gaussian_coverage_contour(mean, covariance, probability=0.95, points=240):
    """Return a 2D Gaussian coverage contour for display only."""
    # Chi-square(df=2) quantile at 95%; the probability argument is kept explicit
    # in the interface so the diagnostic contract is visible at the call site.
    if not np.isclose(float(probability), 0.95):
        raise ValueError("only the frozen 95% diagnostic contour is supported")
    radius = np.sqrt(5.991464547107979)
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    angles = np.linspace(0.0, 2.0 * np.pi, int(points), endpoint=True)
    unit = np.vstack([np.cos(angles), np.sin(angles)])
    contour = np.asarray(mean, dtype=float)[:, None] + (
        eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ unit * radius
    )
    return contour[0], contour[1]


def _lymph_density_panel(ax, fsca, ssca, viable_mask, retain_txt=""):
    """All analyzed events with parent contours and the applied viable gate."""
    x_all, y_all = _subsample(fsca, ssca, n=180_000)
    if len(x_all) < 20:
        ax.set_visible(False)
        return

    (x_lo, x_hi), (y_lo, y_hi) = _data_extent(fsca, ssca, full_scale=True)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    dens = _heat_scatter_density(
        ax,
        fsca,
        ssca,
        bins=260,
        smooth=2.0,
        range_=((x_lo, x_hi), (y_lo, y_hi)),
    )
    if dens is None:
        return

    # Density island shade from the actual gate mask (no FSC×SSC rectangle).
    if viable_mask is not None and int(np.sum(viable_mask)) > 20:
        _shade_island(
            ax, fsca[viable_mask], ssca[viable_mask],
            bins=100, level_frac=0.06,
            color=GATE_FILL, alpha=0.32, edge=GATE_EDGE, lw=1.5,
        )
        _gate_label(ax, "Applied viable gate", 0.98, 0.04, GATE_EDGE,
                    ha="right", va="bottom")

    ax.set_xlabel("FSC-A", fontsize=8)
    ax.set_ylabel("SSC-A", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("1. All analyzed events → viable cellular scatter gate", fontsize=6.7,
                 fontweight="bold", color="#1a3a5c")
    ax.text(
        0.02, 0.98, f"{retain_txt}\ndensity island (not rect)",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
        color="#1a3a5c",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85, lw=0.5),
    )


def _live_cd45_lymph_geometry_panel(
    ax, fsca, ssca, parent_mask, applied_gate_mask, *, retain_txt="",
):
    """Historical-run-compatible lymph geometry diagnostic on Live-CD45 parent."""
    parent = np.asarray(parent_mask, dtype=bool)
    applied = parent & np.asarray(applied_gate_mask, dtype=bool)
    if int(parent.sum()) < 20:
        ax.set_visible(False)
        return _fit_live_cd45_density_modes(fsca, ssca, parent)

    x = np.asarray(fsca, dtype=float) * SCATTER_INSTRUMENT_SCALE
    y = np.asarray(ssca, dtype=float) * SCATTER_INSTRUMENT_SCALE
    (x_lo, x_hi), (y_lo, y_hi) = _data_extent(x[parent], y[parent], full_scale=False)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    _heat_scatter_density(
        ax,
        x[parent],
        y[parent],
        bins=220,
        smooth=2.0,
        range_=((x_lo, x_hi), (y_lo, y_hi)),
        contour_color="#4B5563",
        contour_alpha=0.82,
        contour_widths=(0.75, 0.95, 1.15),
    )

    if int(applied.sum()) > 20:
        _shade_island(
            ax,
            x[applied],
            y[applied],
            bins=100,
            level_frac=0.05,
            color="#2563EB",
            alpha=0.10,
            edge="#2563EB",
            lw=1.7,
        )

    fit = _fit_live_cd45_density_modes(fsca, ssca, parent)
    for mode in fit["modes"]:
        mean = mode["mean"]
        ax.plot(
            [float(mean[0])], [float(mean[1])], marker="o", markersize=6.0,
            markerfacecolor="white", markeredgecolor="#111827",
            markeredgewidth=1.0, linestyle="None", zorder=6,
        )
        ax.text(
            float(mean[0]), float(mean[1]), str(mode["mode_number"]),
            ha="center", va="center", fontsize=5.0, fontweight="bold",
            color="#111827", zorder=7,
        )
        if mode["is_low_ssc_mode"]:
            cx, cy = _gaussian_coverage_contour(
                mode["mean"], mode["covariance"], probability=0.95,
            )
            ax.plot(
                cx, cy, color="#16A34A", linestyle="--", linewidth=1.8,
                alpha=0.95, zorder=5,
            )

    legend_handles = [
        Line2D([0], [0], color="#4B5563", lw=1.0,
               label="Parent-density contours"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#111827", markersize=5.0,
               label="Fitted mode centers"),
        Line2D([0], [0], color="#16A34A", lw=1.8, ls="--",
               label="Low-SSC contour (95%)"),
        Line2D([0], [0], color="#2563EB", lw=1.7,
               label="Applied gate"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=3.9,
        framealpha=0.86, borderpad=0.25, labelspacing=0.18,
        handlelength=1.8, handletextpad=0.35,
    )
    if int(applied.sum()) > 20:
        _gate_label(ax, "Applied lymph gate", 0.98, 0.04, "#2563EB",
                    ha="right", va="bottom")
    if fit["modes"]:
        _gate_label(ax, "Low-SSC mode", 0.02, 0.04, "#15803D",
                    ha="left", va="bottom")

    ax.set_xlabel("FSC-A (instrument units)", fontsize=7)
    ax.set_ylabel("SSC-A (instrument units)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_title("2. Live-CD45 parent → lymph density geometry", fontsize=6.7,
                 fontweight="bold", color="#1a3a5c")
    ax.text(
        0.02, 0.98,
        f"{retain_txt}\nmodes={len(fit['modes'])}  ·  diagnostic only",
        transform=ax.transAxes, va="top", ha="left", fontsize=5.2,
        color="#1a3a5c",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc",
                  alpha=0.86, lw=0.5),
    )
    return fit


def qc_page(pdf, fname, m, g, scat, cuts, pct, timepoint: str | None = None,
            page_number=None, total_pages=None, render_path=None, page_pdf_path=None,
            technical_state="REVIEW", compensation_state=None,
            acquisition_state=None, identity_validated=False):
    """One page per timepoint: full hierarchy (3×4) + retention strip."""
    fsca, ssca = scat[0], scat[1]
    sscw = scat[2] if len(scat) > 2 else None
    ssch = scat[3] if len(scat) > 3 else None
    steps, n_tot = _retention_row(m)
    title_tp = timepoint or fname
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{title_tp}   ·   {fname}", fontsize=11, fontweight="bold",
                 color="#1a3a5c", y=0.965)
    _add_status_banner(fig, technical_state, compensation_state, acquisition_state)

    ax_bar = fig.add_axes([0.07, 0.905, 0.86, 0.045])
    labels = [s[0] for s in steps[1:]]
    vals = [100.0 * s[1] / n_tot if n_tot else 0 for s in steps[1:]]
    colors = ["#1a3a5c", "#2a6f97", "#3d8a6e", "#c17a2b", "#8b4513"]
    ax_bar.bar(labels, vals, color=colors[: len(vals)], width=0.65)
    ax_bar.set_ylim(0, max(vals) * 1.35 + 1 if vals else 1)
    ax_bar.set_ylabel("% tot", fontsize=6)
    ax_bar.tick_params(labelsize=5.5)
    for i, v in enumerate(vals):
        ax_bar.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=5.5, color="#1a3a5c")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    gs = gridspec.GridSpec(
        3, 4, figure=fig, left=0.06, right=0.98, top=0.83, bottom=0.105,
        wspace=0.36, hspace=0.50,
    )
    n_lymph = int(m["lymph_scatter"].sum())
    n_sing = int(m["sing"].sum())
    n_live = int(m["live"].sum())
    n_cd45 = int(m["cd45p"].sum())
    n_lympho = int(m["lympho"].sum())

    # 1. All analyzed events → viable cellular scatter gate.
    ax0 = fig.add_subplot(gs[0, 0])
    _lymph_density_panel(
        ax0, fsca, ssca, m.get("lymph_scatter"),
        retain_txt=_pct_label(n_lymph, n_tot, n_tot),
    )

    # 2. Companion diagnostic: Live-CD45 parent → lymph density geometry.
    ax1 = fig.add_subplot(gs[0, 1])
    applied_geometry_n = int(np.sum(
        np.asarray(m.get("cd45p"), bool) & np.asarray(m.get("lymph_report"), bool)
    ))
    geometry = _live_cd45_lymph_geometry_panel(
        ax1,
        fsca,
        ssca,
        m.get("cd45p"),
        m.get("lymph_report"),
        retain_txt=(
            f"gate n={applied_geometry_n:,}  ·  "
            f"{100.0 * applied_geometry_n / max(n_cd45, 1):.1f}% parent"
        ),
    )

    # 3. Singlets SSC-W × SSC-H
    ax1 = fig.add_subplot(gs[0, 2])
    parent = m["lymph_scatter"]
    if sscw is not None and ssch is not None and parent.sum() >= 10:
        _hb(ax1, ssch, sscw, "SSC-H", "SSC-W", mask=parent, full_axes=True)
        if m["sing"].sum():
            _shade_island(
                ax1, ssch[m["sing"]], sscw[m["sing"]],
                bins=70, level_frac=0.10,
                color="#90EE90", alpha=0.48, edge="#2E7D4F", lw=1.3,
            )
        ax1.set_title(
            f"3. Singlets\n{_pct_label(n_sing, max(n_lymph, 1), n_tot)}",
            fontsize=6.1, fontweight="bold", linespacing=0.95,
        )
    else:
        ax1.set_visible(False)

    # 4. Live
    ax2 = fig.add_subplot(gs[0, 3])
    if "ld" in g:
        ax2.hist(g["ld"][m["sing"]], bins=180, color="gray", alpha=0.85)
        ld_cut = cuts.get("ld")
        if ld_cut is not None:
            _shade_halfplane_x(ax2, ld_cut, side="below",
                               color="#88C999", alpha=0.25, edge="#2E7D4F")
        ax2.set_title(
            f"4. Live L/D−\n{_pct_label(n_live, max(n_sing, 1), n_tot)}",
            fontsize=6.1, fontweight="bold", linespacing=0.95,
        )
        ax2.set_xlabel("UV L/D", fontsize=7)
    else:
        ax2.set_visible(False)

    # 5. CD45+
    ax3 = fig.add_subplot(gs[1, 0])
    if "cd45" in g:
        ax3.hist(g["cd45"][m["live"]], bins=180, color="steelblue", alpha=0.85)
        cd45_cut = cuts.get("cd45")
        if cd45_cut is not None:
            _shade_halfplane_x(ax3, cd45_cut, side="above",
                               color="#5B9BD5", alpha=0.25, edge="#1F6AA5")
        ax3.set_title(
            f"5. CD45+\n{_pct_label(n_cd45, max(n_live, 1), n_tot)}",
            fontsize=6.1, fontweight="bold", linespacing=0.95,
        )
        ax3.set_xlabel("BV510 CD45", fontsize=7)
    else:
        ax3.set_visible(False)

    # 6. CD14±
    ax4 = fig.add_subplot(gs[1, 1])
    if "cd14" in g and m["cd45p"].sum():
        ax4.hist(g["cd14"][m["cd45p"]], bins=180, color="#A0522D", alpha=0.85)
        cd14_cut = cuts.get("cd14")
        if cd14_cut is not None:
            _shade_halfplane_x(ax4, cd14_cut, side="below",
                               color="#5B9BD5", alpha=0.22, edge="#1F6AA5")
            ax4.axvspan(cd14_cut, ax4.get_xlim()[1], facecolor=to_rgba("#A0522D", 0.18),
                        edgecolor="none", zorder=2)
        mono_cd45 = pct.get("%CD14+ (of CD45+)", "—")
        mono_live = pct.get("%CD14+ (of live)", pct.get("%Monocytes (of live)", "—"))
        ax4.set_title(
            f"6. CD14±  mono {mono_cd45}% CD45+ / {mono_live}% live",
            fontsize=7, fontweight="bold",
        )
        ax4.set_xlabel("CD14", fontsize=7)
    else:
        ax4.set_visible(False)

    # 7. CD19 (B vs CD19−)
    ax5 = fig.add_subplot(gs[1, 2])
    if "cd19" in g and m["lympho"].sum() >= 20:
        ax5.hist(g["cd19"][m["lympho"]], bins=180, color="#7E57C2", alpha=0.85)
        cd19_cut = cuts.get("cd19")
        if cd19_cut is not None:
            _shade_halfplane_x(ax5, cd19_cut, side="below",
                               color="#5B9BD5", alpha=0.20, edge="#1F6AA5")
            ax5.axvspan(cd19_cut, ax5.get_xlim()[1], facecolor=to_rgba("#7E57C2", 0.20),
                        edgecolor="none", zorder=2)
        ax5.set_title(
            f"7. CD19  B={pct.get('%B (of lymph)', '—')}% of lymph",
            fontsize=7.5, fontweight="bold",
        )
        ax5.set_xlabel("CD19", fontsize=7)
    else:
        ax5.set_visible(False)

    # 8. CD3 × CD56
    ax6 = fig.add_subplot(gs[1, 3])
    if "cd3" in g and "cd56" in g:
        _hb(ax6, g["cd3"], g["cd56"], "CD3", "CD56", mask=m["cd19n"], full_axes=True)
        c3, c56 = cuts.get("cd3"), cuts.get("cd56")
        if c3 is not None and c56 is not None:
            x0, x1 = ax6.get_xlim()
            y0, y1 = ax6.get_ylim()
            _shade_rect(ax6, c3, x1, y0, c56,
                        color="#E91E8C", alpha=GATE_ALPHA, edge="#C2185B", lw=1.0)
            _shade_rect(ax6, x0, c3, c56, y1,
                        color="#7CB342", alpha=GATE_ALPHA, edge="#558B2F", lw=1.0)
            _gate_label(ax6, "T gate", 0.98, 0.04, "#C2185B", ha="right", va="bottom")
            _gate_label(ax6, "NK gate", 0.02, 0.96, "#558B2F")
        highlights = (
            (m["NK"], "#2E7D32", "^", "NK highlight", PLOT_RNG_SEED + 1),
            (m["T"], "#C2185B", "o", "T highlight", PLOT_RNG_SEED + 2),
        )
        for mask, color, marker, label, seed in highlights:
            if mask.sum():
                idx = _deterministic_mask_indices(mask, 2000, seed)
                ax6.scatter(
                    g["cd3"][idx], g["cd56"][idx], s=5, c=color,
                    marker=marker, edgecolors="black", linewidths=0.12,
                    alpha=0.35, rasterized=True, zorder=5, label=label,
                )
        if ax6.get_legend_handles_labels()[0]:
            ax6.legend(
                loc="upper right", fontsize=4.5, framealpha=0.80,
                borderpad=0.25, labelspacing=0.2, handletextpad=0.25,
            )
        ax6.set_title(
            f"8. T/NK  T={pct.get('%T (of lymph)', '—')}%  "
            f"NK={pct.get('%NK (of lymph)', '—')}%",
            fontsize=6.3, fontweight="bold",
        )
    else:
        ax6.set_visible(False)

    # 9. CD4 × CD8 of T
    ax7 = fig.add_subplot(gs[2, 0])
    if "cd4" in g and m.get("T") is not None and m["T"].sum() >= 20:
        tmask = m["T"]
        if "cd8" in g and cuts.get("cd8") is not None:
            _hb(ax7, g["cd4"], g["cd8"], "CD4", "CD8", mask=tmask, full_axes=True)
            c4, c8 = cuts.get("cd4"), cuts.get("cd8")
            if c4 is not None:
                x0, x1 = ax7.get_xlim()
                y0, y1 = ax7.get_ylim()
                _shade_rect(ax7, c4, x1, y0, y1 if c8 is None else c8,
                            color="#FFC000", alpha=GATE_ALPHA, edge="#E6A800", lw=1.0)
                _gate_label(ax7, "CD4 gate", 0.98, 0.04, "#9A6F00",
                            ha="right", va="bottom")
                if c8 is not None:
                    _shade_rect(ax7, x0, c4, c8, y1,
                                color="#FF5252", alpha=GATE_ALPHA, edge="#C62828", lw=1.0)
                    _gate_label(ax7, "CD8 gate", 0.02, 0.96, "#C62828")
        else:
            ax7.hist(g["cd4"][tmask], bins=160, color="#FFC000", alpha=0.85)
            if cuts.get("cd4") is not None:
                _shade_halfplane_x(ax7, cuts["cd4"], side="above",
                                   color="#FFC000", alpha=0.22, edge="#E6A800")
            ax7.set_xlabel("CD4 (CD8:=CD4−)", fontsize=7)
        # Keep the heading and values as separate text artists. This guarantees the
        # percentages stay inside the panel and avoids a Matplotlib/Poppler clipping
        # artifact observed with a multiline axes title in the UPN27 report.
        ax7.set_title(
            "9. CD4/CD8 of T", fontsize=6.3, fontweight="bold", y=1.08, pad=0,
        )
        ax7.text(
            0.5, 1.025,
            f"CD4={pct.get('%CD4 (of lymph)', '—')}%  ·  "
            f"CD8={pct.get('%CD8 (of lymph)', '—')}%",
            transform=ax7.transAxes, ha="center", va="bottom",
            fontsize=6.3, fontweight="bold", clip_on=False,
        )
    else:
        ax7.axis("off")
        ax7.text(0.1, 0.5, "CD4/CD8\n(insufficient T)", fontsize=9,
                 color="#667788", transform=ax7.transAxes)

    # 10. Donor / CAR
    ax8 = fig.add_subplot(gs[2, 1])
    if "car" in g and "hla" in g and m["NK"].sum() >= 10:
        _hb(ax8, g["hla"], g["car"], "HLA", "CAR (AF647)", mask=m["NK"], full_axes=True)
        hla_c, car_c = cuts.get("hla"), cuts.get("car")
        hla_dim = bool(cuts.get("hla_dim"))
        donor_on_left = hla_dim
        if hla_c is not None:
            x0, x1 = ax8.get_xlim()
            y0, y1 = ax8.get_ylim()
            dx0, dx1 = (x0, hla_c) if hla_dim else (hla_c, x1)
            _shade_rect(ax8, dx0, dx1, y0, y1,
                        color="#4FC3F7", alpha=GATE_ALPHA, edge="#0288D1", lw=0.9)
            gate_x = 0.02 if donor_on_left else 0.98
            gate_ha = "left" if donor_on_left else "right"
            _gate_label(ax8, "Donor gate" if identity_validated else "Configured HLA region",
                        gate_x, 0.04, "#0277BD",
                        ha=gate_ha, va="bottom")
            if car_c is not None:
                _shade_rect(ax8, dx0, dx1, car_c, y1,
                            color="#FFD54F", alpha=GATE_ALPHA, edge="#F9A825", lw=1.0)
                _gate_label(ax8, "CAR+ gate" if identity_validated else "Configured CAR region",
                            gate_x, 0.96, "#9A6A00", ha=gate_ha)
        highlights = (
            (m["Donor"], "#00ACC1", 5, "D",
             "Donor highlight" if identity_validated else "HLA-region highlight", PLOT_RNG_SEED + 3),
            (m["CAR"], "#F9A825", 8, "*",
             "CAR highlight" if identity_validated else "CAR-region highlight", PLOT_RNG_SEED + 4),
        )
        for mask, color, size, marker, label, seed in highlights:
            if mask.sum():
                idx = _deterministic_mask_indices(mask, 1500, seed)
                ax8.scatter(
                    g["hla"][idx], g["car"][idx], s=size, c=color,
                    marker=marker, edgecolors="black", linewidths=0.12,
                    alpha=0.45, rasterized=True, zorder=5, label=label,
                )
        if ax8.get_legend_handles_labels()[0]:
            ax8.legend(
                loc="upper right" if donor_on_left else "upper left",
                fontsize=4.5, framealpha=0.80,
                borderpad=0.25, labelspacing=0.2, handletextpad=0.25,
            )
        ax8.set_title(
            f"10. {'Donor/CAR' if identity_validated else 'Configured HLA/CAR regions'}  "
            f"D-NK={pct.get('%Donor NK (of lymph)', '—')}%  "
            f"CAR={pct.get('%CAR+ (of Donor NK)', '—')}%",
            fontsize=6.3, fontweight="bold",
        )
    else:
        ax8.axis("off")
        ax8.text(0.1, 0.5, "Donor/CAR\n(insufficient NK)", fontsize=9,
                 color="#667788", transform=ax8.transAxes)

    # Subset summary (two compact columns; values and denominators are unchanged).
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis("off")
    lines_left = [
        f"B / lymph             {pct.get('%B (of lymph)', '—'):>8}%",
        f"T / lymph             {pct.get('%T (of lymph)', '—'):>8}%",
        f"CD4 / lymph | T       {pct.get('%CD4 (of lymph)', '—'):>8} | "
        f"{pct.get('%CD4 (of T)', '—')}%",
        f"CD8 / lymph | T       {pct.get('%CD8 (of lymph)', '—'):>8} | "
        f"{pct.get('%CD8 (of T)', '—')}%",
        f"NK / lymph            {pct.get('%NK (of lymph)', '—'):>8}%",
    ]
    lines_right = [
        f"{'Donor NK' if identity_validated else 'HLA region'} / lymph"
        f" {pct.get('%Donor NK (of lymph)', '—'):>8}%",
        f"{'CAR+' if identity_validated else 'CAR region'} / HLA"
        f"   {pct.get('%CAR+ (of Donor NK)', '—'):>8}%",
        f"CD14+ / CD45+ | live  {pct.get('%CD14+ (of CD45+)', '—'):>8} | "
        f"{pct.get('%CD14+ (of live)', pct.get('%Monocytes (of live)', '—'))}%",
        f"lineage purity        {pct.get('lineage_purity', '—'):>8}%",
        f"CD14− lymph n          {n_lympho:>8,}",
        f"viable density        {pct.get('%Lymph_scatter (of total)', '—'):>8}%",
        f"gate             {cuts.get('scatter_gate', 'density_island')}",
    ]
    ax9.text(0.02, 0.95, "Subset summary", fontsize=10, fontweight="bold",
             color="#1a3a5c", transform=ax9.transAxes, va="top")
    for x0 in (0.01, 0.505):
        ax9.add_patch(FancyBboxPatch(
            (x0, 0.27), 0.475, 0.56,
            boxstyle="round,pad=0.012", transform=ax9.transAxes,
            facecolor="#EEF3F8", edgecolor="#CCDDEE", linewidth=0.7,
            clip_on=True,
        ))
    ax9.text(
        0.025, 0.79, "\n".join(str(x) for x in lines_left), va="top",
        family="monospace", fontsize=5.35, linespacing=1.18, color="#1a3a5c",
        transform=ax9.transAxes,
    )
    ax9.text(
        0.52, 0.79, "\n".join(str(x) for x in lines_right), va="top",
        family="monospace", fontsize=5.35, linespacing=1.12, color="#1a3a5c",
        transform=ax9.transAxes,
    )

    fig.text(
        0.5,
        0.025,
        "Panel 2 is a diagnostic view of the downstream Live-CD45 parent, not the second "
        "computational gate. Gray contours = parent density; numbered circles = fitted "
        "mode centers; dashed green = low-SSC fitted mode; solid blue = applied gate.",
        ha="center",
        va="bottom",
        fontsize=5.0,
        color="#556677",
    )
    _add_page_footer(fig, page_number, total_pages)
    _save_page_outputs(
        fig, render_path=render_path, page_pdf_path=page_pdf_path,
    )
    pdf.savefig(fig)
    plt.close(fig)
    return geometry


# ── full QC report ────────────────────────────────────────────────────────────

COMPACT_CITATIONS = (
    "FSC×SSC first gate = density island (UNITO Nat Commun 2025; flowDensity), "
    "not a rectangle/quadrant.\n"
    "Singlets SSC-W×H: Rico Cytometry A 2023.  "
    "Hierarchy + transfer: Finak OpenCyto 2014; HIPC Sci Rep 2016."
)

HIERARCHY_ONELINER = (
    "Density island (FSC×SSC) → singlets (SSC-W×H) → live (L/D−) → CD45+ →\n"
    "CD14± (mono | CD14−∩lymph island) → CD19 → CD3×CD56 (T/NK) → CD4/CD8 → HLA/CAR"
)

HIERARCHY_STAGES = [
    "1 Density\nFSC×SSC\nisland",
    "2 Singlets\nSSC-W×H",
    "3 Live\nL/D−",
    "4 CD45+",
    "5 CD14±\nmono|lymph",
    "6 CD19\nB | CD19−",
    "7 CD3×CD56\nT | NK",
    "8 CD4/CD8\nDonor·CAR",
]

SUBSET_KEYS = [
    ("%B (of lymph)", "B"),
    ("%T (of lymph)", "T"),
    ("%CD4 (of lymph)", "CD4"),
    ("%CD8 (of lymph)", "CD8"),
    ("%NK (of lymph)", "NK"),
    ("%Donor NK (of lymph)", "D-NK"),
    ("%CAR+ (of Donor NK)", "CAR"),
    ("%CD14+ (of CD45+)", "CD14+/45"),
    ("%CD14+ (of live)", "CD14+/live"),
    ("lineage_purity", "purity"),
]


def _draw_hierarchy_boxes(ax, y0=0.55):
    for i, lab in enumerate(HIERARCHY_STAGES):
        x = 0.03 + i * 0.12
        box = FancyBboxPatch(
            (x, y0), 0.108, 0.14, boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#E8F1F8", edgecolor="#1a3a5c", linewidth=1.1,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(x + 0.054, y0 + 0.07, lab, ha="center", va="center",
                fontsize=6, color="#1a3a5c", transform=ax.transAxes, linespacing=1.2)
        if i < len(HIERARCHY_STAGES) - 1:
            ax.annotate(
                "", xy=(x + 0.118, y0 + 0.07), xytext=(x + 0.108, y0 + 0.07),
                arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=1.0),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
            )


def _subset_summary_page(pdf, patient_id, gate_cache, page_number=None,
                         total_pages=None, render_path=None, page_pdf_path=None,
                         technical_state="REVIEW",
                         compensation_state=None, identity_validated=False):
    """All timepoints × all reported subsets."""
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id} — {'subset' if identity_validated else 'configured technical-region'} summary",
                 fontsize=13, fontweight="bold", color="#1a3a5c")
    _add_status_banner(fig, technical_state, compensation_state)
    ax = fig.add_subplot(111)
    ax.axis("off")
    headers = ["TP"] + [s for _, s in SUBSET_KEYS] + ["viable%"]
    cell = []
    for fn, tp, m, g, scat, cuts, pct in gate_cache:
        row = [str(tp)]
        for key, _ in SUBSET_KEYS:
            v = pct.get(key)
            if key == "%CD14+ (of live)" and v is None:
                v = pct.get("%Monocytes (of live)")
            row.append("—" if v is None else f"{float(v):.1f}")
        row.append(f"{float(pct.get('%Lymph_scatter (of total)', 0)):.1f}")
        cell.append(row)
    if cell:
        table = ax.table(cellText=cell, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.05, 1.35)
        for (r, c), cell_obj in table.get_celld().items():
            if r == 0:
                cell_obj.set_facecolor("#1a3a5c")
                cell_obj.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell_obj.set_facecolor("#EEF3F8")
            cell_obj.set_edgecolor("#CCDDEE")
    else:
        ax.text(0.5, 0.5, "No sample gate pages were produced.", ha="center", va="center",
                fontsize=11, color="#556677", transform=ax.transAxes)
    ax.text(
        0.5, 0.04,
        "B/T/NK/CD4/CD8/Donor of CD14− lymph island; CD14+ of CD45+ and of live; "
        "viable% = density-island events / total. Fluorescence labels are configured regions, "
        "not validated identities." if not identity_validated else
        "B/T/NK/CD4/CD8/Donor of CD14− lymph island; CD14+ of CD45+ and of live; "
        "viable% = density-island events / total.",
        ha="center", fontsize=7, color="#556677", transform=ax.transAxes,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    _add_page_footer(fig, page_number, total_pages)
    _save_page_outputs(
        fig, render_path=render_path, page_pdf_path=page_pdf_path,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _acquisition_qc_page(pdf, patient_id, acquisition_rows, page_number=None,
                         total_pages=None, render_path=None, page_pdf_path=None,
                         technical_state="REVIEW",
                         compensation_state=None):
    """Time/event-rate/signal-instability summary; canonical events stay untouched."""
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id} — acquisition stability QC",
                 fontsize=13, fontweight="bold", color="#1a3a5c")
    _add_status_banner(fig, technical_state, compensation_state)
    ax = fig.add_subplot(111)
    ax.axis("off")
    headers = ["File", "State", "Time", "rate CV", "gap bins %", "signal bins %",
               "candidate events %", "removed"]
    cell = []
    row_colors = []
    for row in acquisition_rows or []:
        def pct(key):
            value = row.get(key)
            return "—" if value is None else f"{100.0 * float(value):.2f}"
        state = str(row.get("state") or "NOT_EVALUABLE")
        cell.append([
            str(row.get("file") or "")[:38], state, str(row.get("time_channel") or "—"),
            "—" if row.get("event_rate_cv") is None else f"{float(row['event_rate_cv']):.3f}",
            pct("gap_bin_fraction"), pct("signal_spike_proxy_bin_fraction"),
            pct("candidate_event_fraction"), "NO",
        ])
        row_colors.append(
            "#E8F5E9" if state == "PASS" else
            ("#FFF8E1" if state in {"REVIEW", "NOT_EVALUABLE"} else "#FFEBEE")
        )
    table = ax.table(cellText=cell, colLabels=headers, loc="center", cellLoc="center",
                     colWidths=[0.29, 0.09, 0.06, 0.08, 0.09, 0.10, 0.12, 0.07])
    table.auto_set_font_size(False)
    table.set_fontsize(6.0)
    table.scale(1.0, 1.30)
    for (row, column), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#CCDDEE")
        if row == 0:
            cell_obj.set_facecolor("#1a3a5c")
            cell_obj.set_text_props(color="white", fontweight="bold")
        else:
            cell_obj.set_facecolor(row_colors[row - 1])
    ax.text(
        0.5, 0.06,
        "Signal instability is a Time-binned multi-channel median proxy, not proof of a "
        "within-file voltage change. $P#V is audited separately. Canonical event removal = NO; "
        "flagged intervals are eligible only for one authorized shadow analysis.",
        ha="center", va="center", fontsize=7.2, color="#334455", wrap=True,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc="#F5F7FA", ec="#AAB7C4", lw=0.8),
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    _add_page_footer(fig, page_number, total_pages)
    _save_page_outputs(
        fig, render_path=render_path, page_pdf_path=page_pdf_path,
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_compact_qc_report(pdf, patient_id, gate_cache, cmp_summary=None,
                            compare_df=None, scatter_policy="soft_lock",
                            render_dir=None, page_pdf_dir=None, page_index_path=None,
                            composite_pdf_name=None, acquisition_rows=None,
                            geometry_path=None,
                            technical_state="REVIEW", compensation_state=None,
                            population_identity_state="BLOCKED"):
    """Full QC PDF plus named, page-identical PDF companions and a page index."""
    has_acquisition_page = acquisition_rows is not None
    total_pages = 2 + len(gate_cache) + int(has_acquisition_page)
    render_root = Path(render_dir) if render_dir is not None else None
    if render_root is not None:
        render_root.mkdir(parents=True, exist_ok=True)
    page_pdf_root = Path(page_pdf_dir) if page_pdf_dir is not None else None
    if page_pdf_root is not None:
        page_pdf_root.mkdir(parents=True, exist_ok=True)
    index_target = Path(page_index_path) if page_index_path is not None else None
    geometry_target = Path(geometry_path) if geometry_path is not None else None
    page_records = []
    geometry_records = []

    def page_pdf_path(page, slug):
        if page_pdf_root is None:
            return None
        return page_pdf_root / (
            f"{_safe_filename_token(patient_id)}_QC_page_{int(page):03d}_"
            f"{_safe_filename_token(slug)}.pdf"
        )

    def render_path(page):
        return (render_root / f"page_{int(page):03d}.png") if render_root else None

    def display_path(path):
        if path is None:
            return ""
        if index_target is None:
            return Path(path).as_posix()
        try:
            return Path(path).relative_to(index_target.parent).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def record_page(page, role, slug, *, timepoint="", sample_file=""):
        separate_pdf = page_pdf_path(page, slug)
        page_records.append({
            "composite_pdf": composite_pdf_name or "",
            "composite_page": int(page),
            "composite_total_pages": int(total_pages),
            "page_label": f"{int(page)}/{int(total_pages)}",
            "page_role": role,
            "timepoint": timepoint or "",
            "sample_file": sample_file or "",
            "separate_pdf": display_path(separate_pdf),
            "rendered_png": display_path(render_path(page)),
        })
        return separate_pdf
    # ── Page 1: methods + hierarchy + auto vs manual ─────────────────────────
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id}  ·  CAR-NK gating QC",
                 fontsize=14, fontweight="bold", color="#1a3a5c", y=0.96)
    _add_status_banner(fig, technical_state, compensation_state)
    identity_validated = str(population_identity_state).upper() == "PASS"

    ax = fig.add_axes([0.04, 0.0, 0.92, 0.92])
    ax.axis("off")
    ax.text(0.0, 0.98, "Methods", fontsize=10, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes, va="top")
    ax.text(
        0.0, 0.94,
        "First gate = connected high-density FSC-A×SSC-A island (UNITO/flowDensity); "
        "debris floor excluded.\nSoft-lock transfers the lymph SSC ceiling; FSC is "
        "re-estimated per file via density. Downstream: singlets → live → CD45+ → "
        "CD14± → CD19 → CD3×CD56 → CD4/CD8 → Donor/CAR.\n"
        "Each sample page also shows a diagnostic Live-CD45 lymph-geometry view; "
        "it does not alter gating.",
        fontsize=7.5, color="#334455", transform=ax.transAxes, va="top",
        linespacing=1.35,
    )
    ax.text(0.0, 0.83, "Gating hierarchy", fontsize=10, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes, va="top")
    _draw_hierarchy_boxes(ax, y0=0.64)
    ax.text(0.0, 0.60, HIERARCHY_ONELINER, fontsize=6.5, color="#556677",
            transform=ax.transAxes, va="top")
    ax.text(0.0, 0.53, COMPACT_CITATIONS, fontsize=6, color="#778899",
            transform=ax.transAxes, va="top")
    ax.text(
        0.0, 0.46,
        f"scatter={scatter_policy}  ·  density_island first gate  ·  "
        f"{len(gate_cache)} timepoints  ·  deterministic canonical first run",
        fontsize=6.5, color="#778899", transform=ax.transAxes, va="top",
    )
    ax.text(
        0.0,
        0.42,
        "Paired scatter views: all events → applied viable gate; then Live-CD45 parent "
        "→ parent contours + fitted modes + low-SSC mode + applied lymph gate. "
        "The second view is diagnostic only.",
        fontsize=6.2,
        color="#334455",
        transform=ax.transAxes,
        va="top",
    )

    ax_t = fig.add_axes([0.04, 0.04, 0.92, 0.34])
    ax_t.axis("off")
    ax_t.set_title("Auto vs manual (% of lymph) — all timepoints / metrics",
                   fontsize=10, fontweight="bold", color="#1a3a5c", loc="left", pad=6)

    if compare_df is not None and len(compare_df):
        df = compare_df.copy()
        metrics = ["%B (of lymph)", "%T (of lymph)", "%NK (of lymph)",
                   "%CD4 (of lymph)", "%CD8 (of lymph)",
                   "%Donor NK (of lymph)", "%CAR+ (of Donor NK)"]
        short = {"%B (of lymph)": "B", "%T (of lymph)": "T", "%NK (of lymph)": "NK",
                 "%CD4 (of lymph)": "CD4", "%CD8 (of lymph)": "CD8",
                 "%Donor NK (of lymph)": "D-NK", "%CAR+ (of Donor NK)": "CAR"}
        tps = list(dict.fromkeys(df["timepoint"].tolist()))
        col_labels = ["TP"] + [short[m] for m in metrics]
        cell, colors = [], []
        for tp in tps:
            row, crow = [tp], ["#EEF3F8"]
            for m in metrics:
                sub = df[(df.timepoint == tp) & (df.metric == m)]
                if len(sub) == 0:
                    row.append("—"); crow.append("white"); continue
                r = sub.iloc[0]
                ad = float(r["abs_delta"])
                row.append(f"{r['auto']:.1f}/{r['manual']:.1f}")
                crow.append("#E8F5E9" if ad <= 5 else ("#FFF8E1" if ad <= 10 else "#FFEBEE"))
            cell.append(row); colors.append(crow)
        tbl = ax_t.table(cellText=cell, colLabels=col_labels, loc="upper center",
                         cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.2)
        tbl.scale(1.0, 1.25)
        for (i, j), cell_obj in tbl.get_celld().items():
            cell_obj.set_edgecolor("#CCDDEE")
            if i == 0:
                cell_obj.set_facecolor("#1a3a5c")
                cell_obj.set_text_props(color="white", fontweight="bold", fontsize=6.2)
            else:
                cell_obj.set_facecolor(colors[i - 1][j])
        mae = cmp_summary or {}
        ax_t.text(
            0.0, -0.02,
            f"MAE={mae.get('mae', '—')} pp   ≤5pp={mae.get('within_5pp', '—')}%   "
            f"≤10pp={mae.get('within_10pp', '—')}%   n={mae.get('n', len(df))}",
            transform=ax_t.transAxes, fontsize=8, color="#1a3a5c", va="top",
        )
    else:
        ax_t.text(
            0.5,
            0.62,
            "MANUAL REFERENCE: NOT PROVIDED\nAutomated-versus-manual agreement is not scored.",
            ha="center",
            va="center",
            color="#556677",
            fontsize=9,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="#F5F7FA", ec="#AAB7C4", lw=0.8),
        )

    _add_page_footer(fig, 1, total_pages)
    cover_pdf = record_page(1, "methods_hierarchy_manual_comparison", "methods_hierarchy")
    _save_page_outputs(fig, render_path=render_path(1), page_pdf_path=cover_pdf)
    pdf.savefig(fig)
    plt.close(fig)

    next_page = 2
    if has_acquisition_page:
        acquisition_pdf = record_page(
            next_page, "acquisition_stability_qc", "acquisition_stability_QC"
        )
        _acquisition_qc_page(
            pdf, patient_id, acquisition_rows, page_number=next_page,
            total_pages=total_pages,
            render_path=render_path(next_page), page_pdf_path=acquisition_pdf,
            technical_state=technical_state, compensation_state=compensation_state,
        )
        next_page += 1

    # ── Subset summary table ─────────────────────────────────────────────────
    subset_pdf = record_page(
        next_page, "cross_sample_subset_summary", "temporal_cell_type_summary"
    )
    _subset_summary_page(pdf, patient_id, gate_cache, page_number=next_page,
                         total_pages=total_pages,
                         render_path=render_path(next_page), page_pdf_path=subset_pdf,
                         technical_state=technical_state, compensation_state=compensation_state,
                         identity_validated=identity_validated)

    # ── Remaining pages: full gate sequence for EVERY timepoint ──────────────
    for page_number, (fn, tp, m, g, scat, cuts, pct) in enumerate(
        gate_cache, start=next_page + 1
    ):
        sample_pdf = record_page(
            page_number,
            "sample_gate_sequence",
            f"{tp or Path(fn).stem}_gate_sequence",
            timepoint=tp,
            sample_file=fn,
        )
        geometry = qc_page(
            pdf, fn, m, g, scat, cuts, pct, timepoint=tp,
            page_number=page_number, total_pages=total_pages,
            render_path=render_path(page_number), page_pdf_path=sample_pdf,
            technical_state=technical_state, compensation_state=compensation_state,
            acquisition_state=next(
                (str(row.get("state")) for row in (acquisition_rows or [])
                 if row.get("file") == fn), None,
            ),
            identity_validated=identity_validated,
        )
        applied_gate = np.asarray(m.get("cd45p"), dtype=bool) & np.asarray(
            m.get("lymph_report"), dtype=bool,
        )
        parent_events = int(np.asarray(m.get("cd45p"), dtype=bool).sum())
        applied_events = int(applied_gate.sum())
        base_record = {
            "patient_id": patient_id,
            "timepoint": tp or "",
            "sample_file": fn,
            "parent": "Live-CD45",
            "panel": "lymph_density_geometry",
            "parent_events": parent_events,
            "applied_gate_events": applied_events,
            "applied_gate_percent": round(
                100.0 * applied_events / parent_events, 6,
            ) if parent_events else "",
            "fit_method": geometry.get("fit_method", "NOT_EVALUABLE"),
            "diagnostic_only": True,
        }
        modes = geometry.get("modes", [])
        if not modes:
            geometry_records.append(base_record | {
                "mode_number": "", "is_low_ssc_mode": "", "mode_weight": "",
                "mean_fsc_a_instrument": "", "mean_ssc_a_instrument": "",
                "cov_fsc_fsc": "", "cov_fsc_ssc": "", "cov_ssc_ssc": "",
            })
        for mode in modes:
            mean = np.asarray(mode["mean"], dtype=float)
            covariance = np.asarray(mode["covariance"], dtype=float)
            geometry_records.append(base_record | {
                "mode_number": mode["mode_number"],
                "is_low_ssc_mode": bool(mode["is_low_ssc_mode"]),
                "mode_weight": round(float(mode["weight"]), 9),
                "mean_fsc_a_instrument": round(float(mean[0]), 6),
                "mean_ssc_a_instrument": round(float(mean[1]), 6),
                "cov_fsc_fsc": round(float(covariance[0, 0]), 6),
                "cov_fsc_ssc": round(float(covariance[0, 1]), 6),
                "cov_ssc_ssc": round(float(covariance[1, 1]), 6),
            })

    if index_target is not None:
        index_target.parent.mkdir(parents=True, exist_ok=True)
        with index_target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(page_records[0]))
            writer.writeheader()
            writer.writerows(page_records)
    if geometry_target is not None:
        geometry_target.parent.mkdir(parents=True, exist_ok=True)
        with geometry_target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LYMPH_GEOMETRY_FIELDS)
            writer.writeheader()
            writer.writerows(geometry_records)


def write_longitudinal_composition_report(
    multi_df,
    out_path,
    *,
    patient_id,
    technical_state="REVIEW",
    compensation_state=None,
    population_identity_state="BLOCKED",
    temporal_summary_path=None,
):
    """Create a two-row longitudinal report: percentages above, abundance below.

    The lower row uses exact-date ALC-calibrated K/uL values when the temporal summary
    contains them. Otherwise it falls back to explicitly labeled gated-event counts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(multi_df.to_dict("records"))
    timepoints = [str(record.get("timepoint") or Path(str(record.get("file", ""))).stem)
                  for record in records]
    x = np.arange(len(records), dtype=float)
    identity_validated = str(population_identity_state).upper() == "PASS"

    series = {
        "B (% of CD14-negative lymph region)": ("%B (of lymph)", "n_B"),
        "T (% of CD14-negative lymph region)": ("%T (of lymph)", "n_T"),
        "CD4 (% of CD14-negative lymph region)": ("%CD4 (of lymph)", "n_CD4"),
        "CD8 (% of CD14-negative lymph region)": ("%CD8 (of lymph)", "n_CD8"),
        "NK (% of CD14-negative lymph region)": ("%NK (of lymph)", "n_NK"),
        "Donor NK (% of CD14-negative lymph region)": ("%Donor NK (of lymph)", "n_Donor"),
        "Lymphocytes (% of viable live events)": ("%Lymphocytes (of live)", "n_lympho"),
        "CD14+ (% of viable live events)": ("%CD14+ (of live)", "n_CD14p"),
        "CAR region (% of configured Donor NK region)": ("%CAR+ (of Donor NK)", "n_CAR"),
    }
    left_names = [
        "B (% of CD14-negative lymph region)", "T (% of CD14-negative lymph region)",
        "CD4 (% of CD14-negative lymph region)", "CD8 (% of CD14-negative lymph region)",
        "NK (% of CD14-negative lymph region)", "Donor NK (% of CD14-negative lymph region)",
    ]
    right_names = [
        "Lymphocytes (% of viable live events)", "CD14+ (% of viable live events)",
        "CAR region (% of configured Donor NK region)",
    ]
    colors = {
        "B (% of CD14-negative lymph region)": "#E69F00",
        "T (% of CD14-negative lymph region)": "#0072B2",
        "CD4 (% of CD14-negative lymph region)": "#56B4E9",
        "CD8 (% of CD14-negative lymph region)": "#CC79A7",
        "NK (% of CD14-negative lymph region)": "#009E73",
        "Donor NK (% of CD14-negative lymph region)": "#D55E00",
        "Lymphocytes (% of viable live events)": "#4C78A8",
        "CD14+ (% of viable live events)": "#7F7F7F",
        "CAR region (% of configured Donor NK region)": "#6A3D9A",
    }
    markers = {
        name: marker for name, marker in zip(
            series, ("o", "s", "^", "v", "D", "P", "X", "h", "*")
        )
    }
    population_label = {
        "B (% of CD14-negative lymph region)": "B",
        "T (% of CD14-negative lymph region)": "T",
        "CD4 (% of CD14-negative lymph region)": "CD4",
        "CD8 (% of CD14-negative lymph region)": "CD8",
        "NK (% of CD14-negative lymph region)": "NK",
        "Donor NK (% of CD14-negative lymph region)": "Donor NK",
        "Lymphocytes (% of viable live events)": "Lymphocytes",
        "CD14+ (% of viable live events)": "CD14+",
        "CAR region (% of configured Donor NK region)": "CAR+",
    }
    absolute_lookup = {}
    if temporal_summary_path is not None and Path(temporal_summary_path).is_file():
        with Path(temporal_summary_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    value = float(row.get("absolute_abundance", ""))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    absolute_lookup[(str(row.get("timepoint")), str(row.get("population")))] = value
    absolute_mode = bool(absolute_lookup)

    def numeric(column):
        values = []
        for record in records:
            try:
                value = float(record.get(column))
                values.append(value if np.isfinite(value) else np.nan)
            except (TypeError, ValueError):
                values.append(np.nan)
        return np.asarray(values, dtype=float)

    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    # Keep every legend outside the plotting rectangles.  The percentage
    # legends need their own row because their exact denominator labels are
    # intentionally long; the shorter abundance labels share a figure-level
    # band below both lower panels.  This is layout-only and does not alter any
    # plotted value, axis, timepoint, or visual encoding.
    layout = fig.add_gridspec(
        3,
        2,
        left=0.07,
        right=0.985,
        bottom=0.19,
        top=0.90,
        hspace=0.28,
        wspace=0.22,
        height_ratios=(1.0, 0.22, 1.0),
    )
    axes = np.empty((2, 2), dtype=object)
    percentage_legend_axes = np.empty(2, dtype=object)
    for column in range(2):
        axes[0, column] = fig.add_subplot(layout[0, column])
        percentage_legend_axes[column] = fig.add_subplot(layout[1, column])
        axes[1, column] = fig.add_subplot(
            layout[2, column], sharex=axes[0, column]
        )
        percentage_legend_axes[column].set_axis_off()
    state_label = "validated identities" if identity_validated else "configured technical regions"
    fig.suptitle(
        f"{patient_id} - longitudinal {state_label}",
        fontsize=14,
        fontweight="bold",
        color="#1a3a5c",
        y=0.965,
    )
    _add_status_banner(fig, technical_state, compensation_state)

    for column, names in enumerate((left_names, right_names)):
        pct_ax = axes[0, column]
        count_ax = axes[1, column]
        for name in names:
            pct_col, count_col = series[name]
            pct = numeric(pct_col)
            count = numeric(count_col)
            if absolute_mode:
                lower = np.asarray([
                    absolute_lookup.get((timepoint, population_label[name]), np.nan)
                    for timepoint in timepoints
                ], dtype=float)
            else:
                lower = count
            style = "--" if name in {
                "Donor NK (% of CD14-negative lymph region)",
                "CAR region (% of configured Donor NK region)",
            } else "-"
            pct_ax.plot(
                x, pct, label=name, color=colors[name], marker=markers[name],
                linestyle=style, linewidth=1.5, markersize=4.2,
            )
            if np.isfinite(lower).any():
                count_ax.plot(
                    x, lower, label=population_label[name], color=colors[name],
                    marker=markers[name], linestyle=style, linewidth=1.5, markersize=4.2,
                )

        pct_ax.set_ylim(-2, 102)
        pct_ax.set_ylabel("Percent")
        pct_ax.grid(True, color="#D9E2EA", linewidth=0.6, alpha=0.8)
        handles, labels = pct_ax.get_legend_handles_labels()
        percentage_legend_axes[column].legend(
            handles,
            labels,
            loc="center",
            fontsize=6.3,
            frameon=True,
            ncol=2 if column == 0 else 1,
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=2.2,
            columnspacing=0.9,
        )
        count_ax.set_yscale("symlog", linthresh=10)
        count_ax.set_ylim(bottom=0)
        count_ax.set_ylabel(
            "ALC-calibrated abundance (K/uL, symlog)" if absolute_mode
            else "Gated event count (symlog)"
        )
        count_ax.grid(True, color="#D9E2EA", linewidth=0.6, alpha=0.8)

    lower_handles = []
    lower_labels = []
    for count_ax in axes[1, :]:
        for handle, label in zip(*count_ax.get_legend_handles_labels()):
            if label not in lower_labels:
                lower_handles.append(handle)
                lower_labels.append(label)
    if lower_handles:
        fig.legend(
            lower_handles,
            lower_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            fontsize=6.3,
            frameon=True,
            ncol=min(6, len(lower_handles)),
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=2.2,
            columnspacing=1.1,
        )

    axes[0, 0].set_title("Denominator: CD14-negative lymph region", fontsize=9,
                         fontweight="bold", color="#1a3a5c")
    axes[0, 1].set_title(
        "Lymphocytes/CD14+: viable live events | CAR: configured Donor NK region", fontsize=8.3,
                         fontweight="bold", color="#1a3a5c")
    lower_title = (
        "Exact-date ALC-calibrated abundance estimates" if absolute_mode
        else "Corresponding gated-event counts"
    )
    axes[1, 0].set_title(lower_title, fontsize=9,
                         fontweight="bold", color="#1a3a5c")
    axes[1, 1].set_title(lower_title, fontsize=9,
                         fontweight="bold", color="#1a3a5c")
    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Timepoint")
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    lower_note = (
        "Bottom row: K/uL estimates use only exact flow.csv date matches to clinical ALC; "
        "no interpolation or nearest-date imputation. D3 is unmatched and CD14+ is omitted "
        "because ALC does not calibrate monocyte abundance. "
        if absolute_mode else
        "Bottom row: gated events in the analyzed event set; these are not cells/uL or "
        "biological absolute abundance. "
    )
    fig.text(
        0.5,
        0.012,
        "Top-left denominator: CD14-negative lymph region. Top-right: Lymphocytes and "
        "CD14+ use viable live events; CAR region uses the configured Donor NK region. "
        + lower_note +
        "Dashed donor/CAR traces require upstream compensation, "
        "semantic, and manual-truth validation.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#334455",
        wrap=True,
    )
    fig.savefig(out_path, format="pdf", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)
    return str(out_path)


# Back-compat alias
write_full_qc_report = write_compact_qc_report
