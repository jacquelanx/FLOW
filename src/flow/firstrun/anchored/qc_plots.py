#!/usr/bin/env python3
"""Structured multi-page QC PDF for CAR-NK hierarchical gating.

Page layout
-----------
  1. Cover — methods one-liner, full hierarchy diagram, auto vs manual table
  2. Subset summary — all timepoints × all reported subsets
  3+. One page per timepoint — full gate sequence (scatter → … → CAR+)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap, LogNorm
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from datetime import datetime
from scipy.ndimage import gaussian_filter


# UNITO / Nat Commun–style gate fill (semi-transparent islands, not harsh lines)
GATE_FILL = "#5B9BD5"
GATE_EDGE = "#1F6AA5"
GATE_ALPHA = 0.28
GATE_EDGE_LW = 1.1


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


def _hb(ax, x, y, xlab, ylab, mask=None, bins=160, full_axes=True,
        parent_x=None, parent_y=None):
    """Bivariate density. Optionally show full parent cloud under a gate mask."""
    x_raw = np.asarray(x, float)
    y_raw = np.asarray(y, float)
    if parent_x is not None and parent_y is not None:
        px, py = _subsample(parent_x, parent_y, n=80_000)
        if len(px) >= 10:
            ax.hist2d(px, py, bins=bins, cmap="Greys", cmin=1, alpha=0.55)
    if mask is not None:
        x_plot, y_plot = x_raw[mask], y_raw[mask]
    else:
        x_plot, y_plot = x_raw, y_raw
    if len(x_plot) > 80_000:
        idx = np.random.RandomState(0).choice(len(x_plot), 80_000, replace=False)
        x_plot, y_plot = x_plot[idx], y_plot[idx]
    if len(x_plot) < 10 and (parent_x is None or len(np.asarray(parent_x)) < 10):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        return
    ax.set_facecolor("white")
    if len(x_plot) >= 10:
        ax.hist2d(x_plot, y_plot, bins=bins, cmap="YlOrRd", cmin=1, alpha=0.92)
    if full_axes:
        if parent_x is not None and parent_y is not None:
            (x0, x1), (y0, y1) = _data_extent(parent_x, parent_y)
        else:
            (x0, x1), (y0, y1) = _data_extent(x_raw, y_raw)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
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
        f"scatter={scatter_policy}  ·  {datetime.utcnow():%Y-%m-%d}",
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

def _lymph_density_panel(ax, fsca, ssca, lymph_mask, cuts=None, retain_txt="",
                         report_mask=None):
    """FSA×SSC heat density + UNITO density-island shade (NOT a rectangle).

    Primary shade = viable cellular density island. Dashed line = lymph-SSC
    reporting ceiling used for B/T/NK %. Optional report_mask adds a second
    contour for the lymph density island.
    """
    from .gates import heat_density_2d

    x_all, y_all = _subsample(fsca, ssca, n=180_000)
    if len(x_all) < 20:
        ax.set_visible(False)
        return

    (x_lo, x_hi), (y_lo, y_hi) = _data_extent(fsca, ssca, full_scale=True)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    dens = heat_density_2d(
        fsca, ssca, bins=260, smooth=2.0,
        range_=((x_lo, x_hi), (y_lo, y_hi)),
    )
    if dens is None:
        return
    H, xc, yc, xe, ye = dens
    H_plot = np.ma.masked_where(H <= 0, H)
    vmin = max(float(np.percentile(H[H > 0], 3)) if (H > 0).any() else 0.1, 0.1)
    dens_cmap = plt.cm.YlOrRd.copy()
    dens_cmap.set_bad("white")
    ax.set_facecolor("white")
    ax.pcolormesh(
        xe, ye, H_plot, cmap=dens_cmap,
        norm=LogNorm(vmin=vmin, vmax=max(float(H.max()), vmin * 10)),
        shading="auto", rasterized=True, zorder=1,
    )

    # Density island shade from the actual gate mask (no FSC×SSC rectangle)
    if lymph_mask is not None and int(np.sum(lymph_mask)) > 20:
        _shade_island(
            ax, fsca[lymph_mask], ssca[lymph_mask],
            bins=100, level_frac=0.06,
            color=GATE_FILL, alpha=0.32, edge=GATE_EDGE, lw=1.5,
        )
    if report_mask is not None and int(np.sum(report_mask)) > 20:
        _shade_island(
            ax, fsca[report_mask], ssca[report_mask],
            bins=80, level_frac=0.12,
            color="#C45C26", alpha=0.18, edge="#C45C26", lw=1.0,
        )

    shi = (cuts or {}).get("ssc_hi")
    if shi is not None:
        ax.axhline(float(shi), color="#C45C26", ls="--", lw=1.0, alpha=0.85, zorder=4)
        ax.text(0.98, float(shi), " lymph SSC", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=5.5, color="#C45C26")

    ax.set_xlabel("FSC-A", fontsize=8)
    ax.set_ylabel("SSC-A", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("1. Density island — FSC-A × SSC-A", fontsize=9,
                 fontweight="bold", color="#1a3a5c")
    ax.text(
        0.02, 0.98, f"{retain_txt}\ndensity island (not rect)",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
        color="#1a3a5c",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85, lw=0.5),
    )


def qc_page(pdf, fname, m, g, scat, cuts, pct, timepoint: str | None = None):
    """One page per timepoint: full hierarchy (3×4) + retention strip."""
    fsca, ssca = scat[0], scat[1]
    sscw = scat[2] if len(scat) > 2 else None
    ssch = scat[3] if len(scat) > 3 else None
    steps, n_tot = _retention_row(m)
    title_tp = timepoint or fname
    fig = plt.figure(figsize=(14.0, 10.0))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{title_tp}   ·   {fname}", fontsize=11, fontweight="bold",
                 color="#1a3a5c", y=0.985)

    ax_bar = fig.add_axes([0.06, 0.925, 0.88, 0.035])
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
        3, 4, figure=fig, left=0.05, right=0.98, top=0.90, bottom=0.04,
        wspace=0.30, hspace=0.42,
    )
    n_lymph = int(m["lymph_scatter"].sum())
    n_sing = int(m["sing"].sum())
    n_live = int(m["live"].sum())
    n_cd45 = int(m["cd45p"].sum())
    n_lympho = int(m["lympho"].sum())

    # 1. Density island scatter
    ax0 = fig.add_subplot(gs[0, 0])
    _lymph_density_panel(
        ax0, fsca, ssca, m.get("lymph_scatter"), cuts,
        retain_txt=_pct_label(n_lymph, n_tot, n_tot),
        report_mask=m.get("lymph_report"),
    )

    # 2. Singlets SSC-W × SSC-H
    ax1 = fig.add_subplot(gs[0, 1])
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
            f"2. Singlets  ({_pct_label(n_sing, max(n_lymph, 1), n_tot)})",
            fontsize=7.5, fontweight="bold",
        )
    else:
        ax1.set_visible(False)

    # 3. Live
    ax2 = fig.add_subplot(gs[0, 2])
    if "ld" in g:
        ax2.hist(g["ld"][m["sing"]], bins=180, color="gray", alpha=0.85)
        ld_cut = cuts.get("ld")
        if ld_cut is not None:
            _shade_halfplane_x(ax2, ld_cut, side="below",
                               color="#88C999", alpha=0.25, edge="#2E7D4F")
        ax2.set_title(
            f"3. Live L/D−  ({_pct_label(n_live, max(n_sing, 1), n_tot)})",
            fontsize=7.5, fontweight="bold",
        )
        ax2.set_xlabel("UV L/D", fontsize=7)
    else:
        ax2.set_visible(False)

    # 4. CD45+
    ax3 = fig.add_subplot(gs[0, 3])
    if "cd45" in g:
        ax3.hist(g["cd45"][m["live"]], bins=180, color="steelblue", alpha=0.85)
        cd45_cut = cuts.get("cd45")
        if cd45_cut is not None:
            _shade_halfplane_x(ax3, cd45_cut, side="above",
                               color="#5B9BD5", alpha=0.25, edge="#1F6AA5")
        ax3.set_title(
            f"4. CD45+  ({_pct_label(n_cd45, max(n_live, 1), n_tot)})",
            fontsize=7.5, fontweight="bold",
        )
        ax3.set_xlabel("BV510 CD45", fontsize=7)
    else:
        ax3.set_visible(False)

    # 5. CD14±
    ax4 = fig.add_subplot(gs[1, 0])
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
            f"5. CD14±  mono {mono_cd45}% CD45+ / {mono_live}% live",
            fontsize=7, fontweight="bold",
        )
        ax4.set_xlabel("CD14", fontsize=7)
    else:
        ax4.set_visible(False)

    # 6. CD19 (B vs CD19−)
    ax5 = fig.add_subplot(gs[1, 1])
    if "cd19" in g and m["lympho"].sum() >= 20:
        ax5.hist(g["cd19"][m["lympho"]], bins=180, color="#7E57C2", alpha=0.85)
        cd19_cut = cuts.get("cd19")
        if cd19_cut is not None:
            _shade_halfplane_x(ax5, cd19_cut, side="below",
                               color="#5B9BD5", alpha=0.20, edge="#1F6AA5")
            ax5.axvspan(cd19_cut, ax5.get_xlim()[1], facecolor=to_rgba("#7E57C2", 0.20),
                        edgecolor="none", zorder=2)
        ax5.set_title(
            f"6. CD19  B={pct.get('%B (of lymph)', '—')}% of lymph",
            fontsize=7.5, fontweight="bold",
        )
        ax5.set_xlabel("CD19", fontsize=7)
    else:
        ax5.set_visible(False)

    # 7. CD3 × CD56
    ax6 = fig.add_subplot(gs[1, 2])
    if "cd3" in g and "cd56" in g:
        _hb(ax6, g["cd3"], g["cd56"], "CD3", "CD56", mask=m["cd19n"], full_axes=True)
        c3, c56 = cuts.get("cd3"), cuts.get("cd56")
        if c3 is not None and c56 is not None:
            x0, x1 = ax6.get_xlim()
            y0, y1 = ax6.get_ylim()
            _shade_rect(ax6, c3, x1, y0, c56,
                        color="#E91E8C", alpha=0.18, edge="#C2185B", lw=1.0)
            _shade_rect(ax6, x0, c3, c56, y1,
                        color="#7CB342", alpha=0.22, edge="#558B2F", lw=1.0)
        for mask, color in ((m["NK"], "lime"), (m["T"], "magenta")):
            if mask.sum():
                idx = np.where(mask)[0]
                if len(idx) > 2000:
                    idx = np.random.choice(idx, 2000, replace=False)
                ax6.scatter(g["cd3"][idx], g["cd56"][idx], s=2, c=color,
                            alpha=0.30, rasterized=True, zorder=5)
        ax6.set_title(
            f"7. T/NK  T={pct.get('%T (of lymph)', '—')}%  "
            f"NK={pct.get('%NK (of lymph)', '—')}%",
            fontsize=7, fontweight="bold",
        )
    else:
        ax6.set_visible(False)

    # 8. CD4 × CD8 of T
    ax7 = fig.add_subplot(gs[1, 3])
    if "cd4" in g and m.get("T") is not None and m["T"].sum() >= 20:
        tmask = m["T"]
        if "cd8" in g and cuts.get("cd8") is not None:
            _hb(ax7, g["cd4"], g["cd8"], "CD4", "CD8", mask=tmask, full_axes=True)
            c4, c8 = cuts.get("cd4"), cuts.get("cd8")
            if c4 is not None:
                x0, x1 = ax7.get_xlim()
                y0, y1 = ax7.get_ylim()
                _shade_rect(ax7, c4, x1, y0, y1 if c8 is None else c8,
                            color="#FFC000", alpha=0.18, edge="#E6A800", lw=1.0)
                if c8 is not None:
                    _shade_rect(ax7, x0, c4, c8, y1,
                                color="#FF5252", alpha=0.16, edge="#C62828", lw=1.0)
        else:
            ax7.hist(g["cd4"][tmask], bins=160, color="#FFC000", alpha=0.85)
            if cuts.get("cd4") is not None:
                _shade_halfplane_x(ax7, cuts["cd4"], side="above",
                                   color="#FFC000", alpha=0.22, edge="#E6A800")
            ax7.set_xlabel("CD4 (CD8:=CD4−)", fontsize=7)
        ax7.set_title(
            f"8. CD4/CD8 of T  "
            f"CD4={pct.get('%CD4 (of lymph)', '—')}%  "
            f"CD8={pct.get('%CD8 (of lymph)', '—')}%",
            fontsize=7, fontweight="bold",
        )
    else:
        ax7.axis("off")
        ax7.text(0.1, 0.5, "CD4/CD8\n(insufficient T)", fontsize=9,
                 color="#667788", transform=ax7.transAxes)

    # 9. Donor / CAR
    ax8 = fig.add_subplot(gs[2, 0])
    if "car" in g and "hla" in g and m["NK"].sum() >= 10:
        _hb(ax8, g["hla"], g["car"], "HLA", "CAR (AF647)", mask=m["NK"], full_axes=True)
        hla_c, car_c = cuts.get("hla"), cuts.get("car")
        hla_dim = bool(cuts.get("hla_dim"))
        if hla_c is not None:
            x0, x1 = ax8.get_xlim()
            y0, y1 = ax8.get_ylim()
            dx0, dx1 = (x0, hla_c) if hla_dim else (hla_c, x1)
            _shade_rect(ax8, dx0, dx1, y0, y1,
                        color="#4FC3F7", alpha=0.16, edge="#0288D1", lw=0.9)
            if car_c is not None:
                _shade_rect(ax8, dx0, dx1, car_c, y1,
                            color="#FFD54F", alpha=0.22, edge="#F9A825", lw=1.0)
        for mask, color, s in ((m["Donor"], "cyan", 2), (m["CAR"], "yellow", 4)):
            if mask.sum():
                idx = np.where(mask)[0]
                if len(idx) > 1500:
                    idx = np.random.choice(idx, 1500, replace=False)
                ax8.scatter(g["hla"][idx], g["car"][idx], s=s, c=color,
                            alpha=0.40, rasterized=True, zorder=5)
        ax8.set_title(
            f"9. Donor/CAR  D-NK={pct.get('%Donor NK (of lymph)', '—')}%  "
            f"CAR={pct.get('%CAR+ (of Donor NK)', '—')}%",
            fontsize=7, fontweight="bold",
        )
    else:
        ax8.axis("off")
        ax8.text(0.1, 0.5, "Donor/CAR\n(insufficient NK)", fontsize=9,
                 color="#667788", transform=ax8.transAxes)

    # 10. Subset summary (spans remaining columns)
    ax9 = fig.add_subplot(gs[2, 1:])
    ax9.axis("off")
    lines = [
        f"B (of lymph)           {pct.get('%B (of lymph)', '—'):>8}%",
        f"T (of lymph)           {pct.get('%T (of lymph)', '—'):>8}%",
        f"CD4 (of lymph / of T)  {pct.get('%CD4 (of lymph)', '—'):>8}% / "
        f"{pct.get('%CD4 (of T)', '—')}%",
        f"CD8 (of lymph / of T)  {pct.get('%CD8 (of lymph)', '—'):>8}% / "
        f"{pct.get('%CD8 (of T)', '—')}%",
        f"NK (of lymph)          {pct.get('%NK (of lymph)', '—'):>8}%",
        f"Donor NK (of lymph)    {pct.get('%Donor NK (of lymph)', '—'):>8}%",
        f"CAR+ (of Donor NK)     {pct.get('%CAR+ (of Donor NK)', '—'):>8}%",
        f"CD14+ (of CD45+ / live){pct.get('%CD14+ (of CD45+)', '—'):>8}% / "
        f"{pct.get('%CD14+ (of live)', pct.get('%Monocytes (of live)', '—'))}%",
        f"lineage purity         {pct.get('lineage_purity', '—'):>8}%",
        f"n CD14− lymph = {n_lympho:,}   viable dens% = "
        f"{pct.get('%Lymph_scatter (of total)', '—')}   "
        f"gate={cuts.get('scatter_gate', 'density_island')}",
    ]
    ax9.text(0.02, 0.95, "Subset summary", fontsize=10, fontweight="bold",
             color="#1a3a5c", transform=ax9.transAxes, va="top")
    ax9.text(
        0.02, 0.82, "\n".join(str(x) for x in lines), va="top",
        family="monospace", fontsize=8.5, color="#1a3a5c",
        transform=ax9.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc="#EEF3F8", ec="#CCDDEE"),
    )

    pdf.savefig(fig)
    plt.close(fig)


# ── full QC report ────────────────────────────────────────────────────────────

COMPACT_CITATIONS = (
    "FSC×SSC first gate = density island (UNITO Nat Commun 2025; flowDensity), "
    "not a rectangle/quadrant.  "
    "Singlets SSC-W×H: Rico Cytometry A 2023.  "
    "Hierarchy + transfer: Finak OpenCyto 2014; HIPC Sci Rep 2016."
)

HIERARCHY_ONELINER = (
    "Density island (FSC×SSC) → singlets (SSC-W×H) → live (L/D−) → CD45+ → "
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


def _subset_summary_page(pdf, patient_id, gate_cache):
    """All timepoints × all reported subsets."""
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id} — subset summary (all timepoints)",
                 fontsize=13, fontweight="bold", color="#1a3a5c")
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
    ax.text(
        0.5, 0.04,
        "B/T/NK/CD4/CD8/Donor of CD14− lymph island; CD14+ of CD45+ and of live; "
        "viable% = density-island events / total.",
        ha="center", fontsize=7, color="#556677", transform=ax.transAxes,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def write_compact_qc_report(pdf, patient_id, gate_cache, cmp_summary=None,
                            compare_df=None, scatter_policy="soft_lock"):
    """Full QC PDF: cover + subset table + one full-sequence page per timepoint."""
    # ── Page 1: methods + hierarchy + auto vs manual ─────────────────────────
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{patient_id}  ·  CAR-NK gating QC",
                 fontsize=14, fontweight="bold", color="#1a3a5c", y=0.97)

    ax = fig.add_axes([0.04, 0.0, 0.92, 0.92])
    ax.axis("off")
    ax.text(0.0, 0.98, "Methods", fontsize=10, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes, va="top")
    ax.text(
        0.0, 0.94,
        "First gate = connected high-density FSC-A×SSC-A island (UNITO/flowDensity); "
        "debris floor excluded. Soft-lock transfers lymph SSC ceiling; FSC re-estimated "
        "per file via density. Downstream: singlets → live → CD45+ → CD14± → CD19 → "
        "CD3×CD56 → CD4/CD8 → Donor/CAR.",
        fontsize=7.5, color="#334455", transform=ax.transAxes, va="top",
    )
    ax.text(0.0, 0.86, "Gating hierarchy", fontsize=10, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes, va="top")
    _draw_hierarchy_boxes(ax, y0=0.70)
    ax.text(0.0, 0.66, HIERARCHY_ONELINER, fontsize=6.5, color="#556677",
            transform=ax.transAxes, va="top")
    ax.text(0.0, 0.62, COMPACT_CITATIONS, fontsize=6, color="#778899",
            transform=ax.transAxes, va="top")
    ax.text(
        0.0, 0.57,
        f"scatter={scatter_policy}  ·  density_island first gate  ·  "
        f"{len(gate_cache)} timepoints  ·  {datetime.utcnow():%Y-%m-%d}",
        fontsize=6.5, color="#778899", transform=ax.transAxes, va="top",
    )

    ax_t = fig.add_axes([0.04, 0.04, 0.92, 0.50])
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
        ax_t.text(0.5, 0.5, "No manual reference compare available",
                  ha="center", va="center", color="#8899aa", fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2: subset summary table ─────────────────────────────────────────
    _subset_summary_page(pdf, patient_id, gate_cache)

    # ── Pages 3+: full gate sequence for EVERY timepoint ─────────────────────
    for fn, tp, m, g, scat, cuts, pct in gate_cache:
        qc_page(pdf, fn, m, g, scat, cuts, pct, timepoint=tp)


# Back-compat alias
write_full_qc_report = write_compact_qc_report
