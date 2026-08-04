#!/usr/bin/env python3
"""Per-timepoint QC PDF pages — FSA × SSA is panel 1."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _hb(ax, x, y, xlab, ylab, mask=None, bins=180):
    if mask is not None:
        x, y = x[mask], y[mask]
    if len(x) > 80_000:
        idx = np.random.RandomState(0).choice(len(x), 80_000, replace=False)
        x, y = x[idx], y[idx]
    if len(x) < 10:
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); return
    ax.hist2d(x, y, bins=bins, cmap="viridis", cmin=1)
    ax.set_xlabel(xlab, fontsize=7)
    ax.set_ylabel(ylab, fontsize=7)
    ax.tick_params(labelsize=6)


def _vline(ax, x, color="w", ls="--"):
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=1.0)


def _hline(ax, y, color="w", ls="--"):
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=1.0)


def qc_page(pdf, fname, m, g, scat, cuts, pct):
    fsca, ssca = scat
    fig, ax = plt.subplots(2, 4, figsize=(14, 7.2))
    fig.suptitle(fname, fontsize=11, fontweight="bold", color="#1a3a5c")

    # [0,0] FSA × SSA FIRST — lymph gate
    _hb(ax[0, 0], fsca, ssca, "FSC-A", "SSC-A")
    if m["lymph_scatter"].sum():
        idx = np.where(m["lymph_scatter"])[0]
        if len(idx) > 4000:
            idx = np.random.choice(idx, 4000, replace=False)
        ax[0, 0].scatter(fsca[idx], ssca[idx], s=1, c="red", alpha=0.25, rasterized=True)
    _vline(ax[0, 0], cuts.get("fsc_lo"), color="cyan")
    _vline(ax[0, 0], cuts.get("fsc_hi"), color="cyan")
    _hline(ax[0, 0], cuts.get("ssc_hi"), color="cyan")
    ax[0, 0].set_title("1. Lymphocytes (FSA × SSA)", fontsize=8, fontweight="bold")

    # [0,1] Singlets
    fsch = g.get("_fsch")  # may be absent; plot FSC-A vs ratio proxy via live mask tint
    _hb(ax[0, 1], fsca, ssca, "FSC-A", "SSC-A", mask=m["lymph_scatter"])
    if m["sing"].sum():
        idx = np.where(m["sing"])[0]
        if len(idx) > 3000:
            idx = np.random.choice(idx, 3000, replace=False)
        ax[0, 1].scatter(fsca[idx], ssca[idx], s=1, c="lime", alpha=0.3, rasterized=True)
    ax[0, 1].set_title("2. Singlets (green)", fontsize=8, fontweight="bold")

    # [0,2] CD14 (monocytes) on CD45+
    if "cd14" in g and m["cd45p"].sum():
        ax[0, 2].hist(g["cd14"][m["cd45p"]], bins=200, color="#A0522D", alpha=0.85)
        _vline(ax[0, 2], cuts.get("cd14"), color="r")
        ax[0, 2].set_title("3. CD14+ monocytes", fontsize=8, fontweight="bold")
        ax[0, 2].set_xlabel("CD14", fontsize=7)
    elif "ld" in g:
        ax[0, 2].hist(g["ld"][m["sing"]], bins=200, color="gray", alpha=0.8)
        _vline(ax[0, 2], cuts.get("ld"), color="r")
        ax[0, 2].set_title("3. Live (L/D-)", fontsize=8, fontweight="bold")
        ax[0, 2].set_xlabel("UV L/D", fontsize=7)
    else:
        ax[0, 2].set_visible(False)

    # [0,3] CD45
    if "cd45" in g:
        ax[0, 3].hist(g["cd45"][m["live"]], bins=200, color="steelblue", alpha=0.85)
        _vline(ax[0, 3], cuts.get("cd45"), color="r")
        ax[0, 3].set_title("4. CD45+", fontsize=8, fontweight="bold")
        ax[0, 3].set_xlabel("BV510 CD45", fontsize=7)
    else:
        ax[0, 3].set_visible(False)

    # [1,0] CD3 × CD56
    if "cd3" in g and "cd56" in g:
        _hb(ax[1, 0], g["cd3"], g["cd56"], "CD3", "CD56", mask=m["cd19n"])
        _vline(ax[1, 0], cuts.get("cd3"), color="w")
        _hline(ax[1, 0], cuts.get("cd56"), color="w")
        if m["NK"].sum():
            idx = np.where(m["NK"])[0]
            if len(idx) > 2500:
                idx = np.random.choice(idx, 2500, replace=False)
            ax[1, 0].scatter(g["cd3"][idx], g["cd56"][idx], s=2, c="lime", alpha=0.4, rasterized=True)
        if m["T"].sum():
            idx = np.where(m["T"])[0]
            if len(idx) > 2500:
                idx = np.random.choice(idx, 2500, replace=False)
            ax[1, 0].scatter(g["cd3"][idx], g["cd56"][idx], s=2, c="magenta", alpha=0.35, rasterized=True)
        ax[1, 0].set_title("5. NK(lime) / T(magenta)", fontsize=8, fontweight="bold")

    # [1,1] CD19 × CD14 — B cells + monocytes (FlowJo-style)
    if "cd19" in g and "cd14" in g:
        parent = m["cd45p"] if m["cd45p"].sum() else m["live"]
        _hb(ax[1, 1], g["cd19"], g["cd14"], "CD19", "CD14", mask=parent)
        _vline(ax[1, 1], cuts.get("cd19"), color="w")
        _hline(ax[1, 1], cuts.get("cd14"), color="w")
        if m["B"].sum():
            idx = np.where(m["B"])[0]
            if len(idx) > 2000:
                idx = np.random.choice(idx, 2000, replace=False)
            ax[1, 1].scatter(g["cd19"][idx], g["cd14"][idx], s=3, c="cyan", alpha=0.5, rasterized=True)
        if "CD14p" in m and m["CD14p"].sum():
            idx = np.where(m["CD14p"])[0]
            if len(idx) > 2000:
                idx = np.random.choice(idx, 2000, replace=False)
            ax[1, 1].scatter(g["cd19"][idx], g["cd14"][idx], s=2, c="orange", alpha=0.4, rasterized=True)
        ax[1, 1].set_title("6. B(cyan) / CD14+(orange)", fontsize=8, fontweight="bold")
    elif "hla" in g and "cd56" in g:
        _hb(ax[1, 1], g["hla"], g["cd56"], "HLA (AF488)", "CD56", mask=m["NK"])
        _vline(ax[1, 1], cuts.get("hla"), color="w")
        ax[1, 1].set_title("Donor NK", fontsize=8, fontweight="bold")

    # [1,2] CAR × HLA (Donor context)
    if "car" in g and "hla" in g:
        _hb(ax[1, 2], g["hla"], g["car"], "HLA", "CAR (AF647)", mask=m["NK"])
        _vline(ax[1, 2], cuts.get("hla"), color="w")
        _hline(ax[1, 2], cuts.get("car"), color="w")
        if m["Donor"].sum():
            idx = np.where(m["Donor"])[0]
            if len(idx) > 1500:
                idx = np.random.choice(idx, 1500, replace=False)
            ax[1, 2].scatter(g["hla"][idx], g["car"][idx], s=2, c="cyan", alpha=0.35, rasterized=True)
        if m["CAR"].sum():
            idx = np.where(m["CAR"])[0]
            if len(idx) > 1500:
                idx = np.random.choice(idx, 1500, replace=False)
            ax[1, 2].scatter(g["hla"][idx], g["car"][idx], s=4, c="yellow", alpha=0.6, rasterized=True)
        ax[1, 2].set_title("7. Donor(cyan) / CAR+(yellow)", fontsize=8, fontweight="bold")

    # [1,3] stats text
    ax[1, 3].axis("off")
    lines = [
        f"Scatter lymph : {pct.get('%Lymph_scatter (of total)', '—')}%",
        f"Live          : {pct.get('%Live (of singlets)', '—')}%",
        f"B (of lymph)  : {pct.get('%B (of lymph)', '—')}%",
        f"CD14+ (live)  : {pct.get('%Monocytes (of live)', pct.get('%CD14+ (of live)', '—'))}%",
        f"CD14+ (CD45+) : {pct.get('%CD14+ (of CD45+)', '—')}%",
        f"T (of lymph)  : {pct.get('%T (of lymph)', '—')}%",
        f"NK (of lymph) : {pct.get('%NK (of lymph)', '—')}%",
        f"purity B+T+NK : {pct.get('lineage_purity', '—')}%",
        f"Donor (lymph) : {pct.get('%Donor NK (of lymph)', pct.get('%Donor NK (of NK)', '—'))}%",
        f"CAR (of DNK)  : {pct.get('%CAR+ (of Donor NK)', '—')}%",
        "",
        f"n lymph={m['lympho'].sum():,}  B={m['B'].sum():,}  NK={m['NK'].sum():,}",
        f"n Donor={m['Donor'].sum():,}  CAR={m['CAR'].sum():,}",
        f"n CD14+={m['CD14p'].sum() if 'CD14p' in m else 0:,}",
        f"SSC_hi={cuts.get('ssc_hi') and round(cuts['ssc_hi'], 3)}",
    ]
    if "CD4" in m:
        lines.insert(6, f"CD4 (of lymph): {pct.get('%CD4 (of lymph)', '—')}%")
        lines.insert(7, f"CD8 (of lymph): {pct.get('%CD8 (of lymph)', '—')}%")
    ax[1, 3].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=7.5,
                  transform=ax[1, 3].transAxes, color="#1a3a5c")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)
