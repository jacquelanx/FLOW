#!/usr/bin/env python3
"""
HTML report with automated vs manual gating overlaid.

PHI-safe: titles use study label only (e.g. UPN27). Never include MRN/name.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .compare import METRIC_MAP, summarize


# Display order for main longitudinal plot + comparison table
DISPLAY_METRICS = [
    ("%B (of lymph)", "b_p", "%B"),
    ("%Monocytes (of live)", "mono_p", "%CD14+"),
    ("%T (of lymph)", "t_p", "%T"),
    ("%CD4 (of lymph)", "cd4_p", "%CD4"),
    ("%CD8 (of lymph)", "cd8_p", "%CD8"),
    ("%NK (of lymph)", "nk_p", "%NK"),
    ("%Donor NK (of lymph)", "d_nk_p", "%Donor NK"),
    ("%CAR+ (of Donor NK)", "car_d_p", "%CAR+"),
]

COLORS = {
    "%B": "#4472C4",
    "%CD14+": "#A0522D",
    "%T": "#ED7D31",
    "%CD4": "#FFC000",
    "%CD8": "#C00000",
    "%NK": "#70AD47",
    "%Donor NK": "#00B0F0",
    "%CAR+": "#7030A0",
}


def _auto_frame(auto_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in auto_df.iterrows():
        mono = r.get("%Monocytes (of live)", r.get("%CD14+ (of live)", 0)) or 0
        rows.append({
            "timepoint": r.get("timepoint") or r.get("file"),
            "b_p": float(r.get("%B (of lymph)", 0) or 0),
            "mono_p": float(mono),
            "t_p": float(r.get("%T (of lymph)", 0) or 0),
            "cd4_p": float(r.get("%CD4 (of lymph)", 0) or 0),
            "cd8_p": float(r.get("%CD8 (of lymph)", 0) or 0),
            "nk_p": float(r.get("%NK (of lymph)", 0) or 0),
            "d_nk_p": float(r.get("%Donor NK (of lymph)",
                                  r.get("%Donor NK (of NK)", 0)) or 0),
            "car_d_p": float(r.get("%CAR+ (of Donor NK)", 0) or 0),
        })
    return pd.DataFrame(rows)


def _manual_frame(manual_df: pd.DataFrame) -> pd.DataFrame:
    m = manual_df.copy()
    m.columns = [c.lower().strip() for c in m.columns]
    m["timepoint"] = m["timepoint"].astype(str)
    for col in ("b_p", "t_p", "cd4_p", "cd8_p", "nk_p", "d_nk_p", "car_d_p",
                "mono_p", "cd14_p"):
        if col not in m.columns:
            m[col] = None
    if m["mono_p"].isna().all() and m["cd14_p"].notna().any():
        m["mono_p"] = m["cd14_p"]
    return m


def build_compare_html(
    auto_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    study_label: str = "UPN27",
    compare_df: pd.DataFrame | None = None,
    compare_summary: dict | None = None,
) -> str:
    """Build PHI-safe HTML with auto + manual overlays."""
    auto = _auto_frame(auto_df)
    man = _manual_frame(manual_df).set_index("timepoint")

    # Align timepoints to auto order
    tps = [str(t) for t in auto["timepoint"].tolist()]
    x_json = json.dumps(tps)

    def y_auto(col):
        return [float(v) for v in auto[col].tolist()]

    def y_man(col):
        out = []
        for tp in tps:
            if tp in man.index and pd.notna(man.loc[tp, col]):
                out.append(float(man.loc[tp, col]))
            else:
                out.append(None)
        return out

    traces = []
    for auto_col, man_col, label in DISPLAY_METRICS:
        color = COLORS.get(label, "#666")
        # skip mono if no manual and all auto ~0? still show auto CD14
        ya = y_auto(man_col if man_col in auto.columns else "mono_p")
        if man_col == "mono_p":
            ya = y_auto("mono_p")
        traces.append(
            f"{{x:{x_json},y:{json.dumps(ya)},name:'{label} auto',"
            f"mode:'lines+markers',line:{{color:'{color}',width:2}},"
            f"marker:{{size:7}},"
            f"hovertemplate:'%{{x}}<br>{label} auto: %{{y:.2f}}%<extra></extra>'}}"
        )
        ym = y_man(man_col)
        if any(v is not None for v in ym):
            traces.append(
                f"{{x:{x_json},y:{json.dumps(ym)},name:'{label} manual',"
                f"mode:'lines+markers',"
                f"line:{{color:'{color}',width:2,dash:'dot'}},"
                f"marker:{{size:7,symbol:'diamond'}},"
                f"hovertemplate:'%{{x}}<br>{label} manual: %{{y:.2f}}%<extra></extra>'}}"
            )

    # Comparison table rows
    table_rows = ""
    for tp in tps:
        arow = auto[auto["timepoint"].astype(str) == tp].iloc[0]
        cells = [f"<td><b>{tp}</b></td>"]
        for _, man_col, label in DISPLAY_METRICS:
            av = float(arow[man_col]) if man_col in arow.index else 0.0
            mv = None
            if tp in man.index and man_col in man.columns and pd.notna(man.loc[tp, man_col]):
                mv = float(man.loc[tp, man_col])
            if mv is None:
                cells.append(f"<td>{av:.2f}</td><td class='na'>—</td><td class='na'>—</td>")
            else:
                d = av - mv
                cls = "ok" if abs(d) <= 5 else ("warn" if abs(d) <= 10 else "bad")
                cells.append(
                    f"<td>{av:.2f}</td><td>{mv:.2f}</td>"
                    f"<td class='{cls}'>{d:+.2f}</td>"
                )
        table_rows += "<tr>" + "".join(cells) + "</tr>\n"

    header_metrics = "".join(
        f"<th colspan='3'>{label}</th>" for _, _, label in DISPLAY_METRICS
    )
    subhead = "<th>TP</th>" + "".join(
        "<th>Auto</th><th>Man</th><th>Δ</th>" for _ in DISPLAY_METRICS
    )

    # Long-form delta table from compare_df if provided
    detail_rows = ""
    if compare_df is not None and not compare_df.empty:
        show = compare_df.sort_values("abs_delta", ascending=False)
        for _, r in show.iterrows():
            cls = "ok" if r["abs_delta"] <= 5 else ("warn" if r["abs_delta"] <= 10 else "bad")
            detail_rows += (
                f"<tr><td>{r['timepoint']}</td><td>{r['metric']}</td>"
                f"<td>{r['auto']:.3f}</td><td>{r['manual']:.3f}</td>"
                f"<td class='{cls}'>{r['delta']:+.3f}</td></tr>\n"
            )

    summary_html = ""
    if compare_summary and compare_summary.get("n"):
        w = compare_summary.get("worst") or {}
        summary_html = f"""
  <div class="card metrics">
    <div><div class="k">MAE</div><div class="v">{compare_summary.get('mae')} pp</div></div>
    <div><div class="k">≤5 pp</div><div class="v">{compare_summary.get('within_5pp')}%</div></div>
    <div><div class="k">≤10 pp</div><div class="v">{compare_summary.get('within_10pp')}%</div></div>
    <div><div class="k">n cells</div><div class="v">{compare_summary.get('n')}</div></div>
  </div>
  <p class="worst">Worst: {w.get('timepoint')} · {w.get('metric')} ·
     auto={w.get('auto')} · manual={w.get('manual')} · Δ={w.get('delta')}</p>
"""

    traces_js = ",".join(traces)
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>QC Report — {study_label}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
     margin:0;background:#f0f2f5;color:#212529}}
.hdr{{background:#1a3a5c;color:#fff;padding:18px 28px}}
.hdr h1{{margin:0;font-size:1.4em;letter-spacing:.02em}}
.hdr p{{margin:6px 0 0;opacity:.8;font-size:.88em}}
.body{{padding:18px 28px;max-width:1400px;margin:0 auto}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.1);
      padding:16px;margin-bottom:18px}}
.card h2{{margin:0 0 12px;font-size:1em;color:#1a3a5c;
         border-bottom:2px solid #e9ecef;padding-bottom:6px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.metrics .k{{font-size:.75em;color:#666;text-transform:uppercase}}
.metrics .v{{font-size:1.5em;font-weight:700;color:#1a3a5c}}
.worst{{font-size:.85em;color:#666;margin:-8px 0 18px}}
table{{width:100%;border-collapse:collapse;font-size:.78em}}
th{{background:#1a3a5c;color:#fff;padding:6px 8px;text-align:center}}
td{{padding:5px 8px;border-bottom:1px solid #e9ecef;text-align:right}}
td:first-child{{text-align:left}}
.ok{{color:#198754;font-weight:600}}
.warn{{color:#c27803;font-weight:600}}
.bad{{color:#dc3545;font-weight:600}}
.na{{color:#aaa}}
.legend{{font-size:.85em;color:#555;margin-top:8px}}
.legend span{{display:inline-block;margin-right:14px}}
.note{{font-size:.8em;color:#666;margin-top:10px}}
.scroll{{overflow-x:auto}}
</style></head><body>
<div class="hdr">
  <h1>Flow Cytometry QC — {study_label}</h1>
  <p>FSA×SSA-first automated gating &nbsp;|&nbsp; Manual overlay (dotted) &nbsp;|&nbsp;
     De-identified study label only &nbsp;|&nbsp;
     {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
<div class="body">
  {summary_html}
  <div class="card">
    <h2>Populations Over Time — Auto (solid) vs Manual (dotted)</h2>
    <div id="main" style="height:420px;"></div>
    <div class="legend">
      <span>● Auto</span><span>◇ Manual (dotted)</span>
      <span>Δ color: <span class="ok">≤5 pp</span> /
            <span class="warn">≤10 pp</span> /
            <span class="bad">&gt;10 pp</span></span>
    </div>
  </div>
  <div class="card">
    <h2>Side-by-side Comparison (Auto | Manual | Δ)</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>{header_metrics}</tr>
          <tr>{subhead}</tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    <p class="note">Manual B/T/CD4/CD8/NK/Donor are % of lymphocytes (CD45+ CD14−).
       CAR+ is % of Donor NK. CD14+ is % of live when reported.
       Donor NK manual column is % of lymph (same denominator as NK).</p>
  </div>
  <div class="card">
    <h2>All Metric Deltas (sorted by |Δ|)</h2>
    <div class="scroll">
      <table>
        <thead><tr>
          <th>Timepoint</th><th>Metric</th><th>Auto</th><th>Manual</th><th>Δ</th>
        </tr></thead>
        <tbody>{detail_rows or '<tr><td colspan="5">No compare rows</td></tr>'}</tbody>
      </table>
    </div>
  </div>
</div>
<script>
Plotly.newPlot('main',[{traces_js}],{{
  margin:{{t:10,r:10,b:100,l:50}},
  legend:{{orientation:'h',y:-0.28,font:{{size:10}}}},
  xaxis:{{tickangle:-40,tickfont:{{size:10}}}},
  yaxis:{{title:'%',range:[0,105]}},
  hovermode:'x unified',
  plot_bgcolor:'#fafafa',paper_bgcolor:'white'
}},{{responsive:true}});
</script></body></html>
"""


def write_compare_report(
    auto_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    out_html: Path,
    study_label: str = "UPN27",
    compare_df: pd.DataFrame | None = None,
    compare_summary: dict | None = None,
) -> Path:
    html = build_compare_html(
        auto_df, manual_df, study_label=study_label,
        compare_df=compare_df, compare_summary=compare_summary,
    )
    out_html = Path(out_html)
    out_html.write_text(html, encoding="utf-8")
    return out_html
