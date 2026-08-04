#!/usr/bin/env python3
"""Longitudinal QC HTML report + biological rule flags (from manual QC report)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def run_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Biological consistency rules (same as manual UPN QC report)."""
    flags = []
    # expect columns: timepoint, b_p, t_p, cd4_p, cd8_p, nk_p, d_nk_p, car_d_p, mono_p?
    for i, row in df.iterrows():
        tp = row.get("timepoint", f"TP{i+1}")

        if row.get("d_nk_p", 0) > row.get("nk_p", 0) + 2:
            flags.append({"Timepoint": tp, "Rule": "R1", "Severity": "FLAG",
                          "Message": f"Donor NK ({row['d_nk_p']:.2f}%) > NK ({row['nk_p']:.2f}%)"})
        if row.get("car_d_p", 0) > row.get("d_nk_p", 0) + 2:
            flags.append({"Timepoint": tp, "Rule": "R2", "Severity": "FLAG",
                          "Message": f"CAR+ ({row['car_d_p']:.2f}%) > Donor NK ({row['d_nk_p']:.2f}%)"})
        if row.get("cd4_p", 0) + row.get("cd8_p", 0) > row.get("t_p", 0) + 5:
            flags.append({"Timepoint": tp, "Rule": "R3", "Severity": "FLAG",
                          "Message": f"CD4+CD8 > T+5pp"})
        if row.get("b_p", 0) + row.get("t_p", 0) + row.get("nk_p", 0) > 110:
            flags.append({"Timepoint": tp, "Rule": "R4", "Severity": "FLAG",
                          "Message": f"B+T+NK = {row['b_p']+row['t_p']+row['nk_p']:.1f}%"})
        if row.get("car_d_p", 0) > 1 and row.get("d_nk_p", 0) == 0:
            flags.append({"Timepoint": tp, "Rule": "R5", "Severity": "FLAG",
                          "Message": "CAR+ without Donor NK"})
        if i > 0:
            prev = df.iloc[i - 1]
            for col, label in [("nk_p", "NK"), ("t_p", "T"), ("b_p", "B"), ("mono_p", "CD14+")]:
                if col in row and col in prev and pd.notna(row[col]) and pd.notna(prev[col]):
                    delta = abs(row[col] - prev[col])
                    if delta > 40:
                        flags.append({"Timepoint": tp, "Rule": "R6", "Severity": "WARN",
                                      "Message": f"{label}: Δ{delta:.1f}pp vs prev"})
    return pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["Timepoint", "Rule", "Severity", "Message"])


def auto_to_report_frame(auto_df: pd.DataFrame) -> pd.DataFrame:
    """Map pipeline CSV columns → report columns."""
    rows = []
    for _, r in auto_df.iterrows():
        mono = r.get("%Monocytes (of live)", r.get("%CD14+ (of live)", None))
        rows.append({
            "timepoint": r.get("timepoint", r.get("file")),
            "b_p": r.get("%B (of lymph)", 0) or 0,
            "t_p": r.get("%T (of lymph)", 0) or 0,
            "cd4_p": r.get("%CD4 (of lymph)", 0) or 0,
            "cd8_p": r.get("%CD8 (of lymph)", 0) or 0,
            "nk_p": r.get("%NK (of lymph)", 0) or 0,
            "d_nk_p": r.get("%Donor NK (of lymph)",
                            r.get("%Donor NK (of NK)", 0)) or 0,
            "car_d_p": r.get("%CAR+ (of Donor NK)", 0) or 0,
            "mono_p": (mono if mono is not None else 0) or 0,
            "cd14_cd45_p": r.get("%CD14+ (of CD45+)", 0) or 0,
        })
    return pd.DataFrame(rows)


def build_html(df: pd.DataFrame, flags: pd.DataFrame, patient: str, upn: str,
               compare_summary: dict | None = None) -> str:
    x = df["timepoint"].astype(str).tolist()
    x_json = json.dumps(x)
    C = dict(B="#4472C4", T="#ED7D31", CD4="#FFC000", CD8="#FF0000",
             NK="#70AD47", DNK="#00B0F0", CAR="#7030A0", CD14="#A0522D")

    def series(name, col, color, dash="solid"):
        y = [float(v) for v in df[col].tolist()]
        return (f"{{x:{x_json},y:{json.dumps(y)},name:'{name}',mode:'lines+markers',"
                f"line:{{color:'{color}',width:2,dash:'{dash}'}},marker:{{size:7}},"
                f"hovertemplate:'%{{x}}<br>{name}: %{{y:.2f}}%<extra></extra>'}}")

    traces_main = ",".join([
        series("%B", "b_p", C["B"]),
        series("%T", "t_p", C["T"]),
        series("%NK", "nk_p", C["NK"]),
        series("%CD14+ (of live)", "mono_p", C["CD14"], "dot"),
        series("%Donor NK", "d_nk_p", C["DNK"]),
        series("%CAR+", "car_d_p", C["CAR"], "dot"),
    ])

    if flags.empty:
        flag_rows = ('<tr><td colspan="4" style="text-align:center;color:#198754;'
                     'padding:12px;">✓ No QC flags</td></tr>')
    else:
        flag_rows = ""
        for _, r in flags.iterrows():
            sev = "#dc3545" if r["Severity"] == "FLAG" else "#fd7e14"
            badge = (f'<span style="background:{sev};color:white;padding:2px 8px;'
                     f'border-radius:4px;font-size:11px;">{r["Severity"]}</span>')
            flag_rows += (f'<tr><td>{r["Timepoint"]}</td><td>{r["Rule"]}</td>'
                          f'<td>{badge}</td><td>{r["Message"]}</td></tr>')

    def fmt(v):
        v = float(v)
        return f"{v:.1f}%" if v >= 0.1 else f"{v:.3f}%"

    summary_rows = ""
    for _, row in df.iterrows():
        summary_rows += (
            f'<tr><td>{row["timepoint"]}</td>'
            f'<td>{fmt(row["b_p"])}</td><td>{fmt(row["mono_p"])}</td>'
            f'<td>{fmt(row["t_p"])}</td>'
            f'<td>{fmt(row["cd4_p"])}</td><td>{fmt(row["cd8_p"])}</td>'
            f'<td>{fmt(row["nk_p"])}</td><td>{fmt(row["d_nk_p"])}</td>'
            f'<td>{fmt(row["car_d_p"])}</td></tr>'
        )

    cmp_block = ""
    if compare_summary and compare_summary.get("n"):
        w = compare_summary.get("worst", {})
        cmp_block = f"""
  <div class="card">
    <h2>vs Manual Gating</h2>
    <p>n={compare_summary['n']} &nbsp;|&nbsp; MAE={compare_summary['mae']} pp
       &nbsp;|&nbsp; ≤5pp: {compare_summary['within_5pp']}%
       &nbsp;|&nbsp; ≤10pp: {compare_summary['within_10pp']}%</p>
    <p style="font-size:.85em;color:#666;">Worst: {w.get('timepoint')} {w.get('metric')}
       auto={w.get('auto')} manual={w.get('manual')} Δ={w.get('delta')}</p>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>QC {upn} {patient}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box}} body{{font-family:Arial,sans-serif;margin:0;background:#f0f2f5;color:#212529}}
.hdr{{background:#1a3a5c;color:#fff;padding:16px 28px}} .hdr h1{{margin:0;font-size:1.35em}}
.hdr p{{margin:4px 0 0;opacity:.75;font-size:.85em}} .body{{padding:18px 28px}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.1);padding:16px;margin-bottom:18px}}
.card h2{{margin:0 0 12px;font-size:.95em;color:#1a3a5c;border-bottom:2px solid #e9ecef;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:.82em}}
th{{background:#1a3a5c;color:#fff;padding:6px 10px;text-align:left}}
td{{padding:5px 10px;border-bottom:1px solid #e9ecef}}
</style></head><body>
<div class="hdr">
  <h1>Flow Cytometry QC — {upn}: {patient}</h1>
  <p>FSA×SSA-first pipeline &nbsp;|&nbsp; {len(df)} timepoints &nbsp;|&nbsp;
     {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
<div class="body">
  <div class="card"><h2>Populations Over Time</h2>
    <div id="main" style="height:370px;"></div></div>
  {cmp_block}
  <div class="card"><h2>QC Flags</h2>
    <table><thead><tr><th>Timepoint</th><th>Rule</th><th>Severity</th><th>Detail</th></tr></thead>
    <tbody>{flag_rows}</tbody></table></div>
  <div class="card"><h2>Data Table</h2>
    <table><thead><tr>
      <th>TP</th><th>%B</th><th>%CD14+ (live)</th><th>%T</th><th>%CD4</th><th>%CD8</th>
      <th>%NK</th><th>%Donor NK</th><th>%CAR+</th>
    </tr></thead><tbody>{summary_rows}</tbody></table></div>
</div>
<script>
Plotly.newPlot('main',[{traces_main}],{{
  margin:{{t:10,r:10,b:90,l:50}},legend:{{orientation:'h',y:-0.35}},
  xaxis:{{tickangle:-40,tickfont:{{size:10}}}},
  yaxis:{{title:'%',range:[0,105]}},hovermode:'x unified',
  plot_bgcolor:'#fafafa',paper_bgcolor:'white'
}},{{responsive:true}});
</script></body></html>"""


def write_report(auto_df: pd.DataFrame, out_html: Path, patient: str, upn: str,
                 compare_summary: dict | None = None) -> Path:
    rdf = auto_to_report_frame(auto_df)
    flags = run_qc_flags(rdf)
    out_html.write_text(build_html(rdf, flags, patient, upn, compare_summary), encoding="utf-8")
    return out_html
