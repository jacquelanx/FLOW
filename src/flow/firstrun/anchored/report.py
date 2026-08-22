#!/usr/bin/env python3
"""Longitudinal QC HTML report + biological rule flags (from manual QC report)."""
from __future__ import annotations

import json
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
        # CAR is a percentage of Donor NK while Donor NK is a percentage of lymph.
        # Containment must therefore be checked with event counts, not unlike-denominator
        # percentages.
        if row.get("n_car", 0) > row.get("n_donor", 0):
            flags.append({"Timepoint": tp, "Rule": "R2", "Severity": "FLAG",
                          "Message": f"CAR events ({int(row['n_car'])}) > Donor NK events "
                                     f"({int(row['n_donor'])})"})
        if row.get("cd4_p", 0) + row.get("cd8_p", 0) > row.get("t_p", 0) + 5:
            flags.append({"Timepoint": tp, "Rule": "R3", "Severity": "FLAG",
                          "Message": f"CD4+CD8 > T+5pp"})
        if row.get("b_p", 0) + row.get("t_p", 0) + row.get("nk_p", 0) > 110:
            flags.append({"Timepoint": tp, "Rule": "R4", "Severity": "FLAG",
                          "Message": f"B+T+NK = {row['b_p']+row['t_p']+row['nk_p']:.1f}%"})
        if row.get("n_car", 0) > 0 and row.get("n_donor", 0) == 0:
            flags.append({"Timepoint": tp, "Rule": "R5", "Severity": "FLAG",
                          "Message": "CAR-region events without Donor NK parent events"})
        if i > 0:
            prev = df.iloc[i - 1]
            for col, label in [
                ("nk_p", "NK / CD14-negative lymph region"),
                ("t_p", "T / CD14-negative lymph region"),
                ("b_p", "B / CD14-negative lymph region"),
                ("mono_p", "CD14+ / viable live events"),
            ]:
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
            "lymph_live_p": r.get("%Lymphocytes (of live)", 0) or 0,
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
            "n_donor": r.get("n_Donor", 0) or 0,
            "n_car": r.get("n_CAR", 0) or 0,
        })
    return pd.DataFrame(rows)


def build_html(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    patient: str,
    upn: str,
    compare_summary: dict | None = None,
    *,
    technical_state: str = "REVIEW",
    compensation_state: str | None = None,
) -> str:
    patient_text = str(patient or "").strip()
    upn_text = str(upn or "").strip()
    report_identity = (
        upn_text if not patient_text or patient_text.casefold() == upn_text.casefold()
        else f"{upn_text}: {patient_text}"
    )
    status_parts = [
        "PROVISIONAL / NOT BIOLOGICALLY VALIDATED",
        f"RUN {str(technical_state or 'REVIEW').upper()}",
    ]
    if compensation_state:
        status_parts.append(f"COMPENSATION {str(compensation_state).upper()}")
    status_text = " &middot; ".join(status_parts)
    x = df["timepoint"].astype(str).tolist()
    x_json = json.dumps(x)
    C = dict(B="#4472C4", T="#ED7D31", CD4="#FFC000", CD8="#FF0000",
             NK="#70AD47", DNK="#00B0F0", CAR="#7030A0", CD14="#A0522D")

    def series(name, col, color, dash="solid", symbol="circle"):
        y = [float(v) for v in df[col].tolist()]
        return (f"{{x:{x_json},y:{json.dumps(y)},name:'{name}',mode:'lines+markers',"
                f"line:{{color:'{color}',width:2,dash:'{dash}'}},"
                f"marker:{{size:7,symbol:'{symbol}'}},"
                f"hovertemplate:'%{{x}}<br>{name}: %{{y:.2f}}%<extra></extra>'}}")

    traces_lymph = ",".join([
        series("B (% of CD14-negative lymph region)", "b_p", C["B"], symbol="circle"),
        series("T (% of CD14-negative lymph region)", "t_p", C["T"], symbol="square"),
        series("CD4 (% of CD14-negative lymph region)", "cd4_p", C["CD4"], symbol="triangle-up"),
        series("CD8 (% of CD14-negative lymph region)", "cd8_p", C["CD8"], symbol="triangle-down"),
        series("NK (% of CD14-negative lymph region)", "nk_p", C["NK"], symbol="diamond"),
        series(
            "Donor NK region (% of CD14-negative lymph region)",
            "d_nk_p", C["DNK"], "dash", "cross",
        ),
    ])
    traces_nested = ",".join([
        series("Lymphocytes (% of viable live events)", "lymph_live_p", C["B"], symbol="x"),
        series("CD14+ (% of viable live events)", "mono_p", C["CD14"], symbol="hexagon"),
        series(
            "CAR region (% of configured Donor NK region)",
            "car_d_p", C["CAR"], "dot", "star",
        ),
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
            f'<td>{fmt(row["lymph_live_p"])}</td><td>{fmt(row["mono_p"])}</td>'
            f'<td>{fmt(row["b_p"])}</td>'
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
<title>Interactive longitudinal QC {report_identity}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box}} body{{font-family:Arial,sans-serif;margin:0;background:#f0f2f5;color:#212529}}
.hdr{{background:#1a3a5c;color:#fff;padding:16px 28px}} .hdr h1{{margin:0;font-size:1.35em}}
.hdr p{{margin:4px 0 0;opacity:.75;font-size:.85em}} .body{{padding:18px 28px}}
.status-banner{{background:#fff0f0;color:#9c1111;border-top:2px solid #b32020;
border-bottom:2px solid #b32020;padding:7px 16px;text-align:center;font-size:.78em;
font-weight:700;letter-spacing:.035em;text-transform:uppercase}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.1);padding:16px;margin-bottom:18px}}
.card h2{{margin:0 0 12px;font-size:.95em;color:#1a3a5c;border-bottom:2px solid #e9ecef;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:.82em}}
th{{background:#1a3a5c;color:#fff;padding:6px 10px;text-align:left}}
td{{padding:5px 10px;border-bottom:1px solid #e9ecef}}
.plot-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.plot-panel h3{{font-size:.82em;margin:0 0 4px;color:#334455}}
.denominator-note{{font-size:.78em;color:#5a6570;margin:0 0 12px}}
.table-scroll{{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}}
.data-table{{min-width:1180px}}
@media(max-width:950px){{.plot-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="status-banner" role="status" aria-label="Run validation status">{status_text}</div>
<div class="hdr">
  <h1>Interactive longitudinal QC - {report_identity}</h1>
  <p>FSC-A x SSC-A-first pipeline &nbsp;|&nbsp; {len(df)} timepoints &nbsp;|&nbsp;
     deterministic canonical first run</p>
</div>
<div class="body">
  <div class="card"><h2>Longitudinal percentages by denominator</h2>
    <p class="denominator-note">Curves are grouped only when they share a denominator. Hover labels and table headers state the parent population explicitly.</p>
    <div class="plot-grid">
      <div class="plot-panel"><h3>Denominator: CD14-negative lymph region</h3><div id="lymph" style="height:430px;"></div></div>
      <div class="plot-panel"><h3>Lymphocytes/CD14+: viable live events; CAR region: configured Donor NK region</h3><div id="nested" style="height:430px;"></div></div>
    </div>
  </div>
  {cmp_block}
  <div class="card"><h2>QC Flags</h2>
    <table><thead><tr><th>Timepoint</th><th>Rule</th><th>Severity</th><th>Detail</th></tr></thead>
    <tbody>{flag_rows}</tbody></table></div>
  <div class="card"><h2>Data Table</h2>
    <div class="table-scroll" role="region" aria-label="Longitudinal percentage table" tabindex="0">
    <table class="data-table"><thead><tr>
      <th>TP</th><th>%Lymphocytes / viable live events</th><th>%CD14+ / viable live events</th>
      <th>%B / CD14-negative lymph region</th><th>%T / CD14-negative lymph region</th>
      <th>%CD4 / CD14-negative lymph region</th><th>%CD8 / CD14-negative lymph region</th>
      <th>%NK / CD14-negative lymph region</th><th>%Donor NK region / CD14-negative lymph region</th>
      <th>%CAR region / configured Donor NK region</th>
    </tr></thead><tbody>{summary_rows}</tbody></table></div></div>
</div>
<script>
const sharedLayout={{
  margin:{{t:10,r:10,b:145,l:55}},legend:{{orientation:'h',y:-0.42}},
  xaxis:{{tickangle:-40,tickfont:{{size:10}}}},
  yaxis:{{title:'Percent of stated parent',range:[0,105]}},hovermode:'x unified',
  plot_bgcolor:'#fafafa',paper_bgcolor:'white'
}};
Plotly.newPlot('lymph',[{traces_lymph}],sharedLayout,{{responsive:true}});
Plotly.newPlot('nested',[{traces_nested}],sharedLayout,{{responsive:true}});
</script></body></html>"""


def write_report(
    auto_df: pd.DataFrame,
    out_html: Path,
    patient: str,
    upn: str,
    compare_summary: dict | None = None,
    *,
    technical_state: str = "REVIEW",
    compensation_state: str | None = None,
) -> Path:
    rdf = auto_to_report_frame(auto_df)
    flags = run_qc_flags(rdf)
    out_html.write_text(
        build_html(
            rdf,
            flags,
            patient,
            upn,
            compare_summary,
            technical_state=technical_state,
            compensation_state=compensation_state,
        ),
        encoding="utf-8",
    )
    return out_html
