from __future__ import annotations

import pandas as pd

from flow.firstrun.anchored.report import (
    auto_to_report_frame,
    build_html,
    run_qc_flags,
)


def _fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "file": "Specimen_001_D7.fcs",
            "timepoint": "D7",
            "%Lymphocytes (of live)": 72.0,
            "%CD14+ (of live)": 8.0,
            "%B (of lymph)": 2.0,
            "%T (of lymph)": 10.0,
            "%CD4 (of lymph)": 8.0,
            "%CD8 (of lymph)": 2.0,
            "%NK (of lymph)": 80.0,
            "%Donor NK (of lymph)": 70.0,
            "%CAR+ (of Donor NK)": 20.0,
            "n_Donor": 378,
            "n_CAR": 76,
        }
    ])


def test_html_names_each_percentage_denominator_and_uses_distinct_dashboard_identity():
    report_frame = auto_to_report_frame(_fixture())
    html = build_html(
        report_frame,
        run_qc_flags(report_frame),
        patient="UPN27",
        upn="UPN27",
        technical_state="BLOCKED",
        compensation_state="BLOCKED",
    )
    assert "Interactive longitudinal QC" in html
    assert "UPN27: UPN27" not in html
    assert "B (% of CD14-negative lymph region)" in html
    assert "Lymphocytes (% of viable live events)" in html
    assert "CD14+ (% of viable live events)" in html
    assert "CAR region (% of configured Donor NK region)" in html
    assert "Percentages with other denominators" not in html
    assert 'class="table-scroll"' in html
    assert 'class="data-table"' in html
    assert "PROVISIONAL / NOT BIOLOGICALLY VALIDATED" in html
    assert "RUN BLOCKED" in html
    assert "COMPENSATION BLOCKED" in html
    for symbol in ("circle", "square", "triangle-up", "triangle-down", "diamond", "cross"):
        assert f"symbol:'{symbol}'" in html


def test_car_containment_qc_uses_counts_not_unlike_denominator_percentages():
    report_frame = auto_to_report_frame(_fixture())
    flags = run_qc_flags(report_frame)
    assert "R2" not in set(flags["Rule"])

    invalid = report_frame.copy()
    invalid.loc[0, "n_car"] = 379
    flags = run_qc_flags(invalid)
    assert "R2" in set(flags["Rule"])
