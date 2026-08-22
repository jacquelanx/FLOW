from __future__ import annotations

import csv
import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from flow.firstrun.anchored.flow_outputs import (
    load_exact_alc_matches,
    write_temporal_cell_type_summary,
)
from flow.firstrun.anchored.qc_plots import (
    write_compact_qc_report,
    write_longitudinal_composition_report,
)
from flow.firstrun.anchored.output_layout import FirstRunOutputLayout
from flow.firstrun.anchored.acquisition_cleaning import write_acquisition_cleaning_report


def _multilineage_fixture():
    return pd.DataFrame([
        {
            "file": "Specimen_001_Baseline.fcs", "timepoint": "Baseline",
            "%Lymphocytes (of live)": 20.0, "%B (of lymph)": 5.0,
            "%T (of lymph)": 40.0, "%CD4 (of lymph)": 30.0,
            "%CD8 (of lymph)": 10.0, "%NK (of lymph)": 50.0,
            "%Donor NK (of lymph)": 1.0, "%CAR+ (of Donor NK)": 0.0,
            "%CD14+ (of live)": 3.0, "n_live": 1000, "n_lympho": 200,
            "n_B": 10, "n_T": 80, "n_CD4": 60, "n_CD8": 20, "n_NK": 100,
            "n_Donor": 2, "n_CAR": 0, "n_CD14p": 30,
            "donor_reliable": False, "car_reliable": False,
            "compensation_state": "BLOCKED", "acquisition_qc_state": "PASS",
        },
        {
            "file": "Specimen_001_D7.fcs", "timepoint": "D7",
            "%Lymphocytes (of live)": 60.0, "%B (of lymph)": 2.0,
            "%T (of lymph)": 10.0, "%CD4 (of lymph)": 8.0,
            "%CD8 (of lymph)": 2.0, "%NK (of lymph)": 80.0,
            "%Donor NK (of lymph)": 70.0, "%CAR+ (of Donor NK)": 20.0,
            "%CD14+ (of live)": 8.0, "n_live": 900, "n_lympho": 540,
            "n_B": 11, "n_T": 54, "n_CD4": 43, "n_CD8": 11, "n_NK": 432,
            "n_Donor": 378, "n_CAR": 76, "n_CD14p": 72,
            "donor_reliable": True, "car_reliable": True,
            "compensation_state": "BLOCKED", "acquisition_qc_state": "PASS",
        },
    ])


def test_temporal_summary_has_explicit_denominators_and_no_false_absolute_abundance(tmp_path):
    path = write_temporal_cell_type_summary(
        _multilineage_fixture(), tmp_path, patient_id="UPN27", reference_tp="Baseline"
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    d7_nk = next(row for row in rows if row["timepoint"] == "D7" and row["population"] == "NK")
    assert d7_nk["percentage"] == "80.0"
    assert d7_nk["percentage_denominator"] == "CD14-negative lymph region"
    assert d7_nk["gated_event_count"] == "432"
    assert d7_nk["percentage_point_change_from_reference"] == "30.0"
    assert d7_nk["absolute_abundance_available"] == "False"
    assert d7_nk["population_identity_state"] == "CONFIGURED_TECHNICAL_REGION"
    baseline_lymph = next(
        row for row in rows
        if row["timepoint"] == "Baseline" and row["population"] == "Lymphocytes"
    )
    assert baseline_lymph["percentage_denominator"] == "viable live events"
    d7_car = next(
        row for row in rows if row["timepoint"] == "D7" and row["population"] == "CAR+"
    )
    assert d7_car["percentage_denominator"] == "configured Donor NK region"


def test_temporal_summary_uses_only_exact_date_alc_matches(tmp_path):
    (tmp_path / "flow.csv").write_text(
        "label,date\nBaseline,06/25/25\nD3,07/03/25\nD7,07/07/25\n"
    )
    (tmp_path / "alc.csv").write_text("date,alc\n06/25/25,0.5\n07/07/25,0.38\n")
    matches = load_exact_alc_matches(tmp_path)
    assert set(matches) == {"Baseline", "D7"}
    path = write_temporal_cell_type_summary(
        _multilineage_fixture(), tmp_path, patient_id="UPN27",
        reference_tp="Baseline", alc_matches=matches,
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline_lymph = next(
        row for row in rows if row["timepoint"] == "Baseline"
        and row["population"] == "Lymphocytes"
    )
    d7_nk = next(
        row for row in rows if row["timepoint"] == "D7" and row["population"] == "NK"
    )
    d7_cd14 = next(
        row for row in rows if row["timepoint"] == "D7" and row["population"] == "CD14+"
    )
    assert baseline_lymph["absolute_abundance"] == "0.5"
    assert baseline_lymph["absolute_abundance_unit"] == "K/uL"
    assert d7_nk["absolute_abundance"] == "0.304"
    assert d7_nk["absolute_abundance_source_date"] == "2025-07-07"
    assert d7_cd14["absolute_abundance_available"] == "False"
    assert "lymphoid" in d7_cd14["absolute_abundance_reason"]


def test_longitudinal_percentage_and_event_count_pdf_is_created(tmp_path):
    target = tmp_path / "Longitudinal_Cell_Composition_UPN27.pdf"
    summary = write_temporal_cell_type_summary(
        _multilineage_fixture(), tmp_path, patient_id="UPN27",
        alc_matches={
            "Baseline": {"alc_k_per_uL": 0.5, "source_date": "2025-06-25",
                         "source_file": "alc.csv"},
            "D7": {"alc_k_per_uL": 0.38, "source_date": "2025-07-07",
                   "source_file": "alc.csv"},
        },
    )
    write_longitudinal_composition_report(
        _multilineage_fixture(), target, patient_id="UPN27",
        technical_state="BLOCKED", compensation_state="BLOCKED",
        temporal_summary_path=summary,
    )
    assert target.is_file()
    assert target.stat().st_size > 10_000


def test_longitudinal_legends_use_dedicated_non_data_bands(tmp_path, monkeypatch):
    target = tmp_path / "Longitudinal_Cell_Composition_UPN27.pdf"
    captured = {}
    original_close = plt.close

    def capture_close(fig):
        captured["figure"] = fig

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        write_longitudinal_composition_report(
            _multilineage_fixture(), target, patient_id="UPN27",
            technical_state="BLOCKED", compensation_state="BLOCKED",
        )
        fig = captured["figure"]
        data_axes = [axis for axis in fig.axes if axis.axison]
        legend_axes = [axis for axis in fig.axes if not axis.axison]
        assert len(data_axes) == 4
        assert len(legend_axes) == 2
        assert all(axis.get_legend() is None for axis in data_axes)
        assert all(axis.get_legend() is not None for axis in legend_axes)
        assert len(fig.legends) == 1
        percentage_labels = {
            text.get_text()
            for axis in legend_axes
            for text in axis.get_legend().get_texts()
        }
        assert "B (% of CD14-negative lymph region)" in percentage_labels
        assert "Lymphocytes (% of viable live events)" in percentage_labels
        assert "CAR region (% of configured Donor NK region)" in percentage_labels

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        data_boxes = [axis.get_window_extent(renderer) for axis in data_axes]
        legends = [axis.get_legend() for axis in legend_axes] + list(fig.legends)
        for legend in legends:
            legend_box = legend.get_window_extent(renderer)
            assert all(not legend_box.overlaps(box) for box in data_boxes)
        footer = next(
            text for text in fig.texts
            if text.get_text().startswith("Top-left denominator:")
        )
        assert not fig.legends[0].get_window_extent(renderer).overlaps(
            footer.get_window_extent(renderer)
        )
    finally:
        if "figure" in captured:
            original_close(captured["figure"])


def test_composite_pages_are_also_named_and_indexed(tmp_path):
    composite = tmp_path / "QC_Report_UPN27.pdf"
    with PdfPages(composite) as pdf:
        write_compact_qc_report(
            pdf,
            "UPN27",
            [],
            render_dir=tmp_path / "report_pages",
            page_pdf_dir=tmp_path / "report_pages" / "pdf",
            page_index_path=tmp_path / "report_page_index.csv",
            geometry_path=tmp_path / "lymph_density_geometry.csv",
            composite_pdf_name=composite.name,
            acquisition_rows=None,
            technical_state="BLOCKED",
            compensation_state="BLOCKED",
        )
    separate = sorted((tmp_path / "report_pages" / "pdf").glob("*.pdf"))
    assert [path.name for path in separate] == [
        "UPN27_QC_page_001_methods_hierarchy.pdf",
        "UPN27_QC_page_002_temporal_cell_type_summary.pdf",
    ]
    with (tmp_path / "report_page_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["page_label"] for row in rows] == ["1/2", "2/2"]
    assert rows[1]["page_role"] == "cross_sample_subset_summary"
    assert rows[1]["separate_pdf"].endswith(
        "UPN27_QC_page_002_temporal_cell_type_summary.pdf"
    )
    with (tmp_path / "lymph_density_geometry.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        geometry_rows = list(csv.DictReader(handle))
    assert geometry_rows == []


def test_typed_output_layout_indexes_results_without_duplicate_ssa_aliases(tmp_path):
    layout = FirstRunOutputLayout.create(tmp_path)
    (layout.reports / "QC_Report_UPN27.pdf").write_bytes(b"canonical-pdf")
    (layout.reports / "Interactive_Longitudinal_QC_UPN27.html").write_text(
        "<html>interactive</html>"
    )
    (layout.tables / "multilineage.csv").write_text("file,timepoint\na.fcs,D0\n")
    _, legacy_map, index = layout.write_catalog("UPN27")

    assert not (layout.reports / "QC_Report_UPN27_SSA.pdf").exists()
    assert not (layout.reports / "QC_Report_UPN27_SSA.html").exists()
    mapping = json.loads(legacy_map.read_text())
    assert mapping["policy"] == "deprecated or misleading names are mapped, not duplicated"
    assert mapping["mappings"][0]["canonical_relative_path"] == (
        "reports/QC_Report_UPN27.pdf"
    )
    assert mapping["mappings"][1]["canonical_relative_path"] == (
        "reports/Interactive_Longitudinal_QC_UPN27.html"
    )
    with index.open(newline="", encoding="utf-8") as handle:
        paths = {row["relative_path"] for row in csv.DictReader(handle)}
    assert "reports/QC_Report_UPN27.pdf" in paths
    assert "reports/Interactive_Longitudinal_QC_UPN27.html" in paths
    assert "tables/multilineage.csv" in paths


def test_time_cleaning_report_keeps_legend_outside_data_axes(tmp_path, monkeypatch):
    target = tmp_path / "Acquisition_Cleaning_Sensitivity_UPN27.pdf"
    summary = pd.DataFrame([{
        "file": "Specimen_001_Pre.fcs", "timepoint": "Pre",
        "acquisition_qc_state": "FAIL", "analyzed_event_count": 200_000,
        "excluded_analyzed_event_count": 10_000,
        "excluded_analyzed_event_percent": 5.0,
        "full_file_candidate_event_count": 20_000,
        "full_file_candidate_event_percent": 5.0,
        "candidate_interval_count": 1,
        "sensitivity_result": "EXCLUDED_CANDIDATE_INTERVALS",
    }])
    comparisons = pd.DataFrame([
        {
            "file": "Specimen_001_Pre.fcs", "metric": metric,
            "canonical_percent": canonical, "cleaned_percent": cleaned,
            "delta_percentage_points": cleaned - canonical,
        }
        for metric, canonical, cleaned in [
            ("%Lymph_scatter (of total)", 70.0, 75.0),
            ("%T (of lymph)", 50.0, 49.0),
            ("%NK (of lymph)", 10.0, 11.0),
        ]
    ])
    intervals = pd.DataFrame([{
        "file": "Specimen_001_Pre.fcs", "start": 2.0, "end": 3.0,
        "rate_burst": True, "signal_spike_proxy": True,
    }])
    captured = []
    original_close = plt.close

    def capture_close(fig):
        captured.append(fig)

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        write_acquisition_cleaning_report(
            summary, comparisons, intervals, target,
            patient_id="UPN27", compensation_state="BLOCKED",
        )
        assert target.is_file() and target.stat().st_size > 5_000
        detail = captured[-1]
        detail.canvas.draw()
        renderer = detail.canvas.get_renderer()
        data_axes = [axis for axis in detail.axes if axis.axison]
        assert len(detail.legends) == 1
        legend_box = detail.legends[0].get_window_extent(renderer)
        assert all(
            not legend_box.overlaps(axis.get_window_extent(renderer))
            for axis in data_axes
        )
    finally:
        for figure in captured:
            original_close(figure)
