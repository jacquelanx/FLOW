"""Tests for the cutoff-diagnostics wiring and its design discipline.

Covers the layers that need no container-only dependencies: the marker table, the threshold
provenance rules, the flag contract, the emitters, and the FLOW-side harness wiring.

Several tests here guard DESIGN RULES rather than behaviour, because those rules are the
reason the diagnostics can exist without becoming hardcoded analysis:

  * every threshold must declare where its value came from;
  * every flag must carry the rule that produced it and what would resolve it;
  * a marker's cutoff must be evaluated in the pool the pipeline actually applies it in;
  * the harness hook must stay generic (a project either ships a diagnostics script or not).
"""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from flow.firstrun import (
    ANCHORED_DIAGNOSTICS_TEMPLATE,
    ANCHORED_PACKAGE,
    DIAGNOSTICS_FILENAME,
    SCRIPT_FILENAME,
    anchored_diagnostics_script,
    build_first_run,
    has_diagnostics,
    provision_anchored,
)
from flow.firstrun.anchored.diagnostics import (
    emit,
    flags,
    markers,
    obligations,
    record,
    thresholds,
)

DIAGNOSTICS_PACKAGE = ANCHORED_PACKAGE / "diagnostics"


# ── marker table ──────────────────────────────────────────────────────────────
def test_every_marker_declares_the_pool_its_cutoff_is_applied_in():
    """Auditing a cutoff against the wrong population measures nothing the pipeline uses."""
    valid = {
        markers.POOL_SINGLETS, markers.POOL_LIVE, markers.POOL_CD45P, markers.POOL_CD14N,
        markers.POOL_CD19N, markers.POOL_CD19N_CD3NEG, markers.POOL_T, markers.POOL_NK,
        markers.POOL_DONOR,
    }
    for m in markers.MARKERS:
        assert m.pool in valid, f"{m.key} has an unknown derivation pool {m.pool!r}"
        assert m.applied_pool in valid, f"{m.key} has an unknown applied pool"
        assert m.label and m.positive_label
        assert m.downstream, f"{m.key} declares no dependent metrics to sweep"


def test_cd56_is_derived_in_the_cd3_negative_pool_but_applied_across_cd19_negative():
    """The one marker where derivation and application pools differ (gates.robust_cd56_cut)."""
    cd56 = markers.MARKERS_BY_KEY["cd56"]
    assert cd56.pool == markers.POOL_CD19N_CD3NEG
    assert cd56.applied_pool == markers.POOL_CD19N


def test_cd8_carries_no_auditable_cutoff():
    """CD8 is defined as CD4-negative within T, so placement metrics would audit nothing."""
    assert "cd8" in markers.NO_CUTOFF_KEYS
    assert "cd8" not in markers.MARKERS_BY_KEY


def test_accepted_fraction_bands_are_ordered_and_within_zero_one():
    for m in markers.MARKERS:
        if m.accepted_fraction is None:
            continue
        lo, hi = m.accepted_fraction
        assert 0.0 <= lo < hi <= 1.0, f"{m.key} band {m.accepted_fraction} is malformed"


def test_dim_donor_polarity_inverts_the_hla_positive_side():
    """Getting this backwards would invert the donor control checks in tier 4."""
    hla = markers.MARKERS_BY_KEY["hla"]
    assert markers.resolve_polarity(hla, hla_dim=False).positive_above is True
    dim = markers.resolve_polarity(hla, hla_dim=True)
    assert dim.positive_above is False
    assert "DIM" in dim.positive_label
    # Polarity must not leak into other markers.
    cd3 = markers.MARKERS_BY_KEY["cd3"]
    assert markers.resolve_polarity(cd3, hla_dim=True).positive_above is True


def test_headline_metrics_all_have_a_numerator_and_a_parent():
    for metric in markers.HEADLINE_METRICS:
        assert metric in markers.METRIC_NUMERATORS, f"{metric} has no numerator mask"
        assert metric in markers.METRIC_PARENTS, f"{metric} has no parent mask"


def test_every_headline_metric_is_reachable_from_some_cutoff():
    """A metric no cutoff moves would silently get no sensitivity range."""
    covered = {d for m in markers.MARKERS for d in m.downstream}
    assert set(markers.HEADLINE_METRICS) <= covered


# ── threshold provenance ──────────────────────────────────────────────────────
def test_every_threshold_declares_its_provenance():
    """The design rule: an undocumented threshold is hardcoded analysis in disguise."""
    cost_knobs = {"n_bootstrap", "n_permutation", "n_unimodality_boot", "random_seed"}
    for f in fields(thresholds.Thresholds):
        if f.name == "overrides_applied" or f.name in cost_knobs:
            continue
        assert f.name in thresholds.THRESHOLD_PROVENANCE, (
            f"threshold {f.name!r} has no provenance entry — add one stating where the value "
            "came from and why"
        )
        provenance, rationale = thresholds.THRESHOLD_PROVENANCE[f.name]
        assert provenance in thresholds.PROVENANCE_NOTES
        assert len(rationale) > 40, f"{f.name} rationale is too thin to review"


def test_describe_reports_value_default_and_provenance_for_every_threshold():
    rows = thresholds.Thresholds().describe()
    assert rows
    for r in rows:
        assert {"threshold", "value", "default", "overridden", "provenance", "rationale"} <= set(r)
        assert r["overridden"] is False


def test_pipeline_constants_match_the_pipeline():
    """MIN_PARENT / MIN_POS are reused, not re-invented; drift if they ever diverge."""
    from flow.firstrun.anchored.calibrate import MIN_PARENT, MIN_POS

    t = thresholds.Thresholds()
    assert t.min_parent_events == MIN_PARENT
    assert t.min_positive_events == MIN_POS


def test_metadata_overrides_are_applied_and_recorded():
    t = thresholds.Thresholds.from_metadata(
        {"diagnostics_thresholds": {"drift_negative_sd_max": 0.25}})
    assert t.drift_negative_sd_max == 0.25
    assert t.overrides_applied == {"drift_negative_sd_max": 0.25}
    assert any(r["overridden"] for r in t.describe())


def test_unknown_and_non_numeric_overrides_are_ignored_not_silently_accepted():
    """A typo in a threshold name must not look like a threshold that was set."""
    t = thresholds.Thresholds.from_metadata(
        {"diagnostics_thresholds": {"nope": 1, "drift_se_multiple": "abc"}})
    assert t.overrides_applied == {}
    assert t.drift_se_multiple == thresholds.Thresholds().drift_se_multiple


def test_integer_thresholds_stay_integers_after_override():
    t = thresholds.Thresholds.from_metadata(
        {"diagnostics_thresholds": {"min_parent_events": 80}})
    assert t.min_parent_events == 80
    assert isinstance(t.min_parent_events, int)


def test_missing_or_malformed_threshold_block_falls_back_to_defaults():
    for meta in (None, {}, {"diagnostics_thresholds": None}, {"diagnostics_thresholds": []}):
        t = thresholds.Thresholds.from_metadata(meta)
        assert t.drift_se_multiple == 1.0
        assert t.overrides_applied == {}


# ── flag contract ─────────────────────────────────────────────────────────────
def test_a_flag_must_carry_its_rule_and_a_resolution_hint():
    """A flag without its rule is a verdict, which this module is not allowed to issue."""
    book = flags.FlagBook()
    f = book.add(code="X", tier=2, subject="cd56 @ D28", rule="drift > 0.5 SD",
                 measured={"drift": 1.2}, resolution_hint="check the overlay")
    assert f.rule and f.resolution_hint and f.measured
    row = f.to_row()
    assert row["rule"] == "drift > 0.5 SD"
    assert json.loads(row["measured"]) == {"drift": 1.2}


def test_exceedance_direction_handles_both_failure_senses():
    book = flags.FlagBook()
    high = book.raise_if(True, code="H", tier=1, subject="s", rule="r", measured={},
                         resolution_hint="h", value=1.0, threshold=0.5)
    low = book.raise_if(True, code="L", tier=1, subject="s", rule="r", measured={},
                        resolution_hint="h", value=0.25, threshold=0.5,
                        higher_is_worse=False)
    assert high.exceedance == pytest.approx(2.0)
    assert low.exceedance == pytest.approx(2.0)


def test_raise_if_does_nothing_when_the_condition_is_false():
    book = flags.FlagBook()
    assert book.raise_if(False, code="X", tier=0, subject="s", rule="r", measured={},
                         resolution_hint="h") is None
    assert len(book) == 0


def test_flags_order_by_tier_then_by_exceedance():
    book = flags.FlagBook()
    book.add("A", 3, "s", "r", {}, "h", exceedance=9.0)
    book.add("B", 0, "s", "r", {}, "h", exceedance=1.5)
    book.add("C", 0, "s", "r", {}, "h", exceedance=4.0)
    book.add("D", 0, "s", "r", {}, "h", exceedance=None)
    assert [f.code for f in book.ordered()] == ["C", "B", "D", "A"]


def test_zero_denominator_exceedance_does_not_crash():
    book = flags.FlagBook()
    f = book.raise_if(True, code="X", tier=0, subject="s", rule="r", measured={},
                      resolution_hint="h", value=0.0, threshold=5.0, higher_is_worse=False)
    assert f.exceedance == float("inf")
    assert [x.code for x in book.ordered()] == ["X"]  # infinite exceedance must still sort


def test_measured_values_survive_json_encoding_even_when_numpy():
    np = pytest.importorskip("numpy")
    book = flags.FlagBook()
    f = book.add("X", 1, "s", "r", {"v": np.float64(1.5), "arr": np.array([1, 2])}, "h")
    decoded = json.loads(f.to_row()["measured"])
    assert decoded["v"] == 1.5
    assert decoded["arr"] == [1, 2]


# ── metric records ────────────────────────────────────────────────────────────
def test_records_keep_missing_values_missing():
    """"Missing is not zero" is load-bearing: a false "no drift" is worse than a gap."""
    rec = record.MetricRecorder()
    rec.add(tier=2, metric="drift", value=None, unit=record.U_TRANSFORM)
    rec.add(tier=2, metric="ratio", value=float("nan"))
    assert rec.rows[0]["value"] is None
    assert rec.rows[1]["value"] is None


def test_records_carry_a_dimensionless_twin():
    rec = record.MetricRecorder()
    rec.add(tier=2, metric="drift", value=0.38, unit=record.U_TRANSFORM,
            normalized_value=0.09, normalized_unit=record.U_GAP_FRACTION)
    row = rec.rows[0]
    assert row["value"] == 0.38 and row["unit"] == record.U_TRANSFORM
    assert row["normalized_value"] == 0.09 and row["normalized_unit"]


def test_safe_ratio_refuses_to_divide_by_nothing():
    assert record.safe_ratio(1.0, 0.0) is None
    assert record.safe_ratio(None, 2.0) is None
    assert record.safe_ratio(1.0, 4.0) == pytest.approx(0.25)


def test_rounded_preserves_missingness():
    assert record.rounded(None) is None
    assert record.rounded(float("nan")) is None
    assert record.rounded(1.23456, 2) == 1.23


# ── emitters ──────────────────────────────────────────────────────────────────
def _minimal_report() -> dict:
    return {
        "patient_id": "UPN00",
        "configuration": {"reference_timepoint": "Baseline", "timepoints": ["Baseline", "D7"],
                          "control_tubes": [], "hla_cutoff": 1.0, "car_cutoff": 2.0,
                          "hla_dim": False, "subsample": 100, "anchor_derivation": "negative"},
        "reproduction": {"available": True, "faithful": True, "checked": 10,
                         "max_abs_delta_pp": 0.0, "source": "multilineage.csv",
                         "mismatches": []},
        "notes": [], "skipped": [], "thresholds": thresholds.Thresholds().describe(),
        "tier0_data_adequacy": {"samples": [], "counting": []},
        "tier1_cutoff_foundation": {"available": True, "markers": {}},
        "tier2_transfer_validity": {"available": True, "markers": {}, "gate_geometry": {}},
        "tier3_sensitivity": {"envelope": [], "counterfactual": [], "policy_variance": {},
                              "coverage": {"dropped": [], "reason": None}},
        "tier4_internal_controls": {"host_nk_car": [], "pre_infusion_donor": [],
                                    "control_tubes": {}},
        "tier5_operator_concordance": {"available": False, "reason": "no manual CSV"},
        "tier_failures": [], "limitations": ["cannot see aesthetic wrongness"], "files": {},
    }


def test_summary_opens_with_how_to_read_it_and_states_the_discipline():
    book = flags.FlagBook()
    book.add("NEGATIVE_MODE_DRIFT", 2, "cd56 @ D28", "drift > 0.5 SD",
             {"drift_in_negative_sd": 1.17}, "read the overlay PNG", exceedance=2.3)
    text = emit.render_summary(_minimal_report(), book)
    assert "HOW TO READ THIS FILE" in text
    assert "MEASUREMENT, not a conclusion" in text
    assert "Missing is not zero" in text
    # The flag, its rule, its measurement and its next step must all be present.
    assert "NEGATIVE_MODE_DRIFT" in text and "cd56 @ D28" in text
    assert "drift > 0.5 SD" in text and "read the overlay PNG" in text
    assert "1.17" in text


def test_summary_says_so_when_no_rule_fired_without_claiming_correctness():
    text = emit.render_summary(_minimal_report(), flags.FlagBook())
    assert "FLAGGED ITEMS (0)" in text
    assert "not the same as" in text  # "...'the cutoffs are correct'"


def test_summary_surfaces_a_reproduction_mismatch_prominently():
    report = _minimal_report()
    report["reproduction"].update(
        {"faithful": False, "mismatches": [{"file": "a.fcs", "metric": "%NK (of lymph)",
                                            "delta_pp": 4.2}]})
    text = emit.render_summary(report, flags.FlagBook())
    assert "MISMATCH" in text
    assert "may describe a different gating" in text


def test_summary_reports_a_failed_tier_as_not_performed():
    report = _minimal_report()
    report["tier_failures"] = [{"tier": 3, "name": "sensitivity", "error": "ValueError: x"}]
    text = emit.render_summary(report, flags.FlagBook())
    assert "TIERS THAT FAILED TO COMPUTE" in text
    assert "sensitivity" in text


def test_summary_never_hides_a_coverage_gap():
    """No silent caps: dropped work must be named, not omitted."""
    report = _minimal_report()
    report["tier3_sensitivity"]["coverage"] = {
        "dropped": [{"timepoint": "D7", "marker": "cd56", "reason": "time budget exhausted"}],
        "reason": "wall-clock budget exhausted",
    }
    text = emit.render_summary(report, flags.FlagBook())
    assert "COVERAGE GAP" in text and "cd56" in text


def test_a_coverage_gap_names_the_markers_not_only_the_count():
    """A count nobody can act on is what five trajectories discharged in silence."""
    report = _minimal_report()
    report["tier3_sensitivity"]["coverage"] = {
        "dropped": [{"timepoint": f"T{i}", "marker": "car",
                     "reason": "no offset scale available"} for i in range(12)],
        "reason": None,
    }
    for text in (emit.render_summary(report, flags.FlagBook()),
                 emit.render_digest(report, flags.FlagBook())):
        assert "COVERAGE GAP" in text
        assert "'car'" in text, "the skipped marker must be named"
        # And what it costs, derived from MarkerSpec.downstream rather than listed.
        assert markers.CAR_METRIC in text


def test_a_coverage_gap_states_that_the_reported_range_came_from_other_cutoffs():
    """The trap this closes: 'cutoff' named as the dominant source of an unswept metric."""
    line = obligations.describe_coverage_gap(
        {"dropped": [{"timepoint": "D7", "marker": "car", "reason": "no offset scale"}]})
    assert "OTHER cutoffs" in line


# ── tier 1: where the audit was taken ─────────────────────────────────────────
def test_the_audit_table_says_which_timepoint_carried_each_row():
    """An absent tier-1 row and a clean one look identical; the column is what separates them."""
    rows = emit.cutoff_audit_rows({"markers": {
        "cd45": {"available": True, "label": "CD45", "cutoff": 1.0,
                 "audited_at_timepoint": "Baseline", "audited_at_reference": True},
        "car": {"available": True, "label": "CAR", "cutoff": 2.0,
                "audited_at_timepoint": "D7", "audited_at_reference": False,
                "reference_audit_unavailable_reason": "parent pool has only 2 events",
                "audit_scope_note": "judged against the D7 Donor pool instead"},
    }})
    by_marker = {r["marker"]: r for r in rows}
    assert by_marker["cd45"]["audited_at_reference"] is True
    assert by_marker["car"]["audited_at_reference"] is False
    assert by_marker["car"]["audited_at_timepoint"] == "D7"
    assert "2 events" in by_marker["car"]["reference_audit_unavailable_reason"]


def test_an_unavailable_audit_row_still_says_where_it_tried():
    """The row that carries no measurements is exactly the one that must explain itself."""
    rows = emit.cutoff_audit_rows({"markers": {"car": {
        "available": False, "reason": "parent pool has only 2 events",
        "audited_at_timepoint": None, "audited_at_reference": False,
        "audit_scope_note": "Treat it as unaudited, not as sound.",
    }}})
    assert rows[0]["audited_at_timepoint"] in (None, "")
    assert "unaudited" in rows[0]["audit_scope_note"]


def test_the_summary_says_when_a_marker_was_audited_away_from_the_reference():
    report = _minimal_report()
    report["tier1_cutoff_foundation"] = {"available": True, "markers": {"car": {
        "available": True, "label": "CAR", "cutoff": 1229.0,
        "audited_at_timepoint": "D7", "audited_at_reference": False,
        "reference_audit_unavailable_reason": "parent pool has only 2 events",
        "audit_scope_note": "these numbers judge the same locked cutoff at D7",
    }}}
    text = emit.render_summary(report, flags.FlagBook())
    assert "AUDITED AT D7" in text
    assert "2 events" in text


def test_the_fallback_audit_floor_is_stated_and_exceeds_the_counting_floor():
    """MIN_PARENT is enough to quote a percentage, not to estimate a density."""
    th = thresholds.Thresholds()
    assert th.fallback_audit_min_events > th.min_parent_events
    provenance, rationale = thresholds.THRESHOLD_PROVENANCE["fallback_audit_min_events"]
    assert provenance in thresholds.PROVENANCE_NOTES
    assert "COUNTING floor" in rationale


# ── 2-D gate geometry: all three gates ────────────────────────────────────────
def test_gate_geometry_carries_every_2d_gate_prefixed_by_which_one():
    """The HLA x CAR gate defines %Donor NK and %CAR+; it had no numeric surrogate at all."""
    rows = emit.gate_geometry_rows({"gate_geometry": {"D7": {
        "nk_quadrant": {"cd3_boundary_margin_in_sd": 3.0},
        "donor_car_quadrant": {"car_boundary_margin_in_sd": 1.2,
                               "hla_boundary_margin_donor_in_sd": 4.9},
        "locked_scatter": {"containment": 0.97},
    }}})
    row = rows[0]
    assert row["nk_cd3_boundary_margin_in_sd"] == 3.0
    assert row["donor_car_car_boundary_margin_in_sd"] == 1.2
    assert row["donor_car_hla_boundary_margin_donor_in_sd"] == 4.9
    assert row["scatter_containment"] == 0.97


def test_every_named_geometry_column_is_a_margin_or_a_fraction_never_a_count():
    """``_table_is_read`` needs only one column to match, so a count would discharge the item.

    The defeat this guards: the geometry flags report ``n_nk``, it resolved to an event-count
    column, and printing that count satisfied an obligation about a boundary margin.
    """
    for col in obligations.GATE_GEOMETRY_COLUMNS:
        assert ("margin" in col or "fraction" in col
                or col in ("scatter_containment", "scatter_spillover")), col
        assert not col.startswith("n_") and "_n_" not in col, f"{col} is a count"


def test_csv_writer_unions_heterogeneous_keys(tmp_path: Path):
    """An 'unavailable' row explains itself with a field its siblings lack; keep it."""
    rows = [{"marker": "cd3", "value": 1.0}, {"marker": "car", "reason": "no events"}]
    p = emit._write_csv(tmp_path / "t.csv", rows, preferred=("marker",))
    text = p.read_text()
    assert text.splitlines()[0] == "marker,value,reason"
    assert "no events" in text


def test_csv_writer_serializes_containers_and_blanks_none(tmp_path: Path):
    p = emit._write_csv(tmp_path / "t.csv", [{"a": {"k": 1}, "b": None, "c": True}])
    body = p.read_text().splitlines()[1]
    assert '{""k"": 1}' in body or '{"k": 1}' in body
    assert body.endswith("true")


def test_csv_writer_returns_none_for_no_rows(tmp_path: Path):
    assert emit._write_csv(tmp_path / "t.csv", []) is None


def test_write_all_produces_the_json_and_the_summary(tmp_path: Path):
    written = emit.write_all(tmp_path, _minimal_report(), record.MetricRecorder(),
                             flags.FlagBook())
    assert (tmp_path / emit.JSON_FILE).is_file()
    assert (tmp_path / emit.SUMMARY_TXT).is_file()
    assert "json" in written and "summary" in written
    # Thresholds always have rows, so that table must always be written.
    assert (tmp_path / emit.THRESHOLDS_CSV).is_file()


# ── harness wiring ────────────────────────────────────────────────────────────
def test_diagnostics_template_exists_and_honors_the_first_run_contract():
    src = anchored_diagnostics_script()
    assert src, "diagnostics template is empty/missing"
    for token in ("--data", "--out", "--plots"):
        assert token in src
    assert ANCHORED_DIAGNOSTICS_TEMPLATE.is_file()


def test_diagnostics_template_states_that_it_does_not_modify_the_first_run():
    src = anchored_diagnostics_script().lower()
    assert "does not modify" in src or "not modify" in src
    assert "authoritative" in src


def test_provision_installs_the_diagnostics_entry_point_and_package(tmp_path: Path):
    provision_anchored(tmp_path)
    assert (tmp_path / SCRIPT_FILENAME).is_file()
    assert (tmp_path / DIAGNOSTICS_FILENAME).is_file()
    assert (tmp_path / DIAGNOSTICS_FILENAME).read_text() == ANCHORED_DIAGNOSTICS_TEMPLATE.read_text()
    # The diagnostics subpackage must travel with it, or `import anchored.diagnostics` fails
    # inside the sandbox.
    for mod in ("__init__.py", "estimators.py", "context.py", "foundation.py", "transfer.py",
                "sensitivity.py", "controls.py", "concordance.py", "adequacy.py", "emit.py",
                "orchestrate.py", "markers.py", "thresholds.py", "flags.py", "record.py"):
        assert (tmp_path / "anchored" / "diagnostics" / mod).is_file(), f"missing {mod}"
    assert not (tmp_path / "anchored" / "diagnostics" / "__pycache__").exists()


def test_provision_is_idempotent(tmp_path: Path):
    provision_anchored(tmp_path)
    provision_anchored(tmp_path)
    assert (tmp_path / "anchored" / "diagnostics" / "estimators.py").is_file()


def test_first_run_gains_a_second_cell_only_when_a_diagnostics_script_is_present(tmp_path: Path):
    """The hook is generic: a project either ships one or it does not."""
    assert build_first_run(tmp_path) is None
    (tmp_path / SCRIPT_FILENAME).write_text("print('x')")
    assert not has_diagnostics(tmp_path)
    fr = build_first_run(tmp_path)
    assert fr.kind == "script" and len(fr.cells) == 1
    assert "DIAGNOSTICS" not in fr.prompt_note

    (tmp_path / DIAGNOSTICS_FILENAME).write_text("print('y')")
    assert has_diagnostics(tmp_path)
    fr = build_first_run(tmp_path)
    assert fr.kind == "script+diagnostics" and len(fr.cells) == 2
    assert "DIAGNOSTICS pass" in fr.prompt_note


def test_first_run_mode_none_still_disables_everything(tmp_path: Path):
    (tmp_path / SCRIPT_FILENAME).write_text("print('x')")
    (tmp_path / DIAGNOSTICS_FILENAME).write_text("print('y')")
    assert build_first_run(tmp_path, "none") is None


def test_generated_diagnostics_cell_is_valid_python(tmp_path: Path):
    import ast

    from flow.firstrun import CONTAINER_DIAGNOSTICS, _diagnostics_cell

    cell = _diagnostics_cell()
    ast.parse(cell)  # it is executed verbatim in the sandbox kernel
    assert CONTAINER_DIAGNOSTICS in cell
    assert "__DIAG_SCRIPT__" not in cell  # the placeholder must be substituted
    # It must add only NEW tables, not clobber the first-run's.
    assert "_before" in cell and "first_run_tables" in cell


def test_diagnostics_prompt_note_states_the_measurement_discipline():
    from flow.firstrun import DIAGNOSTICS_NOTE

    lowered = DIAGNOSTICS_NOTE.lower()
    assert "measurement" in lowered and "rule" in lowered
    assert "not conclusions" in lowered
    assert "null is" in lowered or "not zero" in lowered


def test_diagnostics_package_init_is_lazy():
    """Importing the package must not pull flowkit/pandas at module load."""
    init_src = (DIAGNOSTICS_PACKAGE / "__init__.py").read_text()
    assert "__getattr__" in init_src
    for line in init_src.splitlines():
        assert not line.startswith(("from .orchestrate", "import .orchestrate",
                                    "from .context", "from ..gates"))


def test_estimator_core_imports_no_heavy_dependency():
    """estimators.py must stay numpy-only so it is testable in FLOW's own dev env."""
    src = (DIAGNOSTICS_PACKAGE / "estimators.py").read_text()
    for forbidden in ("import pandas", "import scipy", "import sklearn", "from scipy",
                      "from sklearn", "import flowkit"):
        assert forbidden not in src, f"estimators.py must not use {forbidden!r}"


def test_pure_layers_import_no_heavy_dependency():
    """These are the layers the tests above exercise; keep them dependency-light."""
    for name in ("markers.py", "thresholds.py", "flags.py", "record.py", "emit.py"):
        src = (DIAGNOSTICS_PACKAGE / name).read_text()
        for forbidden in ("import pandas", "import numpy", "from ..gates", "from ..fcs_io"):
            assert forbidden not in src, f"{name} must not use {forbidden!r}"


# ── digest (the bounded index the agent actually reads) ───────────────────────
# The full summary runs to ~265 KB on a real study while the agent's observation window is a
# few thousand characters, and the truncator keeps the head and tail — deleting the middle,
# where the flags are. These tests guard the property that makes the digest usable: it is
# bounded BY CONSTRUCTION and still accounts for every finding.
def _synthetic_book(n_codes: int, per_code: int, rule_len: int = 300) -> flags.FlagBook:
    book = flags.FlagBook()
    for c in range(n_codes):
        for i in range(per_code):
            book.add(
                code=f"SYNTHETIC_CODE_{c:03d}",
                tier=c % 6,
                subject=f"marker_{i:02d} @ timepoint_{i:02d}",
                rule="rule text " * (rule_len // 10),
                measured={"value": i},
                resolution_hint="hint " * 40,
                exceedance=float(i + 1),
            )
    return book


def _hidden_groups(text: str) -> tuple[int, int]:
    """Parse the '+N more group(s) covering M flag(s)' line, or (0, 0) if absent."""
    m = re.search(r"\+(\d+) more group\(s\) covering (\d+) flag\(s\)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


@pytest.mark.parametrize(
    "n_codes,per_code",
    [(1, 1), (17, 25), (40, 10), (120, 5), (300, 4)],
)
def test_digest_stays_inside_its_budget_at_every_scale(n_codes: int, per_code: int):
    """A trailing cut would delete the middle; the digest must fit before it is written."""
    text = emit.render_digest(_minimal_report(), _synthetic_book(n_codes, per_code))
    assert len(text) <= emit.DIGEST_MAX_CHARS, (
        f"{n_codes} codes x {per_code} produced {len(text)} chars")


def test_digest_accounts_for_every_flag_even_when_it_cannot_show_them_all():
    """No silent caps: what is not shown must be counted, by group and by flag."""
    book = _synthetic_book(300, 4)
    text = emit.render_digest(_minimal_report(), book)
    assert f"all {len(book)} flag(s)" in text
    hidden_groups, hidden_flags = _hidden_groups(text)
    shown_groups = len(emit._flag_groups(book)) - hidden_groups
    assert hidden_groups > 0, "this scale should force groups to be withheld"
    assert shown_groups > 0
    # Every withheld group is counted, and its flags are counted too.
    assert hidden_groups + shown_groups == len(emit._flag_groups(book))
    assert hidden_flags == len(book) - sum(
        len(fs) for _, _, fs in emit._flag_groups(book)[:shown_groups])
    assert "diagnostics_flags.csv" in text


def test_digest_names_every_group_with_its_full_count_at_realistic_scale():
    """At the scale a real study produces, no group is withheld and every count is exact."""
    book = _synthetic_book(17, 25)
    text = emit.render_digest(_minimal_report(), book)
    assert _hidden_groups(text) == (0, 0)
    for c in range(17):
        assert f"SYNTHETIC_CODE_{c:03d}  x25" in text
    # The rule must travel with the finding — that is the whole discipline.
    assert text.count("rule  :") == 17


def test_digest_degrades_detail_rather_than_dropping_a_group():
    """A missing group reads as 'nothing fired there' — the false negative to avoid."""
    lean = emit.render_digest(_minimal_report(), _synthetic_book(60, 8))
    for c in range(60):
        assert f"SYNTHETIC_CODE_{c:03d}" in lean
    # Detail was shed (no per-group rule lines survive at this scale), counts were not.
    assert "rule  :" not in lean or lean.count("rule  :") < 60
    assert lean.count("x8") >= 60


def test_digest_is_a_bounded_index_that_fits_the_agents_seed_observation_window():
    """The chain that makes the fix work end to end: digest <= printed <= observation."""
    from flow.env.notebook_env import SEED_OBS_CHARS
    from flow.firstrun import DIGEST_PRINT_CHARS

    assert emit.DIGEST_MAX_CHARS <= DIGEST_PRINT_CHARS <= SEED_OBS_CHARS


def test_the_harness_and_the_writer_agree_on_the_digest_filename():
    """The cell reads the file by name; a rename in one place would silently fall back."""
    from flow.firstrun import DIGEST_FILENAME

    assert DIGEST_FILENAME == emit.DIGEST_TXT


def test_digest_says_so_when_no_rule_fired_without_claiming_correctness():
    text = emit.render_digest(_minimal_report(), flags.FlagBook())
    assert "FLAGGED ITEMS (0)" in text
    assert "not the same as" in text


def test_digest_surfaces_a_reproduction_mismatch_before_anything_else():
    report = _minimal_report()
    report["reproduction"].update({"faithful": False, "mismatches": [{"file": "a.fcs"}]})
    text = emit.render_digest(report, flags.FlagBook())
    assert "REPRODUCTION: MISMATCH" in text
    assert text.index("REPRODUCTION: MISMATCH") < text.index("FLAGGED ITEMS")


def test_digest_names_a_tier_that_failed_to_compute():
    report = _minimal_report()
    report["tier_failures"] = [{"tier": 3, "name": "sensitivity", "error": "ValueError: x"}]
    text = emit.render_digest(report, flags.FlagBook())
    assert "NOT PERFORMED" in text and "sensitivity" in text


def test_digest_never_hides_a_coverage_gap():
    report = _minimal_report()
    report["tier3_sensitivity"]["coverage"] = {
        "dropped": [{"timepoint": "D7", "marker": "cd56"}], "reason": "time budget"}
    text = emit.render_digest(report, flags.FlagBook())
    assert "COVERAGE GAP" in text and "NOT " in text


def test_digest_distinguishes_an_unbounded_ratio_from_a_non_numeric_rule():
    """Both are 'no number'; conflating them hides which rule can even be exceeded."""
    book = flags.FlagBook()
    book.add("A_CODE", 0, "x @ D7", "r", {}, "h", exceedance=float("inf"))
    book.add("B_CODE", 0, "y @ D7", "r", {}, "h", exceedance=None)
    text = emit.render_digest(_minimal_report(), book)
    assert "unbounded" in text
    assert "not a numeric rule" in text


def test_digest_is_deterministic():
    book = _synthetic_book(12, 7)
    a = emit.render_digest(_minimal_report(), book)
    b = emit.render_digest(_minimal_report(), book)
    assert a == b


def test_digest_reports_row_counts_so_the_agent_knows_what_it_has_not_read(tmp_path: Path):
    written = emit.write_all(tmp_path, _minimal_report(), record.MetricRecorder(),
                             flags.FlagBook())
    assert (tmp_path / emit.DIGEST_TXT).is_file()
    assert "digest" in written
    text = (tmp_path / emit.DIGEST_TXT).read_text()
    # The thresholds table always has rows; the digest must state how many.
    assert emit.THRESHOLDS_CSV in text
    assert "rows" in text
    assert "DataFrame in this notebook" in text


def test_diagnostics_cell_prints_the_digest_file_not_the_whole_stdout(tmp_path: Path):
    """The report is written to disk in full; only a bounded index is printed."""
    from flow.firstrun import DIGEST_FILENAME, DIGEST_PRINT_CHARS, _diagnostics_cell

    cell = _diagnostics_cell()
    assert DIGEST_FILENAME in cell
    assert str(DIGEST_PRINT_CHARS) in cell
    # The old behaviour: dump 14k of stdout into a 4k window.
    assert "stdout[-14000:]" not in cell
    # Ten wide tables x head(12) is what used to consume the rest of the window.
    assert "head(12)" not in cell
    assert "shape=" in cell and "columns:" in cell


def test_diagnostics_cell_still_shows_something_when_no_digest_was_written(tmp_path: Path):
    """A project whose diagnostics script predates the digest must not go silent."""
    from flow.firstrun import _diagnostics_cell

    cell = _diagnostics_cell()
    assert "no digest file at" in cell
    assert "_d.stdout[-8000:]" in cell
    # A non-zero exit must still surface its stderr.
    assert "DIAGNOSTICS STDERR" in cell


def test_diagnostics_note_tells_the_agent_the_digest_is_an_index_not_the_report():
    """Printing an index while implying it is the whole report invites confident wrong claims."""
    from flow.firstrun import DIAGNOSTICS_NOTE

    assert "DIGEST" in DIAGNOSTICS_NOTE and "INDEX" in DIAGNOSTICS_NOTE
    assert "not the whole report" in DIAGNOSTICS_NOTE
    # It must say how to reach the rows the digest only counted.
    assert "DataFrames" in DIAGNOSTICS_NOTE
    assert "code" in DIAGNOSTICS_NOTE


# ── digest: the sample axis ───────────────────────────────────────────────────
def _report_with_timepoints(off_trend: bool):
    """A tier-0 payload carrying the per-timepoint rollup and one optional off-trend value."""
    r = _minimal_report()
    r["tier0_data_adequacy"] = {
        "samples": [], "counting": [],
        "per_timepoint": [
            {"timepoint": "Baseline", "n_events_analyzed": 577564, "n_lymphocytes": 4106,
             "lineage_purity_pct": 76.693},
            {"timepoint": "Post", "n_events_analyzed": 469586, "n_lymphocytes": 54138,
             "lineage_purity_pct": 84.713},
            {"timepoint": "D3", "n_events_analyzed": 129753, "n_lymphocytes": 2129,
             "lineage_purity_pct": 76.562},
        ],
        "discontinuity": [
            {"timepoint": "Baseline", "metric": "%NK (of lymph)", "assessable": False,
             "reason": "first timepoint in the series — no earlier neighbour"},
            {"timepoint": "Post", "metric": "%NK (of lymph)", "value_pct": 0.076,
             "assessable": True, "discontinuous": off_trend,
             "direction": "below both neighbours" if off_trend else "within envelope",
             "prev_timepoint": "Pre", "prev_value_pct": 19.408,
             "next_timepoint": "D3", "next_value_pct": 14.843,
             "tolerance_pp": 4.6, "excursion_in_tolerances": 3.211 if off_trend else None},
            {"timepoint": "D3", "metric": "%NK (of lymph)", "assessable": False,
             "reason": "last timepoint in the series — no later neighbour"},
        ],
    }
    return r


def test_digest_shows_the_sample_axis_before_the_rule_indexed_flags():
    """A bad timepoint hides inside a rule's tally; the per-sample view has to come first."""
    text = emit.render_digest(_report_with_timepoints(True), _synthetic_book(3, 4))
    assert "PER-TIMEPOINT" in text
    assert text.index("PER-TIMEPOINT") < text.index("FLAGGED ITEMS")
    # Each timepoint appears with its own counts, so none can be passed over silently.
    for tp in ("Baseline", "Post", "D3"):
        assert tp in text
    assert "54138" in text or "54,138" in text


def test_digest_names_every_off_trend_value_with_both_neighbours():
    """The number is unjudgeable without the two values it is being compared against."""
    text = emit.render_digest(_report_with_timepoints(True), flags.FlagBook())
    assert "OFF-TREND VALUES (1)" in text
    assert "0.076" in text and "Post" in text
    assert "Pre=19.408" in text and "D3=14.843" in text   # both neighbours quoted
    assert "3.211" in text                                 # how far past tolerance
    assert "spike, not a step" in text                     # why a trend cannot produce it
    # The caveat that makes a run of flags readable must travel with them.
    assert "read a RUN of these together" in text


def test_digest_states_what_was_not_assessed_rather_than_implying_a_pass():
    """Endpoints cannot be assessed; silence about them would read as 'fine'."""
    text = emit.render_digest(_report_with_timepoints(False), flags.FlagBook())
    assert "OFF-TREND VALUES (0)" in text
    assert "not the same" in text          # "...not the same as passing"
    assert "endpoints" in text
    assert emit.DISCONTINUITY_CSV in text


def test_digest_with_the_sample_axis_still_fits_its_budget():
    """The new section must not push the per-group rule lines off the ladder."""
    text = emit.render_digest(_report_with_timepoints(True), _synthetic_book(17, 25))
    assert len(text) <= emit.DIGEST_MAX_CHARS
    assert "rule  :" in text               # discipline survived the addition
    assert "PER-TIMEPOINT" in text


def test_the_window_chain_still_holds_after_the_digest_grew():
    from flow.env.notebook_env import SEED_OBS_CHARS
    from flow.firstrun import DIGEST_PRINT_CHARS

    assert emit.DIGEST_MAX_CHARS <= DIGEST_PRINT_CHARS <= SEED_OBS_CHARS


def test_write_all_emits_the_two_sample_axis_tables(tmp_path: Path):
    written = emit.write_all(tmp_path, _report_with_timepoints(True), record.MetricRecorder(),
                             flags.FlagBook())
    assert (tmp_path / emit.TIMEPOINTS_CSV).is_file()
    assert (tmp_path / emit.DISCONTINUITY_CSV).is_file()
    # The not-assessable rows must be in the CSV too, with their reason.
    body = (tmp_path / emit.DISCONTINUITY_CSV).read_text()
    assert "no earlier neighbour" in body and "no later neighbour" in body
    assert "timepoints" in written and "discontinuity" in written


def test_diagnostics_note_forbids_dropping_a_timepoint_silently():
    from flow.firstrun import DIAGNOSTICS_NOTE

    assert "PER-TIMEPOINT" in DIAGNOSTICS_NOTE
    assert "including or excluding" in DIAGNOSTICS_NOTE
    assert "Silently leaving a timepoint out" in DIAGNOSTICS_NOTE


# ── tier 3 values vs the headline numbers ─────────────────────────────────────
# Tier 3 re-gates a CAPPED subsample (event_cap_per_file, default 40000) while the pipeline
# gates its full subsample (200000 on the study this was found on). A column named "value_pct"
# in the uncertainty table therefore disagreed with multilineage.csv — Baseline read 30.277
# against a reported 29.834 — with nothing saying so, and the reproduction check could not
# catch it because it validates the replay, not tier 3.
def _envelope_report(headline: float, subsampled: float):
    r = _minimal_report()
    r["tier3_sensitivity"] = {
        "envelope": [{
            "timepoint": "Baseline", "metric": "%NK (of lymph)",
            "headline_value_pct": headline,
            "value_pct_at_sensitivity_subsample": subsampled,
            "headline_minus_subsample_pp": round(headline - subsampled, 4),
            "cutoff_low_pct": 10.847, "cutoff_high_pct": 33.921, "cutoff_range_pp": 23.074,
            "counting_ci_halfwidth_pp": 1.399, "dominant_uncertainty": "cutoff",
            "cutoff_range_driver_low": "cd56:minus", "cutoff_range_driver_high": "cd56:plus",
        }],
        "counterfactual": [{
            "timepoint": "D7", "metric": "%NK (of lymph)", "headline_value_pct": 42.616,
            "locked_value_pct_at_subsample": 42.6, "per_sample_value_pct_at_subsample": 0.099,
            "delta_pp": -42.501,
        }],
        "policy_variance": {}, "coverage": {"dropped": [], "reason": None},
    }
    return r


def test_the_uncertainty_table_separates_the_headline_from_the_subsampled_value(tmp_path: Path):
    """Both must be present and distinguishable by name, plus their difference."""
    report = _envelope_report(29.834, 30.277)
    emit.write_all(tmp_path, report, record.MetricRecorder(), flags.FlagBook())
    header = (tmp_path / emit.UNCERTAINTY_CSV).read_text().splitlines()[0]
    assert "headline_value_pct" in header
    assert "value_pct_at_sensitivity_subsample" in header
    assert "headline_minus_subsample_pp" in header
    # The bare, ambiguous name must be gone: it is what invited quoting the wrong number.
    cols = header.split(",")
    assert "value_pct" not in cols
    body = (tmp_path / emit.UNCERTAINTY_CSV).read_text().splitlines()[1]
    assert "29.834" in body and "30.277" in body and "-0.443" in body


def test_the_counterfactual_table_labels_its_values_as_subsampled(tmp_path: Path):
    emit.write_all(tmp_path, _envelope_report(29.834, 30.277), record.MetricRecorder(),
                   flags.FlagBook())
    header = (tmp_path / emit.COUNTERFACTUAL_CSV).read_text().splitlines()[0].split(",")
    assert "locked_value_pct_at_subsample" in header
    assert "per_sample_value_pct_at_subsample" in header
    assert "headline_value_pct" in header
    assert "locked_value_pct" not in header and "per_sample_value_pct" not in header


def test_the_summary_quotes_the_headline_value_not_the_subsampled_one():
    """The prose table is what a reader copies from; it must carry the reported number."""
    text = emit.render_summary(_envelope_report(29.834, 30.277), flags.FlagBook())
    i = text.index("HEADLINE NUMBERS")
    window = text[i:i + 1400]
    assert "29.834" in window
    assert "30.277" not in window
    # And it must say how to read the span, since the bounds come from the subsample.
    assert "WIDTH" in window


def test_tier3_states_that_its_values_are_not_the_reported_numbers():
    """The distinction has to be written down where the tier documents itself."""
    from flow.firstrun.anchored.diagnostics import sensitivity

    # Fragments chosen to sit inside a single source string literal: the note is wrapped
    # across lines, so a longer phrase would straddle a boundary and never match.
    src = Path(sensitivity.__file__).read_text()
    assert "_at_subsample' are NOT the pipeline's reported numbers" in src
    assert "carries headline_value_pct" in src
    assert "quote THAT as the number" in src
    # Differences stay valid — that is why only the absolute values needed relabelling.
    assert "both sides of every comparison come from" in src
