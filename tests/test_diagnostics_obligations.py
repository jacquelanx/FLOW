"""The obligations worklist: is the list derived, finishable, and honest about its cap?

The defect these tests guard is not a wrong number — it is a correct report nobody finishes.
Two production runs opened the digest, quoted its headline, and between them opened three of
eleven tables while 357 flag rows went unread and unmentioned. So the properties under test
are about USABILITY as much as correctness:

  * derived from what fired, so a clean study yields an empty list rather than busywork;
  * capped per class, so one prolific class cannot starve the rest of the report;
  * counted, so "not promoted" can never be read as "nothing there";
  * pointing only at tables that exist, because a worklist that names a missing file is worse
    than no worklist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow.firstrun.anchored.diagnostics import emit, flags, obligations, record

# Table labels -> the paths ``write_all`` would have recorded. Used as the ``written`` map so
# ``must_read`` and ``how`` resolve to the notebook variable names the agent actually has.
WRITTEN = {
    "flags": "/work/first_run/diagnostics_flags.csv",
    "cutoff_audit": "/work/first_run/diagnostics_cutoff_audit.csv",
    "transfer": "/work/first_run/diagnostics_transfer.csv",
    "uncertainty": "/work/first_run/diagnostics_uncertainty.csv",
    "counterfactual": "/work/first_run/diagnostics_counterfactual.csv",
    "controls": "/work/first_run/diagnostics_controls.csv",
    "timepoints": "/work/first_run/diagnostics_timepoints.csv",
    "discontinuity": "/work/first_run/diagnostics_discontinuity.csv",
    "gate_geometry": "/work/first_run/diagnostics_gate_geometry.csv",
}


def _report(**over) -> dict:
    """A report with nothing wrong in it, so each test adds only the finding it is about."""
    r = {
        "patient_id": "UPN00",
        "configuration": {"reference_timepoint": "Baseline", "timepoints": ["Baseline", "D7"]},
        "reproduction": {"available": True, "faithful": True, "checked": 10,
                         "max_abs_delta_pp": 0.0, "source": "multilineage.csv",
                         "mismatches": []},
        "tier0_data_adequacy": {"per_timepoint": [], "discontinuity": []},
        "tier3_sensitivity": {"envelope": [], "coverage": {"dropped": [], "reason": None}},
        "tier5_operator_concordance": {"available": True},
    }
    r.update(over)
    return r


def _disc(timepoint: str, metric: str, excursion: float, value: float = 42.0) -> dict:
    return {"timepoint": timepoint, "metric": metric, "value_pct": value, "assessable": True,
            "discontinuous": True, "direction": "above both neighbours",
            "prev_timepoint": "P", "prev_value_pct": 1.5,
            "next_timepoint": "N", "next_value_pct": 2.5,
            "tolerance_pp": 0.4, "excursion_in_tolerances": excursion}


def _book(*specs) -> flags.FlagBook:
    """A FlagBook from ``(code, tier, subject, exceedance)`` tuples."""
    b = flags.FlagBook()
    for code, tier, subject, exc in specs:
        b.add(code, tier, subject, f"rule for {code}", {"measured_value": 1.23},
              f"hint for {code}", exceedance=exc)
    return b


def _build(report=None, book=None, **kw):
    return obligations.build(report or _report(), book or flags.FlagBook(),
                             written=WRITTEN, **kw)


# ── derived, not enumerated ───────────────────────────────────────────────────
def test_a_clean_run_produces_an_empty_worklist_not_busywork():
    m = _build()
    assert m["items"] == []
    assert m["n_candidates"] == 0 and m["n_not_promoted"] == 0


def test_an_empty_worklist_does_not_read_as_a_clean_bill_of_health():
    """Zero obligations means no rule fired, which is not the same as 'the cutoffs hold'."""
    text = obligations.render_worklist(_build())
    assert "0 items" in text
    assert "not the same as" in text


def test_a_fired_flag_becomes_an_obligation_carrying_its_full_group_count():
    """The count is the point: 'you have seen 1 of 23' is what turns a tally back into work."""
    book = _book(*[(flags.F_NEG_MODE_DRIFT, 2, f"cd56 @ D{i}", 3.0) for i in range(23)])
    m = _build(book=book)
    item = next(i for i in m["items"] if i["id"].endswith("NEGATIVE_MODE_DRIFT"))
    assert item["n_findings"] == 23
    assert "23 finding(s)" in item["statement"]
    assert "seen 1" in item["statement"]


def test_a_flag_group_carries_the_flags_own_resolution_hint_not_an_invented_one():
    """``why`` must come from the report; inventing a rationale would be concluding."""
    book = _book((flags.F_NO_VALLEY_EVIDENCE, 1, "cd19 @ Baseline", None))
    item = _build(book=book)["items"][0]
    assert item["why"] == f"hint for {flags.F_NO_VALLEY_EVIDENCE}"


def test_a_new_flag_code_is_classified_by_its_tier_without_touching_this_module():
    """The list is derived: an unrecognised code still lands in its tier's class and table."""
    book = _book(("SOME_FUTURE_RULE", 2, "cd99 @ D7", 5.0))
    item = _build(book=book)["items"][0]
    assert item["kind"] == "transfer_validity"
    # The flag rows settle it; the tier's measurement table is supporting context.
    assert item["must_read"] == ["diagnostics_flags"]
    assert item["should_read"] == ["diagnostics_transfer"]


def test_genuine_context_stays_context_even_now_that_promotion_exists():
    """``timepoints`` is not the table the off-trend rule was evaluated on — per-sample
    retention is background to that judgement, not the measurement behind it. Promotion
    follows the rule; it is not licence to require every named table again."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D14", "%NK (of lymph)", 13.7)]})
    off = _build(r)["items"][0]
    assert off["must_read"] == ["diagnostics_discontinuity"]
    assert off["should_read"] == ["diagnostics_timepoints"]


def test_a_table_whose_columns_cannot_be_named_stays_advisory():
    """The demand that bought a throwaway ``.head()`` was "open this table". Without resolved
    columns this module can only make that same demand, so it declines to and says "also"."""
    # WRITTEN points at paths that do not exist, so no header resolves — the honest case.
    book = _book(("NEGATIVE_MODE_DRIFT", 2, "cd56 @ D28", 4.0))
    it = _build(book=book)["items"][0]
    assert it["must_read"] == ["diagnostics_flags"]
    assert it["should_read"] == ["diagnostics_transfer"]
    assert not it["read_columns"].get("diagnostics_transfer")


def test_the_worklist_distinguishes_the_settling_read_from_optional_context():
    """A reader who thinks every listed table is mandatory batches them all into one cell."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D14", "%NK (of lymph)", 13.7)]})
    text = obligations.render_worklist(_build(r))
    assert "read: diagnostics_discontinuity[" in text
    assert "also: diagnostics_timepoints[" in text
    assert "`also:` line is context" in text


# ── the sample axis ───────────────────────────────────────────────────────────
def test_off_trend_samples_are_promoted_one_per_sample_worst_first():
    """The decision — include this sample or exclude it — is made once per sample."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D7", "%NK (of lymph)", 3.2),
        _disc("D14", "%T (of lymph)", 43.8),
        _disc("D14", "%NK (of lymph)", 13.7),
    ]})
    items = [i for i in _build(r)["items"] if i["kind"] == "off_trend_timepoint"]
    assert [i["id"] for i in items] == ["OFF_TREND__D14", "OFF_TREND__D7"]
    # D14 fired on two metrics but gets ONE obligation, which says so.
    assert "+1 more metric(s)" in items[0]["statement"]


def test_an_off_trend_obligation_quotes_both_neighbours_it_is_judged_against():
    """The value alone is unjudgeable; the comparison is what makes it a finding."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D14", "%NK (of lymph)", 13.7, value=0.075)]})
    item = _build(r)["items"][0]
    assert "0.075" in item["statement"]
    assert "P=1.5" in item["statement"] and "N=2.5" in item["statement"]
    assert "13.7" in item["statement"]


def test_a_sample_label_with_spaces_survives_into_the_id_and_the_filter():
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D35 Pleural Fluid", "%T (of lymph)", 5.2)]})
    item = _build(r)["items"][0]
    assert item["id"] == "OFF_TREND__D35_PLEURAL_FLUID"
    assert item["must_mention"] == ["D35 Pleural Fluid"]
    assert "'D35 Pleural Fluid'" in item["how"][0]


def test_a_discontinuity_finding_is_not_promoted_twice():
    """It is one finding; spending a worklist slot on each axis would waste the list."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D14", "%NK (of lymph)", 13.7)]})
    book = _book((flags.F_TEMPORAL_DISCONTINUITY, 0, "%NK (of lymph) @ D14", 13.7))
    m = _build(r, book)
    assert [i["id"] for i in m["items"]] == ["OFF_TREND__D14"]
    assert not any(flags.F_TEMPORAL_DISCONTINUITY in i["id"] for i in m["items"])


# ── the caps ──────────────────────────────────────────────────────────────────
def test_a_prolific_class_cannot_starve_the_rest_of_the_report():
    """14 off-trend samples must not push every tier-1 finding off the list."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc(f"T{i}", "%NK (of lymph)", float(20 - i)) for i in range(14)]})
    book = _book(
        (flags.F_NO_VALLEY_EVIDENCE, 1, "cd19 @ Baseline", None),
        (flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D7", 4.0),
        (flags.F_FRAGILE_METRIC, 3, "%NK (of lymph) @ D7", 6.0),
        (flags.F_CONTROL_MISSING, 4, "host NK CAR check", None),
    )
    kinds = {i["kind"] for i in _build(r, book)["items"]}
    for cls in ("off_trend_timepoint", "cutoff_foundation", "transfer_validity",
                "sensitivity", "internal_controls"):
        assert cls in kinds, f"{cls} was starved off the worklist"


def test_class_caps_sum_to_the_global_cap_so_the_per_class_limits_are_what_bind():
    """If the global cap bit first, the lowest-priority classes could never appear at all."""
    assert sum(obligations.CLASS_CAPS.values()) == obligations.MAX_OBLIGATIONS
    assert set(obligations.CLASS_CAPS) == set(obligations.CLASS_ORDER)


def test_nothing_is_dropped_without_being_counted():
    """No silent caps: promoted + not promoted must account for every candidate."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc(f"T{i}", "%NK (of lymph)", float(20 - i)) for i in range(14)]})
    m = _build(r)
    assert m["n_promoted"] + m["n_not_promoted"] == m["n_candidates"]
    assert m["n_not_promoted"] == 14 - obligations.CLASS_CAPS["off_trend_timepoint"]
    assert "NOT PROMOTED" in m["not_promoted_note"].upper()
    assert "never 'nothing there'" in m["not_promoted_note"]


def test_the_rendered_worklist_says_what_it_did_not_promote():
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc(f"T{i}", "%NK (of lymph)", float(20 - i)) for i in range(14)]})
    text = obligations.render_worklist(_build(r))
    assert "NOT promoted" in text or "NOT PROMOTED" in text.upper()


@pytest.mark.parametrize("n_off_trend,n_codes", [(0, 0), (1, 1), (14, 18), (40, 60)])
def test_the_rendered_worklist_stays_inside_its_window_at_every_scale(n_off_trend, n_codes):
    """It shares a bounded observation with the digest; a trailing cut would eat its tail."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc(f"T{i}", "%NK (of lymph)", float(100 - i)) for i in range(n_off_trend)]})
    book = _book(*[(f"CODE_{i}", i % 6, f"item{i} @ T{i}", float(i + 1))
                   for i in range(n_codes)])
    text = obligations.render_worklist(_build(r, book))
    assert len(text) <= obligations.WORKLIST_MAX_CHARS, \
        f"{n_off_trend}/{n_codes} rendered {len(text)} chars"


def test_the_render_bound_holds_against_pathologically_long_flag_text():
    """The cap on ITEMS does not bound the render; rule and hint text are unbounded prose."""
    book = flags.FlagBook()
    for i in range(20):
        book.add(f"CODE_{i}", i % 6, "subject " * 60, "rule " * 200, {"v": 1},
                 "hint " * 200, exceedance=float(i + 1))
    text = obligations.render_worklist(_build(book=book))
    assert len(text) <= obligations.WORKLIST_MAX_CHARS, f"rendered {len(text)} chars"
    assert "more obligation(s) not shown" in text or text.count("read:") >= 1


def test_the_rendered_worklist_counts_items_it_had_no_room_to_show():
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc(f"T{i}", "%NK (of lymph)", float(100 - i)) for i in range(5)]})
    book = _book(*[(f"CODE_{i}", i % 6, f"item{i} @ T{i}", float(i + 1)) for i in range(20)])
    m = _build(r, book)
    text = obligations.render_worklist(m, max_chars=1800)
    assert "more obligation(s) not shown" in text
    assert obligations.OBLIGATIONS_JSON in text


# ── honesty about where to look ───────────────────────────────────────────────
def test_an_obligation_never_names_a_table_that_was_not_written():
    """A worklist pointing at a missing file is worse than no worklist."""
    book = _book((flags.F_COUNTERFACTUAL_GAP, 3, "%NK (of lymph) @ D7", 8.0))
    m = obligations.build(_report(), book, written={"flags": "/w/diagnostics_flags.csv"})
    item = m["items"][0]
    assert item["must_read"] == ["diagnostics_flags"]
    assert all("counterfactual" not in h for h in item["how"])


def test_every_must_read_entry_is_a_usable_notebook_variable_name():
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D14", "%NK (of lymph)", 13.7)]})
    book = _book((flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D7", 4.0),
                 (flags.F_BOUNDARY_MASS, 2, "NK cloud @ D7", 2.0))
    for item in _build(r, book)["items"]:
        for name in item["must_read"]:
            assert name.isidentifier(), name
            assert name in {Path(p).stem for p in WRITTEN.values()}


def test_a_flag_group_is_addressable_by_the_column_the_agent_would_filter_on():
    book = _book((flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D7", 4.0))
    item = _build(book=book)["items"][0]
    assert f"diagnostics_flags.code == '{flags.F_NEG_MODE_DRIFT}'" in item["how"][0]


def test_must_mention_never_gates_on_the_item_half_of_a_subject():
    """A gate on '%NK (of lymph)' or 'NK cloud' appearing verbatim would fail good prose.

    The scope half (a literal sample label) IS demanded — see the test below. What must never
    be demanded is the metric or population name, which careful writing can phrase any number
    of ways.
    """
    r = _report(
        reproduction={"available": True, "faithful": False, "checked": 140,
                      "max_abs_delta_pp": 4.2, "source": "multilineage.csv",
                      "mismatches": [{"metric": "%NK (of lymph)"}]},
        tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
            _disc("D14", "%NK (of lymph)", 13.7)]},
    )
    book = _book((flags.F_BOUNDARY_MASS, 2, "NK cloud @ D7", 2.0),
                 (flags.F_FRAGILE_METRIC, 3, "%NK (of lymph) @ D7", 5.0))
    for item in _build(r, book)["items"]:
        for token in item["must_mention"]:
            assert "%" not in token and " " not in token, (
                f"{item['id']} would gate on a substring good prose can omit: {token!r}")


def test_a_flag_group_demands_the_sample_its_worst_case_sits_at():
    """Reading was enforced and reporting was not, so reading is what happened.

    The trajectory that discharged all 23 obligations by printing columns wrote the thinnest
    caveats of five. Naming the worst case's SAMPLE is the minimum that distinguishes having
    addressed an item from having opened it.
    """
    book = _book((flags.F_NEG_MODE_DRIFT, 2, "cd45 @ D7", 4.0))
    item = next(i for i in _build(book=book)["items"] if i["code"] == flags.F_NEG_MODE_DRIFT)
    assert item["must_mention"] == ["D7"]


def test_a_tier_one_subjects_reference_suffix_does_not_hide_the_sample():
    """``cd4 @ Baseline (reference)`` names the Baseline sample and must resolve to it."""
    book = _book((flags.F_LOW_PLACEMENT_MARGIN, 1, "cd4 @ Baseline (reference)", 2.0))
    item = next(i for i in _build(book=book)["items"]
                if i["code"] == flags.F_LOW_PLACEMENT_MARGIN)
    assert item["must_mention"] == ["Baseline"]


def test_a_scope_that_is_not_a_sample_demands_nothing():
    """A control tube or a marker is not a sample label; there is no token to hold anyone to."""
    book = _book((flags.F_CONTROL_MISSING, 4, "control tube @ control:ntnk", 1.0),
                 (flags.F_NEG_WIDTH_CHANGE, 2, "cd14 @ SomeUnknownTimepoint", 1.0))
    for item in _build(book=book)["items"]:
        assert item["must_mention"] == [], item["id"]


# ── the findings that invalidate everything else ──────────────────────────────
def test_a_failed_reproduction_check_is_promoted_ahead_of_everything():
    """If the audit recomputed a different gating, every other item may not apply."""
    r = _report(
        reproduction={"available": True, "faithful": False, "checked": 140,
                      "max_abs_delta_pp": 4.2, "source": "multilineage.csv",
                      "mismatches": [{"metric": "%NK (of lymph)"}]},
        tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
            _disc("D14", "%NK (of lymph)", 99.0)]},
    )
    first = _build(r, _book((flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D7", 9.9)))["items"][0]
    assert first["id"] == "REPRODUCTION_MISMATCH"
    assert first["priority"] == 1
    assert "1 of 140" in first["statement"]


def test_an_unverifiable_reproduction_check_is_reported_as_unverified_not_omitted():
    r = _report(reproduction={"available": False})
    item = _build(r)["items"][0]
    assert item["id"] == "REPRODUCTION_UNVERIFIED"
    assert "Unverified is not verified" in item["why"]


def test_unmeasured_operator_concordance_is_itself_an_obligation():
    """Its absence must not be read as agreement."""
    r = _report(tier5_operator_concordance={"available": False, "reason": "no manual CSV"})
    item = next(i for i in _build(r)["items"] if i["kind"] == "operator_concordance")
    assert "no manual CSV" in item["statement"]
    assert "unmeasured" in item["why"]


def test_a_sensitivity_coverage_gap_is_named_rather_than_left_silent():
    r = _report(tier3_sensitivity={"envelope": [], "coverage": {
        "dropped": [{"timepoint": "D7", "marker": "cd56"}] * 14,
        "reason": "wall-clock budget exhausted"}})
    item = next(i for i in _build(r)["items"] if i["kind"] == "coverage_gap")
    assert "14 marker x timepoint" in item["statement"]
    assert "wall-clock budget exhausted" in item["statement"]
    assert "cd56" in item["statement"], "the skipped marker must be named"


def test_a_coverage_gap_on_one_marker_demands_that_marker_be_named():
    """Five trajectories discharged the countless version without reporting the gap."""
    r = _report(tier3_sensitivity={"envelope": [], "coverage": {
        "dropped": [{"timepoint": f"T{i}", "marker": "car",
                     "reason": "no offset scale available"} for i in range(12)],
        "reason": None}})
    item = next(i for i in _build(r)["items"] if i["kind"] == "coverage_gap")
    assert item["must_mention"] == ["car"]
    assert "all 12 are marker 'car'" in item["statement"]


def test_a_coverage_gap_spread_over_markers_demands_no_single_token():
    """No one word would be fair to require, so none is."""
    r = _report(tier3_sensitivity={"envelope": [], "coverage": {
        "dropped": ([{"timepoint": "D7", "marker": "cd56"}] * 3
                    + [{"timepoint": "D7", "marker": "ld"}] * 2),
        "reason": "budget"}})
    item = next(i for i in _build(r)["items"] if i["kind"] == "coverage_gap")
    assert item["must_mention"] == []


def test_a_coverage_gap_is_not_dischargeable_by_opening_the_table_alone(tmp_path: Path):
    """``cutoff_range_pp`` looks complete until you read which cutoff drove it.

    Needs a real CSV on disk: ``read_columns`` is resolved against the written header, and an
    item whose columns cannot be resolved degrades to the table-name check by design.
    """
    csv = tmp_path / "diagnostics_uncertainty.csv"
    csv.write_text("timepoint,metric,cutoff_range_pp,cutoff_range_driver_low,"
                   "cutoff_range_driver_high\nD7,%CAR+ (of Donor NK),2.1,cd14:plus,cd14:minus\n")
    r = _report(tier3_sensitivity={"envelope": [], "coverage": {
        "dropped": [{"timepoint": "D7", "marker": "car"}], "reason": "no offset scale"}})
    m = obligations.build(r, flags.FlagBook(), written={"uncertainty": str(csv)})
    item = next(i for i in m["items"] if i["kind"] == "coverage_gap")
    cols = item["read_columns"]["diagnostics_uncertainty"]
    assert "cutoff_range_driver_low" in cols and "cutoff_range_driver_high" in cols
    # And the runnable line projects them, so following it discharges the item it came from.
    assert any("cutoff_range_driver_low" in line for line in item["how"])


# ── cutoffs tier 1 cannot judge ───────────────────────────────────────────────
def test_an_asserted_cutoff_is_escalated_per_marker_and_demands_its_name():
    """`%Donor NK` rested on a cutoff with no valley evidence and no flag said so."""
    book = _book((flags.F_ASSERTED_CUTOFF, 1, "hla @ Baseline (reference)", 1.0))
    item = next(i for i in _build(book=book)["items"] if i["id"].startswith("ASSERTED_CUTOFF"))
    assert item["id"] == "ASSERTED_CUTOFF__HLA"
    assert item["must_mention"] == ["hla"], "'donor' appears in every write-up already"
    assert item["forced"] is True
    assert item["must_read"] == ["diagnostics_cutoff_audit"]


def test_both_codes_on_one_cutoff_spend_one_slot_not_two():
    """Both make the same request of the reader: judge it from tiers 4 and 5 instead."""
    book = _book((flags.F_ASSERTED_CUTOFF, 1, "car @ D7", 1.0),
                 (flags.F_REFERENCE_AUDIT_UNAVAILABLE, 1, "car @ Baseline (reference)", 1.0))
    items = [i for i in _build(book=book)["items"] if i["id"].startswith("ASSERTED_CUTOFF")]
    assert [i["id"] for i in items] == ["ASSERTED_CUTOFF__CAR"]
    assert items[0]["n_findings"] == 2
    for code in (flags.F_ASSERTED_CUTOFF, flags.F_REFERENCE_AUDIT_UNAVAILABLE):
        assert code in items[0]["code"]
        assert any(code in line for line in items[0]["how"]), f"no read line for {code}"


def test_an_asserted_cutoff_item_survives_a_full_class_cap():
    """It bypasses the caps for the same reason a root gate does — see the module docstring."""
    filler = [(flags.F_LOW_PLACEMENT_MARGIN, 1, f"m{i} @ Baseline", float(20 - i))
              for i in range(9)]
    book = _book(*filler, (flags.F_REFERENCE_AUDIT_UNAVAILABLE, 1, "car @ Baseline", 1.0))
    m = _build(book=book, limit=3)
    assert any(i["id"] == "ASSERTED_CUTOFF__CAR" for i in m["items"])
    assert m["n_forced"] >= 1


def test_an_asserted_cutoff_statement_names_the_headline_metrics_that_depend_on_it():
    book = _book((flags.F_REFERENCE_AUDIT_UNAVAILABLE, 1, "car @ Baseline (reference)", 1.0))
    item = next(i for i in _build(book=book)["items"] if i["id"] == "ASSERTED_CUTOFF__CAR")
    assert "%CAR+ (of Donor NK)" in item["statement"]


def test_an_asserted_cutoff_offers_the_controls_without_requiring_them():
    """Tier 4 may itself be unavailable; requiring an empty table buys a ``.head()``."""
    book = _book((flags.F_ASSERTED_CUTOFF, 1, "hla @ Baseline (reference)", 1.0))
    item = next(i for i in _build(book=book)["items"] if i["id"] == "ASSERTED_CUTOFF__HLA")
    assert "diagnostics_controls" in item["should_read"]
    assert "diagnostics_controls" not in item["must_read"]


def test_an_item_with_nothing_to_check_is_marked_advisory_not_silently_discharged():
    """Otherwise a reader who ignored it would be recorded as having addressed it.

    The real case: an obligation whose only detail lives in the nested JSON, so there is no
    table to open and no token the write-up must contain.
    """
    r = _report(tier3_sensitivity={"envelope": [], "coverage": {
        "dropped": [{"timepoint": "D7"}], "reason": "budget"}})
    # No uncertainty table written, so the coverage item has nothing checkable left.
    m = obligations.build(r, flags.FlagBook(), written={"flags": "/w/diagnostics_flags.csv"})
    item = m["items"][0]
    assert item["id"] == "SENSITIVITY_COVERAGE_GAP"
    assert item["advisory"] is True
    assert m["n_advisory"] == 1
    # It still reaches the reader, flagged as a caveat rather than dropped.
    text = obligations.render_worklist(m)
    assert "SENSITIVITY_COVERAGE_GAP" in text
    assert "advisory" in text


def test_the_same_item_is_enforceable_when_its_table_exists():
    r = _report(tier3_sensitivity={"envelope": [], "coverage": {
        "dropped": [{"timepoint": "D7"}], "reason": "budget"}})
    item = _build(r)["items"][0]
    assert item["advisory"] is False
    assert item["must_read"] == ["diagnostics_uncertainty"]


def test_every_item_that_is_not_advisory_can_actually_be_checked():
    """The invariant the ``advisory`` flag exists to keep: no unenforceable item counts."""
    r = _report(
        reproduction={"available": False},
        tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
            _disc("D14", "%NK (of lymph)", 13.7)]},
        tier3_sensitivity={"envelope": [], "coverage": {"dropped": [{"t": 1}], "reason": "b"}},
        tier5_operator_concordance={"available": False, "reason": "no manual CSV"},
    )
    book = _book(*[(f"CODE_{i}", i % 6, f"item{i} @ T{i}", float(i + 1)) for i in range(12)])
    for item in _build(r, book)["items"]:
        checkable = bool(item["must_read"] or item["must_mention"])
        assert item["advisory"] is not checkable, item["id"]


def test_the_manifest_states_that_an_obligation_is_a_duty_to_read_not_a_verdict():
    """The discipline that keeps this from becoming the analysis has to travel with it."""
    m = _build()
    assert "duty to READ" in m["discipline"]
    assert "not a verdict" in m["discipline"]
    assert obligations.SCHEMA == m["schema"]


# ── wiring into the emitters and the harness ──────────────────────────────────
def _full_report() -> dict:
    r = _report(tier0_data_adequacy={
        "per_timepoint": [{"timepoint": "D14", "n_events_analyzed": 100,
                           "n_lymphocytes": 10, "lineage_purity_pct": 50.0}],
        "discontinuity": [_disc("D14", "%NK (of lymph)", 13.7)]})
    r.update({"notes": [], "skipped": [], "thresholds": [], "tier1_cutoff_foundation": {},
              "tier2_transfer_validity": {}, "tier4_internal_controls": {},
              "tier_failures": [], "limitations": [], "files": {}})
    return r


def test_write_all_emits_both_the_manifest_and_the_rendered_worklist(tmp_path: Path):
    report = _full_report()
    written = emit.write_all(tmp_path, report, record.MetricRecorder(),
                             _book((flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D7", 4.0)))
    assert "obligations" in written and "worklist" in written
    manifest = json.loads(Path(written["obligations"]).read_text())
    assert manifest["schema"] == obligations.SCHEMA
    assert manifest["items"], "a fired flag and an off-trend sample should yield obligations"
    assert "WORKLIST" in Path(written["worklist"]).read_text()
    # And the manifest travels in the nested report, so the JSON is self-contained.
    assert report["obligations"]["n_promoted"] == manifest["n_promoted"]


def test_the_digest_announces_the_worklist_next_to_the_counts_it_makes_actionable():
    report = _full_report()
    book = _book((flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D7", 4.0))
    manifest = obligations.build(report, book, written=WRITTEN)
    text = emit.render_digest(report, book, obligations=manifest)
    assert "WORKLIST:" in text
    assert obligations.OBLIGATIONS_JSON in text
    assert text.index("FLAGGED ITEMS") < text.index("WORKLIST:")


def test_the_digest_is_unchanged_when_no_worklist_was_built():
    """The manifest is optional; a project without one must render exactly as before."""
    assert "WORKLIST:" not in emit.render_digest(_full_report(), flags.FlagBook())


def test_the_diagnostics_cell_prints_the_worklist_last(tmp_path: Path):
    """Last in the observation is the most-read position; the schema dump is not the point."""
    import ast

    from flow.firstrun import (WORKLIST_FILENAME, WORKLIST_PRINT_CHARS, _diagnostics_cell)

    cell = _diagnostics_cell()
    ast.parse(cell)  # the cell is generated by string substitution; it must still parse
    assert WORKLIST_FILENAME in cell
    assert str(WORKLIST_PRINT_CHARS) in cell
    assert cell.index(WORKLIST_FILENAME) > cell.index("diagnostics table(s) loaded")


def test_the_worklist_and_the_digest_both_fit_in_the_seed_observation_window():
    """Both are printed by one cell. If they cannot co-exist, the truncator eats a middle."""
    from flow.env.notebook_env import SEED_OBS_CHARS
    from flow.firstrun import DIGEST_PRINT_CHARS, WORKLIST_PRINT_CHARS

    assert emit.DIGEST_MAX_CHARS <= DIGEST_PRINT_CHARS
    # The print cap must not cut inside the render's own bound, or the worklist would be
    # truncated twice: once by a rule that keeps every item, then again by a raw slice.
    assert obligations.WORKLIST_MAX_CHARS <= WORKLIST_PRINT_CHARS
    # Room for both plus the table inventory and the run log the same cell prints.
    assert DIGEST_PRINT_CHARS + WORKLIST_PRINT_CHARS < SEED_OBS_CHARS


def test_the_note_tells_the_agent_the_worklist_is_a_floor_and_how_to_print_safely():
    from flow.firstrun import DIAGNOSTICS_NOTE

    assert "WORKLIST" in DIAGNOSTICS_NOTE
    assert "minimum, not the maximum" in DIAGNOSTICS_NOTE
    # Silence on an item must be named as the failure, since 'no change' is a valid outcome.
    assert "does not change the" in DIAGNOSTICS_NOTE
    assert "NOT PROMOTED" in DIAGNOSTICS_NOTE
    # And it must say how to print a long frame without the truncator deleting the middle.
    assert "groupby" in DIAGNOSTICS_NOTE
    assert "head and tail" in DIAGNOSTICS_NOTE


# ── root gates: the one stated exception to the cap ────────────────────────────
# A cap that can evict the finding "the cutoff every headline number is measured inside was
# not derived from the data" is not bounding the reader's work, it is choosing their
# conclusion. A production run dropped exactly that: PERCENTILE_FALLBACK_SUSPECTED and
# NO_VALLEY_EVIDENCE on the viability cutoff lost their slots to the fifth off-trend
# timepoint, went unread, and the write-up called the cutoffs "generally reasonable".

def _flood_off_trend(n: int = 14) -> dict:
    """A report with enough off-trend samples to exhaust every generous cap."""
    return _report(tier0_data_adequacy={
        "per_timepoint": [],
        "discontinuity": [_disc(f"D{i}", "%NK (of lymph)", 9.0 - i * 0.1) for i in range(n)],
    })


def test_root_gate_keys_come_from_the_pipelines_own_downstream_declaration():
    """Derived, not listed: a marker is a root gate because every headline metric moves with
    it, which ``markers.py`` already records. Listing keys here would be a second source of
    truth that a change to the panel could silently contradict."""
    from flow.firstrun.anchored.diagnostics.markers import MARKERS, _ALL_LINEAGE

    for m in MARKERS:
        expected = m.valley_derived and set(_ALL_LINEAGE) <= set(m.downstream)
        assert (m.key in obligations.ROOT_GATE_KEYS) is expected, m.key
    # The cleanup gates every lineage number is measured inside — and nothing narrower.
    assert obligations.ROOT_GATE_KEYS == {"ld", "cd45", "cd14", "cd19"}
    for narrower in ("cd3", "cd56", "cd4"):
        assert narrower not in obligations.ROOT_GATE_KEYS
    # HLA and CAR are not valley-derived by design, so a valley finding there is
    # informational and escalating it would promote noise.
    for not_valley in ("hla", "car"):
        assert not_valley not in obligations.ROOT_GATE_KEYS


def test_a_root_gate_foundation_finding_survives_a_class_cap_that_is_already_full():
    """The regression, exactly: 14 off-trend samples plus a full foundation class, and the
    viability finding must still be on the list."""
    book = _book(
        *[(flags.F_CUTOFF_INSIDE_POPULATION, 1, f"m{i} @ Baseline", 9.0) for i in range(9)],
        (flags.F_PERCENTILE_FALLBACK, 1, "ld @ Baseline (reference)", 1.0),
    )
    m = _build(report=_flood_off_trend(), book=book)
    ids = [it["id"] for it in m["items"]]
    assert "ROOT_GATE__PERCENTILE_FALLBACK_SUSPECTED__LD" in ids
    item = next(it for it in m["items"] if it["id"].endswith("__LD"))
    assert item["forced"] is True
    assert m["n_forced"] == 1


def test_the_global_cap_never_evicts_a_root_gate_item():
    """The class-cap bypass would be undone if the global cap could still cut the item."""
    book = _book(
        *[(flags.F_LOW_PARENT, 0, f"x{i} @ D{i}", 5.0) for i in range(4)],
        (flags.F_NO_VALLEY_EVIDENCE, 1, "ld @ Baseline (reference)", None),
    )
    m = _build(report=_flood_off_trend(), book=book, limit=1)
    assert any(it["id"].endswith("NO_VALLEY_EVIDENCE__LD") for it in m["items"])


def test_a_root_gate_statement_says_what_depends_on_the_cutoff():
    """'NO_VALLEY_EVIDENCE on ld' is only actionable if the reader knows ld is upstream of
    every number they are about to report."""
    book = _book((flags.F_NO_VALLEY_EVIDENCE, 1, "ld @ Baseline (reference)", None))
    item = _build(book=book)["items"][0]
    assert "ROOT GATE" in item["statement"]
    assert "%NK (of lymph)" in item["statement"] or "%Live (of singlets)" in item["statement"]
    # And it still may not conclude: ``why`` is the flag's own hint, verbatim.
    assert item["why"] == f"hint for {flags.F_NO_VALLEY_EVIDENCE}"


def test_a_root_gate_item_demands_a_marker_name_a_write_up_would_actually_use():
    """``must_mention`` is matched on word boundaries, so demanding "ld" would be both
    unsatisfiable in prose and dangerous if it ever matched."""
    ld = _build(book=_book((flags.F_NO_VALLEY_EVIDENCE, 1, "ld @ Baseline", None)))["items"][0]
    assert ld["must_mention"] == ["viability"]
    cd45 = _build(book=_book((flags.F_PERCENTILE_FALLBACK, 1, "cd45 @ Baseline", 1.0)))["items"][0]
    assert cd45["must_mention"] == ["cd45"]
    assert obligations._mention_token("ld") == "viability"


def test_one_finding_does_not_spend_two_worklist_slots():
    """A group whose every finding is escalated per-marker is skipped, as
    TEMPORAL_DISCONTINUITY is — otherwise the bypass costs the list twice over."""
    book = _book(
        (flags.F_PERCENTILE_FALLBACK, 1, "ld @ Baseline", 1.0),
        (flags.F_PERCENTILE_FALLBACK, 1, "cd14 @ Baseline", 1.0),
    )
    ids = [it["id"] for it in _build(book=book)["items"]]
    assert sorted(ids) == ["ROOT_GATE__PERCENTILE_FALLBACK_SUSPECTED__CD14",
                           "ROOT_GATE__PERCENTILE_FALLBACK_SUSPECTED__LD"]
    assert "FLAGS__PERCENTILE_FALLBACK_SUSPECTED" not in ids


def test_a_group_only_partly_escalated_keeps_its_group_obligation():
    """The un-escalated findings are on other markers and are still unaddressed; dropping the
    group would lose them silently, which is the one thing this module refuses to do."""
    book = _book(
        (flags.F_NO_VALLEY_EVIDENCE, 1, "ld @ Baseline", None),
        (flags.F_NO_VALLEY_EVIDENCE, 1, "cd56 @ Baseline", None),
    )
    ids = [it["id"] for it in _build(book=book)["items"]]
    assert "ROOT_GATE__NO_VALLEY_EVIDENCE__LD" in ids
    assert "FLAGS__NO_VALLEY_EVIDENCE" in ids


def test_forced_items_are_counted_separately_from_what_merely_fitted():
    """``n_forced`` lets a reader tell a list shaped by priority from one shaped by the cap."""
    m = _build(book=_book((flags.F_PERCENTILE_FALLBACK, 1, "ld @ Baseline", 1.0)))
    assert m["n_forced"] == 1
    assert _build()["n_forced"] == 0


# ── the cap, raised ───────────────────────────────────────────────────────────
def test_the_transfer_class_can_promote_every_tier_two_code_that_fires():
    """The old cap of 2 dropped PLACEMENT_MARGIN_LOST_ON_TRANSFER (56 findings) and
    OVERTON_DISAGREES_WITH_CUTOFF (38) from a study where both fired."""
    codes = [flags.F_NEG_DISTRIBUTION_SHIFT, flags.F_NEG_MODE_DRIFT, flags.F_NEG_WIDTH_CHANGE,
             flags.F_PLACEMENT_MARGIN_LOST, flags.F_BOUNDARY_MASS]
    book = _book(*[(c, 2, f"cd45 @ D{i}", 4.0) for i, c in enumerate(codes)])
    ids = [it["id"] for it in _build(book=book)["items"]]
    for c in codes:
        assert f"FLAGS__{c}" in ids, c


def test_the_cap_is_still_finishable_in_a_trajectorys_step_budget():
    """The raise is bounded by the reader's budget, not abandoned. Items share ``read:``
    lines, so the bound that matters is distinct tables, not item count.

    24 -> 26 buys the two slots the ``gate_geometry`` split needs, and adds one distinct
    table. Measured headroom on the run that motivated the split: 21 obligations discharged
    in 10 of 30 steps."""
    assert obligations.MAX_OBLIGATIONS <= 26
    assert sum(obligations.CLASS_CAPS.values()) == obligations.MAX_OBLIGATIONS


def test_gate_geometry_is_not_starved_by_the_drift_codes_it_shares_a_tier_with():
    """The split's reason for existing. Tier 2 holds seven codes for a transfer cap of five,
    and the two geometry codes fire least often — so before the split they ranked sixth and
    seventh and could never be promoted while any drift code fired, taking the only numeric
    account of the 2-D gates off the worklist entirely."""
    drift = [flags.F_NEG_DISTRIBUTION_SHIFT, flags.F_NEG_MODE_DRIFT, flags.F_NEG_WIDTH_CHANGE,
             flags.F_PLACEMENT_MARGIN_LOST, flags.F_OVERTON_DISAGREES]
    geometry = [flags.F_BOUNDARY_MASS, flags.F_QUADRANT_MARGIN]
    # Drift codes fire far more often, which is what won them the slots.
    book = _book(
        *[(c, 2, f"cd45 @ D{i}", 9.0) for i, c in enumerate(drift)],
        *[(c, 2, f"NK cloud @ D{i}", 1.3) for i, c in enumerate(geometry)],
    )
    m = _build(book=book)
    ids = [it["id"] for it in m["items"]]
    for c in drift + geometry:
        assert f"FLAGS__{c}" in ids, f"{c} was starved off the worklist"
    # And geometry is its own class, so the cap that binds it is its own.
    kinds = {it["id"]: it["kind"] for it in m["items"]}
    for c in geometry:
        assert kinds[f"FLAGS__{c}"] == "gate_geometry"
    for c in drift:
        assert kinds[f"FLAGS__{c}"] == "transfer_validity"


def test_a_geometry_obligation_points_at_the_gate_geometry_table():
    """The class exists to get that table opened; the item has to actually name it."""
    m = _build(book=_book((flags.F_QUADRANT_MARGIN, 2, "NK CD3 boundary @ D7", 1.31)))
    item = next(it for it in m["items"] if it["id"] == f"FLAGS__{flags.F_QUADRANT_MARGIN}")
    named = set(item["must_read"]) | set(item["should_read"])
    assert "diagnostics_gate_geometry" in named
    assert any("diagnostics_gate_geometry" in h for h in item["how"])


# ── pointing at columns, not just tables ──────────────────────────────────────
# ``read: diagnostics_cutoff_audit`` was satisfied by a cell printing three of that table's
# thirty-seven columns — with the two carrying the finding among the thirty-four it projected
# away — and the item was recorded as discharged because the table had been opened.

AUDIT_COLUMNS = ("marker,label,cutoff,valley_supported,suspected_percentile_fallback,"
                 "cutoff_percentile_rank_in_pool,density_at_cutoff_ratio,trough_depth_ratio")
TRANSFER_COLUMNS = ("marker,timepoint,negative_mode_drift,negative_mode_drift_in_negative_sd,"
                    "negative_mode_drift_in_se,placement_in_negative_sd")
FLAGS_COLUMNS = "code,tier,subject,rule,exceedance,measured,resolution_hint"


def _written_with_headers(tmp_path: Path, **extra: str) -> dict:
    """A ``written`` map whose files exist, so headers can be read as in a real run.

    ``extra`` adds or overrides a table, as ``label="col,col,col"``.
    """
    out = {}
    tables = {"flags": FLAGS_COLUMNS, "cutoff_audit": AUDIT_COLUMNS,
              "transfer": TRANSFER_COLUMNS}
    tables.update(extra)
    for label, header in tables.items():
        p = tmp_path / f"diagnostics_{label}.csv"
        p.write_text(header + "\n")
        out[label] = str(p)
    return out


def _measured_book(code: str, tier: int, subject: str, measured: dict) -> flags.FlagBook:
    b = flags.FlagBook()
    b.add(code, tier, subject, f"rule for {code}", measured, f"hint for {code}", exceedance=2.0)
    return b


def test_a_read_line_names_the_columns_the_rule_was_evaluated_on(tmp_path: Path):
    book = _measured_book(flags.F_PERCENTILE_FALLBACK, 1, "ld @ Baseline",
                          {"suspected_percentile": 90.0, "valley_supported": False})
    m = obligations.build(_report(), book, written=_written_with_headers(tmp_path))
    how = " ".join(m["items"][0]["how"])
    assert "suspected_percentile_fallback" in how
    assert "valley_supported" in how


def test_a_measured_key_resolves_to_the_shortest_column_containing_it(tmp_path: Path):
    """``drift`` must resolve to ``negative_mode_drift``, not to one of the longer columns
    that also contain the substring."""
    cols = TRANSFER_COLUMNS.split(",")
    assert obligations._resolve_columns(["drift"], cols) == ["negative_mode_drift"]
    # And a flag's name for a quantity resolves to the emitter's name for the same column.
    assert obligations._resolve_columns(["drift_in_negative_sd"], cols) == [
        "negative_mode_drift_in_negative_sd"]


def test_a_key_that_matches_no_column_is_dropped_rather_than_guessed():
    """A projection is only worth having if it runs. The flags table still carries the full
    ``measured`` payload, so the numbers stay reachable."""
    cols = AUDIT_COLUMNS.split(",")
    assert obligations._resolve_columns(["no_such_measurement"], cols) == []
    assert obligations._projection("t", cols, ["no_such_measurement"]) is None


def test_a_projection_selects_columns_rather_than_indexing_a_tuple(tmp_path: Path):
    """``frame['a', 'b']`` raises; only ``frame[['a', 'b']]`` selects. Every emitted line is
    meant to be pasted into a cell, so a line that cannot run is worse than a bare table."""
    book = _measured_book(flags.F_NEG_MODE_DRIFT, 2, "cd45 @ D32", {"drift": 0.4})
    m = obligations.build(_report(), book, written=_written_with_headers(tmp_path))
    projections = [h for h in m["items"][0]["how"] if "[[" in h or "['" in h]
    assert projections, "no projection was emitted"
    for h in projections:
        expr = h.split("  #")[0].strip()
        # A column selection always closes a doubled bracket; a tuple index never does.
        assert "[['" in expr and "']]" in expr, expr


def test_without_a_readable_header_the_line_falls_back_to_the_bare_table():
    """Guessing column names off the flag's own keys is how an unrunnable line gets emitted."""
    book = _measured_book(flags.F_NEG_MODE_DRIFT, 2, "cd45 @ D32", {"drift": 0.4})
    m = obligations.build(_report(), book, written=WRITTEN)  # paths that do not exist
    how = " ".join(m["items"][0]["how"])
    assert "diagnostics_transfer" in how
    assert "[['" not in how


def test_the_flags_read_line_always_reaches_the_numbers(tmp_path: Path):
    """``measured`` is the one column that holds the quantities whatever the code, so it is
    what makes an unresolvable projection survivable."""
    book = _measured_book("SOME_FUTURE_RULE", 2, "cd99 @ D7", {"unmappable_key": 1.0})
    m = obligations.build(_report(), book, written=_written_with_headers(tmp_path))
    flag_lines = [h for h in m["items"][0]["how"] if "diagnostics_flags" in h]
    assert flag_lines
    assert "measured" in flag_lines[0]


# ── the render gives up detail before it gives up findings ────────────────────
# Raising the cap to 24 achieved nothing on its own: the render kept 14 items at full detail
# and pointed the other ten at diagnostics_obligations.json, which is the file nobody opens.

def _many_items(n: int = 24, *, prose: int = 240) -> dict:
    """A manifest with more obligations than fit at full detail.

    Built directly rather than through ``build``, because the class caps bind long before the
    render does — a fixture that goes through ``build`` cannot reach the overflow path this
    render logic exists for. ``prose`` approximates a real ``statement``/``why``: the ones in
    production are a rule text and a resolution hint, not a test label.
    """
    return {
        "n_not_promoted": 2,
        "not_promoted_note": "2 candidate(s) were NOT promoted.",
        "items": [{
            "priority": i + 1,
            "id": f"FLAGS__SOME_RULE_{i:02d}",
            "statement": f"[tier 2] SOME_RULE_{i:02d}: 40 finding(s); " + "m" * prose,
            "why": f"hint {i:02d}: " + "h" * prose,
            "must_read": ["diagnostics_flags"],
            "should_read": ["diagnostics_transfer"],
            "how": [f"diagnostics_flags[diagnostics_flags.code == 'SOME_RULE_{i:02d}']"],
            "must_mention": [],
            "advisory": False,
        } for i in range(n)],
    }


def test_every_promoted_obligation_reaches_the_render_even_when_detail_must_go():
    m = _many_items()
    full = obligations.render_worklist(m, max_chars=10**9)
    assert len(full) > obligations.WORKLIST_MAX_CHARS, "fixture no longer overflows"
    text = obligations.render_worklist(m)
    for it in m["items"]:
        assert it["id"] in text, f"{it['id']} was promoted but never rendered"
    assert "not shown here" not in text
    assert len(text) <= obligations.WORKLIST_MAX_CHARS


def test_a_render_that_dropped_the_hints_says_so():
    """An item shown without its hint must not read as an item that has none."""
    text = obligations.render_worklist(_many_items())
    assert "why:" not in text
    assert "omitted" in text and obligations.OBLIGATIONS_JSON in text


def test_a_short_worklist_still_gets_its_hints():
    """Compacting is a response to overflow, not the default."""
    text = obligations.render_worklist(_many_items(n=2))
    assert "why: hint 00" in text
    assert "omitted" not in text


def test_items_are_only_counted_off_the_end_once_compacting_is_not_enough():
    """The last resort still keeps the highest-priority items and still counts the rest."""
    text = obligations.render_worklist(_many_items(), max_chars=2500)
    assert "not shown here" in text
    assert "FLAGS__SOME_RULE_00" in text


# ── the measurement table is required when its columns can be named ───────────
# The narrow rule — ``must_read`` is the flags index alone — was measured in production and
# cost the thing it was protecting: 15 of 23 items collapsed onto ``diagnostics_flags``, one
# ``groupby('code').size()`` discharged all 15, and ``diagnostics_transfer`` and
# ``diagnostics_gate_geometry`` went unopened across five trajectories. These tests pin the
# replacement rule and, just as importantly, its limit.
GEOMETRY_COLUMNS = "timepoint,nk_cd3_boundary_margin_in_sd,nk_nk_mode_cd3,nk_n_nk"


def test_the_measurement_table_is_required_once_its_columns_resolve(tmp_path: Path):
    book = _measured_book(flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D28",
                          {"drift_in_negative_sd": 0.38})
    it = obligations.build(_report(), book,
                           written=_written_with_headers(tmp_path))["items"][0]
    assert it["must_read"] == ["diagnostics_flags", "diagnostics_transfer"]
    assert it["should_read"] == [], "a required table must not also be listed as optional"
    assert it["read_columns"]["diagnostics_transfer"] == [
        "negative_mode_drift_in_negative_sd"]
    # The flags entry names the column carrying the numbers for any code, whatever it is.
    assert "measured" in it["read_columns"]["diagnostics_flags"]


def test_read_columns_never_includes_an_identity_column(tmp_path: Path):
    """A reader who prints ``[['marker', 'timepoint']]`` has named the rows, not read them."""
    book = _measured_book(flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D28", {"drift": 0.38})
    it = obligations.build(_report(), book,
                           written=_written_with_headers(tmp_path))["items"][0]
    cols = it["read_columns"]["diagnostics_transfer"]
    assert cols, "the projection resolved, so there is something to require"
    for ident in obligations.IDENTITY_COLUMNS:
        assert ident not in cols


def test_the_geometry_table_becomes_required_not_merely_mentioned(tmp_path: Path):
    """The scatter panels' numeric equivalent. It was advisory-only, and no trajectory in
    run a17be587a070 opened it across five attempts."""
    book = _measured_book(flags.F_QUADRANT_MARGIN, 2, "NK cd56 boundary @ D7",
                          {"margin_in_sd": 1.14, "nk_mode_cd3": 501.0})
    m = obligations.build(_report(), book,
                          written=_written_with_headers(tmp_path,
                                                        gate_geometry=GEOMETRY_COLUMNS))
    it = m["items"][0]
    assert "diagnostics_gate_geometry" in it["must_read"]
    assert it["read_columns"]["diagnostics_gate_geometry"]


# ── the per-sample adequacy table ─────────────────────────────────────────────
# ``timepoints`` was advisory on every item that named it, and four of the five trajectories
# in run 7e81a8e6038c never opened it — while the profile's own QC expectation asks the reader
# to flag samples with few CD45+ events, low viability or unstable acquisition, which is the
# only place those numbers live. These tests pin the promotion and the reason it needs a
# literal column list rather than the usual ``measured``-key resolution.
TIMEPOINTS_COLUMNS = ("timepoint,filename,n_total_events,n_events_analyzed,n_lymphocytes,"
                      "lineage_purity_pct,retention_cd45p_pct,retention_live_pct,n_cd45p,"
                      "n_live,retention_lymphocytes_pct,acquisition_signal_drift_sd,"
                      "acquisition_rate_cv")
DISCONTINUITY_COLUMNS = ("timepoint,metric,value_pct,prev_value_pct,next_value_pct,"
                         "excursion_in_tolerances,excursion_beyond_tolerance_pp")


def _written_with_sample_tables(tmp_path: Path) -> dict:
    return _written_with_headers(tmp_path, timepoints=TIMEPOINTS_COLUMNS,
                                 discontinuity=DISCONTINUITY_COLUMNS)


def test_an_off_trend_sample_requires_the_per_sample_adequacy_table(tmp_path: Path):
    """"Spike or artefact" is not answerable from the discontinuity table alone: the numbers
    that settle it — events analyzed, retention, purity, acquisition drift — are per SAMPLE."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D3", "%T (of lymph)", 5.087)]})
    off = obligations.build(r, flags.FlagBook(),
                            written=_written_with_sample_tables(tmp_path))["items"][0]
    assert off["must_read"] == ["diagnostics_discontinuity", "diagnostics_timepoints"]
    assert off["should_read"] == [], "a required table must not also be listed as optional"
    assert "retention_cd45p_pct" in off["read_columns"]["diagnostics_timepoints"]


def test_every_off_trend_read_line_can_actually_discharge_its_item(tmp_path: Path):
    """The submission check reads the agent's SOURCE, so a bare row filter names no
    measurement and leaves the item pending. A ``read:`` line that cannot discharge the
    obligation it belongs to is worse than no line: it sends an honest reader in a circle,
    and the reader who escapes does so by projecting columns nothing told them to project.

    Asserted for EVERY required table rather than the one that motivated it: both of this
    item's ``read:`` lines were bare row filters, and only one was noticed."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D3", "%T (of lymph)", 5.087)]})
    off = obligations.build(r, flags.FlagBook(),
                            written=_written_with_sample_tables(tmp_path))["items"][0]
    assert len(off["must_read"]) > 1, "the test proves little if only one table is required"
    for table in off["must_read"]:
        line = next((h for h in off["how"] if table in h), None)
        assert line, f"{table} is required but no how: line points at it"
        required = off["read_columns"].get(table)
        assert required, f"{table} is required but the item names no column in it"
        assert any(c in line for c in required), (
            f"following {line!r} literally would not satisfy the columns it demands")


def test_timepoints_is_required_even_though_its_flags_report_keys_it_lacks(tmp_path: Path):
    """``LOW_PARENT_EVENTS`` measures ``n_parent`` on a GATE; every column in this table
    measures the SAMPLE. So nothing resolves from ``measured`` and the usual rule would leave
    the table advisory for a reason that says nothing about whether it is worth reading."""
    book = _measured_book(flags.F_LOW_PARENT, 0, "%CAR+ (of Donor NK) @ Baseline",
                          {"metric": "%CAR+ (of Donor NK)", "n_parent": 2,
                           "parent_mask": "Donor"})
    it = obligations.build(_report(), book,
                           written=_written_with_sample_tables(tmp_path))["items"][0]
    assert it["id"] == f"FLAGS__{flags.F_LOW_PARENT}"
    assert "diagnostics_timepoints" in it["must_read"]
    assert it["should_read"] == []
    assert it["read_columns"]["diagnostics_timepoints"]
    # ``filename`` names the row, it does not measure the sample.
    assert "filename" not in it["read_columns"]["diagnostics_timepoints"]


def test_the_adequacy_table_stays_advisory_when_its_header_is_unreadable():
    """The fallback is a literal column list, not a licence to demand an unwritten table.
    WRITTEN points at paths that do not exist, so nothing resolves — the honest case."""
    r = _report(tier0_data_adequacy={"per_timepoint": [], "discontinuity": [
        _disc("D3", "%T (of lymph)", 5.087)]})
    off = _build(r)["items"][0]
    assert off["must_read"] == ["diagnostics_discontinuity"]
    assert off["should_read"] == ["diagnostics_timepoints"]


def test_every_column_a_required_table_names_exists_in_that_table(tmp_path: Path):
    """A projection is only worth enforcing if it runs."""
    written = _written_with_headers(tmp_path, gate_geometry=GEOMETRY_COLUMNS)
    book = flags.FlagBook()
    for code, tier, subj, measured in (
        (flags.F_NEG_MODE_DRIFT, 2, "cd56 @ D28", {"drift": 0.1}),
        (flags.F_NEG_DISTRIBUTION_SHIFT, 2, "ld @ D91", {"emd_negative_subset": 0.2}),
        (flags.F_NO_VALLEY_EVIDENCE, 1, "cd19 @ Baseline", {"n_modes": 1}),
        (flags.F_QUADRANT_MARGIN, 2, "NK cd56 @ D7", {"margin_in_sd": 1.1}),
    ):
        book.add(code, tier, subj, f"rule for {code}", measured, "hint", exceedance=2.0)
    m = obligations.build(_report(), book, written=written)
    headers = {f"diagnostics_{label}": Path(path).read_text().strip().split(",")
               for label, path in written.items()}
    checked = 0
    for it in m["items"]:
        for table, cols in (it.get("read_columns") or {}).items():
            assert table in it["must_read"], f"{it['id']} names columns for an unrequired table"
            for c in cols:
                assert c in headers[table], f"{it['id']}: {table} has no column {c}"
                checked += 1
    assert checked, "the test proved nothing if no item named a column"
