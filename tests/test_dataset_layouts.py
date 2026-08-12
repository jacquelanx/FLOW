"""Dataset-layout tolerance for the anchored first-run template.

A study only reaches the pipeline if FLOW can (a) get every FCS onto disk without losing
files to basename collisions and (b) recognise which file is which specimen. These tests
pin that behaviour against the layouts we have actually been handed:

  * UPN27 — flat folder, ``Specimen_001_<timepoint>.fcs``, one control of each kind.
  * UPN25 — a zip of one acquisition folder per timepoint, repeating the same basenames in
    every folder, with a full set of instrument compensation controls per folder, and
    timepoints (D16/D17/D32/D35, plus a second D35 specimen) that no fixed token map covers.

The parsing helpers live in ``anchored/compare.py``, which imports pandas for the
manual-gating comparison. pandas is a container dependency, so it is stubbed here — the
functions under test use only ``re`` and ``pathlib``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

COMPARE_PY = (Path(__file__).resolve().parent.parent
              / "src" / "flow" / "firstrun" / "anchored" / "compare.py")


@pytest.fixture(scope="module")
def compare():
    """Load ``anchored.compare``, standing pandas in only when it is genuinely absent."""
    stubbed = importlib.util.find_spec("pandas") is None
    if stubbed:
        sys.modules["pandas"] = types.ModuleType("pandas")
    try:
        spec = importlib.util.spec_from_file_location("_anchored_compare", COMPARE_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        if stubbed:
            sys.modules.pop("pandas", None)


# ── the layout the pipeline was written against must not move ───────────────────
UPN27_FILES = {
    "Specimen_001_Baseline.fcs": "Baseline",
    "Specimen_001_Pre.fcs": "Pre",
    "Specimen_001_Post.fcs": "Post",
    "Specimen_001_D3.fcs": "D3",
    "Specimen_001_D7.fcs": "D7",
    "Specimen_001_D14.fcs": "D14",
    "Specimen_001_D21.fcs": "D21",
    "Specimen_001_D28.fcs": "D28",
    "Specimen_001_week8.fcs": "week8",
    "Specimen_001_D81.fcs": "D81",
    "Specimen_001_D86.fcs": "D86",
    "Specimen_001_D91.fcs": "D91",
    # Controls carry no timepoint.
    "Specimen_001_CBMC.fcs": None,
    "Specimen_001_Car.fcs": None,
    "Specimen_001_NT-NK.fcs": None,
}


@pytest.mark.parametrize("fname,expected", sorted(UPN27_FILES.items()))
def test_reference_study_labels_are_unchanged(compare, fname, expected):
    """UPN27's labels are the keys of its manual-gating CSV — they must stay exact."""
    assert compare.timepoint_from_file(fname) == expected


def test_reference_study_pre_infusion_flags_are_unchanged(compare):
    assert compare.is_pre_infusion("Specimen_001_Baseline.fcs")
    assert compare.is_pre_infusion("Specimen_001_Pre.fcs")
    assert compare.is_baseline("Specimen_001_Baseline.fcs")
    assert not compare.is_baseline("Specimen_001_Pre.fcs")
    assert not compare.is_pre_infusion("Specimen_001_Post.fcs")
    assert not compare.is_pre_infusion("Specimen_001_D14.fcs")


# ── timepoints outside the fixed token map ──────────────────────────────────────
@pytest.mark.parametrize("fname,expected", [
    ("Specimen_001_D16.fcs", "D16"),
    ("Specimen_001_D17.fcs", "D17"),
    ("Specimen_001_D32.fcs", "D32"),
    ("Specimen_001_D35.fcs", "D35"),
    ("Specimen_001_Week8.fcs", "week8"),
    ("patient_wk12.fcs", "week12"),
    ("sample_month3.fcs", "month3"),
    ("UPN30_D14_PBMC.fcs", "D14 PBMC"),
    # Timepoint first, boilerplate after — the boilerplate is not a specimen qualifier.
    ("D7_Specimen_001.fcs", "D7"),
    ("no_timepoint_here.fcs", None),
])
def test_unmapped_timepoints_still_parse(compare, fname, expected):
    assert compare.timepoint_from_file(fname) == expected


@pytest.mark.parametrize("fname", [
    "Specimen_001_D35 Pleural Fluid.fcs",   # as acquired
    "Specimen_001_D35_Pleural_Fluid.fcs",   # after upload sanitises the space
])
def test_second_specimen_on_one_day_stays_distinct(compare, fname):
    """D35 blood and D35 pleural fluid are different samples, not one duplicated row."""
    assert compare.timepoint_from_file(fname) == "D35 Pleural Fluid"
    assert compare.timepoint_from_file(fname) != compare.timepoint_from_file(
        "Specimen_001_D35.fcs")


# ── compensation controls are not specimens ─────────────────────────────────────
@pytest.mark.parametrize("fname", [
    "Compensation Controls_Alexa Fluor 488 Stained Control.fcs",
    "Compensation Controls_Unstained Control.fcs",
    "Compensation_Controls_UV_450_L_2f_D_Stained_Control.fcs",
    "UPN_25_D14__Compensation_Controls_PE_Stained_Control.fcs",
    "Comp_PE_D14.fcs",
])
def test_compensation_controls_are_recognised(compare, fname):
    assert compare.is_comp_control(fname)
    # ...and therefore carry no timepoint, so nothing downstream gates them as a sample.
    assert compare.timepoint_from_file(fname) is None


@pytest.mark.parametrize("fname", [
    "Specimen_001_D14.fcs", "Specimen_001_NT-NK.fcs", "Specimen_001_Car.fcs",
    "Specimen_001_CBMC.fcs", "Specimen_001_Baseline.fcs",
])
def test_specimens_and_assay_controls_are_not_mistaken_for_comp(compare, fname):
    assert not compare.is_comp_control(fname)


# ── the "<source folder>__" provenance prefix is transparent ────────────────────
def test_source_prefix_does_not_change_interpretation(compare):
    assert compare.strip_source_prefix(
        "UPN_25_D14__Specimen_001_D14.fcs") == "Specimen_001_D14.fcs"
    assert compare.timepoint_from_file("UPN_25_D14__Specimen_001_D14.fcs") == "D14"
    assert compare.timepoint_from_file("UPN_25_Baseline__Specimen_001_Baseline.fcs") == "Baseline"


def test_source_prefix_cannot_leak_into_the_pre_infusion_flag(compare):
    """The "Pre + Post" folder holds both samples — the folder must not make Post pre."""
    assert compare.is_pre_infusion("UPN_25_Pre_Post__Specimen_001_Pre.fcs")
    assert not compare.is_pre_infusion("UPN_25_Pre_Post__Specimen_001_Post.fcs")
    assert not compare.is_baseline("UPN_25_Baseline__Specimen_001_Car.fcs")


# ── composition_shift.csv must survive labels the parser could not resolve ──────
_HAVE_ANALYSIS_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("pandas", "numpy", "matplotlib"))
FIRSTRUN = Path(__file__).resolve().parent.parent / "src" / "flow" / "firstrun"


@pytest.mark.skipif(not _HAVE_ANALYSIS_DEPS,
                    reason="needs pandas/numpy/matplotlib (container deps)")
def test_composition_shift_tolerates_unresolved_and_repeated_labels(tmp_path):
    """Rows are addressed positionally, so a NaN or repeated timepoint can't collapse them.

    Addressing by label previously raised ("truth value of a Series is ambiguous") the
    moment two rows shared a timepoint — which any file the token map didn't cover did,
    because they all landed on NaN together.
    """
    sys.path.insert(0, str(FIRSTRUN))
    try:
        import pandas as pd

        from anchored.flow_outputs import write_composition_shift
    finally:
        sys.path.remove(str(FIRSTRUN))

    multi = pd.DataFrame([
        {"file": "s_Baseline.fcs", "timepoint": "Baseline", "%NK (of lymph)": 30.0},
        {"file": "s_D7.fcs", "timepoint": "D7", "%NK (of lymph)": 45.0},
        {"file": "mystery_a.fcs", "timepoint": None, "%NK (of lymph)": 50.0},
        {"file": "mystery_b.fcs", "timepoint": None, "%NK (of lymph)": 60.0},
        {"file": "s_D7_repeat.fcs", "timepoint": "D7", "%NK (of lymph)": 47.0},
    ])
    path = write_composition_shift(multi, str(tmp_path), reference_tp="Baseline")
    import csv as _csv

    rows = [r for r in _csv.DictReader(open(path)) if r["population"] == "%NK (of lymph)"]
    assert len(rows) == 5, "every sample must be reported, including unlabelled ones"
    assert [r["timepoint"] for r in rows] == [
        "Baseline", "D7", "mystery_a.fcs", "mystery_b.fcs", "D7 #2"]
    assert [r["value"] for r in rows] == ["30.0", "45.0", "50.0", "60.0", "47.0"]
    # Deltas are still measured against the reference row.
    assert rows[1]["delta_from_reference"] == "15.0"
    assert rows[1]["delta_from_previous"] == "15.0"


# ── ordering fallback when a project ships no flow.csv ──────────────────────────
def test_timepoints_sort_chronologically_without_a_flow_csv(compare):
    labels = ["week8", "D3", "Baseline", "D35 Pleural Fluid", "Post", "D14", "Pre",
              "D35", "D7", None, "D110"]
    assert sorted(labels, key=compare.timepoint_sort_key) == [
        "Baseline", "Pre", "Post", "D3", "D7", "D14", "D35", "D35 Pleural Fluid",
        "week8", "D110", None,
    ]
