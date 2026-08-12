"""Validation of the operator's manual-gating CSV.

These pin the failure modes measured against the pre-validation code, where a malformed
CSV either crashed with a raw ``KeyError``/``TypeError`` from inside the calibration, or —
worse — succeeded and reported a confident MAE computed over the handful of rows that
happened to match. A well-formed file must still pass through untouched.

``manual_check`` imports pandas (a container dependency), so these skip in the dev venv and
run in the sandbox image.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FIRSTRUN = Path(__file__).resolve().parent.parent / "src" / "flow" / "firstrun"
_HAVE_PANDAS = importlib.util.find_spec("pandas") is not None

pytestmark = pytest.mark.skipif(not _HAVE_PANDAS, reason="needs pandas (container dep)")


@pytest.fixture(scope="module")
def mc():
    sys.path.insert(0, str(FIRSTRUN))
    try:
        from anchored import manual_check

        return manual_check
    finally:
        sys.path.remove(str(FIRSTRUN))


@pytest.fixture
def dataset(tmp_path):
    """An FCS folder whose filenames parse to Baseline / D14 / week8."""
    fcs = tmp_path / "fcs"
    fcs.mkdir()
    for tp in ("Baseline", "D14", "Week8"):
        (fcs / f"Specimen_001_{tp}.fcs").write_bytes(b"FCS3.0")
    return fcs


def _csv(tmp_path, text: str) -> Path:
    p = tmp_path / "manual_gating.csv"
    p.write_text(text)
    return p


GOOD = (
    "timepoint,b_p,t_p,cd4_p,cd8_p,nk_p,d_nk_p,car_d_p\n"
    "Baseline,1.24,44.0,8.11,28.7,42.5,0.064,0.0\n"
    "D14,0.51,63.5,20.0,40.0,27.9,12.8,7.2\n"
    "week8,0.70,4.6,1.0,3.0,85.0,32.2,9.7\n"
)


def test_well_formed_csv_passes_with_no_warnings(mc, tmp_path, dataset):
    check = mc.check_manual_gating(_csv(tmp_path, GOOD), dataset, "Baseline")
    assert check.matched == ["Baseline", "D14", "week8"]
    assert check.unmatched_manual == []
    assert check.ungated_fcs == []
    assert check.warnings == []


# ── the silent failures ─────────────────────────────────────────────────────────
def test_partly_misspelled_labels_are_reported_not_swallowed(mc, tmp_path, dataset):
    """Two of three rows misspelled used to yield a clean-looking MAE over the third."""
    check = mc.check_manual_gating(_csv(tmp_path, GOOD.replace("D14,", "Day 14,")
                                        .replace("week8,", "Week 8,")), dataset, "Baseline")
    assert check.matched == ["Baseline"]
    assert sorted(check.unmatched_manual) == ["Day 14", "Week 8"]
    blob = check.report()
    assert "Day 14" in blob and "Week 8" in blob
    # ...and the report must name the labels that would have worked.
    assert "D14" in blob and "week8" in blob


def test_all_labels_misspelled_is_called_out_loudly(mc, tmp_path, dataset):
    text = GOOD.replace("Baseline,", "Day 0,").replace("D14,", "Day 14,").replace(
        "week8,", "Week 8,")
    with pytest.raises(mc.ManualGatingError) as e:
        # No reference row either, since 'Baseline' was renamed — that is the fatal one.
        mc.check_manual_gating(_csv(tmp_path, text), dataset, "Baseline")
    assert "reference timepoint" in str(e.value)


def test_reference_row_missing_some_metrics_warns_instead_of_narrowing_silently(
        mc, tmp_path, dataset):
    text = ("timepoint,b_p,nk_p\n"
            "Baseline,1.24,42.5\n"
            "D14,0.51,27.9\n")
    check = mc.check_manual_gating(_csv(tmp_path, text), dataset, "Baseline")
    warn = " ".join(check.warnings)
    assert "t_p" in warn and "cd4_p" in warn and "cd8_p" in warn
    assert "calibration will target only" in warn


def test_low_coverage_mae_is_qualified(mc):
    import pandas as pd

    cmp_df = pd.DataFrame([{"timepoint": "Baseline", "metric": "%NK (of lymph)",
                            "abs_delta": 0.0}])
    check = mc.ManualCheck(Path("m.csv"), ["Baseline", "D14", "week8"],
                           ["Baseline", "Day 14", "Week 8"], ["Baseline"], [])
    note = mc.coverage_note(cmp_df, check)
    assert "1 of 3 manual timepoint(s)" in note
    assert "LOW COVERAGE" in note


def test_empty_comparison_is_not_mistaken_for_agreement(mc):
    import pandas as pd

    note = mc.coverage_note(pd.DataFrame(), None)
    assert "NOT a perfect agreement" in note


# ── the cryptic failures ────────────────────────────────────────────────────────
def test_missing_timepoint_column_explains_the_header(mc, tmp_path, dataset):
    """Previously a bare KeyError: 'timepoint' from inside load_manual."""
    text = "sample,b_p,t_p,nk_p\nBaseline,1.24,44.0,42.5\n"
    with pytest.raises(mc.ManualGatingError) as e:
        mc.check_manual_gating(_csv(tmp_path, text), dataset, "Baseline")
    msg = str(e.value)
    assert "no 'timepoint' column" in msg
    assert "sample" in msg          # shows what was actually found
    assert "timepoint,b_p" in msg   # ...and what was expected


def test_blank_reference_row_explains_itself(mc, tmp_path, dataset):
    """Previously a TypeError: float() argument ... not 'NoneType' from the calibration."""
    text = ("timepoint,b_p,t_p,cd4_p,cd8_p,nk_p\n"
            "Baseline,,,,,\n"
            "D14,0.51,63.5,20.0,40.0,27.9\n")
    with pytest.raises(mc.ManualGatingError) as e:
        mc.check_manual_gating(_csv(tmp_path, text), dataset, "Baseline")
    assert "no usable value" in str(e.value)


def test_missing_reference_row_names_the_rows_it_did_find(mc, tmp_path, dataset):
    text = "timepoint,b_p,t_p,nk_p\nD14,0.51,63.5,27.9\n"
    with pytest.raises(mc.ManualGatingError) as e:
        mc.check_manual_gating(_csv(tmp_path, text), dataset, "Baseline")
    msg = str(e.value)
    assert "no row for the reference timepoint 'Baseline'" in msg
    assert "D14" in msg
    assert "metadata.json" in msg  # tells them where the setting lives


def test_case_or_spacing_slip_in_the_reference_label_is_diagnosed(mc, tmp_path, dataset):
    text = "timepoint,b_p,t_p,nk_p\nbase line,1.24,44.0,42.5\n"
    with pytest.raises(mc.ManualGatingError) as e:
        mc.check_manual_gating(_csv(tmp_path, text.replace("base line", "BaseLine")),
                               dataset, "Baseline")
    assert "Did you mean" in str(e.value)


def test_empty_file_is_rejected(mc, tmp_path, dataset):
    with pytest.raises(mc.ManualGatingError) as e:
        mc.check_manual_gating(_csv(tmp_path, "timepoint,b_p,t_p,nk_p\n"), dataset, "Baseline")
    assert "no rows" in str(e.value)


def test_duplicate_timepoint_rows_warn(mc, tmp_path, dataset):
    check = mc.check_manual_gating(_csv(tmp_path, GOOD + "D14,9,9,9,9,9,9,9\n"),
                                   dataset, "Baseline")
    assert any("duplicate" in w for w in check.warnings)
