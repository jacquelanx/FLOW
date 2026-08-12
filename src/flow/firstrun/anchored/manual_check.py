#!/usr/bin/env python3
"""FLOW addition: validate the operator's manual-gating CSV before it steers a run.

The manual CSV is ground truth — it calibrates the reference timepoint and it is what the
MAE is reported against. A malformed one used to fail in two unhelpful ways:

  * **Silently.** ``compare()`` drops any row whose timepoint label doesn't match a label
    parsed from the FCS filenames. Misspell two of three timepoints and the run still
    succeeds, reporting a confident MAE computed over the one row that happened to match.
  * **Cryptically.** A missing header or a blank cell in the reference row surfaced as a
    raw ``KeyError: 'timepoint'`` or ``TypeError: float() argument must be ... not
    'NoneType'`` from deep inside the calibration.

This module checks the file up front and says what is wrong in the operator's terms. It
only inspects — it never edits the CSV or the numbers, so a well-formed file produces
exactly the run it always did.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compare import METRIC_MAP, load_manual, timepoint_from_file

# Metrics the reference timepoint is calibrated against (see reference_anchor.REF_METRICS).
REF_METRICS = ("b_p", "t_p", "nk_p", "cd4_p", "cd8_p")
# Every column compare() can use, for the "unused column" hint.
KNOWN_METRICS = tuple(METRIC_MAP.values())
# Coverage below this fraction of the supplied rows makes the MAE misleading on its own.
LOW_COVERAGE_FRAC = 0.75


class ManualGatingError(Exception):
    """The manual CSV cannot be used as ground truth. Message is operator-facing."""


class ManualCheck:
    """Outcome of the check: what matched, what didn't, and what to warn about."""

    def __init__(self, path: Path, expected: list[str], supplied: list[str],
                 matched: list[str], warnings: list[str]):
        self.path = path
        self.expected = expected          # timepoints parsed from the FCS files
        self.supplied = supplied          # timepoints listed in the manual CSV
        self.matched = matched            # the intersection, in FCS order
        self.warnings = warnings

    @property
    def unmatched_manual(self) -> list[str]:
        """Manual rows that correspond to no FCS — these are silently dropped downstream."""
        return [t for t in self.supplied if t not in self.matched]

    @property
    def ungated_fcs(self) -> list[str]:
        """Samples with no manual row — legitimate, but they get no MAE."""
        return [t for t in self.expected if t not in self.matched]

    def report(self) -> str:
        lines = [f"manual_gating.csv: matched {len(self.matched)} of {len(self.supplied)} "
                 f"manual timepoint(s) to FCS files."]
        if self.unmatched_manual:
            lines.append(f"  UNMATCHED manual rows (ignored): {_fmt(self.unmatched_manual)}")
            lines.append(f"  Expected one of: {_fmt(self.expected)}")
        if self.ungated_fcs:
            lines.append(f"  Samples with no manual row (not compared): "
                         f"{_fmt(self.ungated_fcs)}")
        lines.extend(f"  WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def _fmt(items, limit: int = 12) -> str:
    items = [str(i) for i in items]
    shown = ", ".join(items[:limit])
    return shown + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() == ""


def expected_timepoints(fcs_dir: Path) -> list[str]:
    """Timepoint labels FLOW parses from the FCS filenames, in filename order."""
    out: list[str] = []
    for f in sorted(Path(fcs_dir).glob("*.fcs")):
        tp = timepoint_from_file(f.name)
        if tp is not None and tp not in out:
            out.append(tp)
    return out


def check_manual_gating(manual_path, fcs_dir, reference_tp: str) -> ManualCheck:
    """Validate the manual CSV against the dataset. Raises ManualGatingError if unusable.

    Fatal: unreadable/empty file, no ``timepoint`` column, no row for the reference
    timepoint, or a reference row with no usable metric. Everything else is a warning —
    partial manual coverage is legitimate and only limits what can be compared.
    """
    manual_path = Path(manual_path)
    try:
        man = load_manual(manual_path)
    except KeyError:
        raise ManualGatingError(
            f"{manual_path.name} has no 'timepoint' column.\n"
            f"  Expected a header row like: "
            f"timepoint,{','.join(REF_METRICS)},d_nk_p,car_d_p\n"
            f"  Found: {', '.join(_read_header(manual_path)) or '(no header)'}")
    except Exception as e:
        raise ManualGatingError(f"{manual_path.name} could not be read: {e}")

    if man.empty:
        raise ManualGatingError(f"{manual_path.name} has a header but no rows.")

    warnings: list[str] = []
    supplied = [str(t).strip() for t in man["timepoint"]]

    dupes = sorted({t for t in supplied if supplied.count(t) > 1})
    if dupes:
        warnings.append(f"duplicate timepoint row(s) {_fmt(dupes)} — the first of each is used")

    expected = expected_timepoints(fcs_dir)
    matched = [t for t in expected if t in supplied]

    # ── the reference row: fatal problems, because the whole calibration rests on it ──
    ref_rows = man[man["timepoint"].astype(str).str.strip() == reference_tp]
    if ref_rows.empty:
        hint = ""
        near = [t for t in supplied if t.lower().replace(" ", "") ==
                reference_tp.lower().replace(" ", "")]
        if near:
            hint = (f"\n  Did you mean the row labelled {near[0]!r}? Labels must match "
                    f"exactly (case and spacing included).")
        raise ManualGatingError(
            f"{manual_path.name} has no row for the reference timepoint {reference_tp!r}.\n"
            f"  That row is what the whole calibration is built on, so it is required.\n"
            f"  Rows found: {_fmt(supplied)}{hint}\n"
            f"  (The reference timepoint is set by 'reference_timepoint' in metadata.json.)")

    ref = ref_rows.iloc[0]
    present = [m for m in REF_METRICS if m in man.columns and not _is_blank(ref.get(m))]
    if not present:
        blank = [m for m in REF_METRICS if m in man.columns]
        raise ManualGatingError(
            f"{manual_path.name}: the {reference_tp!r} row has no usable value in any of "
            f"{', '.join(REF_METRICS)}.\n"
            f"  Columns present but blank: {_fmt(blank) or '(none)'}\n"
            f"  Missing columns: {_fmt([m for m in REF_METRICS if m not in man.columns])}\n"
            f"  Fill in at least one of these for {reference_tp!r} — they are the "
            f"percentages the automated gating is calibrated to match.")
    missing = [m for m in REF_METRICS if m not in present]
    if missing:
        warnings.append(
            f"the {reference_tp!r} row is missing or blank for {_fmt(missing)} — "
            f"calibration will target only {_fmt(present)}")

    # ── label mismatches: the silent failure this module exists to surface ──
    unmatched = [t for t in supplied if t not in expected]
    if unmatched and not matched:
        warnings.append(
            f"NONE of the manual timepoints match any FCS file, so nothing will be "
            f"compared and the reported MAE will be empty. Manual labels: "
            f"{_fmt(sorted(set(supplied)))}; expected: {_fmt(expected)}")
    elif unmatched:
        warnings.append(
            f"{len(unmatched)} manual row(s) match no FCS file and are ignored: "
            f"{_fmt(unmatched)}")

    unused = [c for c in man.columns
              if c not in KNOWN_METRICS and c != "timepoint" and not c.startswith("date")]
    if unused:
        warnings.append(f"column(s) not used by the comparison: {_fmt(unused)}")

    return ManualCheck(manual_path, expected, supplied, matched, warnings)


def coverage_note(cmp_df: pd.DataFrame, check: ManualCheck | None) -> str:
    """One line qualifying an MAE with how much of the manual data it actually covers."""
    if cmp_df is None or cmp_df.empty:
        return ("no manual timepoint matched an FCS file — MAE is empty, NOT a perfect "
                "agreement")
    n_tp = int(cmp_df["timepoint"].nunique())
    supplied = len(set(check.supplied)) if check else n_tp
    note = f"over {len(cmp_df)} metric comparison(s) spanning {n_tp} of {supplied} manual timepoint(s)"
    if supplied and n_tp < supplied * LOW_COVERAGE_FRAC:
        note += (f" — LOW COVERAGE: {supplied - n_tp} manual timepoint(s) were dropped, so "
                 f"this MAE does not describe them")
    return note


def _read_header(path: Path) -> list[str]:
    try:
        with open(path) as f:
            return [c.strip() for c in f.readline().strip().split(",")]
    except Exception:
        return []
