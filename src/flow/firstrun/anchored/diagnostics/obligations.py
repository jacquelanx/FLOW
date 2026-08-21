"""The obligations worklist — turning a counted report into a finite amount of work.

The diagnostics pass measures well and indexes honestly. ``cutoff_diagnostics_digest.txt``
names every flag group with its full count and tells the reader, in as many words, that a
group printed as ``x90`` means ninety findings of which they have seen one. On a real study
that is 375 flags in 18 groups spread across 11 tables.

Two consecutive production runs showed what a reader actually does with that. Both opened the
digest, both quoted its headline, and between them they opened three of the eleven tables:
357 flag rows were never read, and neither write-up said so. The instruction to read them was
present, specific, and ignored. More of the same instruction is not a fix.

What is missing is a WORKLIST — a list short enough to finish. This module derives one from
the flags that actually fired and states, per item, what would discharge it.

Three rules keep this from becoming the analysis:

  * **An obligation is a duty to READ, never a verdict.** ``statement`` restates a
    measurement that already exists in the report and ``why`` carries the flag's own
    resolution hint. Neither concludes that a cutoff is wrong — that judgement stays with the
    reader, exactly as in ``flags.py``.
  * **The list is derived, never enumerated.** Items come from ``FlagBook`` and the tier-0
    discontinuity rollup, so a new flag code produces obligations without editing this file,
    and a study where nothing fires produces an empty list rather than a reassuring one.
  * **A cap is stated, not silent.** Promoting everything would rebuild the 375-item problem,
    so each class has a cap and the manifest records how many candidates were not promoted.
    ``n_not_promoted > 0`` means "there is more", never "that was all".

    The cap has ONE stated exception. A tier-1 finding that a ROOT GATE's cutoff was not
    derived from the data — the cutoff every headline number is measured inside — is promoted
    past both caps (``ROOT_GATE_CODES``, ``ROOT_GATE_KEYS``, ``forced``), and the count is
    reported as ``n_forced``. The exception exists because the run that motivated it dropped
    exactly that finding: the viability cutoff sat at a literal fallback percentile with no
    valley evidence, ``ld`` is the parent of CD45+ -> lymph -> NK, and the item lost its slot
    to the fifth off-trend timepoint. A cap that can evict the denominator of every reported
    number is not bounding the reader's work, it is choosing their conclusion.

A fourth rule governs how an item POINTS at its rows: **name the columns, not just the
table.** ``read: diagnostics_cutoff_audit`` was satisfied by a cell that printed three of
that table's thirty-seven columns and projected away the two carrying the finding — and the
item was recorded as discharged, because the table had been opened. So ``how`` lines are
projected to the columns the rule was evaluated on, resolved against the CSV's real header
(see ``_resolve_columns``) so that every line RUNS. A key that resolves to no column is
dropped rather than guessed; the flags-table line always carries the full ``measured``
payload, so the numbers are reachable even when a projection cannot be built.

``must_mention`` deserves its own note, because it is the field a submission check enforces
against the answer TEXT — and it is the only field that can require the reader to WRITE
something rather than print something. It is populated where the required token is a word the
write-up has to use anyway (a literal sample label; the word "reproduction"; the name of the
marker whose cutoff the item is about) and where the digest already states the naming
requirement. For a root gate the token comes from the pipeline's own display label rather than
the cut key, because the check matches on word boundaries and nobody writes "ld" as a word —
they write "viability" (see ``_mention_token``, which declines to demand a token shorter than
three characters rather than impose one that cannot be satisfied honestly).

Flag groups once got ``must_read`` alone, on the argument that the worst offender in a group is
an item like ``%NK (of lymph)`` or ``NK cloud`` and requiring such a substring verbatim would
fail careful answers while rewarding keyword stuffing. That argument was right about the ITEM
half of a subject and wrong to stop there, and the next run measured the cost: seventeen of
twenty-three obligations carried no token, and the trajectory that discharged all twenty-three
— by printing columns — produced the THINNEST write-up of the five, having read the host-NK
false-positive rate in a cell and reported nothing about it. Reading was enforced, reporting
was not, so reading is what happened. So a flag group now demands the SCOPE half: the literal
sample label its worst case sits at, when the run has a sample by that name (``_scope_token``).
That is the same token class the off-trend items already use, and the requirement good writing
can fail is still refused — the item name is still never demanded.

``must_read`` names the tables holding the measurements the rule was evaluated on; genuine
supporting context goes in ``should_read``, which is surfaced but never blocks.

That split was once drawn more narrowly — ONE table per item, the flags index — because the
first version required every named table at once and a production run showed the cost: a
trajectory opened the off-trend rows and the flag rows in its first cell, was told "15 of 15
still undischarged" because it had not also opened the supporting tables, and responded with a
single cell whose own comment read "I will aggregate them to show I have reviewed them" —
``.head()`` on five tables, no new information. Requiring the supporting table bought a
``.head()`` call; refusing credit for the primary read cost the honest work that preceded it.

Narrowing it to the index alone then cost the opposite thing, and the next run measured it: 15
of 23 items resolved to ``must_read == ['diagnostics_flags']``, one ``groupby('code').size()``
discharged all 15, and ``diagnostics_transfer`` and ``diagnostics_gate_geometry`` — the tables
holding the per-timepoint and 2-D geometry measurements — went unopened across five
trajectories. A tally of a tally is not a read. The run before it, which still required the
detail table, opened 7-8 of 11 tables on the same model.

So the rule is now: the detail table is required WHEN ITS MEASUREMENT COLUMNS CAN BE NAMED
(``read_columns``), and advisory when they cannot. That is what makes this different from the
first version rather than a revert to it — the demand that bought a ``.head()`` was "open this
table"; the demand here is "print these columns", which a ``.head()`` does not satisfy. Where
the columns cannot be resolved the old caution still holds and the table stays a suggestion.

``read_columns`` maps a required table to the columns carrying the measurement, with the
identity columns removed: a reader who prints ``[['marker', 'timepoint']]`` has named the rows
without reading any of them. Enforcement is the consumer's business — see
``flow.env.notebook_env`` — and an item with no ``read_columns`` degrades to the table-name
check, so a manifest built by an older pass still works.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

from . import flags as F
from .flags import Flag, FlagBook
from .markers import HEADLINE_METRICS, MARKERS, _ALL_LINEAGE

SCHEMA = "flow_diagnostics_obligations/v1"
OBLIGATIONS_JSON = "diagnostics_obligations.json"

# How many obligations are promoted. Sized against the agent's STEP budget rather than the
# flag count: a trajectory gets ~30 steps, each obligation costs at least one cell to read
# plus a sentence to address, and a list much longer than this cannot be finished — at which
# point it is ignored whole, which is the failure this module exists to fix. Many items share
# a ``read:`` line (every flag group is a filter on the same table), so this is closer to
# a dozen cells of work than twenty-four.
#
# Raised from 16 after a production run showed the cost of the old figure. It promoted 15 of
# 24 candidates; the trajectory discharged all 15 in nine steps — finishing with two thirds of
# its budget unspent — and the nine it never saw held 194 of the run's 375 flags, including
# both tier-1 findings on the viability gate. The cap was binding on the report, not on the
# reader. The step budget is the real constraint and it was not close to reached.
#
# Raised again from 24 when ``gate_geometry`` was split out of ``transfer_validity`` (see
# TABLE_CLASS): the new class needs two slots of its own. The same evidence applies — the run
# that motivated the split discharged 21 obligations in 10 of its 30 steps — and the bound
# that actually matters is DISTINCT TABLES rather than item count, of which the split adds
# exactly one.
MAX_OBLIGATIONS = 26

# Classes in the order they are promoted. Reproduction leads because a failed reproduction
# check means every other measurement may describe a different gating than the one being
# reported. Off-trend samples come next because dropping one silently is the single failure
# the digest calls out by name.
CLASS_ORDER = (
    "reproduction",
    "off_trend_timepoint",
    "data_adequacy",
    "cutoff_foundation",
    # Directly after the 1-D cutoff foundation, because it asks the same kind of question one
    # dimension up: whether a GATE boundary sits inside the cloud it is meant to divide.
    "gate_geometry",
    "transfer_validity",
    "sensitivity",
    "internal_controls",
    "operator_concordance",
    "coverage_gap",
)

# Per-class caps, applied BEFORE the global cap so one prolific class cannot crowd out the
# others: a study with fourteen off-trend samples must still surface its tier-1 findings, and
# a low-priority class that fires once (an unavailable control invalidating every CAR+ number)
# must still reach the list. The caps sum to MAX_OBLIGATIONS by construction, so the global
# cap is a backstop and these are what actually shape the list.
#
# The per-class shares follow where the flags actually are. On the run that motivated the
# raise, ``transfer_validity`` held 145 flags in five codes and could promote two of them, so
# ``PLACEMENT_MARGIN_LOST_ON_TRANSFER`` (56 findings) and ``OVERTON_DISAGREES_WITH_CUTOFF``
# (38) were dropped from a list that had room for neither. A class cap below its own code
# count is the mechanism that turns a counted flag back into an uncounted one.
CLASS_CAPS = {
    "reproduction": 1,
    "off_trend_timepoint": 5,
    "data_adequacy": 2,
    "cutoff_foundation": 6,
    "gate_geometry": 2,
    "transfer_validity": 5,
    "sensitivity": 2,
    "internal_controls": 1,
    "operator_concordance": 1,
    "coverage_gap": 1,
}

# ── root gates: the cutoffs every headline number is measured against ─────────
# A cutoff whose movement moves EVERY headline lineage metric is not one finding among many
# — it is the denominator of all of them. ``MarkerSpec.downstream`` already records which
# reported metrics each cutoff can move, so this is read off the pipeline's own declaration
# rather than listed here: a marker added to ``markers.py`` is classified without touching
# this file, and a marker whose downstream shrinks stops being escalated.
#
# ``valley_derived`` is required too. For HLA and CAR the pipeline does not claim to derive a
# cutoff from a valley (they come from controls or a longitudinal fit), so valley-shape
# findings there are informational by construction — escalating them would promote noise.
ROOT_GATE_KEYS = frozenset(
    m.key for m in MARKERS
    if m.valley_derived and set(_ALL_LINEAGE) <= set(m.downstream)
)

# The tier-1 codes that say a cutoff was NOT derived from the data's own structure — as
# opposed to derived and then placed marginally, which the other foundation codes cover. On a
# root gate this is the strongest statement the diagnostics can make about a headline number,
# so it is promoted regardless of how many off-trend samples are competing for the list.
#
# The run that motivated this had PERCENTILE_FALLBACK_SUSPECTED on the viability cutoff (at
# exactly the 90.0th percentile, ``valley_supported=False``) and NO_VALLEY_EVIDENCE on the
# same marker. Live is the parent of CD45+ -> lymph -> NK, so every reported NK% rested on it.
# Both lost the cap race to the fifth off-trend timepoint, went unread, and the write-up
# called the cutoffs "generally reasonable".
ROOT_GATE_CODES = frozenset({F.F_NO_VALLEY_EVIDENCE, F.F_PERCENTILE_FALLBACK})

# ── asserted cutoffs: the ones placement metrics cannot judge ─────────────────
# The mirror image of a root gate. A root gate is a cutoff the pipeline DERIVES from a valley,
# where the finding is that the derivation did not hold up. These two codes cover cutoffs the
# pipeline never claimed to derive from the data at all (HLA from control brackets, CAR from
# the NT-NK control's p99.9), and cutoffs the reference pool could not audit — where the
# finding is that the tier-1 measurements CANNOT answer the question, so the answer has to come
# from tiers 4 and 5 or be reported as unmeasured.
#
# Escalated per MARKER and forced past the caps for the same reason root gates are: on the run
# this was written against, ``%Donor NK`` rested on a cutoff with no valley evidence and a
# separation of 0.92 pooled SD, ``%CAR+`` rested on one with no tier-1 row at all, and neither
# fact reached any of five write-ups. Unlike a group obligation these carry ``must_mention``,
# because an answer that discusses "the cutoffs" generically has not addressed the specific
# one whose evidence is missing.
ASSERTED_CUTOFF_CODES = frozenset({F.F_ASSERTED_CUTOFF, F.F_REFERENCE_AUDIT_UNAVAILABLE})

# Which tier a flag belongs to decides its class, so a flag code added later is classified
# without touching this map.
TIER_CLASS = {
    0: "data_adequacy",
    1: "cutoff_foundation",
    2: "transfer_validity",
    3: "sensitivity",
    4: "internal_controls",
    5: "operator_concordance",
}

# Where the rows behind a flag live, by ``write_all`` label. Keyed on the flag-code constants
# rather than string literals so a rename in ``flags.py`` fails loudly here instead of
# silently pointing a reader at the wrong table.
CODE_TABLE = {
    F.F_LOW_PARENT: "timepoints",
    F.F_LOW_POSITIVE: "uncertainty",
    F.F_WIDE_COUNTING_CI: "uncertainty",
    F.F_ACQ_DRIFT: "timepoints",
    F.F_ACQ_RATE: "timepoints",
    F.F_NO_TIME_CHANNEL: "timepoints",
    F.F_QUADRANT_MARGIN: "gate_geometry",
    F.F_BOUNDARY_MASS: "gate_geometry",
    F.F_SCATTER_CONTAINMENT: "gate_geometry",
    F.F_FRAGILE_METRIC: "uncertainty",
    F.F_COUNTERFACTUAL_GAP: "counterfactual",
    F.F_MANUAL_DIVERGENCE: "concordance",
    F.F_MANUAL_COVERAGE: "concordance",
}

# A flag's TIER decides its class by default, but one tier can hold codes that answer
# different questions. Tier 2 carries both the marker-drift codes and the 2-D gate-geometry
# codes, and the drift codes fire an order of magnitude more often — so sharing one cap makes
# geometry structurally unpromotable rather than merely deprioritized. On the run that
# motivated this split, tier 2 fired SEVEN codes for a cap of five:
#
#     65 NEGATIVE_DISTRIBUTION_SHIFT      29 NEGATIVE_WIDTH_CHANGE
#     36 NEGATIVE_MODE_DRIFT              21 OVERTON_DISAGREES_WITH_CUTOFF
#     33 PLACEMENT_MARGIN_LOST_ON_TRANSFER
#     12 HIGH_BOUNDARY_EVENT_MASS          1 QUADRANT_BOUNDARY_IN_CLOUD
#
# The two geometry codes ranked sixth and seventh by count, so they were last in line BY
# CONSTRUCTION and could never win a slot while any drift code fired: 13 findings, and the
# only numeric account of the 2-D gates, permanently uncounted. That is the mechanism
# CLASS_CAPS' own comment names — a class cap below its own code count turns a counted flag
# back into an uncounted one — reached by the tier map instead of the cap.
#
# Keyed on the DETAIL TABLE rather than on the code, so a geometry flag added later to
# ``flags.py`` and ``CODE_TABLE`` is classified without touching this map, in keeping with
# TIER_CLASS above. A label absent here keeps its tier's class.
TABLE_CLASS = {
    "gate_geometry": "gate_geometry",
}

# Fallback when a code carries no explicit table: the tier's own flat table.
TIER_TABLE = {
    0: "timepoints",
    1: "cutoff_audit",
    2: "transfer",
    3: "uncertainty",
    4: "controls",
    5: "concordance",
}

# ``TEMPORAL_DISCONTINUITY`` is deliberately excluded from the flag-group pass: the same
# findings are promoted per SAMPLE by the off-trend class, which is the axis a reader
# reasons about, and promoting both would spend two worklist slots on one finding.
SKIP_CODES = frozenset({F.F_TEMPORAL_DISCONTINUITY})


# ── helpers ───────────────────────────────────────────────────────────────────
def _one_line(text: Any, limit: int = 220) -> str:
    """Collapse to a single clipped line. Statements are printed one-per-line downstream."""
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _num(v: Any, places: int = 3) -> str:
    """Render a measurement, naming missingness rather than printing a misleading zero."""
    if v is None or v == "":
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "—"
    return f"{round(f, places):g}"


def _slug(text: str) -> str:
    """An id fragment: uppercase, alphanumerics and underscores only."""
    out = []
    for ch in str(text or ""):
        out.append(ch.upper() if (ch.isalnum() or ch == "_") else "_")
    return "_".join(x for x in "".join(out).split("_") if x) or "ITEM"


def _split_subject(subject: str) -> tuple[str, str]:
    """Split ``item @ scope`` the way the flag emitters write it.

    Duplicated from ``emit`` rather than imported: ``emit`` imports this module to build the
    manifest, so the dependency has to run one way only.
    """
    s = str(subject or "").strip()
    if " @ " in s:
        item, scope = s.split(" @ ", 1)
        return item.strip(), scope.strip()
    return s, ""


def _groups(book: FlagBook) -> list[tuple[int, str, list[Flag]]]:
    """Group flags by (tier, code), preserving ``book.ordered()`` so the first is the worst.

    Worklist order therefore matches digest order — a reader comparing the two sees the same
    findings in the same sequence.
    """
    groups: dict[tuple[int, str], list[Flag]] = {}
    for f in book.ordered():
        groups.setdefault((f.tier, f.code), []).append(f)
    return [(tier, code, fs) for (tier, code), fs in groups.items()]


def _filter_expr(var: Optional[str], column: str, value: Any,
                 columns: Sequence[str] = ()) -> Optional[str]:
    """A runnable one-liner selecting the rows behind an obligation, or None if no table.

    ``columns`` projects the selection. Without it the line runs but names no measurement,
    and the submission check reads the agent's SOURCE — so a reader who follows a bare row
    filter literally has printed the row and still not satisfied the item that asked for the
    column. A ``read:`` line has to be able to discharge the obligation that emitted it.
    """
    if not var:
        return None
    sel = f"{var}[{var}.{column} == {value!r}]"
    return f"{sel}[{list(columns)!r}]" if columns else sel


# ── column projection ─────────────────────────────────────────────────────────
# Naming a table is not naming the measurement. A production run was told to read
# ``diagnostics_cutoff_audit``, did exactly that, and printed
# ``[['marker', 'cutoff', 'valley_supported']]`` — three of the table's thirty-seven columns,
# with the two that carried the finding among the thirty-four it projected away. The item was
# recorded as discharged because the table had been opened. So a ``read:`` line names the
# COLUMNS the rule was evaluated on.

# Columns that say WHICH row, prepended so a projection is readable on its own. Ordered by
# how a reader scans the table; only those actually present are used.
IDENTITY_COLUMNS = ("marker", "label", "timepoint", "metric", "check", "subject", "code")

# Beyond this a projection stops being a pointer and becomes the table again.
MAX_PROJECTED_COLUMNS = 9


def _read_header(path: Any) -> tuple[str, ...]:
    """The column names of a written CSV, or () if it cannot be read.

    Only the first line is read: the file was written moments ago by this same pass, and the
    alternative — trusting the flag's own key names — is what makes a projection unrunnable.
    """
    try:
        with open(str(path), "r", encoding="utf-8", newline="") as fh:
            line = fh.readline()
    except OSError:
        return ()
    if not line:
        return ()
    return tuple(c.strip().strip('"') for c in line.rstrip("\r\n").split(",") if c.strip())


def _headers(written: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """``{notebook variable: columns}`` for every written CSV."""
    out: dict[str, tuple[str, ...]] = {}
    for path in written.values():
        stem = Path(str(path)).stem
        if str(path).endswith(".csv") and stem.isidentifier():
            cols = _read_header(path)
            if cols:
                out[stem] = cols
    return out


def _resolve_columns(keys: Sequence[str], header: Sequence[str]) -> list[str]:
    """Map a flag's ``measured`` keys onto real column names in ``header``.

    A flag names its measurements for a reader of the flag, and the emitter names the same
    quantity for a reader of the table, so the two drift: ``drift_in_negative_sd`` is the
    column ``negative_mode_drift_in_negative_sd``, and ``cutoff_percentile_rank`` is
    ``cutoff_percentile_rank_in_pool``. An exact match wins; otherwise the SHORTEST column
    containing the key wins, which resolves the bare key ``drift`` to ``negative_mode_drift``
    rather than to one of the six longer columns that also contain it.

    A key that resolves to nothing is DROPPED, never guessed. A projection is only useful if
    it runs, and the flags table always carries the full ``measured`` payload as a fallback.
    """
    cols = list(header)
    out: list[str] = []
    for key in keys:
        k = str(key)
        if k in cols:
            match = k
        else:
            candidates = [c for c in cols if k in c]
            if not candidates:
                continue
            match = min(candidates, key=lambda c: (len(c), c))
        if match not in out:
            out.append(match)
    return out


def _projection(var: Optional[str], header: Sequence[str], keys: Sequence[str]) -> Optional[str]:
    """``var[['id', ..., 'measurement', ...]]`` — the columns the rule turns on.

    Returns None when there is no table, no header, or nothing resolved: the caller then
    names the bare table, which is where this module started and is still honest.
    """
    if not var or not header:
        return None
    resolved = _resolve_columns(keys, header)
    if not resolved:
        return None
    ident = [c for c in IDENTITY_COLUMNS if c in header and c not in resolved]
    cols = (ident + resolved)[:MAX_PROJECTED_COLUMNS]
    return f"{var}[{cols!r}]"


def _measured_keys(fs: Sequence[Flag]) -> list[str]:
    """Every ``measured`` key the group's flags use, in first-seen order.

    Taken across the group rather than from the worst flag alone: a rule that could not be
    evaluated on one subject reports a different key set there, and a projection that omits
    those columns hides exactly the rows a reader most needs.
    """
    keys: list[str] = []
    for f in fs:
        for k in (f.measured or {}):
            if k not in keys:
                keys.append(str(k))
    return keys


# ── candidate builders ────────────────────────────────────────────────────────
def _reproduction_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The fidelity check, when it failed or could not be run.

    Promoted first because it is the only finding that can invalidate the whole pass: if the
    numbers this audit recomputed are not the numbers in the headline tables, the audit is
    describing a different gating than the one being reported.
    """
    rep = report.get("reproduction") or {}
    if not rep.get("available"):
        return [{
            "kind": "reproduction",
            "id": "REPRODUCTION_UNVERIFIED",
            "statement": (
                "The reproduction check could not run — there was no headline table to "
                "compare this audit's recomputed values against."
            ),
            "why": (
                "Unverified is not verified. Every measurement below assumes it describes the "
                "gating that produced the reported numbers, and that assumption was not "
                "tested. State this rather than reporting the audit as clean."
            ),
            "must_mention": ["reproduction"],
            "must_read": [],
            "should_read": [],
            "how": ["cutoff_diagnostics.json -> ['reproduction']"],
        }]
    if rep.get("faithful"):
        return []
    n_bad = len(rep.get("mismatches") or [])
    return [{
        "kind": "reproduction",
        "id": "REPRODUCTION_MISMATCH",
        "statement": (
            f"{n_bad} of {rep.get('checked')} reported values did not reproduce from "
            f"{rep.get('source')} (max delta {_num(rep.get('max_abs_delta_pp'))} pp)."
        ),
        "why": (
            "The audit recomputed the gating and did not land on the reported numbers, so "
            "every diagnostic below may describe a different gating than the headline "
            "tables. Report this before any conclusion drawn from those tables."
        ),
        "must_mention": ["reproduction"],
        "must_read": [],
        "should_read": [],
        "how": ["cutoff_diagnostics.json -> ['reproduction']['mismatches']"],
    }]


# The discontinuity columns that carry the off-trend judgement — the value and the envelope
# it sits outside. Named literally rather than resolved from ``measured`` keys: an off-trend
# item is built from the tier-0 report rows, not from a Flag, so there are no keys to resolve.
OFF_TREND_COLUMNS = ("value_pct", "prev_value_pct", "next_value_pct",
                     "excursion_in_tolerances", "excursion_beyond_tolerance_pp")

# The per-SAMPLE adequacy columns: whether a sample can carry a reading at all. Named
# literally, for a reason specific to this table — the flags that point here report keys it
# does not hold (``LOW_PARENT_EVENTS`` measures ``n_parent`` on a GATE, while every column
# here measures the SAMPLE), so nothing resolves from ``measured`` and the table would stay
# advisory on a resolution failure rather than on a judgement about its worth. These are the
# quantities the profile's QC expectation already asks the reader to act on: few CD45+
# events, low viability, unstable acquisition.
TIMEPOINT_COLUMNS = ("n_events_analyzed", "n_lymphocytes", "lineage_purity_pct",
                     "retention_cd45p_pct", "retention_live_pct",
                     "retention_lymphocytes_pct", "acquisition_signal_drift_sd",
                     "acquisition_rate_cv")

# The gate-geometry measurements, named literally — and for a sharper reason than the two
# above. Resolution here does not fail, it succeeds WRONGLY. A geometry flag reports
# ``margin_in_sd``, three different gates each have a column ending in that, and
# ``_resolve_columns`` breaks the tie on column-name length — so a finding on the HLA x CAR
# boundary resolved to ``nk_cd3_boundary_margin_in_sd``, a column about a different gate. The
# same flag reports ``n_nk``, which resolved cleanly to an EVENT COUNT, and since
# ``_table_is_read`` needs only one column to match, printing that count discharged the item
# without printing a single margin.
#
# So these are named: every column that carries a MARGIN, a boundary EVENT MASS, or a
# CONTAINMENT fraction, across all three gates. Counts and cut values are deliberately absent
# — they are context, not the quantity any geometry rule is evaluated on.
GATE_GEOMETRY_COLUMNS = (
    "nk_cd3_boundary_margin_in_sd",
    "nk_cd56_boundary_margin_in_sd",
    "nk_fraction_within_one_sd_of_a_boundary",
    "donor_car_hla_boundary_margin_host_in_sd",
    "donor_car_hla_boundary_margin_donor_in_sd",
    "donor_car_car_boundary_margin_in_sd",
    "donor_car_fraction_nk_within_one_sd_of_hla_boundary",
    "donor_car_fraction_donor_within_one_sd_of_car_boundary",
    "scatter_containment",
    "scatter_spillover",
)


def _off_trend_items(report: dict[str, Any], var,
                     headers: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    """One obligation per SAMPLE carrying an off-trend value, worst excursion first.

    Grouped by timepoint rather than by (timepoint, metric) because the decision a reader has
    to make — include this sample or exclude it, and say which — is made once per sample.
    """
    t0 = report.get("tier0_data_adequacy") or {}
    disc = [d for d in (t0.get("discontinuity") or []) if d.get("discontinuous")]
    if not disc:
        return []

    by_tp: dict[str, list[dict[str, Any]]] = {}
    for d in disc:
        by_tp.setdefault(str(d.get("timepoint")), []).append(d)

    def worst_excursion(rows: Sequence[dict[str, Any]]) -> float:
        vals = [r.get("excursion_in_tolerances") or 0 for r in rows]
        return max((float(v) for v in vals), default=0.0)

    disc_var = var("discontinuity")
    tp_var = var("timepoints")
    # Resolved against the real header, because a PROJECTED ``read:`` line has to run and a
    # column named from a literal tuple alone might not be in the file. ``read_columns`` keeps
    # the literal as its fallback: naming what settles the item is still right when no header
    # is at hand, and the emitter wrote these columns, so they are there.
    disc_cols = _measurement_columns(headers.get(disc_var or "", ()), OFF_TREND_COLUMNS)
    tp_cols = _measurement_columns(headers.get(tp_var or "", ()), TIMEPOINT_COLUMNS)
    items: list[dict[str, Any]] = []
    for tp, rows in sorted(by_tp.items(), key=lambda kv: -worst_excursion(kv[1])):
        rows = sorted(rows, key=lambda r: -(r.get("excursion_in_tolerances") or 0))
        w = rows[0]
        extra = f" (+{len(rows) - 1} more metric(s) at this sample)" if len(rows) > 1 else ""
        items.append({
            "kind": "off_trend_timepoint",
            "id": f"OFF_TREND__{_slug(tp)}",
            "statement": _one_line(
                f"{tp}: {_num(w.get('value_pct'))}% {w.get('metric')} sits outside BOTH "
                f"neighbours' envelope ({w.get('prev_timepoint')}="
                f"{_num(w.get('prev_value_pct'))}%, {w.get('next_timepoint')}="
                f"{_num(w.get('next_value_pct'))}%) by "
                f"{_num(w.get('excursion_in_tolerances'))}x their combined counting "
                f"tolerance{extra}."
            ),
            # Terse by design: the reason a spike is not noise is stated once in the worklist
            # header, and repeating it per sample would cost five slots' worth of window.
            "why": f"Say explicitly whether {tp} is included or excluded, and why.",
            "must_mention": [tp],
            # ``timepoints`` is REQUIRED. The off-trend rule is evaluated on the discontinuity
            # table, which is why this was long carried as context — but the question an
            # off-trend sample poses is "spike or artefact", and the only numbers that bear on
            # it (events analyzed, retention, purity, acquisition drift) are in THIS table. So
            # it is the measurement behind the judgement the item actually demands, not
            # background to it. It still falls back to advisory when its columns do not
            # resolve: a demand this module cannot point at is the one that buys a ``.head()``.
            "must_read": [v for v in (disc_var, tp_var if tp_cols else None) if v],
            "should_read": [v for v in (tp_var if not tp_cols else None,) if v],
            "read_columns": {
                **({disc_var: disc_cols or list(OFF_TREND_COLUMNS)} if disc_var else {}),
                **({tp_var: list(tp_cols)} if tp_var and tp_cols else {}),
            },
            "how": [e for e in (
                _filter_expr(disc_var, "timepoint", tp, disc_cols),
                _filter_expr(tp_var, "timepoint", tp, tp_cols),
            ) if e],
        })
    return items


# The flags-table columns that carry a finding's numbers, as opposed to naming its row.
# ``measured`` holds the quantities the rule was evaluated on for EVERY code, so it is the
# one column that makes a flags-table read informative whatever the finding is.
FLAGS_MEASUREMENT_COLUMNS = ("measured", "rule", "exceedance")
_FLAGS_READ_COLUMNS = ("subject", "rule", "exceedance", "measured", "resolution_hint")


def _flags_read_line(flags_var: Optional[str], header: Sequence[str], code: str) -> Optional[str]:
    """The flags-table read for one code, projected to the columns carrying the numbers.

    ``measured`` is the only column that always holds the quantities the rule was evaluated
    on, whatever the code — so where a detail-table projection cannot be resolved, this line
    still puts the numbers in front of the reader.
    """
    expr = _filter_expr(flags_var, "code", code)
    if not expr:
        return None
    cols = [c for c in _FLAGS_READ_COLUMNS if c in (header or ())]
    # Double brackets: ``frame[['a', 'b']]`` selects columns, ``frame['a', 'b']`` raises.
    return f"{expr}[{cols!r}]" if cols else expr


def _measurement_columns(header: Sequence[str], keys: Sequence[str]) -> list[str]:
    """The resolved columns that carry MEASUREMENTS, with the identity columns removed.

    ``read_columns`` exists to be checked, and checking an identity column proves nothing: a
    reader who prints ``[['marker', 'timepoint']]`` has named the rows without reading any of
    them. So the enforceable set is the projection minus ``IDENTITY_COLUMNS``.
    """
    return [c for c in _resolve_columns(keys, header) if c not in IDENTITY_COLUMNS]


def _scope_token(subject: str, labels: Sequence[str]) -> Optional[str]:
    """The SAMPLE LABEL a flag's worst case sits at, if the run has a sample by that name.

    This is the one token a flag group can fairly demand, and the distinction is the whole
    argument. What ``must_mention`` may not gate on is the ITEM half of a subject — a
    requirement that ``%NK (of lymph)`` or ``NK cloud`` appear verbatim fails careful prose and
    rewards keyword stuffing, which is why flag groups carried no token at all. The SCOPE half
    is a different thing: a literal sample label, matched on word boundaries, and already the
    accepted token class for the off-trend items.

    The evidence for closing the gap: on the run this was written against, seventeen of
    twenty-three obligations had no token, and the trajectory that discharged all twenty-three
    — by printing the columns and saying nothing — was the one whose write-up carried the
    FEWEST caveats of the five. It had read the host-NK false-positive rate in a cell and
    reported nothing about it. Reading was enforced and reporting was not, so reading was what
    happened. Naming the worst case's sample is the minimum a reader must write to have
    addressed it rather than opened it.

    Returns None when the scope is not a sample — a control tube, a marker, a tier — because
    there is then no label to hold anyone to.
    """
    known = {str(x) for x in labels if x}
    if not known:
        return None
    item, scope = _split_subject(subject)
    for cand in (scope, item):
        # Tier-1 subjects carry the reference marker inline: "cd4 @ Baseline (reference)".
        cleaned = re.sub(r"\s*\(reference\)\s*$", "", str(cand or "")).strip()
        if cleaned in known:
            return cleaned
    return None


# Detail tables whose columns are named literally rather than resolved from the flags' own
# ``measured`` keys. Both entries have a comment at their definition explaining which failure
# mode they answer; keyed by ``write_all`` label so a table added later opts in explicitly.
_LITERAL_COLUMNS = {
    "timepoints": TIMEPOINT_COLUMNS,
    "gate_geometry": GATE_GEOMETRY_COLUMNS,
}


def _flag_group_items(
    book: FlagBook, var, headers: dict[str, tuple[str, ...]],
    escalated: Sequence[Flag] = (), labels: Sequence[str] = (),
) -> dict[str, list[dict[str, Any]]]:
    """One obligation per (tier, code) group, keyed by class.

    The group's own count is the point: an obligation that says "23 findings, you have seen
    one" is what turns a tally back into work.

    ``escalated`` are flags already promoted per-marker by ``_root_gate_items`` and
    ``_asserted_cutoff_items``. A group whose findings are ALL escalated is skipped, the same
    way ``TEMPORAL_DISCONTINUITY`` is: two worklist slots on one finding is the cost the cap
    exists to avoid. A group only partly escalated is kept, because its remaining findings are
    on other markers and unaddressed.

    ``labels`` are the run's sample labels, used to demand the worst case's sample be named in
    the answer — see ``_scope_token`` for why that is the one token this class may require.
    """
    pools: dict[str, list[dict[str, Any]]] = {}
    flags_var = var("flags")
    flags_header = headers.get(flags_var or "", ())
    tp_var = var("timepoints")
    done = {(f.code, f.subject) for f in escalated}
    for tier, code, fs in _groups(book):
        if code in SKIP_CODES:
            continue
        if all((f.code, f.subject) in done for f in fs):
            continue
        # TABLE_CLASS first: a code whose detail table has a class of its own is classified
        # by that table, not by the tier it shares with unrelated codes. See TABLE_CLASS.
        cls = (TABLE_CLASS.get(CODE_TABLE.get(code) or "")
               or TIER_CLASS.get(tier, "data_adequacy"))
        worst = fs[0]
        item, scope = _split_subject(worst.subject)
        detail_var = var(CODE_TABLE.get(code) or TIER_TABLE.get(tier) or "")
        detail_header = headers.get(detail_var or "", ())
        detail_read = _projection(detail_var, detail_header, _measured_keys(fs))
        detail_cols = _measurement_columns(detail_header, _measured_keys(fs))
        # ``timepoints`` is the one detail table whose flags report keys it does not hold:
        # LOW_PARENT_EVENTS measures ``n_parent`` on a GATE, while every column here measures
        # the SAMPLE. Resolution therefore fails for a reason that says nothing about whether
        # the table is worth requiring, so its columns are named literally instead — the same
        # escape OFF_TREND_COLUMNS takes for a finding built from report rows rather than a
        # Flag. Without this the table is unpromotable BY CONSTRUCTION, which is the mechanism
        # CLASS_CAPS' own comment names: a counted finding turned back into an uncounted one.
        detail_label = CODE_TABLE.get(code) or TIER_TABLE.get(tier) or ""
        literal = _LITERAL_COLUMNS.get(detail_label)
        # ``timepoints`` only when nothing resolved (its keys genuinely do not appear in the
        # table); ``gate_geometry`` always, because there resolution succeeds on the wrong
        # gate's column and on an event count. See each constant's own comment.
        if detail_var and literal and (detail_label == "gate_geometry" or not detail_cols):
            detail_read = _projection(detail_var, detail_header, literal) or detail_read
            detail_cols = _measurement_columns(detail_header, literal) or detail_cols
        how = [e for e in (
            _flags_read_line(flags_var, flags_header, code),
            f"{detail_read or detail_var}  # the measurements the rule was evaluated on"
            if detail_var else None,
        ) if e]
        # The flags table is an INDEX of findings; the detail table holds the measurements the
        # rule was actually evaluated on. Requiring only the index is what let a run discharge
        # fifteen items by printing one ``groupby('code').size()`` — a tally of the tally. So
        # the detail table is required too, but ONLY when its columns resolved: naming a table
        # whose measurement columns we cannot name is the demand that buys a ``.head()``, and
        # this module already learned that lesson once.
        promote = bool(detail_var and detail_var != flags_var and detail_cols)
        read_columns = {v: list(cols) for v, cols in (
            (flags_var, [c for c in FLAGS_MEASUREMENT_COLUMNS if c in (flags_header or ())]),
            (detail_var if promote else None, detail_cols),
        ) if v and cols}
        pools.setdefault(cls, []).append({
            "kind": cls,
            "id": f"FLAGS__{_slug(code)}",
            "statement": _one_line(
                f"[tier {tier}] {code}: {len(fs)} finding(s); you have seen 1 in the digest. "
                f"Worst is {worst.subject or item or '—'} — rule: {worst.rule}"
            ),
            "why": _one_line(worst.resolution_hint, 260),
            # The worst case's SAMPLE, when the run has one by that name. Not the item — see
            # ``_scope_token`` for why that distinction is what makes this fair to require.
            "must_mention": [t for t in (_scope_token(worst.subject, labels),) if t],
            "must_read": [v for v in (flags_var, detail_var if promote else None) if v],
            "should_read": [v for v in (detail_var,) if v and v != flags_var and not promote],
            "read_columns": read_columns,
            "how": how,
            "n_findings": len(fs),
            "code": code,
            "tier": tier,
        })
    return pools


_MARKER_LABEL = {m.key: m.label for m in MARKERS}
_MARKER_DOWNSTREAM = {m.key: m.downstream for m in MARKERS}


def _mention_token(marker: str) -> Optional[str]:
    """A word a reader would actually write for ``marker``, or None if there is none.

    ``must_mention`` is matched on word boundaries, so a two-letter key like ``ld`` is both
    unenforceable in prose (nobody writes "ld" as a word; they write "viability" or "L/D")
    and dangerous if it ever did match. Prefer the longest alphabetic run in the pipeline's
    own display label — ``Viability(L/D)`` -> ``viability``, ``CD45`` -> ``cd45`` — and give
    up rather than demand a token that cannot be satisfied honestly.
    """
    label = _MARKER_LABEL.get(marker, marker)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(label))
    best = max(words, key=len, default="")
    for cand in (best, marker):
        if len(cand) >= 3:
            return cand.lower()
    return None


def _asserted_token(marker: str) -> Optional[str]:
    """The word a reader writes for a cutoff they are discussing BY NAME.

    Differs from ``_mention_token`` in which name it prefers. That one wants the population a
    marker identifies, so ``ld`` becomes "viability" — right for a root gate, because the
    write-up talks about the viability gate rather than about "ld". This one wants the marker
    as it appears in the cutoff's own name, so ``hla`` stays "hla": a reader addressing the
    donor cutoff writes "the HLA cutoff", and ``_mention_token`` would have demanded "donor"
    off the label ``HLA (Donor)`` — a word every one of these write-ups already contains, which
    is a requirement that cannot fail and therefore cannot help.
    """
    key = str(marker or "").strip().lower()
    return key if len(key) >= 3 else _mention_token(key)


def _asserted_cutoff_items(
    book: FlagBook, var, headers: dict[str, tuple[str, ...]],
) -> tuple[list[dict[str, Any]], list[Flag]]:
    """One FORCED obligation per MARKER whose cutoff tier 1 cannot judge.

    Returns the items and the flags they cover, so ``_flag_group_items`` can skip a group it
    has fully absorbed — same contract as ``_root_gate_items``.

    These bypass the caps on the same argument root gates do, and the argument is if anything
    stronger: a root gate at least HAS placement measurements to argue with. Here the finding
    is that the measurements do not exist or do not apply, which is the one thing a reader
    cannot discover by reading the table — an absent row and a clean row look identical.

    Per MARKER and not per (marker, code), which is where this differs from
    ``_root_gate_items``. Both codes make the SAME request of the reader — judge this cutoff
    from tiers 4 and 5, because tier 1 cannot — so on the CAR cutoff, which fires both, two
    items would spend two slots and two paragraphs of a bounded worklist on one decision. A
    root gate's two codes are genuinely different findings (no valley versus a fallback
    percentile) and stay separate.
    """
    items: list[dict[str, Any]] = []
    covered: list[Flag] = []
    flags_var = var("flags")
    flags_header = headers.get(flags_var or "", ())
    audit_var = var("cutoff_audit")
    audit_header = headers.get(audit_var or "", ())
    controls_var = var("controls")
    by_marker: dict[str, list[tuple[int, str, Flag]]] = {}
    for tier, code, fs in _groups(book):
        if code not in ASSERTED_CUTOFF_CODES:
            continue
        for f in fs:
            marker, _scope = _split_subject(f.subject)
            covered.append(f)
            by_marker.setdefault(marker.strip().lower(), []).append((tier, code, f))
    for key, found in by_marker.items():
        tier = min(t for t, _c, _f in found)
        token = _asserted_token(key)
        downstream = [m for m in (_MARKER_DOWNSTREAM.get(key) or ()) if m in HEADLINE_METRICS]
        keys = _measured_keys([f for _t, _c, f in found])
        detail_read = _projection(audit_var, audit_header, keys)
        audit_cols = _measurement_columns(audit_header, keys)
        items.append({
            "kind": "cutoff_foundation",
            "id": f"ASSERTED_CUTOFF__{_slug(key)}",
            "statement": _one_line(
                f"[tier {tier}] tier 1 CANNOT judge the {_MARKER_LABEL.get(key, key)} cutoff "
                f"from its placement"
                + (f", and {len(downstream)} headline metric(s) depend on it "
                   f"({', '.join(downstream)})" if downstream else "")
                + ". " + "; ".join(f"{c} — {f.rule}" for _t, c, f in found),
                340,
            ),
            # The flags' own hints, verbatim — same discipline as the root-gate items. Both
            # codes already point the reader at tiers 4 and 5, so there is nothing this module
            # needs to add and nothing it may.
            "why": _one_line(" ".join(f.resolution_hint for _t, _c, f in found), 300),
            "must_mention": [t for t in (token,) if t],
            "must_read": [v for v in (audit_var,) if v],
            # The internal controls are where a cutoff like this is actually checkable, so they
            # are offered rather than required: tier 4 may itself be unavailable, and requiring
            # a table that has nothing in it for this marker buys a ``.head()``.
            "should_read": [v for v in (controls_var, flags_var) if v and v != audit_var],
            "read_columns": {audit_var: audit_cols} if audit_var and audit_cols else {},
            "how": [e for e in (
                f"{detail_read}  # the measurements the rule was evaluated on"
                if detail_read else _filter_expr(audit_var, "marker", key),
                _filter_expr(audit_var, "marker", key) if detail_read else None,
            ) + tuple(
                _flags_read_line(flags_var, flags_header, c)
                for c in dict.fromkeys(c for _t, c, _f in found)
            ) if e],
            "n_findings": len(found),
            "code": ",".join(dict.fromkeys(c for _t, c, _f in found)),
            "tier": tier,
            "marker": key,
            # Bypasses CLASS_CAPS and the global cap. See the docstring.
            "forced": True,
        })
    return items, covered


def _root_gate_items(
    book: FlagBook, var, headers: dict[str, tuple[str, ...]],
) -> tuple[list[dict[str, Any]], list[Flag]]:
    """One FORCED obligation per (root-gate marker, foundation code) that fired.

    Returns the items and the flags they cover, so ``_flag_group_items`` can skip a group it
    has fully absorbed.

    These bypass the class cap. Everything else in this module competes for a slot on the
    argument that a list must stay finishable; this class does not, because the finding is
    about the denominator of every headline number rather than about one sample or one
    marker among nine. There are at most ``len(ROOT_GATE_KEYS) * len(ROOT_GATE_CODES)`` of
    them and in practice one or two, so the list stays finishable anyway.

    Per MARKER, not per code: the group obligation names only its worst subject, so a
    ``NO_VALLEY_EVIDENCE`` group whose worst case is CD19 never puts the word "viability" in
    front of the reader even though the same group holds that finding too.
    """
    items: list[dict[str, Any]] = []
    covered: list[Flag] = []
    flags_var = var("flags")
    flags_header = headers.get(flags_var or "", ())
    audit_var = var("cutoff_audit")
    audit_header = headers.get(audit_var or "", ())
    for tier, code, fs in _groups(book):
        if code not in ROOT_GATE_CODES:
            continue
        for f in fs:
            marker, scope = _split_subject(f.subject)
            key = marker.strip().lower()
            if key not in ROOT_GATE_KEYS:
                continue
            covered.append(f)
            token = _mention_token(key)
            downstream = _MARKER_DOWNSTREAM.get(key) or ()
            detail_read = _projection(audit_var, audit_header, _measured_keys([f]))
            audit_cols = _measurement_columns(audit_header, _measured_keys([f]))
            row_read = _filter_expr(audit_var, "marker", key)
            items.append({
                "kind": "cutoff_foundation",
                "id": f"ROOT_GATE__{_slug(code)}__{_slug(key)}",
                "statement": _one_line(
                    f"[tier {tier}] {code} on {_MARKER_LABEL.get(key, key)} — a ROOT GATE: "
                    f"every headline lineage number is measured inside it "
                    f"({len(downstream)} reported metric(s) move with this cutoff, including "
                    f"{', '.join(downstream[:3])}). "
                    f"{f.subject or key} — rule: {f.rule}",
                    380,
                ),
                # The flag's own hint, verbatim. The root-gate framing — that this cutoff is
                # upstream of every reported population — belongs in ``statement``, where it
                # is derived from ``MarkerSpec.downstream``, and is there. Appending a
                # rationale of this module's own invention here is the one thing ``why`` may
                # not do, however true the sentence would be.
                "why": _one_line(f.resolution_hint, 260),
                # The marker itself, so an answer that discusses the cutoffs generically
                # without ever naming this one does not discharge the item.
                "must_mention": [t for t in (token,) if t],
                "must_read": [v for v in (audit_var,) if v],
                "should_read": [v for v in (flags_var,) if v and v != audit_var],
                # This is the item whose defeat is quoted in the column-projection note above:
                # the audit table WAS opened, and the two columns carrying the finding were
                # among the thirty-four projected away.
                "read_columns": {audit_var: audit_cols} if audit_var and audit_cols else {},
                "how": [e for e in (
                    f"{detail_read}  # the measurements the rule was evaluated on"
                    if detail_read else row_read,
                    row_read if detail_read else None,
                    _flags_read_line(flags_var, flags_header, code),
                ) if e],
                "n_findings": 1,
                "code": code,
                "tier": tier,
                "marker": key,
                # Bypasses CLASS_CAPS and the global cap. See the docstring.
                "forced": True,
            })
    return items, covered


def _concordance_items(report: dict[str, Any], var) -> list[dict[str, Any]]:
    """Tier 5 being unavailable is itself a finding: nothing external checked the gating."""
    t5 = report.get("tier5_operator_concordance") or {}
    if t5.get("available") is not False:
        return []
    controls_var = var("controls")
    return [{
        "kind": "operator_concordance",
        "id": "NO_OPERATOR_CONCORDANCE",
        "statement": _one_line(
            "Operator concordance was not measured: "
            + str(t5.get("reason") or "no manual-gating comparison table was produced.")
        ),
        "why": (
            "With no manual gating to compare against, the internal controls are the only "
            "external check on this gating. Say that the agreement is unmeasured rather than "
            "leaving its absence to be read as agreement."
        ),
        "must_mention": [],
        "must_read": [v for v in (controls_var,) if v],
        "should_read": [],
        "how": [f"{controls_var}" if controls_var else
                "cutoff_diagnostics.json -> ['tier5_operator_concordance']"],
    }]


def coverage_gap_markers(coverage: Optional[dict[str, Any]]) -> list[tuple[str, int]]:
    """``[(marker, n_dropped), ...]``, most-dropped first."""
    counts: dict[str, int] = {}
    for d in (coverage or {}).get("dropped") or []:
        key = str(d.get("marker") or "—")
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def describe_coverage_gap(coverage: Optional[dict[str, Any]]) -> str:
    """One line naming WHICH sweeps were skipped and which reported numbers that costs.

    The old line said only how many: "12 marker x timepoint sensitivity sweep(s) were NOT run
    (see cutoff_diagnostics.json)". Five trajectories discharged that item and not one of them
    learned that all twelve were the CAR marker — so none of them learned that the
    cutoff-attributable range on ``%CAR+ (of Donor NK)`` was computed by moving CD14, CD45 and
    the viability gate while the CAR cutoff itself was never perturbed. The count was honest
    and told the reader nothing they could act on.

    Naming the markers is enough to fix that, because ``MarkerSpec.downstream`` already knows
    which reported metrics each cutoff moves: the affected headline numbers are derived, not
    listed here.
    """
    dropped = (coverage or {}).get("dropped") or []
    if not dropped:
        return ""
    n = len(dropped)
    ranked = coverage_gap_markers(coverage)
    if len(ranked) == 1:
        who = f"all {n} are marker {ranked[0][0]!r}"
    else:
        who = "markers: " + ", ".join(f"{k} x{v}" for k, v in ranked[:4])
        if len(ranked) > 4:
            who += f", +{len(ranked) - 4} more"
    reasons = [str(d.get("reason") or "") for d in dropped if d.get("reason")]
    why = (coverage or {}).get("reason") or (reasons[0] if reasons else "")
    metrics = sorted({
        m for k, _ in ranked
        for m in (_MARKER_DOWNSTREAM.get(k) or ())
        if m in HEADLINE_METRICS
    })
    parts = [f"{n} marker x timepoint sensitivity sweep(s) were NOT run — {who}"]
    if why:
        parts.append(f"({_one_line(why, 110)})")
    line = " ".join(parts) + "."
    if metrics:
        line += (f" So {', '.join(metrics)} carr{'ies' if len(metrics) == 1 else 'y'} no range "
                 "attributable to the un-swept cutoff(s) above — any 'cutoff' uncertainty "
                 "reported for them comes from moving OTHER cutoffs.")
    return line


"""The uncertainty columns that say WHICH cutoff drove each reported range — so a reader can
see the one that never appears. Named literally: the coverage item is built from the tier-3
report rather than from a Flag, so there are no ``measured`` keys to resolve."""
COVERAGE_COLUMNS = ("cutoff_range_pp", "cutoff_range_driver_low", "cutoff_range_driver_high")


def _coverage_items(report: dict[str, Any], var,
                    headers: Optional[dict[str, tuple[str, ...]]] = None
                    ) -> list[dict[str, Any]]:
    """Work the sensitivity tier did not do. Named, because silence would read as a pass."""
    cov = ((report.get("tier3_sensitivity") or {}).get("coverage")) or {}
    dropped = cov.get("dropped") or []
    if not dropped:
        return []
    # Discharged by opening the uncertainty table: that is where the ranges which WERE
    # measured live, so it is also where their absence for these pairs is visible. Without a
    # condition like this the item would be unenforceable — see ``advisory`` in ``build``.
    unc_var = var("uncertainty")
    unc_header = (headers or {}).get(unc_var or "", ())
    unc_cols = _measurement_columns(unc_header, COVERAGE_COLUMNS)
    unc_read = _projection(unc_var, unc_header, COVERAGE_COLUMNS) or unc_var
    ranked = coverage_gap_markers(cov)
    # A token only when ONE marker accounts for every skipped sweep. Then the marker's name is
    # a word the write-up has to use to state the gap honestly, and demanding it is the
    # difference between the item being discharged and the gap being reported — five
    # trajectories discharged the old version and none of them mentioned it. With the drops
    # spread over several markers there is no single word that would be fair to require.
    token = _asserted_token(ranked[0][0]) if len(ranked) == 1 else None
    return [{
        "kind": "coverage_gap",
        "id": "SENSITIVITY_COVERAGE_GAP",
        "statement": _one_line(describe_coverage_gap(cov), 300),
        "why": (
            "These pairs carry no cutoff-attributable range, so a number that looks "
            "unqualified here is unmeasured rather than robust. Do not read the absence of a "
            "sensitivity result as a small one — and where the un-swept cutoff is the one that "
            "defines the metric, the 'cutoff' uncertainty reported for that metric was "
            "produced by moving different cutoffs entirely."
        ),
        "must_mention": [t for t in (token,) if t],
        "must_read": [v for v in (unc_var,) if v],
        "should_read": [],
        # Opening the uncertainty table is not enough: its ``cutoff_range_pp`` column looks
        # like a complete accounting until you read the DRIVER columns and notice which cutoff
        # never appears in them.
        "read_columns": ({unc_var: unc_cols} if unc_var and unc_cols else {}),
        "how": [e for e in (
            "cutoff_diagnostics.json -> ['tier3_sensitivity']['coverage']['dropped']",
            f"{unc_read}  # which cutoff actually drove each range — and which never appears"
            if unc_read else None,
        ) if e],
    }]


# ── assembly ──────────────────────────────────────────────────────────────────
def build(
    report: dict[str, Any],
    book: FlagBook,
    *,
    written: Optional[dict[str, str]] = None,
    limit: int = MAX_OBLIGATIONS,
) -> dict[str, Any]:
    """Derive the worklist manifest.

    ``written`` is ``write_all``'s ``{label: path}`` map. It is what makes ``must_read`` and
    ``how`` honest: a table that was not written (no rows, so no file) is dropped from both
    rather than named as somewhere to look.
    """
    written = written or {}

    def var(label: str) -> Optional[str]:
        """The notebook variable for a table label — its CSV stem — or None if unwritten."""
        path = written.get(label)
        if not path:
            return None
        stem = Path(str(path)).stem
        return stem if stem.isidentifier() else None

    headers = _headers(written)
    # The run's own sample labels, so a flag group can require its worst case's sample be
    # named in the answer without inventing a token. See ``_scope_token``.
    labels = [str(x) for x in ((report.get("configuration") or {}).get("timepoints") or [])]

    # Derived first so the flag-group pass can skip a group it fully absorbs.
    root_items, root_covered = _root_gate_items(book, var, headers)
    asserted_items, asserted_covered = _asserted_cutoff_items(book, var, headers)

    pools: dict[str, list[dict[str, Any]]] = {c: [] for c in CLASS_ORDER}
    pools["reproduction"].extend(_reproduction_items(report))
    pools["off_trend_timepoint"].extend(_off_trend_items(report, var, headers))
    pools["cutoff_foundation"].extend(root_items + asserted_items)
    for cls, items in _flag_group_items(
        book, var, headers, list(root_covered) + list(asserted_covered), labels
    ).items():
        pools.setdefault(cls, []).extend(items)
    pools["operator_concordance"].extend(_concordance_items(report, var))
    pools["coverage_gap"].extend(_coverage_items(report, var, headers))

    promoted: list[dict[str, Any]] = []
    n_candidates = 0
    n_dropped = 0
    for cls in CLASS_ORDER:
        pool = pools.get(cls) or []
        n_candidates += len(pool)
        cap = CLASS_CAPS.get(cls, 1)
        # Forced items are promoted whole and do not consume the class cap. A cap is an
        # argument about the reader's budget; it is not an argument for dropping a tier-1
        # finding on the cutoff every headline number is measured against.
        forced = [it for it in pool if it.get("forced")]
        rest = [it for it in pool if not it.get("forced")]
        promoted.extend(forced + rest[:cap])
        n_dropped += max(0, len(rest) - cap)
    # Any class not named in CLASS_ORDER would be dropped silently, which is exactly what
    # this module refuses to do — count it.
    for cls, pool in pools.items():
        if cls not in CLASS_ORDER:
            n_candidates += len(pool)
            n_dropped += len(pool)

    # The global cap is a backstop on the CAPPED classes; it never evicts a forced item, or
    # the class-cap bypass above would be undone by the next line. Order is preserved so the
    # worklist still reads in class order.
    before_global = len(promoted)
    budget = max(0, int(limit)) - sum(1 for it in promoted if it.get("forced"))
    kept: list[dict[str, Any]] = []
    for it in promoted:
        if it.get("forced"):
            kept.append(it)
        elif budget > 0:
            kept.append(it)
            budget -= 1
    promoted = kept
    n_dropped += before_global - len(promoted)

    for i, item in enumerate(promoted, 1):
        item["priority"] = i
        # An item with neither a table to open nor a token to write cannot be checked, so a
        # reader who ignored it would still be recorded as having discharged it. Mark it
        # ADVISORY instead: it stays in the worklist, where it is genuinely useful, and is
        # excluded from any enforced total rather than inflating one. This normally happens
        # only when the table an item points at was not written for this run.
        item["advisory"] = not (item["must_read"] or item["must_mention"])

    note = (
        f"{n_dropped} of {n_candidates} candidate obligation(s) were NOT promoted — the list "
        f"is capped per class and at {limit} overall so that it stays short enough to finish. "
        f"Not promoted means NOT ADDRESSED, never 'nothing there': the rows are in "
        f"diagnostics_flags.csv and cutoff_diagnostics.json."
    ) if n_dropped else "Every candidate obligation was promoted; none were held back."

    # The full table inventory, so a reader of this manifest can report coverage across
    # EVERY table this run produced and not merely across the ones an obligation named.
    # "three of eleven tables opened" is the statistic that exposed the problem this module
    # exists to fix; it cannot be computed from the obligations alone.
    tables = sorted({
        Path(str(p)).stem for p in written.values()
        if str(p).endswith(".csv") and Path(str(p)).stem.isidentifier()
    })

    return {
        "schema": SCHEMA,
        "generated_by": "anchored.diagnostics",
        "tables": tables,
        "cap": int(limit),
        "n_candidates": n_candidates,
        "n_promoted": len(promoted),
        "n_not_promoted": n_dropped,
        # Promoted but uncheckable (see ``advisory`` above). Stated so a reader of this
        # manifest can tell "nothing was left undischarged" from "nothing could be checked".
        "n_advisory": sum(1 for it in promoted if it["advisory"]),
        # Promoted past the caps because the finding is on a root gate. Stated so a reader can
        # tell a list shaped by priority from one shaped by what happened to fit.
        "n_forced": sum(1 for it in promoted if it.get("forced")),
        "not_promoted_note": note,
        "discipline": (
            "An obligation is a duty to READ a measurement that already exists, not a verdict "
            "on it. Discharging one means reading the rows named in 'how' and saying what you "
            "concluded — including 'this does not change the answer', which is a valid "
            "outcome. Leaving one unread and unmentioned is not."
        ),
        "items": promoted,
    }


# The render's own bound, exported so the print budget in ``firstrun`` can be checked against
# it rather than against a literal that can drift out of step with it.
WORKLIST_MAX_CHARS = 15000


def render_worklist(manifest: dict[str, Any], *, max_chars: int = WORKLIST_MAX_CHARS) -> str:
    """The worklist as text, for printing into the agent's opening observation.

    Bounded like the digest is: if the items would overrun, later ones are replaced by a line
    that counts them. Their existence and count always survive.

    Detail is given up BEFORE items are. ``why`` is the flag's resolution hint — useful, but
    secondary to ``statement`` (the measurement) and ``read:`` (the route to its rows), which
    are the irreducible content of an obligation. Dropping the hint from every item costs the
    reader a sentence each; dropping the item costs them the finding, and a finding that
    reaches only ``diagnostics_obligations.json`` is one the raised cap did not actually
    deliver. So the render tries full detail, then compact, and only then starts counting
    items off the end.
    """
    items = (manifest or {}).get("items") or []
    n_more = int((manifest or {}).get("n_not_promoted") or 0)
    head = [
        "WORKLIST — the obligations this run leaves you, in priority order.",
        "  Each item names measurements that ALREADY EXIST in the tables above and that the",
        "  digest only COUNTED. Run EVERY `read:` line an item lists — they are different",
        "  measurements, not alternatives, and the columns are named because printing the",
        "  table without them settles nothing. An `also:` line is context you may skip.",
        "  Discharge an item by running its `read:` lines and saying what you",
        "  concluded in your answer — 'this does not change the result' is a valid conclusion;",
        "  silence is not. A conclusion that holds only because a contradicting measurement",
        "  went unread and unmentioned is worse than no conclusion. These are duties to read,",
        "  not verdicts: nothing here says a cutoff is wrong.",
        "  An off-trend sample is a value outside the envelope of BOTH its temporal neighbours",
        "  by more than their combined counting intervals — a spike, not a step, so sampling",
        "  noise does not explain it and something about that sample differs.",
    ]
    if not items:
        head.append("")
        head.append("  (0 items — no rule fired and no sample went off-trend. That is not the "
                    "same as 'the cutoffs are correct'.)")
        return "\n".join(head) + "\n"

    def build_blocks(with_why: bool) -> list[list[str]]:
        return [_block(it, with_why=with_why) for it in items]

    def _block(it: dict[str, Any], *, with_why: bool) -> list[str]:
        b = [f"  [{it.get('priority')}] {it.get('id')}", f"      {it.get('statement')}"]
        if with_why and it.get("why"):
            b.append(f"      why: {it['why']}")
        # ``read:`` is what settles the item; ``also:`` is context. Labelling the lines
        # rather than adding a separate "settled by" line keeps the distinction visible
        # without spending a line per item — the render shares a bounded window with the
        # digest. A reader who believes every listed table is mandatory batches them all
        # into one throwaway cell instead of reading the one that matters.
        must, should = it.get("must_read") or [], it.get("should_read") or []
        for expr in it.get("how") or []:
            supporting = (any(t in expr for t in should)
                          and not any(t in expr for t in must))
            b.append(f"      {'also' if supporting else 'read'}: {expr}")
        if it.get("must_mention"):
            b.append(f"      name in your answer: {', '.join(it['must_mention'])}")
        if it.get("advisory"):
            b.append("      (advisory: no table of its own to open — carry it into the "
                     "write-up as a caveat)")
        return b

    def assemble(blocks: list[list[str]], keep: int, *, compact: bool) -> str:
        body: list[str] = [""]
        for b in blocks[:keep]:
            body.extend(b)
        hidden = len(blocks) - keep
        if hidden > 0:
            body.append(f"  [+{hidden} more obligation(s) not shown here to stay inside this "
                        f"window — all of them are in {OBLIGATIONS_JSON}]")
        tail = [""]
        if compact:
            # Said plainly: an item rendered without its hint is not an item with no hint.
            tail.append(f"  [the 'why' line — each flag's own resolution hint — is omitted "
                        f"above to fit every obligation in this window; all of them are in "
                        f"{OBLIGATIONS_JSON}]")
        if n_more:
            tail.append(f"  {manifest.get('not_promoted_note')}")
        return "\n".join(head + body + tail) + "\n"

    # Full detail for every item, if it fits.
    full = build_blocks(with_why=True)
    text = assemble(full, len(full), compact=False)
    if len(text) <= max_chars:
        return text
    # Then every item without its hint, which is what the raised cap is for.
    compact = build_blocks(with_why=False)
    text = assemble(compact, len(compact), compact=True)
    if len(text) <= max_chars:
        return text
    # Only then start counting items off the end — the most detail at the most items.
    for keep in range(len(full), 0, -1):
        text = assemble(full, keep, compact=False)
        if len(text) <= max_chars:
            return text
    for keep in range(len(compact), 0, -1):
        text = assemble(compact, keep, compact=True)
        if len(text) <= max_chars:
            return text
    return assemble(compact, 1, compact=True)
