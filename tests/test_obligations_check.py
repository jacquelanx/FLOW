"""The submission check: does an unread obligation actually cost something, and safely?

The failure being corrected is a run that reads a summary counting 375 findings, opens three
of eleven tables, and submits. Guidance alone did not fix it, so the environment now refuses
such a submission — which means these tests have to guard the refusal in BOTH directions:

  * it must bite (an unread obligation blocks the first submission, with the specific list);
  * it must let go (a hard ceiling on refusals, and never a refusal so late in the step
    budget that the trajectory ends with no answer at all);
  * it must stay generic (no manifest ⇒ no enforcement, so every non-anchored project and
    every existing test path behaves exactly as before);
  * it must record what it let through (the ceiling is a concession, not an erasure).

The last one carries the most weight. A refusal that silently gives up would turn "the answer
skipped half its evidence" into "the answer looked fine", which is worse than not checking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow.aviary.tool import ToolCall
from flow.env.notebook_env import (MAX_SUBMIT_REFUSALS, MIN_STEPS_LEFT_TO_REFUSE,
                                   REFUSAL_ITEMS_SHOWN)

ANSWER = "NK% rises from 29.8% at Baseline to 90.3% by D28."


def _manifest(tmp_path: Path, items: list[dict], tables: list[str] | None = None) -> Path:
    p = tmp_path / "diagnostics_obligations.json"
    p.write_text(json.dumps({
        "schema": "flow_diagnostics_obligations/v1",
        "tables": tables if tables is not None else ["diagnostics_flags", "diagnostics_transfer"],
        "items": items,
    }))
    return p


def _item(oid: str, *, must_read=(), must_mention=(), statement="a measurement", how=()) -> dict:
    return {"id": oid, "statement": statement, "how": list(how),
            "must_read": list(must_read), "must_mention": list(must_mention)}


def _seeded(make_env, manifest: Path | None, **kw):
    """An env with one read-only first-run cell, so ``n_protected_cells`` is realistic."""
    env = make_env(seed_cells=["first_run_tables = {'diagnostics_flags': 1}"],
                   obligations_file=manifest, **kw)
    env.reset()
    return env


def _write_cell(env, source: str):
    return env.step(ToolCall(name="edit_cell",
                             arguments={"index": "new", "source": source, "execute": False}))


def _submit(env, answer: str = ANSWER):
    return env.step(ToolCall(name="submit_answer", arguments={"answer": answer}))


# ── it bites ──────────────────────────────────────────────────────────────────
def test_an_unopened_table_refuses_the_first_submission(make_env, tmp_path):
    m = _manifest(tmp_path, [_item("FLAGS__DRIFT", must_read=["diagnostics_transfer"])])
    env = _seeded(make_env, m)
    obs, _reward, done, _info = _submit(env)
    assert done is False, "the trajectory must continue so the agent can fix it"
    assert env.answer is None, "a refused submission must not be recorded as the answer"
    assert "ANSWER NOT ACCEPTED" in obs
    assert "diagnostics_transfer" in obs


def test_an_unnamed_sample_refuses_even_when_the_table_was_read(make_env, tmp_path):
    """Reading the rows and then saying nothing about them is the failure, not a fix."""
    m = _manifest(tmp_path, [_item("OFF_TREND__D14", must_read=["diagnostics_discontinuity"],
                                   must_mention=["D14"])])
    env = _seeded(make_env, m)
    _write_cell(env, "print(diagnostics_discontinuity)")
    obs, _r, done, _i = _submit(env)
    assert done is False
    assert "STILL NEEDED — name in your answer" in obs
    assert "D14" in obs


def test_the_refusal_names_the_expression_that_discharges_the_item(make_env, tmp_path):
    """A refusal with no route to compliance is just an obstacle."""
    how = "diagnostics_flags[diagnostics_flags.code == 'NEGATIVE_MODE_DRIFT']"
    m = _manifest(tmp_path, [_item("FLAGS__DRIFT", must_read=["diagnostics_flags"], how=[how])])
    env = _seeded(make_env, m)
    obs, *_ = _submit(env)
    assert how in obs


def test_the_refusal_says_that_no_change_is_a_valid_conclusion(make_env, tmp_path):
    """Otherwise the check reads as a demand to find something, which invites invention."""
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    obs, *_ = _submit(_seeded(make_env, m))
    assert "does not change the result" in obs


def test_reading_and_naming_lets_the_same_answer_through(make_env, tmp_path):
    m = _manifest(tmp_path, [_item("OFF_TREND__D14", must_read=["diagnostics_discontinuity"],
                                   must_mention=["D14"])])
    env = _seeded(make_env, m)
    _write_cell(env, "print(diagnostics_discontinuity.head())")
    obs, reward, done, _i = _submit(env, ANSWER + " D14 is excluded as off-trend.")
    assert done is True and reward == 1.0
    assert env.answer is not None
    assert "Answer submitted" in obs
    assert env.evidence_coverage()["obligations_undischarged"] == 0


def test_a_named_token_matches_regardless_of_case(make_env, tmp_path):
    """'Pre' in a manifest must not force the write-up to capitalise mid-sentence."""
    m = _manifest(tmp_path, [_item("OFF_TREND__PRE", must_mention=["Pre"])])
    env = _seeded(make_env, m)
    _obs, _r, done, _i = _submit(env, "The pre timepoint is excluded.")
    assert done is True


# ── naming a sample vs. merely containing its letters ─────────────────────────
@pytest.mark.parametrize("token,answer,names_it", [
    # The exact production failure: 'Pre' satisfied by the word "preventing".
    ("Pre", "The metadata lacks HLA-A3, preventing discrimination of donor cells.", False),
    ("Pre", "These fragile metrics should be interpreted with their ranges.", False),
    ("D14", "Diagnostics flagged cd14 and cd45 with shallow troughs.", False),
    ("D7", "The cd7 channel is unused.", False),
    ("Post", "The composite gate is unstable.", False),
    # Hyphens and underscores are word breaks, so these DO name the sample.
    ("Post", "NK% fell to near zero post-conditioning, then recovered by D16.", True),
    ("Pre", "The pre-infusion sample is the anchor.", True),
    ("Pre", "Excluding Pre, the trend is monotone.", True),
    ("D7", "(pre, post, d7) were off-trend.", True),
    ("week8", "Sustained at 85% at week8.", True),
    ("D35 Pleural Fluid", "D35 Pleural Fluid is excluded (different compartment).", True),
])
def test_a_token_must_be_NAMED_not_merely_contained(make_env, tmp_path, token, answer,
                                                    names_it):
    m = _manifest(tmp_path, [_item("OFF_TREND", must_mention=[token])])
    env = _seeded(make_env, m)
    _obs, _r, done, _i = _submit(env, answer)
    assert done is names_it, f"{token!r} vs {answer!r}"


def test_the_exact_production_false_discharge_is_now_caught(make_env, tmp_path):
    """Regression: run 2215c35e345f traj 0 discharged OFF_TREND__PRE via "preventing"."""
    answer = (
        "The metadata lacks a Donor HLA marker (HLA-A3), preventing the discrimination of "
        "Donor NK cells from Patient NK cells. Several timepoints (e.g., D7, D14, Post, "
        "D35 Pleural Fluid) show discontinuous, off-trend shifts."
    )
    m = _manifest(tmp_path, [
        _item("OFF_TREND__PRE", must_mention=["Pre"]),
        _item("OFF_TREND__D14", must_mention=["D14"]),
        _item("OFF_TREND__POST", must_mention=["Post"]),
    ])
    env = _seeded(make_env, m)
    obs, _r, done, _i = _submit(env, answer)
    assert done is False, "Pre is never named in this answer"
    assert "OFF_TREND__PRE" in obs
    # The two samples the answer really does name must NOT be re-demanded.
    assert "OFF_TREND__D14" not in obs and "OFF_TREND__POST" not in obs
    assert "2 of 3 obligation(s)" in obs and "1 remain" in obs


# ── supporting tables inform, they do not block ───────────────────────────────
def test_a_supporting_table_is_surfaced_but_never_blocks(make_env, tmp_path):
    """Requiring every named table at once produced a throwaway batch-read cell."""
    item = _item("OFF_TREND__D14", must_read=["diagnostics_discontinuity"],
                 must_mention=["D14"])
    item["should_read"] = ["diagnostics_timepoints"]
    m = _manifest(tmp_path, [item])
    env = _seeded(make_env, m)
    _write_cell(env, "print(diagnostics_discontinuity[diagnostics_discontinuity.discontinuous])")
    _obs, reward, done, _i = _submit(env, ANSWER + " D14 is excluded as off-trend.")
    assert done is True and reward == 1.0, "the primary read plus the naming must suffice"


def test_a_refusal_reports_progress_rather_than_a_flat_count(make_env, tmp_path):
    """'15 of 15 undischarged' told a run that none of its reading counted."""
    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"]),
                             _item("B", must_read=["diagnostics_transfer"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags.head())")
    obs, _r, done, _i = _submit(env)
    assert done is False
    assert "1 of 2 obligation(s)" in obs and "1 remain" in obs
    # Only the outstanding item is re-listed; ids are printed one per line.
    listed = {line.strip() for line in obs.splitlines()}
    assert "B" in listed and "A" not in listed
    assert "diagnostics_transfer" in obs


def test_a_partly_done_item_shows_what_it_already_has(make_env, tmp_path):
    """A half-satisfied item that reads as untouched invites starting over, not finishing."""
    m = _manifest(tmp_path, [_item("OFF_TREND__D14", must_read=["diagnostics_discontinuity"],
                                   must_mention=["D14"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_discontinuity)")
    obs, *_ = _submit(env)
    assert "already opened: diagnostics_discontinuity" in obs
    assert "STILL NEEDED — name in your answer: D14" in obs


def test_the_refusal_marks_optional_context_as_optional(make_env, tmp_path):
    item = _item("A", must_read=["diagnostics_flags"])
    item["should_read"] = ["diagnostics_transfer"]
    m = _manifest(tmp_path, [item])
    obs, *_ = _submit(_seeded(make_env, m))
    assert "optional context, not required: diagnostics_transfer" in obs


def test_the_first_run_cells_do_not_discharge_anything_by_themselves(make_env, tmp_path):
    """The seed cells name every table by construction; counting them would void the check."""
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    env = make_env(seed_cells=["first_run_tables = {}  # diagnostics_flags loaded here"],
                   obligations_file=m)
    env.reset()
    obs, _r, done, _i = _submit(env)
    assert done is False, "a table named only in the read-only first-run cell is not read"
    assert "diagnostics_flags" in obs


# ── it lets go ────────────────────────────────────────────────────────────────
def test_the_answer_is_accepted_once_the_refusal_ceiling_is_reached(make_env, tmp_path):
    """A check that never yields trades a thin answer for no answer, which is worse."""
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=40)
    for i in range(MAX_SUBMIT_REFUSALS):
        _obs, _r, done, _i = _submit(env)
        assert done is False, f"refusal {i + 1} should not end the run"
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0
    assert env.answer is not None
    assert env._submit_refusals == MAX_SUBMIT_REFUSALS


def test_the_last_refusal_says_it_is_the_last(make_env, tmp_path):
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=40)
    for _ in range(MAX_SUBMIT_REFUSALS - 1):
        _submit(env)
    obs, *_ = _submit(env)
    assert "last refusal" in obs


def test_no_refusal_when_the_step_budget_is_nearly_spent(make_env, tmp_path):
    """Refusing here reliably produces a FAILED trajectory instead of an evidenced one."""
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=MIN_STEPS_LEFT_TO_REFUSE)
    # First step consumed by the submit itself, leaving fewer than the required margin.
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0, "a late refusal would cost the whole answer"
    assert env._submit_refusals == 0


def test_a_refusal_never_ends_the_run_by_exhausting_the_budget(make_env, tmp_path):
    """The margin exists so a refusal always leaves room to act on it."""
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=MIN_STEPS_LEFT_TO_REFUSE + 1)
    _obs, _r, done, info = _submit(env)
    assert done is False
    assert not info.get("budget_exhausted")
    assert env.max_steps - env.steps_taken >= 2, "no room left to read and resubmit"


def test_an_empty_answer_is_still_rejected_on_its_own_terms(make_env, tmp_path):
    """The obligations check must not shadow the pre-existing empty-answer guard."""
    m = _manifest(tmp_path, [_item("X", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m)
    obs, _r, done, _i = _submit(env, "   ")
    assert done is False
    assert "non-empty answer" in obs


# ── it stays generic ──────────────────────────────────────────────────────────
def test_no_manifest_means_no_enforcement_at_all(make_env):
    env = _seeded(make_env, None)
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0
    cov = env.evidence_coverage()
    assert cov["enforced"] is False
    assert cov["obligations_total"] == 0


def test_an_env_with_no_seed_cells_behaves_exactly_as_before(make_env):
    """Every pre-existing test path constructs the env without any of this."""
    env = make_env()
    env.reset()
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0
    assert env.evidence_coverage()["enforced"] is False


@pytest.mark.parametrize("payload", [
    "not json at all",
    '{"items": "not a list"}',
    '["a", "list", "not", "a", "dict"]',
    '{"items": [{"no_id": true}, 42, null]}',
    '{}',
])
def test_a_malformed_manifest_disables_the_check_rather_than_failing_the_run(
    make_env, tmp_path, payload
):
    """An audit that cannot run must never be reported as an analysis failure."""
    p = tmp_path / "diagnostics_obligations.json"
    p.write_text(payload)
    env = _seeded(make_env, p)
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0
    assert env.evidence_coverage()["obligations_total"] == 0


def test_an_advisory_item_is_not_enforced_and_does_not_inflate_the_total(make_env, tmp_path):
    """It has nothing to check, so counting it would record it as discharged by default."""
    m = _manifest(tmp_path, [
        dict(_item("ADVISORY_ONLY", statement="a caveat with no table"), advisory=True),
        _item("REAL", must_read=["diagnostics_flags"]),
    ])
    env = _seeded(make_env, m, max_steps=40)
    obs, _r, done, _i = _submit(env)
    assert done is False
    assert "0 of 1 obligation(s)" in obs, "the advisory item must not be counted"
    assert "1 remain" in obs
    assert "ADVISORY_ONLY" not in obs
    for _ in range(MAX_SUBMIT_REFUSALS):
        _submit(env)
    assert env.evidence_coverage()["obligations_total"] == 1


def test_an_item_with_no_discharge_condition_is_not_enforced_even_unflagged(make_env, tmp_path):
    """Defence in depth: the harness does not rely on the writer having set the flag."""
    m = _manifest(tmp_path, [_item("NOTHING_TO_CHECK")])
    env = _seeded(make_env, m)
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0
    assert env.evidence_coverage()["enforced"] is False


def test_a_missing_manifest_file_disables_the_check(make_env, tmp_path):
    env = _seeded(make_env, tmp_path / "never_written.json")
    _obs, _r, done, _i = _submit(env)
    assert done is True


def test_the_environment_learns_nothing_about_what_a_measurement_MEANS(make_env, tmp_path):
    """Generic by construction: an invented domain enforces exactly like the real one."""
    m = _manifest(tmp_path, [_item("SOMETHING_ELSE_ENTIRELY", must_read=["telescope_pointing"],
                                   must_mention=["Betelgeuse"])],
                  tables=["telescope_pointing"])
    env = _seeded(make_env, m)
    obs, _r, done, _i = _submit(env)
    assert done is False
    assert "telescope_pointing" in obs and "Betelgeuse" in obs


# ── it records what it let through ────────────────────────────────────────────
def test_what_the_ceiling_let_through_is_recorded_not_erased(make_env, tmp_path):
    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"], statement="unread A"),
                             _item("B", must_mention=["D14"])])
    env = _seeded(make_env, m, max_steps=40)
    for _ in range(MAX_SUBMIT_REFUSALS):
        _submit(env)
    _submit(env)
    cov = env.evidence_coverage()
    assert cov["enforced"] is True
    assert cov["obligations_total"] == 2
    assert cov["obligations_undischarged"] == 2
    assert cov["submit_refusals"] == MAX_SUBMIT_REFUSALS
    assert {u["id"] for u in cov["undischarged"]} == {"A", "B"}
    assert any(u["unread_tables"] == ["diagnostics_flags"] for u in cov["undischarged"])
    assert any(u["unnamed"] == ["D14"] for u in cov["undischarged"])


def test_coverage_separates_partial_progress_from_none(make_env, tmp_path):
    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"]),
                             _item("B", must_read=["diagnostics_transfer"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags.head())")
    for _ in range(MAX_SUBMIT_REFUSALS + 1):
        _submit(env)
    cov = env.evidence_coverage()
    assert cov["discharged_ids"] == ["A"]
    assert [u["id"] for u in cov["undischarged"]] == ["B"]


def test_coverage_reports_which_of_the_runs_tables_were_opened_at_all(make_env, tmp_path):
    """'three of eleven tables' is the statistic that exposed the problem; keep computing it."""
    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"])],
                  tables=["diagnostics_flags", "diagnostics_transfer", "diagnostics_uncertainty"])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags.head(), diagnostics_uncertainty.shape)")
    _submit(env)
    cov = env.evidence_coverage()
    assert cov["tables_available"] == [
        "diagnostics_flags", "diagnostics_transfer", "diagnostics_uncertainty"]
    assert cov["tables_opened"] == ["diagnostics_flags", "diagnostics_uncertainty"]


def test_a_long_undischarged_list_is_capped_but_counted(make_env, tmp_path):
    n = REFUSAL_ITEMS_SHOWN + 4
    m = _manifest(tmp_path, [_item(f"ITEM_{i}", must_read=["diagnostics_flags"])
                             for i in range(n)])
    env = _seeded(make_env, m, max_steps=40)
    obs, *_ = _submit(env)
    assert f"0 of {n} obligation(s)" in obs and f"{n} remain" in obs
    assert f"+{n - REFUSAL_ITEMS_SHOWN} more undischarged item(s)" in obs


# ── the artifact and the consensus ────────────────────────────────────────────
def test_the_trajectory_writes_the_coverage_artifact_only_when_enforced(make_env, tmp_path):
    from flow.trajectory import Trajectory

    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=40)
    for _ in range(MAX_SUBMIT_REFUSALS + 1):
        _submit(env)

    traj = Trajectory(tmp_path / "run")
    coverage = traj.write_evidence_coverage(env)
    written = json.loads((tmp_path / "run" / "evidence_coverage.json").read_text())
    assert written["obligations_undischarged"] == 1
    assert coverage["enforced"] is True

    # An ungated run writes no file at all, so it cannot be mistaken for a clean sweep.
    plain = make_env()
    plain.reset()
    traj2 = Trajectory(tmp_path / "run2")
    traj2.write_evidence_coverage(plain)
    assert not (tmp_path / "run2" / "evidence_coverage.json").exists()


def test_run_json_reports_the_undischarged_count_only_when_the_check_ran(make_env, tmp_path):
    from flow.agent.react import AgentResult
    from flow.trajectory import RunMetadata, Trajectory

    def _meta() -> RunMetadata:
        return RunMetadata(run_id="r", question="q", provider="mock", model="m", image="i",
                           image_digest=None, max_steps=40, limits={}, started_at=0.0)

    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=40)
    for _ in range(MAX_SUBMIT_REFUSALS + 1):
        _submit(env)
    traj = Trajectory(tmp_path / "gated")
    traj.finalize(env, AgentResult(submitted=True, answer=ANSWER), _meta())
    assert traj.read_run_json()["obligations_undischarged"] == 1
    assert traj.read_run_json()["obligations_total"] == 1

    plain = make_env()
    plain.reset()
    traj2 = Trajectory(tmp_path / "plain")
    traj2.finalize(plain, AgentResult(submitted=True, answer=ANSWER), _meta())
    # Null, not zero: the question was never asked.
    assert traj2.read_run_json()["obligations_undischarged"] is None
    assert traj2.read_run_json()["obligations_total"] is None


def test_the_consensus_is_told_what_an_analyst_did_not_read():
    from flow.consensus import META_SYSTEM_PROMPT, _build_user_message

    evidence = [
        {"enforced": True, "obligations_total": 4,
         "undischarged": [{"id": "OFF_TREND__D14", "statement": "D14 sits off-trend"}]},
        {"enforced": True, "obligations_total": 4, "undischarged": []},
    ]
    msg = _build_user_message("q", ["conclusion one", "conclusion two"], evidence)
    assert "EVIDENCE NOT READ" in msg
    assert "1 of 4" in msg and "D14 sits off-trend" in msg
    # Only the analyst who left something unread gets a note.
    assert msg.count("EVIDENCE NOT READ") == 1
    # And the reviewer is told what to do with it, or the note is decoration.
    assert "agreement about an assumption" in META_SYSTEM_PROMPT


def test_the_consensus_note_is_absent_when_there_is_nothing_to_report():
    from flow.consensus import _build_user_message, _evidence_note

    assert _evidence_note(None) == ""
    assert _evidence_note({"enforced": False, "undischarged": [{"id": "X"}]}) == ""
    assert _evidence_note({"enforced": True, "undischarged": []}) == ""
    assert "EVIDENCE NOT READ" not in _build_user_message("q", ["a", "b"], [None, None])


def test_the_consensus_counts_analysts_with_unread_evidence():
    from flow.consensus import synthesize_consensus
    from flow.providers.mock import MockProvider

    results = [
        {"idx": 0, "submitted": True, "answer": "a",
         "evidence": {"enforced": True, "obligations_total": 2,
                      "undischarged": [{"id": "A", "statement": "s"}]}},
        {"idx": 1, "submitted": True, "answer": "b",
         "evidence": {"enforced": True, "obligations_total": 2, "undischarged": []}},
        {"idx": 2, "submitted": False, "answer": None, "evidence": {}},
    ]
    out = synthesize_consensus(MockProvider(), "q", results)
    assert out.n_submitted == 2
    assert out.n_with_unread_evidence == 1


def test_evidence_records_stay_aligned_when_a_trajectory_did_not_submit():
    """Filtering unsubmitted answers must not shift the notes onto the wrong analyst."""
    from flow.consensus import _build_user_message

    results = [
        {"submitted": False, "answer": None, "evidence": {
            "enforced": True, "obligations_total": 1,
            "undischarged": [{"id": "WRONG", "statement": "belongs to the failed run"}]}},
        {"submitted": True, "answer": "the only conclusion", "evidence": {
            "enforced": True, "obligations_total": 1,
            "undischarged": [{"id": "RIGHT", "statement": "belongs to the real run"}]}},
    ]
    kept = [(r, r["answer"]) for r in results if r["submitted"]]
    msg = _build_user_message("q", [a for _r, a in kept], [r.get("evidence") for r, _a in kept])
    assert "belongs to the real run" in msg
    assert "belongs to the failed run" not in msg


# ── naming a table is not reading it ──────────────────────────────────────────
# ``read_columns`` closes the gap that "open the table" left open. Two production failures
# sit behind it: a run told to read ``diagnostics_cutoff_audit`` printed three of its
# thirty-seven columns with the finding among the thirty-four it projected away, and a run
# discharged fifteen items with one ``groupby('code').size()`` — a count of the findings,
# containing none of their numbers. Both satisfy every table-name check ever written.
def _col_item(oid: str, table: str, columns: list[str], **kw) -> dict:
    item = _item(oid, must_read=[table], **kw)
    item["read_columns"] = {table: columns}
    return item


def test_a_head_call_no_longer_discharges_an_item_that_names_its_columns(make_env, tmp_path):
    m = _manifest(tmp_path, [_col_item("A", "diagnostics_transfer",
                                       ["negative_mode_drift_in_negative_sd"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_transfer.head())")
    obs, _r, done, _i = _submit(env)
    assert done is False, ".head() shows the table without the measurement"
    assert "negative_mode_drift_in_negative_sd" in obs, "the refusal must name the column"


def test_the_exact_groupby_size_false_discharge_is_caught(make_env, tmp_path):
    """Regression: a17be587a070 traj 0 discharged 13 items with this one expression."""
    m = _manifest(tmp_path, [
        _col_item("FLAGS__NEGATIVE_MODE_DRIFT", "diagnostics_flags", ["measured"]),
        _col_item("FLAGS__OVERTON_DISAGREES_WITH_CUTOFF", "diagnostics_flags", ["measured"]),
    ])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags[diagnostics_flags.code.isin("
                     "['NEGATIVE_MODE_DRIFT'])].groupby('code').size())")
    obs, _r, done, _i = _submit(env)
    assert done is False, "a tally of the findings is not a read of their numbers"
    assert "0 of 2 obligation(s)" in obs


def test_one_measurement_column_is_enough(make_env, tmp_path):
    """All-of would refuse an agent that read four of five relevant columns — honest work."""
    m = _manifest(tmp_path, [_col_item("A", "diagnostics_transfer",
                                       ["negative_mode_drift", "emd_negative_subset",
                                        "overton_positive_vs_reference_negative"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_transfer[['marker', 'emd_negative_subset']])")
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0


def test_an_identity_only_projection_does_not_discharge(make_env, tmp_path):
    """Naming the rows is not reading them."""
    m = _manifest(tmp_path, [_col_item("A", "diagnostics_transfer", ["negative_mode_drift"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_transfer[['marker', 'timepoint', 'label']])")
    _obs, _r, done, _i = _submit(env)
    assert done is False


def test_the_refusal_distinguishes_opened_from_never_touched(make_env, tmp_path):
    """'still needed' on a table the agent just printed reads as though the read did not
    count, which is the message that produced the throwaway batch-read cell."""
    m = _manifest(tmp_path, [_col_item("A", "diagnostics_transfer", ["negative_mode_drift"]),
                             _col_item("B", "diagnostics_controls", ["false_positive_rate"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_transfer.shape)")
    obs, *_ = _submit(env)
    assert "projected away every column the rule turns on" in obs
    assert "diagnostics_transfer" in obs
    # B was never opened at all, so it must NOT carry that clause.
    b_block = obs.split("  B")[1] if "  B" in obs else ""
    assert "projected away" not in b_block


def test_an_item_without_read_columns_keeps_the_old_strength(make_env, tmp_path):
    """A manifest from an older diagnostics pass must keep working, not fail closed."""
    m = _manifest(tmp_path, [_item("A", must_read=["diagnostics_flags"])])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags.head())")
    _obs, reward, done, _i = _submit(env)
    assert done is True and reward == 1.0


def test_malformed_read_columns_degrade_rather_than_crash(make_env, tmp_path):
    item = _item("A", must_read=["diagnostics_flags"])
    item["read_columns"] = "not a dict"
    m = _manifest(tmp_path, [item])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags.head())")
    _obs, _r, done, _i = _submit(env)
    assert done is True, "a malformed field disables the stricter check, it does not fail the run"


def test_coverage_separates_opened_from_actually_read(make_env, tmp_path):
    """The gap between these two numbers is the .head() rate, and it is the thing to watch."""
    m = _manifest(tmp_path, [_col_item("A", "diagnostics_flags", ["measured"]),
                             _col_item("B", "diagnostics_transfer", ["negative_mode_drift"])],
                  tables=["diagnostics_flags", "diagnostics_transfer"])
    env = _seeded(make_env, m, max_steps=40)
    _write_cell(env, "print(diagnostics_flags[['measured']], diagnostics_transfer.head())")
    for _ in range(MAX_SUBMIT_REFUSALS + 1):
        _submit(env)
    cov = env.evidence_coverage()
    assert cov["tables_opened"] == ["diagnostics_flags", "diagnostics_transfer"]
    assert cov["tables_read_with_measurements"] == ["diagnostics_flags"]
