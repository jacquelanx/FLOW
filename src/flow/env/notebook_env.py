"""NotebookEnvironment — the Aviary-style environment Finch acts within.

Owns the notebook state and a persistent kernel (inside Docker for real runs). Exposes
**exactly two** tools to the agent:

  * ``edit_cell(index, source, execute)`` — create/replace a cell and optionally run it.
  * ``submit_answer(answer)``            — finalize the analytical conclusion.

The environment NEVER performs analysis. It only edits/executes the agent's code,
truncates observations, summarizes images as ``[image]``, builds ``notebook.ipynb``, and
enforces ``max_steps`` plus per-cell / per-trajectory timeouts. If the agent never submits,
the run fails — the environment will not complete the analysis for it.

Observations are truncated at two different budgets: the deterministic first-run cells get
``SEED_OBS_CHARS`` because they run once and carry the evidence base, while the agent's own
per-step cells get the smaller ``MAX_OBS_CHARS``.

It also enforces an OBLIGATIONS CHECK, and that deserves its rationale here because it is the
one place the environment pushes back on the agent's judgement:

A project's first-run may leave an obligations manifest — a short list of measurements the
reader is expected to open, each naming the table that holds them and, sometimes, a token the
write-up must contain. Two production runs read the summary that counted 375 such findings,
opened three of eleven tables, and submitted without mentioning the rest. Guidance telling
them to read the rows was already present and specific. So on ``submit_answer`` this
environment checks the agent's own cells and its answer text against the manifest and, when
items are undischarged, REFUSES the submission once or twice with the specific list.

Three properties keep that from doing more harm than good:

  * **It is generic.** The environment never learns what a measurement means. It reads a
    manifest a project supplied and enforces exactly what that manifest asked for. No
    manifest — the overwhelmingly common case — means no enforcement whatsoever.
  * **It has a hard ceiling.** ``MAX_SUBMIT_REFUSALS`` refusals, and never when fewer than
    ``MIN_STEPS_LEFT_TO_REFUSE`` steps remain. A check that can refuse indefinitely converts a
    partial answer into NO answer, which is strictly worse than an under-evidenced one.
  * **It records what it let through.** When the ceiling is hit the answer is accepted and
    the still-undischarged items are kept in ``evidence_coverage()``, so the gap lands in the
    run's artifacts and in the consensus instead of disappearing.

The check is honest-agent-grade, not adversary-grade. It asks that the columns a finding turns
on appear in a cell the agent wrote; it cannot tell a real read from a print statement built to
satisfy it, and does not try. The failure being corrected is a model that forgets, not one that
cheats — but "name the table" turned out to be below the floor even for that, because a tally
of the findings (``groupby('code').size()``) contains none of their numbers while satisfying
every table-name check. Hence ``read_columns``; see ``_table_is_read``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flow.aviary.environment import Environment
from flow.aviary.tool import Tool, ToolCall
from flow.env.kernel import CellResult, Kernel, KernelTimeout

# Truncate each stream so the context stays bounded over a long trajectory. Raised from 4000
# after a production run printed a 99-row diagnostics table at 4058 chars — the agent's own
# drill-down into a flagged group is exactly the cell that runs long, and losing its middle
# defeats the point of asking for it. Wide tables should still be aggregated before printing;
# this limit buys headroom, not permission to dump.
MAX_OBS_CHARS = 8000

# The deterministic first-run cells get a larger window than the agent's own cells. They run
# ONCE, before the agent acts, and their output is the entire evidence base the agent
# interprets — a first-run table dump and a diagnostics digest do not fit in 8000 chars, and
# what the truncator drops is the middle, where the findings are. Raised from 20000 once the
# diagnostics cell began printing its worklist: the same run's first-run cell was already
# overrunning 20000 by 7000 chars, so the ceiling was silently eating evidence before the
# worklist existed. Per-step cells keep the smaller limit.
SEED_OBS_CHARS = 32000

# How many times a submission may be refused for undischarged obligations. Two: the first
# refusal is the one that works when the agent simply had not looked, the second covers a
# partial fix, and a third would start trading a whole answer for a citation.
MAX_SUBMIT_REFUSALS = 2
# Never refuse with fewer than this many steps left. A refusal costs one step and the fix
# costs at least two more (read, then resubmit), so refusing near the budget's end reliably
# produces a FAILED trajectory instead of an evidenced one.
MIN_STEPS_LEFT_TO_REFUSE = 4
# Undischarged items named in a single refusal. Beyond this the refusal stops being a
# to-do list and becomes another wall of text to skim; the count is always stated.
REFUSAL_ITEMS_SHOWN = 6


@dataclass
class NotebookCell:
    """A single notebook cell and its most recent execution outputs."""

    source: str = ""
    cell_type: str = "code"
    outputs: list[dict[str, Any]] = field(default_factory=list)
    execution_count: int | None = None


def _names(text_lower: str, token: str) -> bool:
    """Does ``text_lower`` NAME ``token``, rather than merely contain its letters?

    A plain substring test is wrong here, and a production run showed exactly how: the token
    ``Pre`` was satisfied by the word "preventing", so a sample the write-up never mentioned
    was recorded as addressed. Short labels are the common case, so the failure is the common
    case too — ``D14`` matches "cd14", ``Pre`` matches "interpreted".

    The boundary is "not a letter or digit", which deliberately treats a hyphen as a word
    break: "post-conditioning" DOES name the ``Post`` sample and "pre-infusion" DOES name the
    ``Pre`` sample, while "preventing" names neither.
    """
    return re.search(
        r"(?<![0-9a-z])" + re.escape(token.lower()) + r"(?![0-9a-z])", text_lower
    ) is not None


def _truncate(text: str, limit: int = MAX_OBS_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    omitted = len(text) - limit
    return f"{head}\n... [truncated {omitted} chars] ...\n{tail}"


class NotebookEnvironment(Environment):
    """Stateful notebook environment with a persistent kernel backend."""

    def __init__(
        self,
        *,
        kernel: Kernel,
        question: str,
        dataset_description: str,
        max_steps: int = 30,
        per_cell_timeout: float = 120.0,
        per_trajectory_timeout: float = 1800.0,
        seed_cells: list[str] | None = None,
        first_run_note: str = "",
        seed_cell_timeout: float = 900.0,
        system_prompt_extra: str = "",
        seed_obs_chars: int = SEED_OBS_CHARS,
        obligations_file: str | Path | None = None,
    ):
        self.kernel = kernel
        self.question = question
        self.dataset_description = dataset_description
        self.max_steps = max_steps
        self.per_cell_timeout = per_cell_timeout
        self.per_trajectory_timeout = per_trajectory_timeout
        # Deterministic first-run cells executed at reset() before the agent acts. They do
        # NOT count against the agent's step budget; their output primes the first observation.
        self.seed_cells = seed_cells or []
        # Cells 0..n_seed-1 are the deterministic first-run and are READ-ONLY. Overwriting one
        # destroys the audit trail for every headline number the agent goes on to report, and
        # models reach for index=0 by default rather than 'new' — observed on two different
        # models in a row, so this is a guard, not a nicety.
        self.n_protected_cells = len(self.seed_cells)
        self.first_run_note = first_run_note
        self.seed_cell_timeout = seed_cell_timeout
        # Observation limit for the first-run cells only (see SEED_OBS_CHARS).
        self.seed_obs_chars = max(MAX_OBS_CHARS, int(seed_obs_chars))
        # Optional user-supplied guidance, appended to the agent's system prompt.
        self.system_prompt_extra = system_prompt_extra
        # Host-side path where the first-run may have left an obligations manifest. Read at
        # reset(), AFTER the seed cells run — the file does not exist before then.
        self.obligations_file = Path(obligations_file) if obligations_file else None

        self.cells: list[NotebookCell] = []
        self.steps_taken = 0
        self.done = False
        self.answer: str | None = None
        self._started_at: float | None = None
        self._tools = [self._edit_cell_tool(), self._submit_answer_tool()]
        # Obligations submission-check state.
        self._obligations: list[dict[str, Any]] = []
        self._obligation_tables: list[str] = []
        self._submit_refusals = 0
        self._undischarged: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ Aviary API
    def tools(self) -> list[Tool]:
        return self._tools

    def reset(self) -> tuple[str, list[Tool]]:
        self.kernel.start()
        self._started_at = time.time()

        # Execute the deterministic first-run cells (if any) before the agent acts. These
        # become cells 0..k-1 of the notebook; their output primes the first observation.
        seed_outputs: list[str] = []
        for src in self.seed_cells:
            cell = NotebookCell(source=src)
            self.cells.append(cell)
            idx = len(self.cells) - 1
            try:
                res = self.kernel.execute(src, timeout=self.seed_cell_timeout)
                cell.outputs = self._cell_outputs(res)
                seed_outputs.append(
                    self._format_observation(idx, res, limit=self.seed_obs_chars))
            except KernelTimeout as e:
                seed_outputs.append(f"[first-run cell {idx} timed out: {e}]")

        # The manifest is written BY the seed cells, so it can only be read after them.
        self._load_obligations()

        if self.seed_cells:
            intro = (
                "You are working in a Jupyter notebook running inside a sandboxed Docker "
                "container. A standardized FIRST-RUN analysis has ALREADY been executed in "
                "the opening cell(s) below.\n\n"
                f"RESEARCH QUESTION:\n{self.question}\n\n"
                f"DATASET (read-only, mounted at FLOW_DATA_DIR):\n{self.dataset_description}\n\n"
            )
            if self.first_run_note:
                intro += self.first_run_note + "\n\n"
            intro += "FIRST-RUN OUTPUT:\n" + "\n\n".join(seed_outputs)
            intro += (
                "\n\nContinue from here with `edit_cell` (state persists; the first-run "
                "variables are available). Submit your conclusion with `submit_answer`."
            )
        else:
            intro = (
                "You are working in a fresh Jupyter notebook running inside a sandboxed "
                "Docker container.\n\n"
                f"RESEARCH QUESTION:\n{self.question}\n\n"
                f"DATASET (read-only, mounted at the path FLOW_DATA_DIR):\n"
                f"{self.dataset_description}\n\n"
                "Notebook is empty. Use `edit_cell` to write and run code, inspecting the "
                "data and building your analysis step by step. When you have a justified, "
                "evidence-backed conclusion, call `submit_answer`."
            )
        return intro, self._tools

    def step(self, action: ToolCall) -> tuple[str, float, bool, dict[str, Any]]:
        if self.done:
            return "The trajectory has already ended.", 0.0, True, {"already_done": True}

        # Per-trajectory wall-clock budget.
        if (
            self._started_at is not None
            and time.time() - self._started_at > self.per_trajectory_timeout
        ):
            self.done = True
            return (
                "Per-trajectory time budget exceeded. The run is being terminated "
                "without a submitted answer.",
                0.0,
                True,
                {"timeout": "trajectory"},
            )

        self.steps_taken += 1
        info: dict[str, Any] = {"step": self.steps_taken, "tool": action.name}

        if action.name == "edit_cell":
            obs = self._edit_cell(**action.arguments)
            done = False
        elif action.name == "submit_answer":
            obs = self._submit_answer(**action.arguments)
            done = self.done
        else:
            obs = (
                f"Unknown tool '{action.name}'. Only 'edit_cell' and 'submit_answer' "
                "are available."
            )
            done = False
            info["error"] = "unknown_tool"

        # Step-budget enforcement (the environment never finishes the work itself).
        if not done and self.steps_taken >= self.max_steps:
            self.done = True
            done = True
            obs += (
                f"\n\n[step budget of {self.max_steps} reached without submit_answer; "
                "the trajectory has FAILED]"
            )
            info["budget_exhausted"] = True

        reward = 1.0 if (done and self.answer is not None) else 0.0
        return obs, reward, done, info

    # ------------------------------------------------------------------ Tool impls
    def _edit_cell_tool(self) -> Tool:
        return Tool(
            name="edit_cell",
            description=(
                "Create, modify, and execute a cell in the Jupyter notebook. Pass "
                "index='new' to append — this is what you want almost every time. Pass an "
                "integer index ONLY to fix a cell you wrote yourself earlier; any opening "
                "first-run cells are read-only. State persists across cells (variables, "
                "imports, loaded data), so appending never loses anything. Returns the "
                "cell's stdout, stderr, last-expression value, and traceback (truncated)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "index": {
                        "description": "'new' to append, or an integer cell index to overwrite.",
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                    },
                    "source": {"type": "string", "description": "The cell's code."},
                    "execute": {
                        "type": "boolean",
                        "description": "Execute the cell after editing (default true).",
                        "default": True,
                    },
                },
                "required": ["index", "source"],
            },
            fn=self._edit_cell,
        )

    def _submit_answer_tool(self) -> Tool:
        return Tool(
            name="submit_answer",
            description=(
                "Finalize and submit your analytical conclusion to the research "
                "question. Provide a clear, evidence-backed answer. This ends the run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final conclusion, with supporting reasoning.",
                    }
                },
                "required": ["answer"],
            },
            fn=self._submit_answer,
        )

    def _edit_cell(
        self, index: Any = "new", source: str = "", execute: bool = True
    ) -> str:
        # Resolve target cell.
        if index == "new" or index is None:
            cell = NotebookCell(source=source)
            self.cells.append(cell)
            idx = len(self.cells) - 1
        else:
            try:
                idx = int(index)
            except (TypeError, ValueError):
                return f"Invalid cell index {index!r}. Use 'new' or an integer."
            if idx < 0 or idx >= len(self.cells):
                return (
                    f"Use index='new' to append. (Cell index {idx} is out of range: the "
                    f"notebook has {len(self.cells)} cell(s), so valid indices are "
                    f"0-{len(self.cells) - 1}.)"
                )
            if idx < self.n_protected_cells:
                return (
                    f"Cell {idx} is READ-ONLY: cells 0-{self.n_protected_cells - 1} are the "
                    "deterministic first-run and must stay in the notebook as the audit trail "
                    "for the numbers you report. Nothing was changed. Call edit_cell with "
                    "index='new' to append your own cell instead — every variable those cells "
                    "created (including the loaded tables) is already available to it."
                )
            cell = self.cells[idx]
            cell.source = source

        if not execute:
            return f"Cell {idx} updated (not executed)."

        try:
            res = self.kernel.execute(source, timeout=self.per_cell_timeout)
        except KernelTimeout as e:
            self.done = True
            cell.outputs = [{"output_type": "error", "ename": "Timeout", "evalue": str(e)}]
            return f"Cell {idx} timed out: {e}"

        cell.execution_count = self.steps_taken
        cell.outputs = self._cell_outputs(res)
        return self._format_observation(idx, res)

    def _submit_answer(self, answer: str = "") -> str:
        if not answer or not answer.strip():
            return (
                "submit_answer requires a non-empty answer. Continue the analysis and "
                "submit a justified conclusion."
            )
        pending = self._undischarged_obligations(answer)
        if pending and self._may_refuse():
            self._submit_refusals += 1
            return self._refusal(pending)
        # Accepted. Anything still pending is recorded rather than discarded: the ceiling
        # exists so a thin answer beats no answer, not so the gap can go unreported.
        self._undischarged = pending
        self.answer = answer.strip()
        self.done = True
        return "Answer submitted. The trajectory is complete."

    # ------------------------------------------------------ Obligations check
    def _load_obligations(self) -> None:
        """Read the manifest the first-run may have left, or leave the check disabled.

        Fails soft in every direction. A missing, unreadable, or malformed manifest means no
        enforcement: an audit that cannot run must never be reported as an analysis failure, and
        the same logic applies to the check built on top of it.
        """
        if not self.obligations_file:
            return
        try:
            data = json.loads(Path(self.obligations_file).read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        items = data.get("items")
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            # An item the manifest marks advisory, or that arrives with nothing to check
            # against, is skipped rather than counted. Counting it would make it discharged
            # by default and inflate the total it can never be missing from.
            if it.get("advisory") or not (it.get("must_read") or it.get("must_mention")):
                continue
            raw_cols = it.get("read_columns")
            self._obligations.append({
                "id": str(it["id"]),
                "statement": str(it.get("statement") or ""),
                "how": [str(h) for h in (it.get("how") or []) if h],
                "must_read": [str(t) for t in (it.get("must_read") or []) if t],
                "must_mention": [str(t) for t in (it.get("must_mention") or []) if t],
                # Supporting context. Surfaced in a refusal, never a reason for one — see
                # ``_undischarged_obligations``.
                "should_read": [str(t) for t in (it.get("should_read") or []) if t],
                # ``{table: [columns]}`` — the measurement columns behind this item. Absent or
                # malformed means the table-name check alone, so a manifest from an older pass
                # keeps working at its own strength rather than failing closed.
                "read_columns": {
                    str(t): [str(c) for c in cols if c]
                    for t, cols in (raw_cols.items() if isinstance(raw_cols, dict) else ())
                    if t and isinstance(cols, list)
                },
            })
        tables = data.get("tables")
        if isinstance(tables, list):
            self._obligation_tables = [str(t) for t in tables if t]

    def _agent_cell_sources(self) -> str:
        """Every cell the AGENT wrote. The read-only first-run cells prove nothing here —
        they name every table by construction, so counting them would discharge the whole
        manifest before the agent had done anything."""
        return "\n".join(c.source for c in self.cells[self.n_protected_cells:])

    def _table_is_read(self, table: str, columns: list[str], src: str) -> bool:
        """Was ``table`` actually read, rather than merely named?

        Naming a table is not reading it. A production run was told to read
        ``diagnostics_cutoff_audit``, did open it, and printed three of its thirty-seven
        columns — with the two carrying the finding among the thirty-four it projected away.
        A second run discharged fifteen items with one ``groupby('code').size()``: a count of
        the findings, not one of the numbers behind them.

        So when the manifest names the measurement columns, at least ONE of them must appear.
        One rather than all: an agent that reads four of five relevant columns has done the
        work, and demanding the full set would refuse honest reading. One is still enough to
        rule out ``.head()``, a bare mention, and an identity-columns-only projection, which
        are the failures this exists to catch.
        """
        if table not in src:
            return False
        return not columns or any(c in src for c in columns)

    def _undischarged_obligations(self, answer: str) -> list[dict[str, Any]]:
        """Which manifest items this answer and this notebook do not yet satisfy.

        Only ``must_read`` blocks. The manifest puts the tables whose measurement columns it
        can name there, and genuine supporting context in ``should_read`` — see
        ``obligations.py``, which owns that judgement and explains both directions it has been
        wrong in. This side enforces exactly what arrives and infers nothing.
        """
        if not self._obligations:
            return []
        src = self._agent_cell_sources()
        low = (answer or "").lower()
        pending: list[dict[str, Any]] = []
        for it in self._obligations:
            cols = it.get("read_columns") or {}
            read = [t for t in it["must_read"]
                    if self._table_is_read(t, cols.get(t) or [], src)]
            unread = [t for t in it["must_read"] if t not in read]
            unnamed = [t for t in it["must_mention"] if not _names(low, t)]
            if unread or unnamed:
                pending.append({
                    **it,
                    "unread_tables": unread,
                    "unnamed": unnamed,
                    # Distinguish "never opened" from "opened, but none of the columns the
                    # rule turns on". The second is a projection away from a fix, and a
                    # refusal that does not say so reads as though the read did not count.
                    "opened_without_measurements": [
                        t for t in unread if t in src and (cols.get(t) or [])],
                    # What this item DOES already have, so a refusal can report progress
                    # rather than a flat count that reads as "nothing you did counted".
                    "read_tables": read,
                    "named": [t for t in it["must_mention"] if _names(low, t)],
                })
        return pending

    def _may_refuse(self) -> bool:
        """Refuse only while there is both an allowance and room to act on the refusal."""
        return (
            self._submit_refusals < MAX_SUBMIT_REFUSALS
            and (self.max_steps - self.steps_taken) >= MIN_STEPS_LEFT_TO_REFUSE
        )

    def _refusal(self, pending: list[dict[str, Any]]) -> str:
        """The refusal observation: what is missing, how to fix it, and when this stops.

        Says how many refusals remain on purpose. An agent that cannot tell whether the check
        is a wall or a nudge has no way to plan, and the honest answer is that it is a nudge.
        """
        remaining = MAX_SUBMIT_REFUSALS - self._submit_refusals
        n_done = len(self._obligations) - len(pending)
        lines = [
            "ANSWER NOT ACCEPTED — the trajectory continues.",
            "",
            # Progress first. A bare "15 of 15 undischarged" told one run that none of the
            # reading it had just done counted, and it replied with a cell that named the
            # remaining tables without reading them.
            f"{n_done} of {len(self._obligations)} obligation(s) from this project's first-run "
            f"are discharged; {len(pending)} remain. Each remaining one names a measurement "
            "that already exists in a table loaded in this notebook, and names the COLUMNS "
            "holding it — printing the table without those columns does not discharge the "
            "item. Print them in a cell, then say in your answer what you concluded — 'this "
            "does not change the result' is a valid conclusion and stating it discharges the "
            "item; leaving it out does not.",
            "",
        ]
        for it in pending[:REFUSAL_ITEMS_SHOWN]:
            lines.append(f"  {it['id']}")
            if it["statement"]:
                lines.append(f"      {it['statement']}")
            for expr in it["how"]:
                lines.append(f"      run: {expr}")
            # Name what is already satisfied, so a partly-done item does not read as untouched.
            done_bits = []
            if it.get("read_tables"):
                done_bits.append("already opened: " + ", ".join(it["read_tables"]))
            if it.get("named"):
                done_bits.append("already named: " + ", ".join(it["named"]))
            if done_bits:
                lines.append("      " + "; ".join(done_bits))
            if it["unread_tables"]:
                cols = it.get("read_columns") or {}
                # Name the columns, not just the table. "Open diagnostics_transfer" is
                # satisfied by ``.head()``; "print negative_mode_drift_in_negative_sd" is not,
                # and only the second tells the agent what would actually settle the item.
                needed = []
                for t in it["unread_tables"]:
                    want = cols.get(t) or []
                    needed.append(f"{t}[{', '.join(want)}]" if want else t)
                lines.append("      STILL NEEDED — print these columns in a cell you write: "
                             + "; ".join(needed))
                if it.get("opened_without_measurements"):
                    lines.append(
                        "      (you opened " + ", ".join(it["opened_without_measurements"])
                        + " but projected away every column the rule turns on — one of the "
                        "columns above is enough)")
            if it["unnamed"]:
                lines.append("      STILL NEEDED — name in your answer: "
                             + ", ".join(it["unnamed"]))
            if it.get("should_read"):
                lines.append("      (optional context, not required: "
                             + ", ".join(it["should_read"]) + ")")
        if len(pending) > REFUSAL_ITEMS_SHOWN:
            lines.append(f"  [+{len(pending) - REFUSAL_ITEMS_SHOWN} more undischarged item(s); "
                         "the full list is in this project's obligations manifest]")
        lines += [
            "",
            (f"This check will refuse at most {remaining} more submission(s); after that your "
             "answer is accepted as written and the items left undischarged are recorded "
             "against the run. Resubmit the same conclusion with the evidence added, or "
             "state plainly why an item does not apply.")
            if remaining
            else "This was the last refusal; your next submission is accepted as written.",
        ]
        return "\n".join(lines)

    def evidence_coverage(self) -> dict[str, Any]:
        """What the run did with the evidence it was given. Written to the run's artifacts.

        ``enforced`` distinguishes "nothing was left undischarged" from "there was nothing to
        discharge" — without it, every ungated project would report perfect coverage.
        """
        src = self._agent_cell_sources()
        pending_ids = {p["id"] for p in self._undischarged}
        return {
            "enforced": bool(self._obligations),
            "obligations_total": len(self._obligations),
            "obligations_undischarged": len(self._undischarged),
            "submit_refusals": self._submit_refusals,
            "discharged_ids": [o["id"] for o in self._obligations
                               if o["id"] not in pending_ids],
            "undischarged": [
                {"id": p["id"], "statement": p["statement"],
                 "unread_tables": p["unread_tables"], "unnamed": p["unnamed"]}
                for p in self._undischarged
            ],
            "tables_available": list(self._obligation_tables),
            # Kept as-is — the table NAME appearing — so this number stays comparable with the
            # runs recorded before ``read_columns`` existed.
            "tables_opened": [t for t in self._obligation_tables if t in src],
            # The stricter measure: opened AND at least one column some item needs from it.
            # This is the number to watch, and the gap between the two is the ``.head()`` rate.
            "tables_read_with_measurements": [
                t for t in self._obligation_tables
                if any(self._table_is_read(t, (o.get("read_columns") or {}).get(t) or [], src)
                       for o in self._obligations if t in o["must_read"])
            ],
        }

    # ------------------------------------------------------------------ Helpers
    def _cell_outputs(self, res: CellResult) -> list[dict[str, Any]]:
        """Build nbformat-style outputs from a CellResult (for notebook.ipynb)."""
        outputs: list[dict[str, Any]] = []
        if res.stdout:
            outputs.append(
                {"output_type": "stream", "name": "stdout", "text": res.stdout}
            )
        if res.stderr:
            outputs.append(
                {"output_type": "stream", "name": "stderr", "text": res.stderr}
            )
        if res.result:
            outputs.append(
                {
                    "output_type": "execute_result",
                    "data": {"text/plain": res.result},
                    "metadata": {},
                    "execution_count": self.steps_taken,
                }
            )
        for img in res.images:
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"text/plain": f"[image: {img}]"},
                    "metadata": {"flow_image_path": img},
                }
            )
        if res.error and res.traceback:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": "Error",
                    "evalue": "",
                    "traceback": res.traceback.splitlines(),
                }
            )
        return outputs

    def _format_observation(
        self, idx: int, res: CellResult, limit: int | None = None
    ) -> str:
        """Render the agent-visible observation: truncated text, images summarized.

        ``limit`` overrides the per-stream character budget; the first-run cells pass the
        larger ``seed_obs_chars`` because they run once and carry the evidence base.
        """
        lim = MAX_OBS_CHARS if limit is None else limit
        parts = [f"[cell {idx} executed]"]
        if res.stdout:
            parts.append("STDOUT:\n" + _truncate(res.stdout, lim))
        if res.stderr:
            parts.append("STDERR:\n" + _truncate(res.stderr, lim))
        if res.result:
            parts.append("RESULT:\n" + _truncate(res.result, lim))
        if res.images:
            # Named, never rendered: "not shown inline" invited the reading that the image was
            # visible somewhere. It is not — the kernel never returns image bytes.
            parts.append(f"IMAGES: produced {len(res.images)} figure(s): " +
                         ", ".join(res.images) +
                         " — saved to disk for the human reader. You CANNOT see their contents; "
                         "do not describe what they look like.")
        if res.error:
            parts.append("TRACEBACK:\n" + _truncate(res.traceback, lim))
        if len(parts) == 1:
            parts.append("(no output)")
        return "\n\n".join(parts)

    def to_notebook(self) -> dict[str, Any]:
        """Serialize the current state to an nbformat v4 notebook dict."""
        nb_cells = []
        for c in self.cells:
            nb_cells.append(
                {
                    "cell_type": c.cell_type,
                    "metadata": {},
                    "source": c.source.splitlines(keepends=True),
                    "outputs": c.outputs,
                    "execution_count": c.execution_count,
                }
            )
        return {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
                "flow": {"question": self.question},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
