"""NotebookEnvironment — the Aviary-style environment Finch acts within.

Owns the notebook state and a persistent kernel (inside Docker for real runs). Exposes
**exactly two** tools to the agent:

  * ``edit_cell(index, source, execute)`` — create/replace a cell and optionally run it.
  * ``submit_answer(answer)``            — finalize the analytical conclusion.

The environment NEVER performs analysis. It only edits/executes the agent's code,
truncates observations, summarizes images as ``[image]``, builds ``notebook.ipynb``,
and enforces ``max_steps`` plus per-cell / per-trajectory timeouts. If the agent never
submits, the run fails — the environment will not complete the analysis for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from flow.aviary.environment import Environment
from flow.aviary.tool import Tool, ToolCall
from flow.env.kernel import CellResult, Kernel, KernelTimeout

MAX_OBS_CHARS = 4000  # truncate each stream so the context stays bounded


@dataclass
class NotebookCell:
    """A single notebook cell and its most recent execution outputs."""

    source: str = ""
    cell_type: str = "code"
    outputs: list[dict[str, Any]] = field(default_factory=list)
    execution_count: int | None = None


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
        self.first_run_note = first_run_note
        self.seed_cell_timeout = seed_cell_timeout
        # Optional user-supplied guidance, appended to the agent's system prompt.
        self.system_prompt_extra = system_prompt_extra

        self.cells: list[NotebookCell] = []
        self.steps_taken = 0
        self.done = False
        self.answer: str | None = None
        self._started_at: float | None = None
        self._tools = [self._edit_cell_tool(), self._submit_answer_tool()]

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
                seed_outputs.append(self._format_observation(idx, res))
            except KernelTimeout as e:
                seed_outputs.append(f"[first-run cell {idx} timed out: {e}]")

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
                "Create, modify, and execute a cell in the Jupyter notebook. Use "
                "index='new' to append a new cell, or an integer index to overwrite an "
                "existing cell. State persists across cells (variables, imports, loaded "
                "data). Returns the cell's stdout, stderr, last-expression value, and "
                "traceback (truncated)."
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
                    f"Cell index {idx} out of range (notebook has {len(self.cells)} "
                    "cells). Use 'new' to append."
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
        self.answer = answer.strip()
        self.done = True
        return "Answer submitted. The trajectory is complete."

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

    def _format_observation(self, idx: int, res: CellResult) -> str:
        """Render the agent-visible observation: truncated text, images summarized."""
        parts = [f"[cell {idx} executed]"]
        if res.stdout:
            parts.append("STDOUT:\n" + _truncate(res.stdout))
        if res.stderr:
            parts.append("STDERR:\n" + _truncate(res.stderr))
        if res.result:
            parts.append("RESULT:\n" + _truncate(res.result))
        if res.images:
            parts.append(f"IMAGES: produced {len(res.images)} figure(s): " +
                         ", ".join(res.images) + " (rendered as [image]; not shown inline)")
        if res.error:
            parts.append("TRACEBACK:\n" + _truncate(res.traceback))
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
