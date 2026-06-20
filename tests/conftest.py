"""Shared pytest fixtures.

Provides helpers to build a NotebookEnvironment backed by the host-side InProcessKernel,
so the full ReAct loop can be exercised WITHOUT Docker. Real runs never use this path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from flow.env.kernel import InProcessKernel
from flow.env.notebook_env import NotebookEnvironment


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "events.csv").write_text(
        "sample_id,label,DET-A\nS01,Baseline,1.0\nS01,Baseline,2.0\n"
    )
    (d / "metadata.json").write_text('{"panel_notes": "synthetic"}')
    return d


@pytest.fixture
def make_env(tmp_path: Path, tmp_data_dir: Path):
    def _make(question: str = "What is the shape of the data?", **kw) -> NotebookEnvironment:
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        # The mock/agent cells discover the data dir via this env var when not in Docker.
        os.environ["FLOW_DATA_DIR"] = str(tmp_data_dir)
        kernel = InProcessKernel(plots_dir=work / "plots")
        return NotebookEnvironment(
            kernel=kernel,
            question=question,
            dataset_description=f"Data dir at {tmp_data_dir}",
            **kw,
        )

    return _make
