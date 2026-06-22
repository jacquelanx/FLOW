"""Tests for the consensus meta-analysis and the batch orchestration (no Docker)."""

from pathlib import Path

from flow.consensus import synthesize_consensus
from flow.providers.mock import MockProvider


def test_consensus_none_when_no_submissions():
    res = synthesize_consensus(
        MockProvider(), "Q?", [{"idx": 0, "submitted": False, "answer": None}]
    )
    assert res.consensus is None
    assert res.synthesized is False
    assert res.n_submitted == 0
    assert res.failure_reason


def test_consensus_passthrough_for_single_submission():
    res = synthesize_consensus(
        MockProvider(), "Q?", [{"idx": 0, "submitted": True, "answer": "the answer"}]
    )
    assert res.consensus == "the answer"
    assert res.synthesized is False  # nothing to reconcile
    assert res.n_submitted == 1


def test_consensus_synthesizes_multiple():
    results = [
        {"idx": 0, "submitted": True, "answer": "A increases over time."},
        {"idx": 1, "submitted": True, "answer": "A trends upward."},
        {"idx": 2, "submitted": False, "answer": None},
    ]
    res = synthesize_consensus(MockProvider(), "Q?", results)
    assert res.synthesized is True
    assert res.n_submitted == 2
    assert res.n_total == 3
    assert "MOCK PROVIDER" in res.consensus  # came from MockProvider.complete


def test_run_batch_isolation_and_consensus(tmp_path, monkeypatch):
    """run_batch must give each trajectory its own dir and synthesize a consensus."""
    import flow.runner as runner
    from flow.agent.react import AgentResult, StepRecord
    from flow.config import FlowConfig
    from flow.trajectory import RunMetadata, Trajectory

    monkeypatch.setattr(runner, "require_docker", lambda image=None: None)

    seen_dirs = []

    def fake_run_analysis(*, data_dir, config, run_dir, run_id=None, image=None,
                          temperature=0.0, on_step=None):
        seen_dirs.append(Path(run_dir))
        traj = Trajectory(run_dir)
        rec = StepRecord(1, "submit_answer", {}, "done", True, 1.0)
        if on_step:
            on_step(rec)
        result = AgentResult(submitted=True, answer=f"answer-{run_id}", steps=[rec])
        meta = RunMetadata(
            run_id=run_id, question=config.question, provider=config.runtime.provider,
            model=config.runtime.model, image="img", image_digest=None,
            max_steps=config.runtime.max_steps, limits={}, started_at=0.0,
        )
        traj.finalize(
            type("E", (), {"to_notebook": lambda self: {"cells": [], "nbformat": 4,
             "nbformat_minor": 5, "metadata": {}}})(),
            result, meta,
        )
        return result, meta

    monkeypatch.setattr(runner, "run_analysis", fake_run_analysis)

    cfg = FlowConfig(question="What is the trend?")
    cfg.runtime.provider = "mock"
    cfg.runtime.model = "mock"

    batch_dir = tmp_path / "batch"
    phases = []
    bres = runner.run_batch(
        data_dir=tmp_path / "data", config=cfg, batch_dir=batch_dir,
        n_trajectories=3, batch_id="bX", on_phase=phases.append,
    )

    # Each trajectory ran in its own isolated directory.
    assert len(seen_dirs) == 3
    assert len(set(seen_dirs)) == 3
    for idx in range(3):
        assert (batch_dir / "trajectories" / str(idx)).exists()

    # Consensus synthesized from the 3 answers; artifacts written.
    assert bres.consensus["synthesized"] is True
    assert bres.consensus["n_submitted"] == 3
    assert (batch_dir / "consensus.txt").read_text()
    assert (batch_dir / "batch.json").exists()
    assert phases == ["running", "synthesizing", "done"]


def test_run_batch_uses_separate_meta_model(tmp_path, monkeypatch):
    """A configured meta_provider/meta_model is used for the consensus, not the trajectories."""
    import json

    import flow.runner as runner
    from flow.agent.react import AgentResult, StepRecord
    from flow.config import FlowConfig
    from flow.trajectory import RunMetadata, Trajectory

    monkeypatch.setattr(runner, "require_docker", lambda image=None: None)

    built: list[tuple] = []
    real_build = runner.build_provider

    def spy_build(provider, model, temperature=0.0):
        built.append((provider, model, temperature))
        return real_build(provider, model, temperature)

    monkeypatch.setattr(runner, "build_provider", spy_build)

    def fake_run_analysis(*, data_dir, config, run_dir, run_id=None, image=None,
                          temperature=0.0, on_step=None):
        # The trajectory provider is built inside run_analysis, which we bypass here, so
        # record it explicitly to mimic real behavior.
        built.append((config.runtime.provider, config.runtime.model, temperature))
        traj = Trajectory(run_dir)
        result = AgentResult(submitted=True, answer="a", steps=[StepRecord(1, "submit_answer", {}, "x", True, 1.0)])
        meta = RunMetadata(run_id=run_id, question=config.question, provider=config.runtime.provider,
                           model=config.runtime.model, image="i", image_digest=None,
                           max_steps=config.runtime.max_steps, limits={}, started_at=0.0)
        traj.finalize(type("E", (), {"to_notebook": lambda self: {"cells": [], "nbformat": 4,
                      "nbformat_minor": 5, "metadata": {}}})(), result, meta)
        return result, meta

    monkeypatch.setattr(runner, "run_analysis", fake_run_analysis)

    cfg = FlowConfig(question="Q?")
    cfg.runtime.provider = "mock"
    cfg.runtime.model = "trajectory-model"
    cfg.runtime.meta_provider = "mock"
    cfg.runtime.meta_model = "consensus-model"

    batch_dir = tmp_path / "batch"
    runner.run_batch(data_dir=tmp_path / "d", config=cfg, batch_dir=batch_dir,
                     n_trajectories=2, batch_id="bX")

    # The meta model was built exactly once, with the configured consensus model + temp 0.
    assert ("mock", "consensus-model", 0.0) in built
    record = json.loads((batch_dir / "batch.json").read_text())
    assert record["meta_model"] == "consensus-model"
    assert record["model"] == "trajectory-model"
