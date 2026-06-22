"""Run orchestration — wires provider + Docker kernel + environment + agent + artifacts.

This is the single entry point used by both the CLI and the API. It enforces the safety
guarantee: real runs always execute inside Docker via ``DockerKernel``. Tests construct
the environment directly with ``InProcessKernel`` and never go through here.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from flow.agent.react import AgentResult, ReActAgent, StepRecord
from flow.config import FlowConfig
from flow.consensus import ConsensusResult, synthesize_consensus
from flow.env.docker_runtime import DEFAULT_IMAGE, DockerLimits, image_digest, require_docker
from flow.env.kernel import DockerKernel
from flow.env.notebook_env import NotebookEnvironment
from flow.providers.registry import build_provider
from flow.trajectory import RunMetadata, Trajectory
from flow.validation import profile_project

StepHook = Callable[[StepRecord], None]

# Default sampling temperature for multi-trajectory runs, so independent trajectories
# explore diverse analytical paths (single runs stay deterministic at 0.0).
DEFAULT_BATCH_TEMPERATURE = 0.7


def run_analysis(
    *,
    data_dir: str | Path,
    config: FlowConfig,
    run_dir: str | Path,
    run_id: Optional[str] = None,
    image: str = DEFAULT_IMAGE,
    temperature: float = 0.0,
    on_step: Optional[StepHook] = None,
) -> tuple[AgentResult, RunMetadata]:
    """Execute one full trajectory inside Docker and persist all artifacts.

    Raises DockerUnavailableError (with guidance) if Docker/image is missing — FLOW
    never falls back to host execution.
    """
    data_dir = Path(data_dir).resolve()
    run_dir = Path(run_dir).resolve()
    run_id = run_id or uuid.uuid4().hex[:12]

    # Hard safety check: refuse to proceed without Docker.
    require_docker(image)

    # Profile the dataset to build a neutral, structural description for the agent.
    profile = profile_project(data_dir)
    dataset_description = config.dataset.description or ""
    dataset_description = (dataset_description + "\n\n" + profile.dataset_description()).strip()

    limits = DockerLimits(
        memory=config.safety.memory,
        cpus=config.safety.cpus,
        pids_limit=config.safety.pids_limit,
        network="bridge" if config.safety.allow_network else "none",
    )

    trajectory = Trajectory(run_dir)
    kernel = DockerKernel(
        data_dir=data_dir,
        work_dir=run_dir,
        container_name=f"flow-{run_id}",
        image=image,
        limits=limits,
    )
    env = NotebookEnvironment(
        kernel=kernel,
        question=config.question,
        dataset_description=dataset_description,
        max_steps=config.runtime.max_steps,
        per_cell_timeout=config.runtime.per_cell_timeout,
        per_trajectory_timeout=config.runtime.per_trajectory_timeout,
    )
    provider = build_provider(config.runtime.provider, config.runtime.model, temperature)

    meta = RunMetadata(
        run_id=run_id,
        question=config.question,
        provider=config.runtime.provider,
        model=config.runtime.model,
        image=image,
        image_digest=image_digest(image),
        max_steps=config.runtime.max_steps,
        limits={
            "memory": limits.memory,
            "cpus": limits.cpus,
            "pids_limit": limits.pids_limit,
            "per_cell_timeout": config.runtime.per_cell_timeout,
            "per_trajectory_timeout": config.runtime.per_trajectory_timeout,
        },
        started_at=time.time(),
        network=limits.network,
    )

    def _hook(rec: StepRecord) -> None:
        # Persist each step immediately so the UI can tail it live.
        trajectory.append_action(
            {
                "step": rec.step,
                "tool": rec.tool,
                "arguments": rec.arguments,
                "observation": rec.observation,
                "done": rec.done,
                "reward": rec.reward,
            }
        )
        trajectory.write_notebook(env)
        if on_step:
            on_step(rec)

    agent = ReActAgent(provider, env, on_step=_hook)
    try:
        result = agent.run()
    finally:
        kernel.shutdown()

    trajectory.finalize(env, result, meta)
    return result, meta


# --------------------------------------------------------------------------- batches
TrajectoryStatusHook = Callable[[int, str, dict], None]
BatchStepHook = Callable[[int, StepRecord], None]
PhaseHook = Callable[[str], None]


@dataclass
class BatchResult:
    """Outcome of a consensus batch: per-trajectory summaries + the consensus."""

    batch_id: str
    n_trajectories: int
    temperature: float
    trajectories: list[dict[str, Any]] = field(default_factory=list)
    consensus: Optional[dict[str, Any]] = None


def run_batch(
    *,
    data_dir: str | Path,
    config: FlowConfig,
    batch_dir: str | Path,
    n_trajectories: int,
    batch_id: Optional[str] = None,
    image: str = DEFAULT_IMAGE,
    temperature: Optional[float] = None,
    on_phase: Optional[PhaseHook] = None,
    on_trajectory_status: Optional[TrajectoryStatusHook] = None,
    on_step: Optional[BatchStepHook] = None,
) -> BatchResult:
    """Run N independent trajectories, then synthesize a consensus meta-analysis.

    Each trajectory gets a **fully isolated** environment: its own deep-copied config, its
    own artifact directory (``trajectories/<idx>``), and its own Docker container
    (``flow-<batch_id>-<idx>``) with its own persistent kernel. A failure in one trajectory
    is recorded and does not abort the others. After all complete, the meta-analysis reads
    only the submitted conclusions (text) and produces a consensus — no code, no dataset.
    """
    data_dir = Path(data_dir).resolve()
    batch_dir = Path(batch_dir).resolve()
    batch_id = batch_id or uuid.uuid4().hex[:12]
    n = max(1, int(n_trajectories))
    temp = temperature if temperature is not None else (0.0 if n == 1 else DEFAULT_BATCH_TEMPERATURE)

    # Fail fast and clearly if Docker is unavailable, before launching any trajectory.
    require_docker(image)

    traj_root = batch_dir / "trajectories"
    traj_root.mkdir(parents=True, exist_ok=True)

    if on_phase:
        on_phase("running")

    results: list[dict[str, Any]] = []
    for idx in range(n):
        traj_dir = traj_root / str(idx)
        run_id = f"{batch_id}-{idx}"
        # Deep-copy the config so trajectories share no mutable state whatsoever.
        cfg_i = config.model_copy(deep=True)
        if on_trajectory_status:
            on_trajectory_status(idx, "running", {})
        try:
            result, _meta = run_analysis(
                data_dir=data_dir,
                config=cfg_i,
                run_dir=traj_dir,
                run_id=run_id,
                image=image,
                temperature=temp,
                on_step=(lambda rec, i=idx: on_step(i, rec)) if on_step else None,
            )
            summary = {
                "idx": idx,
                "submitted": result.submitted,
                "answer": result.answer,
                "failure_reason": result.failure_reason,
                "steps": len(result.steps),
                "artifact_dir": str(traj_dir),
            }
            status = "completed" if result.submitted else "failed"
        except Exception as e:  # kernel/start failure — isolate it, keep going
            summary = {
                "idx": idx,
                "submitted": False,
                "answer": None,
                "failure_reason": str(e),
                "steps": 0,
                "artifact_dir": str(traj_dir),
            }
            status = "error"
        results.append(summary)
        if on_trajectory_status:
            on_trajectory_status(idx, status, summary)

    # --- consensus meta-analysis (text synthesis only) ---
    # Use a dedicated meta model if configured, else fall back to the trajectory model.
    meta_provider_name = config.runtime.meta_provider or config.runtime.provider
    meta_model_name = config.runtime.meta_model or config.runtime.model
    if on_phase:
        on_phase("synthesizing")
    try:
        meta_provider = build_provider(meta_provider_name, meta_model_name, 0.0)
        consensus = synthesize_consensus(meta_provider, config.question, results)
    except Exception as e:  # e.g. missing API key for the synthesis call
        n_sub = sum(1 for r in results if r["submitted"])
        consensus = ConsensusResult(
            consensus=None, synthesized=False, n_submitted=n_sub, n_total=n,
            failure_reason=f"Consensus synthesis failed: {e}",
        )

    # --- persist batch artifacts (audit trail) ---
    (batch_dir / "consensus.txt").write_text(consensus.consensus or "")
    batch_record = {
        "batch_id": batch_id,
        "question": config.question,
        "provider": config.runtime.provider,
        "model": config.runtime.model,
        "meta_provider": meta_provider_name,
        "meta_model": meta_model_name,
        "n_trajectories": n,
        "temperature": temp,
        "consensus": asdict(consensus),
        "trajectories": [
            {k: v for k, v in r.items() if k != "answer"} | {"answer": r["answer"]}
            for r in results
        ],
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch_record, indent=2, default=str))

    if on_phase:
        on_phase("done")

    return BatchResult(
        batch_id=batch_id,
        n_trajectories=n,
        temperature=temp,
        trajectories=results,
        consensus=asdict(consensus),
    )
