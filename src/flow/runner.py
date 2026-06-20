"""Run orchestration — wires provider + Docker kernel + environment + agent + artifacts.

This is the single entry point used by both the CLI and the API. It enforces the safety
guarantee: real runs always execute inside Docker via ``DockerKernel``. Tests construct
the environment directly with ``InProcessKernel`` and never go through here.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from flow.agent.react import AgentResult, ReActAgent, StepRecord
from flow.config import FlowConfig
from flow.env.docker_runtime import DEFAULT_IMAGE, DockerLimits, image_digest, require_docker
from flow.env.kernel import DockerKernel
from flow.env.notebook_env import NotebookEnvironment
from flow.providers.registry import build_provider
from flow.trajectory import RunMetadata, Trajectory
from flow.validation import profile_project

StepHook = Callable[[StepRecord], None]


def run_analysis(
    *,
    data_dir: str | Path,
    config: FlowConfig,
    run_dir: str | Path,
    run_id: Optional[str] = None,
    image: str = DEFAULT_IMAGE,
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
    provider = build_provider(config.runtime.provider, config.runtime.model)

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
