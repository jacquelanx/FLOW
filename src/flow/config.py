"""Configuration schema for a FLOW run.

The user-editable ``config.yaml`` is **analysis-free** by design. It only declares:
  * where the dataset files live,
  * the research question,
  * runtime settings (provider/model/budgets),
  * safety settings (resource limits, network policy, privacy boundary).

It contains NO analysis logic — no gates, populations, thresholds, transforms, or
dataset-specific branches. Experiment FACTS belong in ``metadata.json`` (what the data
IS), and the agent derives all analysis at runtime. The loader actively rejects any
attempt to smuggle analysis fields into config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class RuntimeConfig(BaseModel):
    """How the agent runs (not what it analyzes)."""

    provider: str = "mock"
    model: str = "mock"
    max_steps: int = 30
    per_cell_timeout: float = 120.0
    per_trajectory_timeout: float = 1800.0
    # Consensus meta-analysis model. Blank = use the same provider/model as the
    # trajectories. Set these to synthesize the consensus with a different (e.g. stronger)
    # model than the one that ran the trajectories. This is a runtime choice, not analysis.
    meta_provider: str = ""
    meta_model: str = ""


class SafetyConfig(BaseModel):
    """Sandbox + privacy settings."""

    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 256
    allow_network: bool = False  # opt-in; enables dependency installs in-container
    # Privacy boundary: by default only schemas/summaries may be sent to a hosted model.
    allow_raw_data_to_model: bool = False


class DatasetConfig(BaseModel):
    """Where the data lives (relative to the project directory)."""

    # The agent receives the whole directory read-only; these are hints/labels only.
    root: str = "."
    description: str = ""


class FlowConfig(BaseModel):
    """Top-level config for a run."""

    question: str = Field(..., description="The research question to answer.")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @field_validator("question")
    @classmethod
    def _question_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("config.question must be a non-empty research question.")
        return v.strip()


# Keys that would indicate analysis logic leaking into config. The loader refuses these
# so config.yaml can never become a place to hardcode the science.
_FORBIDDEN_ANALYSIS_KEYS = {
    "gates",
    "gating",
    "populations",
    "thresholds",
    "transform",
    "transforms",
    "markers",
    "marker_map",
    "pipeline",
    "steps",
    "analysis",
    "compensation",
}


class ConfigError(ValueError):
    """Raised when config is malformed or contains analysis logic."""


def _assert_no_analysis_logic(raw: dict[str, Any]) -> None:
    found = _FORBIDDEN_ANALYSIS_KEYS.intersection({str(k).lower() for k in raw.keys()})
    # also check nested under any top-level mapping
    for v in raw.values():
        if isinstance(v, dict):
            found |= _FORBIDDEN_ANALYSIS_KEYS.intersection(
                {str(k).lower() for k in v.keys()}
            )
    if found:
        raise ConfigError(
            "config.yaml must not contain analysis logic. Remove these keys: "
            f"{sorted(found)}. The agent derives all analysis from the data + question + "
            "metadata.json at runtime."
        )


def load_config(path: str | Path) -> FlowConfig:
    """Load and validate a config.yaml, rejecting any analysis logic."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must be a mapping at the top level.")
    _assert_no_analysis_logic(raw)
    try:
        return FlowConfig.model_validate(raw)
    except Exception as e:  # pydantic ValidationError
        raise ConfigError(f"Invalid config.yaml: {e}") from e


def default_config_yaml(question: str = "State your research question here.") -> str:
    """Return a commented default config.yaml string for the UI/CLI to seed."""
    return f"""\
# FLOW run configuration — ANALYSIS-FREE.
# This file declares WHERE the data is, WHAT to ask, and HOW to run safely.
# It must NOT contain gates, populations, thresholds, transforms, or any analysis
# logic. Experiment facts (which detector carries which stain, key dates, etc.) go in
# metadata.json. The agent derives ALL analysis at runtime.

question: "{question}"

dataset:
  root: "."
  description: ""   # optional free-text note about the dataset (not analysis)

runtime:
  provider: mock           # mock | ollama | gemini | groq | openrouter | deepseek | openai
  model: mock              # e.g. gemini-2.0-flash, qwen2.5-coder, llama-3.3-70b-versatile
  max_steps: 30
  per_cell_timeout: 120
  per_trajectory_timeout: 1800
  meta_provider: ""        # consensus synthesis provider (blank = same as provider above)
  meta_model: ""           # consensus synthesis model (blank = same as model above)

safety:
  memory: "4g"
  cpus: "2"
  pids_limit: 256
  allow_network: false             # opt-in to let the agent install dependencies
  allow_raw_data_to_model: false   # privacy: keep raw event data off hosted models
"""
