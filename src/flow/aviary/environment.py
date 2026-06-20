"""Environment abstraction — mirrors Aviary's ``Environment`` contract.

The agent never executes code itself; it proposes ``ToolCall``s and the Environment
executes them and returns observations. This is the heart of the Aviary design and
of Finch: the agent reasons, the environment acts and observes.

Contract (intentionally identical to fhaviary so it can be swapped in):
    reset() -> (observation: str, tools: list[Tool])
    step(action: ToolCall) -> (observation: str, reward: float, done: bool, info: dict)
    tools() -> list[Tool]
"""

from __future__ import annotations

import abc
from typing import Any

from flow.aviary.tool import Tool, ToolCall


class Environment(abc.ABC):
    """Abstract base for FLOW environments."""

    @abc.abstractmethod
    def reset(self) -> tuple[str, list[Tool]]:
        """Initialise the environment and return (initial observation, tools)."""

    @abc.abstractmethod
    def step(self, action: ToolCall) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute one tool call; return (observation, reward, done, info)."""

    @abc.abstractmethod
    def tools(self) -> list[Tool]:
        """Return the tools advertised to the agent."""
