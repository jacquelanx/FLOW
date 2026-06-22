"""Provider base class and the generate() contract."""

from __future__ import annotations

import abc

from flow.aviary.message import Message
from flow.aviary.tool import Tool, ToolCall


class Provider(abc.ABC):
    """Abstract LLM provider: messages + tools -> a single ToolCall."""

    name: str = "provider"
    model: str = ""

    @abc.abstractmethod
    def generate(self, messages: list[Message], tools: list[Tool]) -> ToolCall:
        """Return exactly one ToolCall for the next agent step.

        Implementations should force tool use. If the underlying model returns plain
        text instead of a tool call, wrap it so the agent loop can recover (e.g. by
        nudging the model), but never fabricate analysis here.
        """

    def complete(self, messages: list[Message]) -> str:
        """Return a free-text completion (no tools).

        Used by the consensus meta-analysis to synthesize trajectory conclusions. This is
        text synthesis, not data analysis — it executes no code and touches no dataset.
        Providers that cannot do plain completions may leave this unimplemented.
        """
        raise NotImplementedError(
            f"{self.name} does not support free-text completion."
        )
