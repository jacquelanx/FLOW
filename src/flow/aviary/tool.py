"""Tool and ToolCall abstractions — mirror Aviary tools / OpenAI function calling.

A ``Tool`` wraps a Python callable with a JSON-schema parameter description that is
advertised to the LLM. A ``ToolCall`` is the agent's request to invoke one tool with
arguments. In FLOW the Environment exposes **exactly two** tools (``edit_cell`` and
``submit_answer``) — this is a hard fidelity constraint to Finch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """A callable advertised to the model.

    Attributes:
        name: the function name the model calls.
        description: natural-language description shown to the model.
        parameters: JSON Schema (object) describing the arguments.
        fn: the host-side implementation invoked by the Environment.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def to_openai(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible ``tools`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A request by the agent to call a tool.

    Attributes:
        name: tool name.
        arguments: parsed keyword arguments.
        id: provider-assigned call id (used to correlate the tool result message).
        raw: the provider's original tool-call object, preserved verbatim. Some providers
            (e.g. Gemini) attach fields like ``thought_signature`` that MUST be echoed back
            on the next request, so the agent loop replays ``raw`` when present instead of
            a reconstructed minimal call.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    def to_openai_assistant_toolcall(self) -> dict[str, Any]:
        """Render this call as it would appear on an assistant message.

        Prefers the provider's original object (``raw``) so provider-specific fields the
        API requires on replay (e.g. Gemini's ``thought_signature``) are not lost.
        """
        if self.raw:
            return self.raw
        import json

        return {
            "id": self.id or f"call_{self.name}",
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }
