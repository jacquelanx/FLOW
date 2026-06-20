"""Message abstraction — mirrors Aviary/OpenAI chat messages.

A ``Message`` is the unit of conversation state passed between the agent loop and
the provider. It carries a role, optional text content, optional tool calls (for
assistant turns), and an optional ``tool_call_id`` (for tool-result turns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single chat message.

    Attributes:
        role: one of system/user/assistant/tool.
        content: free-text content (may be empty when the turn is purely tool calls).
        tool_calls: list of tool-call dicts emitted by an assistant turn.
        tool_call_id: links a ``tool`` message back to the assistant tool call it answers.
        name: optional tool/function name for tool messages.
    """

    role: Role
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_openai(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible chat message dict."""
        msg: dict[str, Any] = {"role": self.role}
        # Tool messages require content; assistant tool-call turns may omit it.
        if self.role == "tool":
            msg["content"] = self.content
            if self.tool_call_id:
                msg["tool_call_id"] = self.tool_call_id
            if self.name:
                msg["name"] = self.name
            return msg
        msg["content"] = self.content or ""
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg
