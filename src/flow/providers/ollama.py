"""Ollama provider — local models such as qwen2.5-coder.

Talks to a local Ollama daemon (default http://localhost:11434) via /api/chat with
tool support. Fully offline once the model is pulled (``ollama pull qwen2.5-coder``).
No API key required.
"""

from __future__ import annotations

import os

import httpx

from flow.aviary.message import Message
from flow.aviary.tool import Tool, ToolCall
from flow.providers.base import Provider
from flow.providers.openai_compat import ProviderError


class OllamaProvider(Provider):
    """Local Ollama chat provider with tool calling."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen2.5-coder",
        base_url: str | None = None,
        timeout: float = 180.0,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, messages: list[Message], tools: list[Tool]) -> ToolCall:
        payload = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "tools": [t.to_openai() for t in tools],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.HTTPError as e:  # pragma: no cover - network dependent
            raise ProviderError(
                f"Could not reach Ollama at {self.base_url}: {e}. Is `ollama serve` running?"
            ) from e
        if r.status_code != 200:
            raise ProviderError(f"Ollama returned HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            raise ProviderError(
                "Ollama model did not return a tool call. It said: "
                + (msg.get("content") or "")[:500]
            )
        fn = tool_calls[0].get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            import json

            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return ToolCall(name=fn.get("name", ""), arguments=args or {})
