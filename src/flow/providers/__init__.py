"""LLM provider abstraction.

A provider turns a conversation + tool schemas into a single ``ToolCall`` (Finch emits
exactly one tool call per step). Providers run **host-side**; API keys live on the host
and never enter the container.

Implementations:
  * ``MockProvider``     — deterministic, offline; drives tests. Emits only a MINIMAL,
    GENERIC cell (load data + print shape) then submits — never any real analysis.
  * ``OllamaProvider``   — local models (e.g. qwen2.5-coder) via the Ollama API.
  * ``OpenAICompatibleProvider`` — any OpenAI-compatible endpoint: Google Gemini,
    Groq, OpenRouter, DeepSeek, etc. (base_url + key from env).
"""

from flow.providers.base import Provider
from flow.providers.mock import MockProvider
from flow.providers.registry import build_provider

__all__ = ["Provider", "MockProvider", "build_provider"]
