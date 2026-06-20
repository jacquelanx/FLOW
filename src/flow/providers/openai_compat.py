"""OpenAI-compatible provider — Gemini, Groq, OpenRouter, DeepSeek, OpenAI, etc.

Any endpoint that speaks the OpenAI Chat Completions API with function calling works
here. Configure via ``base_url`` + an API key read from the environment. Free tiers:

  * Google Gemini  : base_url=https://generativelanguage.googleapis.com/v1beta/openai
                     model=gemini-2.0-flash, key env GEMINI_API_KEY (or GOOGLE_API_KEY)
  * Groq           : base_url=https://api.groq.com/openai/v1, key env GROQ_API_KEY
  * OpenRouter     : base_url=https://openrouter.ai/api/v1, key env OPENROUTER_API_KEY
  * DeepSeek       : base_url=https://api.deepseek.com, key env DEEPSEEK_API_KEY

Keys are read from the host environment only and are never sent to the container.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import httpx

from flow.aviary.message import Message
from flow.aviary.tool import Tool, ToolCall
from flow.providers.base import Provider

# HTTP statuses worth retrying: server overload / transient unavailability, plus 429
# (rate limit) UNLESS it's a hard "limit: 0" quota refusal, which is handled separately.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """Raised on transport/auth/parse failures from a hosted provider."""


class OpenAICompatibleProvider(Provider):
    """Generic OpenAI-compatible chat provider with forced tool use."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        name: str = "openai-compatible",
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        max_backoff: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.name = name
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise ProviderError(
                f"Missing API key: set the {api_key_env} environment variable on the host."
            )

    def generate(self, messages: list[Message], tools: list[Tool]) -> ToolCall:
        payload = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "tools": [t.to_openai() for t in tools],
            "tool_choice": "required",
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as e:  # transient network failure — retry
                last_err = f"Request to {self.name} failed: {e}"
                if attempt < self.max_retries:
                    time.sleep(self._backoff_delay(attempt, None))
                    continue
                raise ProviderError(last_err) from e

            if r.status_code == 200:
                return self._parse(r.json())

            body = r.text[:500]

            # Some providers reject tool_choice="required"; retry once with "auto".
            if r.status_code in (400, 422) and payload.get("tool_choice") == "required":
                payload["tool_choice"] = "auto"
                continue

            # Retry transient overload/rate errors — but NOT a hard "limit: 0" quota
            # refusal, which will never succeed on retry. For 429/503 we honor the
            # server's Retry-After (e.g. Gemini's per-minute rate window) so a multi-step
            # run rides out throttling instead of failing the whole trajectory.
            hard_quota = r.status_code == 429 and "limit: 0" in body
            if r.status_code in _TRANSIENT_STATUSES and not hard_quota and attempt < self.max_retries:
                time.sleep(self._backoff_delay(attempt, r))
                last_err = f"{self.name} returned HTTP {r.status_code}: {body}"
                continue

            kind = "rate limit / overload" if r.status_code in (429, 503) else "error"
            raise ProviderError(
                f"{self.name} returned HTTP {r.status_code} ({kind}) after "
                f"{attempt + 1} attempt(s): {body}"
            )

        raise ProviderError(last_err or f"{self.name}: exhausted retries")

    def _backoff_delay(self, attempt: int, response: object | None) -> float:
        """Seconds to wait before the next attempt.

        Uses jittered exponential backoff, but defers to the server's Retry-After hint
        (header or a ``retryDelay: "26s"`` in the error body) when it asks for longer —
        capped at ``max_backoff`` so a run never blocks unboundedly.
        """
        delay = self.backoff_base ** attempt + random.uniform(0, 0.5)
        hinted = self._retry_after_seconds(response)
        if hinted is not None:
            delay = max(delay, hinted)
        return min(delay, self.max_backoff)

    @staticmethod
    def _retry_after_seconds(response: object | None) -> float | None:
        """Extract a server-suggested retry delay from headers or body, if any."""
        if response is None:
            return None
        headers = getattr(response, "headers", {}) or {}
        try:
            ra = headers.get("retry-after")
        except AttributeError:
            ra = None
        if ra:
            try:
                return float(ra)
            except (TypeError, ValueError):
                pass
        text = getattr(response, "text", "") or ""
        m = re.search(r'ret[rR]y[dD]elay"?\s*:?\s*"?(\d+(?:\.\d+)?)s', text)
        if m:
            return float(m.group(1))
        return None

    def _parse(self, data: dict) -> ToolCall:
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Malformed response from {self.name}: {data}") from e

        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            # Model replied with text instead of a tool call — surface it so the loop
            # can nudge the model; do not invent analysis.
            text = choice.get("content") or ""
            raise ProviderError(
                "Model did not return a tool call. It said: " + text[:500]
            )
        call = tool_calls[0]
        fn = call.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {}
        # Preserve the provider's original tool-call object so fields the API requires on
        # replay (e.g. Gemini's thought_signature) are echoed back verbatim next turn.
        return ToolCall(name=name, arguments=args, id=call.get("id"), raw=call)
