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

    def _headers(self) -> dict:
        """Auth + content headers for the request (overridable per vendor)."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_url(self) -> str:
        """Full chat/completions URL (overridable per vendor)."""
        return f"{self.base_url}/chat/completions"

    def _post(self, payload: dict) -> "httpx.Response":
        """POST to chat/completions with transient-error retry + backoff.

        Returns the final response (the caller inspects its status). Retries 429/503/5xx
        and network blips with honored Retry-After; never retries a hard 'limit: 0' quota
        refusal. Raises ProviderError only on exhausted network failures.
        """
        headers = self._headers()
        url = self._chat_url()
        last_err = ""
        r = None
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(
                    url,
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
                return r
            body = r.text[:500]
            hard_quota = r.status_code == 429 and "limit: 0" in body
            if (
                r.status_code in _TRANSIENT_STATUSES
                and not hard_quota
                and attempt < self.max_retries
            ):
                time.sleep(self._backoff_delay(attempt, r))
                last_err = f"{self.name} returned HTTP {r.status_code}: {body}"
                continue
            return r  # non-transient, or retries exhausted — caller handles status
        return r  # pragma: no cover - loop always returns

    @staticmethod
    def _http_error(name: str, r: "httpx.Response") -> ProviderError:
        body = r.text[:500]
        kind = "rate limit / overload" if r.status_code in (429, 503) else "error"
        return ProviderError(f"{name} returned HTTP {r.status_code} ({kind}): {body}")

    def generate(self, messages: list[Message], tools: list[Tool]) -> ToolCall:
        payload = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "tools": [t.to_openai() for t in tools],
            "tool_choice": "required",
            "temperature": self.temperature,
        }
        r = self._post(payload)
        # Some providers reject tool_choice="required"; retry once with "auto".
        if r.status_code in (400, 422) and payload.get("tool_choice") == "required":
            payload["tool_choice"] = "auto"
            r = self._post(payload)
        if r.status_code != 200:
            raise self._http_error(self.name, r)
        return self._parse(r.json())

    def complete(self, messages: list[Message]) -> str:
        """Free-text chat completion (no tools) — used by the consensus synthesis."""
        payload = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self.temperature,
        }
        r = self._post(payload)
        if r.status_code != 200:
            raise self._http_error(self.name, r)
        data = r.json()
        try:
            return data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Malformed response from {self.name}: {data}") from e

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


# Default Azure REST API version. Overridable via AZURE_OPENAI_API_VERSION. This GA
# version supports chat completions + function calling on gpt-35-turbo / gpt-4* deployments.
DEFAULT_AZURE_API_VERSION = "2024-10-21"


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Azure OpenAI Service provider.

    Azure speaks the same Chat Completions request/response shape as OpenAI, so all the
    payload building, retry, and parsing logic is inherited. Only the wire details differ:

      * URL: ``{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...``
      * Auth: an ``api-key`` header (not ``Authorization: Bearer``).
      * The model is identified by the **deployment name** in the URL, not the request body.

    ``model`` here is the Azure *deployment name* found in the portal, which is often 
    different from the base model id. The agent forces tool use, so the deployment must 
    be a chat model that supports function calling (gpt-35-turbo 0613+ or any gpt-4* model).
    Legacy completion models such as text-davinci-003 will not work.
    """

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        api_version: str = DEFAULT_AZURE_API_VERSION,
        auth_header: str = "api-key",
        name: str = "azure",
        **kwargs,
    ):
        if not endpoint:
            raise ProviderError(
                "Missing Azure endpoint: set the AZURE_OPENAI_ENDPOINT environment "
                "variable on the host (e.g. https://my-resource.openai.azure.com). Behind "
                "an API Management gateway this includes the route prefix, e.g. "
                "https://apimd.mdanderson.edu/dig/foundry"
            )
        if not model:
            raise ProviderError(
                "Azure requires a deployment name: set runtime.model to the deployment "
                "you created in the Azure portal (not the base model id)."
            )
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        # Direct Azure resources authenticate with an "api-key" header; API Management
        # gateways usually want "Ocp-Apim-Subscription-Key". Configurable so both work.
        self.auth_header = auth_header or "api-key"
        # base_url is unused for Azure (URL is built from endpoint + deployment), but the
        # parent stores it; pass the endpoint through for a sensible repr.
        super().__init__(
            model=model,
            base_url=self.endpoint,
            api_key_env=api_key_env,
            name=name,
            **kwargs,
        )

    def _headers(self) -> dict:
        return {self.auth_header: self.api_key, "Content-Type": "application/json"}

    def _chat_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.model}"
            f"/chat/completions?api-version={self.api_version}"
        )
