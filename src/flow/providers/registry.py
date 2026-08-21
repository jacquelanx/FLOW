"""Provider registry — build a Provider from a name + model.

Keeps provider wiring in one place so the CLI, API, and runner stay decoupled from any
specific vendor. Hosted endpoints are configured by (base_url, api_key_env) pairs; the
key is read from the host environment at call time.
"""

from __future__ import annotations

from flow.providers.base import Provider
from flow.providers.mock import MockProvider

# Known OpenAI-compatible endpoints. base_url is overridable via the matching *_BASE_URL
# env var if a provider changes its endpoint.
_OPENAI_COMPAT = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}

# Azure OpenAI is handled separately from the plain OpenAI-compatible endpoints because
# its URL/auth differ and it is configured per-resource from the host environment:
#   AZURE_OPENAI_ENDPOINT      e.g. https://my-resource.openai.azure.com   (required)
#   AZURE_OPENAI_API_KEY       the resource key                            (required)
#   AZURE_OPENAI_API_VERSION   REST api-version                            (optional)
_AZURE_PROVIDERS = {"azure", "azure_openai"}

SUPPORTED_PROVIDERS = [
    "mock",
    "ollama",
    "azure",
    *sorted(_OPENAI_COMPAT.keys()),
]

# Curated, commonly-available models per provider, for the UI dropdowns. This is a
# convenience catalog only — any model id the provider serves still works (the UI offers a
# "custom" escape), and model availability changes over time, so treat this as defaults.
MODEL_CATALOG: dict[str, list[str]] = {
    "mock": ["mock"],
    "ollama": ["qwen2.5-coder", "qwen2.5", "llama3.1", "deepseek-coder-v2"],
    # Every id previously listed here (2.0-flash, 2.5-flash, 1.5-flash, 1.5-pro) is now dead:
    # 2.5-flash answers 404 "no longer available" and the rest are absent from the live model
    # list. These were verified against GET /v1beta/models and probed 5x each with retries
    # disabled. Ordered by what a FREE-TIER key can actually sustain, which is NOT the same as
    # newest-first: the newest Flash has a tiny free-tier request quota (3.7-flash managed 2/5
    # calls, and the "-latest" alias, which tracks whichever Flash is newest, managed 0/5 —
    # both 429 RESOURCE_EXHAUSTED). Pro models are zero-quota on the free tier
    # ("free_tier_requests, limit: 0"), so they need billing enabled.
    # Ordered by DAILY free-tier request budget, because a FLOW trajectory spends one request
    # per step (up to max_steps) and a model that cannot cover that never finishes:
    #   3.1-flash-lite  35+ requests/day observed, no cap hit  -> can complete a trajectory
    #   3.6-flash       "free_tier_requests, limit: 20"        -> dies mid-trajectory
    #   3.7-flash       tighter still (2/5 calls succeeded)
    #   *-latest        aliases the newest, so it inherits the tightest budget (0/5)
    #   *-pro           "limit: 0" on the free tier           -> needs billing
    # With billing enabled the order should be reversed: the larger models are more capable,
    # and 3.6-flash handled FLOW's context and tool calls cleanly for as long as its quota
    # lasted. Latency is a rounding error next to the daily cap.
    "gemini": [
        "gemini-3.1-flash-lite",   # ~1s/call,  free-tier budget covers a full run
        "gemini-3.6-flash",        # ~10s/call, more capable, 20 requests/day free
        "gemini-3.5-flash",        # ~5s/call
        "gemini-3.7-flash",        # newest; free-tier budget too small to be usable
    ],
    # Groq retires model ids without notice — the llama-3.3/3.1 ids that used to head this
    # list now 404 with model_not_found. These were verified against GET /v1/models and each
    # completed a trajectory through the ReAct loop. Note that groq/compound* are excluded on
    # purpose: they reject tool calling outright ("`tool calling` is not supported with this
    # model"), which FLOW's loop requires. Also note the FREE-tier request-size cap: a free key
    # rejects any request over ~8k tokens with HTTP 413 "Request too large" (a permanent
    # rejection, not a retryable 429), and a FLOW trajectory starts around 6k tokens and grows
    # every step. Groq needs a paid tier for this workload regardless of which model is chosen.
    "groq": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ],
    "openrouter": [
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
        "google/gemini-2.0-flash-exp:free",
    ],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "openai": ["gpt-4o-mini", "gpt-4o", "o4-mini"],
    # For Azure the "model" is your deployment name, so these are placeholders/examples —
    # replace with whatever you named your deployment in the Azure portal.
    "azure": ["gpt-35-turbo", "gpt-4o-mini", "gpt-4o"],
}

# Providers shown in the UI (excludes the "google" alias of "gemini" to avoid confusion).
UI_PROVIDERS: list[str] = [p for p in SUPPORTED_PROVIDERS if p != "google"]


def default_model(provider: str) -> str:
    """Return a sensible default model id for a provider (first in the catalog)."""
    models = MODEL_CATALOG.get((provider or "").strip().lower())
    return models[0] if models else ""


def build_provider(provider: str, model: str, temperature: float = 0.0) -> Provider:
    """Construct a Provider instance for ``provider``/``model``.

    ``temperature`` lets multi-trajectory runs diversify (the consensus pipeline samples
    several independent trajectories at a higher temperature, then synthesizes them).

    Raises ValueError for unknown providers and ProviderError for missing keys.
    """
    import os

    key = (provider or "").strip().lower()
    if key == "mock":
        return MockProvider(model=model or "mock")
    if key == "ollama":
        from flow.providers.ollama import OllamaProvider

        return OllamaProvider(model=model or "qwen2.5-coder", temperature=temperature)
    if key in _AZURE_PROVIDERS:
        from flow.providers.openai_compat import (
            DEFAULT_AZURE_API_VERSION,
            AzureOpenAIProvider,
        )

        return AzureOpenAIProvider(
            model=model,
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION
            ),
            # Direct Azure uses "api-key"; APIM gateways use "Ocp-Apim-Subscription-Key".
            auth_header=os.environ.get("AZURE_OPENAI_AUTH_HEADER", "api-key"),
            temperature=temperature,
        )
    if key in _OPENAI_COMPAT:
        from flow.providers.openai_compat import OpenAICompatibleProvider

        cfg = _OPENAI_COMPAT[key]
        base_url = os.environ.get(f"{key.upper()}_BASE_URL", cfg["base_url"])
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key_env=cfg["api_key_env"],
            name=key,
            temperature=temperature,
        )
    raise ValueError(
        f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
