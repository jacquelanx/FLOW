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
    "gemini": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct",
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
