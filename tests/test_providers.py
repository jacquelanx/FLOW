"""Provider tests: registry wiring, mock behavior, OpenAI-compatible parsing."""

import pytest

from flow.providers.mock import MockProvider
from flow.providers.registry import SUPPORTED_PROVIDERS, build_provider


def test_registry_builds_mock():
    p = build_provider("mock", "mock")
    assert isinstance(p, MockProvider)


def test_registry_unknown_provider():
    with pytest.raises(ValueError):
        build_provider("not-a-provider", "x")


def test_registry_missing_key_errors(monkeypatch):
    from flow.providers.openai_compat import ProviderError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        build_provider("gemini", "gemini-2.0-flash")


def test_supported_providers_listed():
    for name in ["mock", "ollama", "gemini", "groq", "openrouter", "deepseek"]:
        assert name in SUPPORTED_PROVIDERS


def test_mock_emits_generic_then_submits():
    from flow.aviary.message import Message

    p = MockProvider()
    # First call -> edit_cell.
    c1 = p.generate([], [])
    assert c1.name == "edit_cell"
    # Simulate one prior assistant tool call -> submit_answer.
    prior = [Message(role="assistant", tool_calls=[{"id": "x"}])]
    c2 = p.generate(prior, [])
    assert c2.name == "submit_answer"
    assert "MOCK" in c2.arguments["answer"]


def test_openai_compatible_parse(monkeypatch):
    from flow.providers.openai_compat import OpenAICompatibleProvider

    monkeypatch.setenv("FAKE_KEY", "abc")
    prov = OpenAICompatibleProvider(
        model="m", base_url="http://x", api_key_env="FAKE_KEY", name="fake"
    )
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "edit_cell",
                                "arguments": '{"index": "new", "source": "1+1"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    call = prov._parse(data)
    assert call.name == "edit_cell"
    assert call.arguments["source"] == "1+1"


def test_openai_compatible_retries_transient_503(monkeypatch):
    """A transient 503 should be retried and then succeed (no failed run)."""
    import httpx

    from flow.providers.openai_compat import OpenAICompatibleProvider

    monkeypatch.setenv("FAKE_KEY", "abc")
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real backoff in tests

    calls = {"n": 0}
    ok_body = {
        "choices": [
            {"message": {"tool_calls": [{"id": "c1", "function": {"name": "submit_answer", "arguments": "{}"}}]}}
        ]
    }

    class Resp:
        def __init__(self, status, data=None, text=""):
            self.status_code = status
            self._data = data
            self.text = text

        def json(self):
            return self._data

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            return Resp(503, text="high demand")
        return Resp(200, data=ok_body)

    monkeypatch.setattr(httpx, "post", fake_post)
    prov = OpenAICompatibleProvider(
        model="m", base_url="http://x", api_key_env="FAKE_KEY", name="fake", backoff_base=0.0
    )
    call = prov.generate([], [])
    assert call.name == "submit_answer"
    assert calls["n"] == 3  # two 503s, then success


def test_openai_compatible_hard_quota_not_retried(monkeypatch):
    """A 429 'limit: 0' refusal must fail fast — retrying would never help."""
    import httpx

    from flow.providers.openai_compat import OpenAICompatibleProvider, ProviderError

    monkeypatch.setenv("FAKE_KEY", "abc")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    class Resp:
        status_code = 429
        text = '{"error":{"message":"Quota exceeded ... limit: 0, model: gemini-2.0-flash"}}'

        def json(self):
            return {}

    def fake_post(*a, **k):
        calls["n"] += 1
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    prov = OpenAICompatibleProvider(
        model="m", base_url="http://x", api_key_env="FAKE_KEY", name="fake", backoff_base=0.0
    )
    with pytest.raises(ProviderError):
        prov.generate([], [])
    assert calls["n"] == 1  # no retries on a hard quota refusal


def test_retry_after_parsing():
    """Server-suggested retry delays (header or body retryDelay) are honored."""
    from flow.providers.openai_compat import OpenAICompatibleProvider as P

    class H:
        def __init__(self, d):
            self._d = d

        def get(self, k):
            return self._d.get(k)

    class Resp:
        def __init__(self, headers=None, text=""):
            self.headers = H(headers or {})
            self.text = text

    assert P._retry_after_seconds(Resp(headers={"retry-after": "12"})) == 12.0
    assert P._retry_after_seconds(Resp(text='... "retryDelay": "26s" ...')) == 26.0
    assert P._retry_after_seconds(Resp(text="no hint here")) is None
    assert P._retry_after_seconds(None) is None


def test_openai_compatible_no_toolcall_raises(monkeypatch):
    from flow.providers.openai_compat import OpenAICompatibleProvider, ProviderError

    monkeypatch.setenv("FAKE_KEY", "abc")
    prov = OpenAICompatibleProvider(
        model="m", base_url="http://x", api_key_env="FAKE_KEY", name="fake"
    )
    with pytest.raises(ProviderError):
        prov._parse({"choices": [{"message": {"content": "no tools here"}}]})
