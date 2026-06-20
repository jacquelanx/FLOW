"""Tests for the minimal .env loader."""

from pathlib import Path

from flow.envfile import load_dotenv


def test_loads_pairs_without_overriding(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "GEMINI_API_KEY=abc123\n"
        'GROQ_API_KEY="gsk_with_quotes"\n'
        "export OPENAI_API_KEY=exported_form\n"
        "ALREADY_SET=from_file\n"
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from_environment")

    assert load_dotenv(env) is True

    import os

    assert os.environ["GEMINI_API_KEY"] == "abc123"
    assert os.environ["GROQ_API_KEY"] == "gsk_with_quotes"  # quotes stripped
    assert os.environ["OPENAI_API_KEY"] == "exported_form"  # 'export ' prefix handled
    # Real environment wins over the file.
    assert os.environ["ALREADY_SET"] == "from_environment"


def test_missing_file_returns_false(tmp_path: Path):
    assert load_dotenv(tmp_path / "nope.env") is False


def test_env_file_var_override(tmp_path: Path, monkeypatch):
    custom = tmp_path / "secrets.env"
    custom.write_text("DEEPSEEK_API_KEY=zzz\n")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("FLOW_ENV_FILE", str(custom))

    assert load_dotenv() is True

    import os

    assert os.environ["DEEPSEEK_API_KEY"] == "zzz"
