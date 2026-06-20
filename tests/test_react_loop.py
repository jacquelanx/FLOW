"""Tests for the ReAct loop with the MockProvider (no Docker, InProcessKernel)."""

from flow.agent.react import ReActAgent
from flow.providers.mock import MockProvider


def test_mock_loop_completes_and_submits(make_env):
    env = make_env(max_steps=10)
    agent = ReActAgent(MockProvider(), env)
    result = agent.run()
    assert result.submitted is True
    assert result.answer is not None
    assert "MOCK PROVIDER" in result.answer
    # Exactly: one exploratory edit_cell, then submit_answer.
    tools_used = [s.tool for s in result.steps]
    assert tools_used == ["edit_cell", "submit_answer"]


def test_mock_loop_records_steps_and_messages(make_env):
    env = make_env()
    seen = []
    agent = ReActAgent(MockProvider(), env, on_step=seen.append)
    result = agent.run()
    assert len(seen) == len(result.steps) == 2
    # Conversation has system + task + reset-obs + (assistant/tool)*2.
    roles = [m.role for m in result.messages]
    assert roles[0] == "system"
    assert "assistant" in roles and "tool" in roles


def test_no_submit_means_failure(make_env):
    """A provider that never submits must fail when the step budget is exhausted."""
    from flow.aviary.tool import ToolCall
    from flow.providers.base import Provider

    class NeverSubmits(Provider):
        name = "never"

        def generate(self, messages, tools):
            return ToolCall(name="edit_cell", arguments={"index": "new", "source": "1+1"})

    env = make_env(max_steps=3)
    result = ReActAgent(NeverSubmits(), env).run()
    assert result.submitted is False
    assert result.answer is None
    assert "budget" in (result.failure_reason or "").lower()
    assert env.steps_taken == 3  # the environment never finished the work itself


def test_provider_error_ends_run_gracefully(make_env):
    from flow.providers.base import Provider

    class Boom(Provider):
        name = "boom"

        def generate(self, messages, tools):
            raise RuntimeError("provider exploded")

    env = make_env()
    result = ReActAgent(Boom(), env).run()
    assert result.submitted is False
    assert "provider exploded" in (result.failure_reason or "")
