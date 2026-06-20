"""Tests for the Aviary-shaped core abstractions."""

from flow.aviary import Environment, Message, Tool, ToolCall


def test_message_to_openai_assistant_with_tool_calls():
    m = Message(role="assistant", tool_calls=[{"id": "x", "type": "function"}])
    d = m.to_openai()
    assert d["role"] == "assistant"
    assert d["tool_calls"][0]["id"] == "x"


def test_message_tool_role_carries_id_and_name():
    m = Message(role="tool", content="obs", tool_call_id="c1", name="edit_cell")
    d = m.to_openai()
    assert d == {"role": "tool", "content": "obs", "tool_call_id": "c1", "name": "edit_cell"}


def test_tool_to_openai_schema():
    t = Tool(name="f", description="d", parameters={"type": "object"}, fn=lambda: None)
    schema = t.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "f"


def test_toolcall_render_roundtrip():
    tc = ToolCall(name="edit_cell", arguments={"index": "new", "source": "1+1"}, id="c9")
    rendered = tc.to_openai_assistant_toolcall()
    assert rendered["id"] == "c9"
    assert rendered["function"]["name"] == "edit_cell"
    import json

    assert json.loads(rendered["function"]["arguments"])["source"] == "1+1"


def test_environment_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        Environment()  # type: ignore[abstract]
