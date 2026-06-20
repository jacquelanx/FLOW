"""Tests for the two tools: edit_cell semantics/truncation and submit_answer validation."""

from flow.aviary.tool import ToolCall
from flow.env.notebook_env import MAX_OBS_CHARS, NotebookEnvironment


def test_exactly_two_tools(make_env):
    env: NotebookEnvironment = make_env()
    names = {t.name for t in env.tools()}
    assert names == {"edit_cell", "submit_answer"}


def test_edit_cell_new_and_execute(make_env):
    env = make_env()
    env.reset()
    obs, reward, done, info = env.step(
        ToolCall(name="edit_cell", arguments={"index": "new", "source": "print('hi'); 21*2"})
    )
    assert "hi" in obs
    assert "42" in obs  # last-expression value
    assert not done
    assert len(env.cells) == 1


def test_edit_cell_state_persists_across_cells(make_env):
    env = make_env()
    env.reset()
    env.step(ToolCall(name="edit_cell", arguments={"index": "new", "source": "x = 100"}))
    obs, *_ = env.step(
        ToolCall(name="edit_cell", arguments={"index": "new", "source": "print(x + 1)"})
    )
    assert "101" in obs  # notebook semantics: x survives


def test_edit_cell_overwrite_index(make_env):
    env = make_env()
    env.reset()
    env.step(ToolCall(name="edit_cell", arguments={"index": "new", "source": "a=1"}))
    env.step(ToolCall(name="edit_cell", arguments={"index": 0, "source": "a=2\nprint(a)"}))
    assert len(env.cells) == 1
    assert "2" in env.cells[0].outputs[0]["text"]


def test_edit_cell_out_of_range(make_env):
    env = make_env()
    env.reset()
    obs, *_ = env.step(ToolCall(name="edit_cell", arguments={"index": 5, "source": "x=1"}))
    assert "out of range" in obs


def test_edit_cell_captures_traceback(make_env):
    env = make_env()
    env.reset()
    obs, *_ = env.step(
        ToolCall(name="edit_cell", arguments={"index": "new", "source": "1/0"})
    )
    assert "TRACEBACK" in obs
    assert "ZeroDivisionError" in obs


def test_observation_truncation(make_env):
    env = make_env()
    env.reset()
    obs, *_ = env.step(
        ToolCall(name="edit_cell", arguments={"index": "new", "source": "print('A'*20000)"})
    )
    assert "truncated" in obs
    # Observation is bounded well under the raw 20k chars.
    assert len(obs) < MAX_OBS_CHARS * 2 + 500


def test_submit_answer_validates_nonempty(make_env):
    env = make_env()
    env.reset()
    obs, reward, done, info = env.step(
        ToolCall(name="submit_answer", arguments={"answer": "   "})
    )
    assert not done
    assert env.answer is None
    assert "non-empty" in obs


def test_submit_answer_finalizes(make_env):
    env = make_env()
    env.reset()
    obs, reward, done, info = env.step(
        ToolCall(name="submit_answer", arguments={"answer": "The trend increases."})
    )
    assert done
    assert reward == 1.0
    assert env.answer == "The trend increases."


def test_unknown_tool_is_rejected(make_env):
    env = make_env()
    env.reset()
    obs, reward, done, info = env.step(ToolCall(name="run_pipeline", arguments={}))
    assert "Unknown tool" in obs
    assert info.get("error") == "unknown_tool"


def test_notebook_serialization(make_env):
    env = make_env()
    env.reset()
    env.step(ToolCall(name="edit_cell", arguments={"index": "new", "source": "print('x')"}))
    nb = env.to_notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "code"
