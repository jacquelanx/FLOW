"""Tests for the first-run harness (runs the project's script), no Docker needed."""

from pathlib import Path

from flow.env.kernel import InProcessKernel
from flow.env.notebook_env import NotebookEnvironment
from flow.firstrun import build_first_run, example_script


def test_none_without_a_project_script(tmp_path: Path):
    # Data present but no first_run.py -> there is nothing to run.
    (tmp_path / "fcs").mkdir()
    (tmp_path / "fcs" / "a.fcs").write_bytes(b"FCS3.0")
    assert build_first_run(tmp_path, "auto") is None


def test_runs_the_project_script_when_present(tmp_path: Path):
    (tmp_path / "first_run.py").write_text("print('hello from the biologist script')")
    fr = build_first_run(tmp_path, "auto")
    assert fr is not None and fr.kind == "script"
    # The seed cell invokes the project's script (mounted at /data) per the contract.
    assert fr.cells and "/data/first_run.py" in fr.cells[0]
    assert "--out" in fr.cells[0] and "first_run_tables" in fr.cells[0]
    assert "INTERPRETIVE" in fr.prompt_note.upper()


def test_none_mode_disables_even_with_a_script(tmp_path: Path):
    (tmp_path / "first_run.py").write_text("print('x')")
    assert build_first_run(tmp_path, "none") is None


def test_example_template_is_available_and_follows_the_contract():
    s = example_script()
    # The editor is seeded with a runnable example that honors the --data/--out/--plots contract.
    assert s and "--data" in s and "--out" in s and "--plots" in s


def test_custom_prompt_appended_to_system(tmp_path: Path):
    from flow.agent.react import ReActAgent
    from flow.providers.mock import MockProvider

    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=tmp_path / "p"),
        question="Q?",
        dataset_description="d",
        system_prompt_extra="MY CUSTOM GATING RULES",
    )
    result = ReActAgent(MockProvider(), env).run()
    system_msg = result.messages[0]
    assert system_msg.role == "system"
    assert "MY CUSTOM GATING RULES" in system_msg.content
    assert "USER-PROVIDED GUIDANCE" in system_msg.content


def test_seed_cells_execute_at_reset_and_prime_observation(tmp_path: Path):
    kernel = InProcessKernel(plots_dir=tmp_path / "plots")
    env = NotebookEnvironment(
        kernel=kernel,
        question="What is the trend?",
        dataset_description="some data",
        seed_cells=["print('SEEDED_OK')\nfirst_run = 123"],
        first_run_note="YOUR ROLE IS INTERPRETIVE.",
    )
    obs, tools = env.reset()
    # The seed cell ran, its output + the interpretive note are in the first observation.
    assert "SEEDED_OK" in obs
    assert "INTERPRETIVE" in obs.upper()
    # Seed cells become real notebook cells but do NOT consume the agent's step budget.
    assert len(env.cells) == 1
    assert env.steps_taken == 0
    assert {t.name for t in tools} == {"edit_cell", "submit_answer"}

    # State from the seed cell persists for the agent's first real step.
    from flow.aviary.tool import ToolCall

    out, *_ = env.step(
        ToolCall(name="edit_cell", arguments={"index": "new", "source": "print(first_run)"})
    )
    assert "123" in out
