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


def test_seed_cells_get_a_larger_observation_window_than_the_agents_own_cells(tmp_path: Path):
    """The first-run cells carry the evidence base and run once, so they are truncated later.

    The agent's per-step cells keep the smaller budget: the point is to spend context on the
    one-time evidence, not to raise every limit.
    """
    from flow.env.notebook_env import MAX_OBS_CHARS, SEED_OBS_CHARS
    from flow.aviary.tool import ToolCall

    # A payload that fits the seed window but not the per-step one.
    size = MAX_OBS_CHARS * 2
    marker_mid = "MIDDLE_OF_THE_REPORT"
    src = (
        f"print('A' * {size // 2} + {marker_mid!r} + 'B' * {size // 2})"
    )
    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=tmp_path / "plots"),
        question="Q?",
        dataset_description="d",
        seed_cells=[src],
    )
    obs, _ = env.reset()
    # The middle survives in the seed observation — which is where a digest's findings live.
    assert marker_mid in obs
    assert "[truncated" not in obs

    # The same output from an agent cell is truncated head+tail, losing the middle.
    step_obs, *_ = env.step(ToolCall(name="edit_cell", arguments={"index": "new", "source": src}))
    assert marker_mid not in step_obs
    assert "[truncated" in step_obs
    assert SEED_OBS_CHARS > MAX_OBS_CHARS


def test_seed_observation_window_can_be_overridden_but_never_below_the_step_window(tmp_path: Path):
    from flow.env.notebook_env import MAX_OBS_CHARS

    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=tmp_path / "plots"),
        question="Q?", dataset_description="d", seed_obs_chars=10,
    )
    assert env.seed_obs_chars == MAX_OBS_CHARS


def test_the_agent_cannot_overwrite_the_first_run_cells(tmp_path: Path):
    """The first-run cells ARE the audit trail for every number the agent reports.

    A real run lost them: the model's first two actions were edit_cell(index=0) and
    edit_cell(index=1), which replaced both seed cells, and a third step was burned on the
    resulting out-of-range error. Two different models reached for an integer index rather
    than 'new', so this is guarded in the environment rather than left to the prompt.
    """
    from flow.aviary.tool import ToolCall

    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=tmp_path / "plots"),
        question="Q?",
        dataset_description="d",
        seed_cells=["first_run_value = 42\nprint('FIRST_RUN_RAN')",
                    "diagnostics_value = 7\nprint('DIAGNOSTICS_RAN')"],
    )
    env.reset()
    assert env.n_protected_cells == 2
    original = [c.source for c in env.cells]

    for idx in (0, 1):
        obs, _r, done, _i = env.step(ToolCall(
            name="edit_cell", arguments={"index": idx, "source": "raise SystemExit('clobber')"}))
        assert "READ-ONLY" in obs
        assert "index='new'" in obs          # says what to do instead
        assert not done
    # Nothing was modified, and the kernel state the seed cells built is intact.
    assert [c.source for c in env.cells] == original
    out, *_ = env.step(ToolCall(
        name="edit_cell",
        arguments={"index": "new", "source": "print(first_run_value + diagnostics_value)"}))
    assert "49" in out
    assert len(env.cells) == 3


def test_out_of_range_message_leads_with_the_fix(tmp_path: Path):
    """The model repeated the same index mistake after the old message; lead with the action."""
    from flow.aviary.tool import ToolCall

    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=tmp_path / "plots"),
        question="Q?", dataset_description="d", seed_cells=["x = 1"],
    )
    env.reset()
    obs, *_ = env.step(ToolCall(name="edit_cell", arguments={"index": 5, "source": "pass"}))
    assert obs.startswith("Use index='new' to append.")


def test_a_project_without_a_first_run_protects_nothing(tmp_path: Path):
    """No seed cells means every cell is the agent's own and stays editable."""
    from flow.aviary.tool import ToolCall

    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=tmp_path / "plots"),
        question="Q?", dataset_description="d",
    )
    env.reset()
    assert env.n_protected_cells == 0
    env.step(ToolCall(name="edit_cell", arguments={"index": "new", "source": "y = 1"}))
    obs, *_ = env.step(ToolCall(name="edit_cell", arguments={"index": 0, "source": "print(99)"}))
    assert "99" in obs and "READ-ONLY" not in obs
