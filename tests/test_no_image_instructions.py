"""Acceptance test: nothing instructs the agent to LOOK at an image.

The agent has no vision path — ``kernel_server`` never returns image bytes, ``Message.content``
is plain text, and no provider builds image content blocks. So an instruction to "open the
overlay PNG and judge" can only be satisfied by fabrication, which is the most damaging failure
mode available to this system: a confident, unfalsifiable claim about a figure nobody checked.

The guarded set includes the first-run scripts' ``DATA_DICTIONARY`` — the text written to
``first_run_summary.txt``. That file is guidance in every sense that matters: the agent cats
it and acts on it. Its omission here is why "it MAY inspect unified_cutoffs.csv + the overlay
plots to AUDIT the cutoff" survived in ``anchored_nk_panel`` after the same instruction was
removed from the profile, and a production trajectory read it back verbatim.

This test guards the rule in both directions:
  * no guidance text may direct the agent to view, open, or judge from an image; and
  * the guidance must say plainly that image content is not visible, because a model that is
    merely not TOLD to look at a figure will still describe one if the filename is in context.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from flow.agent.prompts import SYSTEM_PROMPT
from flow.analysis_profile import compile_guidance, nk_seed_profile
from flow.env.kernel import CellResult
from flow.env.notebook_env import NotebookEnvironment
from flow.firstrun import DIAGNOSTICS_NOTE, PROMPT_NOTE

# A verb that only makes sense if the reader can see pixels, close to something that IS pixels.
# Deliberately window-based rather than sentence-based: the instruction this test exists to
# catch was "open unified_cutoffs.csv and the overlay_<marker>.png plots and judge ...", where
# a filename's own dot defeats any sentence splitter.
LOOK_AT_AN_IMAGE = re.compile(
    r"\b(open|look at|looking at|view|viewing|inspect|inspecting|examine|examining|read|"
    r"reading|judge from)\b.{0,70}?"
    r"(\.png|\.jpe?g|\.pdf|\.svg|overlay|figure|plot)",
    re.IGNORECASE | re.DOTALL,
)
# Text that makes the limitation explicit. One of these must be near any such phrasing.
DISCLAIMS = re.compile(
    r"cannot see|can not see|never describe|not describe|not visible|human reader|"
    r"filename only|reaches you as a filename",
    re.IGNORECASE,
)
DISCLAIM_WINDOW = 250

# The first-run scripts, read as SOURCE rather than imported: importing them pulls in pandas,
# matplotlib and a ``sys.path`` mutation, and none of that is needed to read a string constant.
FIRSTRUN_DIR = Path(__file__).resolve().parents[1] / "src" / "flow" / "firstrun"
FIRSTRUN_DICTIONARIES = ("anchored_nk_panel.py", "example_nk_panel.py")


def _module_constant(filename: str, name: str) -> str:
    """The value of a module-level string assignment, by AST — no import side effects."""
    tree = ast.parse((FIRSTRUN_DIR / filename).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise AssertionError(f"{filename} has no module-level string {name}; renamed or removed?")


def _guidance_texts() -> dict[str, str]:
    texts = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "PROMPT_NOTE": PROMPT_NOTE,
        "DIAGNOSTICS_NOTE": DIAGNOSTICS_NOTE,
        "compiled seed profile guidance": compile_guidance(nk_seed_profile()),
    }
    for fn in FIRSTRUN_DICTIONARIES:
        texts[f"{fn}:DATA_DICTIONARY"] = _module_constant(fn, "DATA_DICTIONARY")
    return texts


def _look_at_image_hits(name: str, text: str) -> list[str]:
    """Every place ``text`` pairs a looking verb with an image and does not disclaim it."""
    hits = []
    for m in LOOK_AT_AN_IMAGE.finditer(text):
        lo = max(0, m.start() - DISCLAIM_WINDOW)
        window = text[lo:m.end() + DISCLAIM_WINDOW]
        if not DISCLAIMS.search(window):
            hits.append(f"{name}: ...{m.group(0)[:120]}...")
    return hits


def test_no_guidance_tells_the_agent_to_look_at_an_image():
    """A 'look at X' where X is an image is an instruction only fabrication can satisfy."""
    hits = []
    for name, text in _guidance_texts().items():
        hits += _look_at_image_hits(name, text)
    assert not hits, (
        "Guidance directs the agent to consult an image it cannot see:\n  " + "\n  ".join(hits))


def test_the_guard_catches_the_instruction_it_was_written_for():
    """A guard that passes on the original defect guards nothing."""
    regression = (
        "YOUR JOB is to (1) EVALUATE the cutoff: open unified_cutoffs.csv and the "
        "overlay_<marker>.png plots and judge whether each cutoff is derived reasonably and "
        "holds across timepoints."
    )
    assert _look_at_image_hits("regression", regression), (
        "the detector no longer catches the analysis_profile instruction that motivated it")
    # And it must not fire on the fixed phrasing, which names the same file with a disclaimer.
    fixed = (
        "EVALUATE the cutoff from the diagnostics MEASUREMENTS, since you cannot see the "
        "overlay_<marker>.png figures — they are drawn for the human reader."
    )
    assert not _look_at_image_hits("fixed", fixed)


def test_the_system_prompt_states_plainly_that_image_content_is_invisible():
    low = SYSTEM_PROMPT.lower()
    assert "cannot see" in low
    assert "filename" in low
    # And it must say what to do instead, or the constraint just blocks the work.
    assert "compute the numbers" in low


def test_the_anchored_profile_routes_the_cutoff_judgement_to_measurements():
    """The judgement the spec asks for must be reachable: name the tables that answer it."""
    text = compile_guidance(nk_seed_profile())
    for table in ("diagnostics_cutoff_audit.csv", "diagnostics_transfer.csv",
                  "diagnostics_uncertainty.csv"):
        assert table in text, f"the spec asks for a cutoff judgement but never names {table}"
    # The figures may still be mentioned — only as the human's artifact, never as the agent's.
    for m in re.finditer(r"overlay_<marker>\.png", text):
        window = text[max(0, m.start() - 200):m.end() + 200].lower()
        assert "cannot see" in window or "human reader" in window


def test_the_first_run_dictionary_routes_the_cutoff_audit_to_measurements():
    """The anchored dictionary is where the defect lived. Removing the instruction is only
    half a fix: the audit it asked for is a real requirement, so the text has to say where
    the evidence comes from instead."""
    text = _module_constant("anchored_nk_panel.py", "DATA_DICTIONARY")
    assert "MAY inspect unified_cutoffs.csv + the overlay plots" not in text
    for table in ("diagnostics_cutoff_audit.csv", "diagnostics_transfer.csv"):
        assert table in text, f"the dictionary asks for a cutoff audit but never names {table}"


def test_every_first_run_dictionary_disclaims_the_figures_it_names():
    """A filename in context is enough to make a model describe the figure, so naming one
    without the disclaimer is the same defect with the verb removed."""
    for fn in FIRSTRUN_DICTIONARIES:
        text = _module_constant(fn, "DATA_DICTIONARY")
        for m in re.finditer(r"\.png", text):
            window = text[max(0, m.start() - 200):m.end() + 400].lower()
            assert "cannot see" in window or "human reader" in window, (
                f"{fn} names a .png with no disclaimer within reach")


def test_the_profile_never_asks_for_a_judgement_it_gives_no_route_to():
    """If the spec wants a verdict on the cutoff, it must say where the evidence comes from."""
    p = nk_seed_profile()
    blob = (p.qc_expectations + " " + p.extra_instructions).lower()
    if "evaluate the unified cutoff" in blob or "evaluate the cutoff" in blob:
        assert "diagnostics_" in blob


def test_the_image_observation_does_not_imply_the_figure_is_viewable(tmp_path: Path):
    """The observation names the file; it must not read as 'shown elsewhere'."""
    env = NotebookEnvironment(kernel=object(), question="q", dataset_description="d")
    obs = env._format_observation(0, CellResult(images=["/work/plots/overlay_cd56.png"]))
    assert "overlay_cd56.png" in obs
    low = obs.lower()
    assert "cannot see" in low
    assert "not shown inline" not in low
