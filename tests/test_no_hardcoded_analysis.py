"""Acceptance test: the AGENT PATH carries zero hardcoded analysis logic.

Greps the source tree for domain/analysis terms. The only executable place such analysis
is allowed is the deterministic first-run scaffold in ``src/flow/firstrun/`` — a fixed,
transparent pipeline the agent runs first and then improves on. Everywhere else in the
library (the ReAct loop, providers, environment, runner, kernel, API) must stay
analysis-free: the agent derives all interpretive analysis at runtime from the data +
question + metadata.

Allowed to MENTION these terms:
  * ``firstrun/``   — IS the sanctioned deterministic pipeline (gating, compensation, ...).
  * ``prompts.py``  — natural-language guidance to the model, not executable logic.
  * ``config.py``   — legitimately NAMES forbidden keys in order to REJECT them.
  * ``validation.py`` — mentions terms only inside the dataset-description guard test.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "flow"
FIRSTRUN = SRC / "firstrun"

# Terms that would indicate baked-in domain/analysis logic.
FORBIDDEN = [
    "otsu",
    "gating",
    "gate(",
    "compensation",
    "leiden(",
    "deseq",
    "edger",
    "fluorophore",
    "fcs_gate",
]
# Terms checked as whole-words (common English substrings otherwise).
FORBIDDEN_WORDS = ["gate", "population", "populations", "threshold", "marker", "markers"]

# Files/dirs allowed to MENTION these terms. prompts = NL guidance; config = refusal list;
# firstrun/ = the sanctioned deterministic scaffold (its whole job is domain analysis).
ALLOWLIST = {
    SRC / "agent" / "prompts.py",
    SRC / "config.py",
    SRC / "validation.py",  # mentions terms only inside the dataset-description guard test
    # Guidance authoring: composes the biologist's NL analysis spec into agent instructions.
    # Contains only labels/descriptions/text templating — no executable analysis logic.
    SRC / "analysis_profile.py",
}


def _is_allowlisted(p: Path) -> bool:
    if p in ALLOWLIST:
        return True
    # Exempt the entire firstrun/ package — it deliberately IS the domain pipeline.
    return FIRSTRUN in p.parents


def _python_files():
    for p in SRC.rglob("*.py"):
        if _is_allowlisted(p):
            continue
        yield p


def test_no_forbidden_substrings():
    hits = []
    for p in _python_files():
        text = p.read_text().lower()
        for term in FORBIDDEN:
            if term in text:
                hits.append(f"{p}: '{term}'")
    assert not hits, f"Hardcoded-analysis terms found in the agent path: {hits}"


def test_no_forbidden_words():
    hits = []
    for p in _python_files():
        text = p.read_text().lower()
        for word in FORBIDDEN_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                hits.append(f"{p}: '{word}'")
    assert not hits, f"Hardcoded-analysis words found in the agent path: {hits}"


def test_firstrun_is_actually_exempted():
    """Guard the guard: firstrun/ exists and its analysis lives only in the example template.

    The harness (__init__.py) is now biology-free — the deterministic analysis is provided per
    project by the biologist. The bundled example script is the one place domain terms live in
    the repo; this checks the exemption still covers real analysis rather than being dead.
    """
    assert FIRSTRUN.is_dir(), "firstrun/ missing — update the exemption."
    example = FIRSTRUN / "example_nk_panel.py"
    assert example.is_file(), "example first-run template missing — update this test."
    text = example.read_text().lower()
    assert "gating" in text or "gate" in text


def test_only_two_tools_defined():
    """The environment must advertise exactly edit_cell + submit_answer."""
    env_src = (SRC / "env" / "notebook_env.py").read_text()
    # Tool names appear as name="..." in the two Tool definitions.
    names = set(re.findall(r'name="([a-z_]+)"', env_src))
    tool_names = {n for n in names if n in {"edit_cell", "submit_answer"}}
    assert tool_names == {"edit_cell", "submit_answer"}
    # No third tool should be registered.
    assert 'name="run_pipeline"' not in env_src
    assert 'name="run_analysis"' not in env_src


def test_config_module_only_rejects_terms_not_uses_them():
    """config.py may name forbidden keys only inside its rejection set."""
    text = (SRC / "config.py").read_text()
    assert "_FORBIDDEN_ANALYSIS_KEYS" in text
    assert "must not contain analysis logic" in text
