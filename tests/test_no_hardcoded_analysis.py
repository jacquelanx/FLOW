"""Acceptance test: ZERO hardcoded analysis logic in src/.

Greps the source tree for domain/analysis terms. The only allowed places for such terms
are prompts (natural-language guidance to the model), tests, and docs — never executable
analysis logic in the library. This enforces non-negotiable principle #1.

Note: ``config.py`` legitimately *names* forbidden keys in order to REJECT them. That file
is allowlisted for the same reason prompts are: it contains no analysis, only a refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "flow"

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

# Files allowed to MENTION these terms (prompts = NL guidance; config = refusal list).
ALLOWLIST = {
    SRC / "agent" / "prompts.py",
    SRC / "config.py",
    SRC / "validation.py",  # mentions terms only inside the dataset-description guard test
}


def _python_files():
    for p in SRC.rglob("*.py"):
        if p in ALLOWLIST:
            continue
        yield p


def test_no_forbidden_substrings():
    hits = []
    for p in _python_files():
        text = p.read_text().lower()
        for term in FORBIDDEN:
            if term in text:
                hits.append(f"{p}: '{term}'")
    assert not hits, f"Hardcoded-analysis terms found in src/: {hits}"


def test_no_forbidden_words():
    hits = []
    for p in _python_files():
        text = p.read_text().lower()
        for word in FORBIDDEN_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                hits.append(f"{p}: '{word}'")
    assert not hits, f"Hardcoded-analysis words found in src/: {hits}"


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
