"""Biologist-owned Analysis Profile.

The domain science in FLOW should be owned by the biologists who understand it — not
hard-coded by whoever maintains the software. This module lets an expert biologist specify,
in the web UI, exactly what analysis they want: the panel (which detector carries which
marker), the gating strategy, the populations of interest, the comparisons that matter, QC
expectations, and what the write-up should contain.

That specification is **not** analysis logic in the executable sense — it is natural-language
guidance that gets compiled into the agent's instructions (``prompt.md``, which the runner
appends to the system prompt). The agent then performs the biologist-specified analysis on
top of the deterministic first-run. So:

  * ``config.yaml`` stays analysis-free (this profile lives in its own sidecar file),
  * the first-run script is untouched,
  * and the biology is authored by biologists, not baked into the codebase.

This module contains only labels, descriptions, and text templating — no gating computation,
no thresholds-as-logic, no data access. It is guidance authoring, in the same category as
``agent/prompts.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROFILE_FILENAME = "analysis_profile.json"
# The compiled guidance is written here; the runner already reads prompt.md as extra guidance.
GUIDANCE_FILENAME = "prompt.md"
# A marker so a compiled prompt.md can be recognised/overwritten without clobbering a
# hand-written one the user created directly.
GUIDANCE_HEADER = "<!-- FLOW analysis profile (generated; edit in the Analysis page) -->"


class MarkerDef(BaseModel):
    """One detector → marker assignment (a FACT about the panel, authored by the biologist)."""

    detector: str = ""
    marker: str = ""
    notes: str = ""


class GateStep(BaseModel):
    """One step of the gating strategy, described in the biologist's own terms."""

    name: str = ""
    definition: str = ""  # e.g. "CD3+ : APC-Cy7-A above the negative/positive valley"


class AnalysisProfile(BaseModel):
    """A biologist's specification of the intended analysis (compiled to agent guidance)."""

    panel_name: str = ""
    markers: list[MarkerDef] = Field(default_factory=list)
    gating: list[GateStep] = Field(default_factory=list)
    populations_of_interest: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    qc_expectations: str = ""
    desired_outputs: list[str] = Field(default_factory=list)
    biological_context: str = ""
    plain_language: bool = False
    extra_instructions: str = ""


# ── Seed: the biology currently hard-coded in the NK first-run, exposed for editing ─────
# This lets a biologist SEE and TAKE OVER exactly what the software currently assumes. It is
# descriptive guidance (labels), not the executable pipeline (which stays in firstrun/).
def nk_seed_profile() -> AnalysisProfile:
    """Return the NK cell-therapy panel as an editable starting profile."""
    return AnalysisProfile(
        panel_name="NK cell-therapy (18-color) — starting point; edit for your assay",
        markers=[
            MarkerDef(detector="BV510-A", marker="CD45", notes="leukocyte gate"),
            MarkerDef(detector="UV 450 L/D-A", marker="Viability (Live/Dead)", notes="exclude dead"),
            MarkerDef(detector="APC-Cy7-A", marker="CD3", notes="T lineage"),
            MarkerDef(detector="Qdot 800-A", marker="CD19", notes="B lineage"),
            MarkerDef(detector="PE-Cy5-A", marker="CD14", notes="monocyte dump"),
            MarkerDef(detector="PE-Texas Red-A", marker="CD56", notes="NK lineage"),
            MarkerDef(detector="Alexa Fluor 488-A", marker="HLA-A3 (Donor)",
                      notes="distinguishes Donor vs Patient NK"),
            MarkerDef(detector="PE-Cy7-A", marker="CD70 (CAR)", notes="CAR readout on Donor NK"),
            MarkerDef(detector="Alexa Fluor 647-A", marker="CD27", notes="on Patient NK"),
        ],
        gating=[
            GateStep(name="Singlets", definition="Exclude doublets on FSC-H vs FSC-A"),
            GateStep(name="Debris removal", definition="Exclude low-FSC debris"),
            GateStep(name="Live cells", definition="Viability (L/D) negative"),
            GateStep(name="CD45+ leukocytes", definition="CD45 (BV510-A) positive"),
            GateStep(name="Lymphocytes", definition="Low-FSC / low-SSC leukocytes"),
            GateStep(name="T cells", definition="CD3 (APC-Cy7-A) positive"),
            GateStep(name="B cells", definition="CD19 (Qdot 800-A) positive"),
            GateStep(name="NK cells", definition="CD3- CD19- CD14- and CD56 (PE-Texas Red-A) positive"),
            GateStep(name="Donor NK", definition="NK and HLA-A3 (Alexa Fluor 488-A) positive"),
            GateStep(name="CAR+ Donor NK", definition="Donor NK and CD70 (PE-Cy7-A) positive"),
            GateStep(name="Patient NK", definition="NK and HLA-A3 negative"),
            GateStep(name="CD27+ Patient NK", definition="Patient NK and CD27 (Alexa Fluor 647-A) positive"),
        ],
        populations_of_interest=[
            "NK cells", "Donor NK", "CAR+ Donor NK", "Patient NK", "CD27+ Patient NK",
        ],
        comparisons=[
            "Change across timepoints (map samples to dates via flow.csv)",
            "Donor NK vs Patient NK over time",
            "Peak, expansion, and persistence of the CAR+ Donor NK population",
        ],
        qc_expectations=(
            "Flag samples with few CD45+ events, low viability, or unstable acquisition; "
            "down-weight unreliable timepoints and say which."
        ),
        desired_outputs=[
            "A per-timepoint table of population % and absolute counts (K/µL)",
            "Trend plots of the key populations over time",
            "A short plain-language summary of what the data show",
        ],
        biological_context="",
        plain_language=True,
        extra_instructions="",
    )


# ── Persistence ────────────────────────────────────────────────────────────────
def load_profile(project_dir: str | Path) -> AnalysisProfile | None:
    """Load a saved profile from the project dir, or None if none has been saved."""
    p = Path(project_dir) / PROFILE_FILENAME
    if not p.exists():
        return None
    try:
        return AnalysisProfile.model_validate(json.loads(p.read_text()))
    except Exception:
        return None


def save_profile(project_dir: str | Path, profile: AnalysisProfile) -> None:
    """Persist the profile and compile it into the agent guidance (prompt.md)."""
    d = Path(project_dir)
    (d / PROFILE_FILENAME).write_text(json.dumps(profile.model_dump(), indent=2))
    _write_guidance(d, compile_guidance(profile))


def split_guidance(text: str) -> tuple[str, str]:
    """Split prompt.md into (hand-written prefix, generated block).

    The generated block starts at GUIDANCE_HEADER. Everything before it is the user's own
    free-form prompt (edited via the Configure page); the block is owned by the Analysis
    page. Either may be empty.
    """
    if GUIDANCE_HEADER in text:
        i = text.index(GUIDANCE_HEADER)
        return text[:i].rstrip(), text[i:]
    return text.rstrip(), ""


def read_handwritten(project_dir: str | Path) -> str:
    """Return only the hand-written portion of prompt.md (no generated block)."""
    p = Path(project_dir) / GUIDANCE_FILENAME
    if not p.exists():
        return ""
    return split_guidance(p.read_text())[0]


def write_handwritten(project_dir: str | Path, text: str) -> None:
    """Replace the hand-written portion of prompt.md, preserving the generated block."""
    p = Path(project_dir) / GUIDANCE_FILENAME
    generated = split_guidance(p.read_text())[1] if p.exists() else ""
    prefix = text.rstrip()
    if prefix and generated:
        p.write_text(prefix + "\n\n" + generated)
    elif generated:
        p.write_text(generated)
    else:
        p.write_text(prefix + ("\n" if prefix else ""))


def _write_guidance(project_dir: Path, text: str) -> None:
    """Write the compiled guidance to prompt.md, preserving any hand-written content.

    The generated block always starts at GUIDANCE_HEADER. Anything the user wrote ABOVE that
    header is kept verbatim across regenerations; only the generated block is replaced. A
    hand-written prompt with no generated block yet has the block appended below it.
    """
    p = project_dir / GUIDANCE_FILENAME
    prefix = ""
    if p.exists():
        existing = p.read_text()
        if GUIDANCE_HEADER in existing:
            # Keep everything the user wrote before our generated block; replace the block.
            prefix = existing.split(GUIDANCE_HEADER)[0]
        elif existing.strip():
            # A hand-written prompt with no generated block yet — append below it.
            prefix = existing.rstrip() + "\n\n"
    p.write_text(prefix + text)


# ── Compiler: profile -> natural-language agent guidance ────────────────────────
def compile_guidance(profile: AnalysisProfile) -> str:
    """Render the profile as the natural-language guidance the agent will follow."""
    lines: list[str] = [
        GUIDANCE_HEADER,
        "# Analysis specification (authored by the lab biologist)",
        "",
        "Follow this specification for the biology. The deterministic first-run gives you a "
        "starting point; verify it against this spec and perform the analysis described here.",
    ]
    if profile.panel_name:
        lines += ["", f"**Panel:** {profile.panel_name}"]

    if profile.markers:
        lines += ["", "## Panel — detector → marker", ""]
        for m in profile.markers:
            if not (m.detector or m.marker):
                continue
            note = f" — {m.notes}" if m.notes else ""
            lines.append(f"- `{m.detector}` = **{m.marker}**{note}")

    if profile.gating:
        lines += ["", "## Gating strategy (in order)", ""]
        for i, g in enumerate(profile.gating, 1):
            if not (g.name or g.definition):
                continue
            defn = f": {g.definition}" if g.definition else ""
            lines.append(f"{i}. **{g.name}**{defn}")

    if profile.populations_of_interest:
        lines += ["", "## Populations of interest", ""]
        lines += [f"- {p}" for p in profile.populations_of_interest if p.strip()]

    if profile.comparisons:
        lines += ["", "## Comparisons / questions to answer", ""]
        lines += [f"- {c}" for c in profile.comparisons if c.strip()]

    if profile.qc_expectations.strip():
        lines += ["", "## Quality-control expectations", "", profile.qc_expectations.strip()]

    if profile.desired_outputs:
        lines += ["", "## The write-up should include", ""]
        lines += [f"- {o}" for o in profile.desired_outputs if o.strip()]

    if profile.biological_context.strip():
        lines += ["", "## Biological context", "", profile.biological_context.strip()]

    if profile.plain_language:
        lines += ["", "## Presentation",
                  "Explain findings in plain language a non-specialist can follow; define "
                  "cytometry/biology terms briefly the first time you use them."]

    if profile.extra_instructions.strip():
        lines += ["", "## Additional instructions", "", profile.extra_instructions.strip()]

    return "\n".join(lines) + "\n"


def profile_payload(project_dir: str | Path) -> dict[str, Any]:
    """Return the stored profile (or the NK seed), plus a compiled-guidance preview.

    ``seeded`` is True when no profile has been saved yet and the NK starting point is shown.
    """
    saved = load_profile(project_dir)
    profile = saved or nk_seed_profile()
    return {
        "profile": profile.model_dump(),
        "guidance_preview": compile_guidance(profile),
        "seeded": saved is None,
    }
