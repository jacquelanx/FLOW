"""Consensus meta-analysis over multiple independent trajectories.

Mirrors the Nature paper: after N agents independently analyze the same data in separate
Jupyter trajectories, a meta-analysis synthesizes their conclusions into a single
consensus-driven answer — exploring diverse analytical paths while delivering a consistent
end result.

This step is **text synthesis**, not data analysis: it reads the trajectories' submitted
conclusions and produces a consensus. It executes no code, reads no dataset, and contains
no domain/analysis logic — all scientific reasoning happened inside the trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from flow.aviary.message import Message
from flow.providers.base import Provider

META_SYSTEM_PROMPT = """\
You are a careful scientific reviewer. Several independent analysts were each given the \
same dataset and the same research question, and each worked in their own notebook and \
submitted a conclusion. Your job is to synthesize their independent conclusions into a \
single consensus answer.

Guidelines:
  • Identify where the analysts AGREE — those points form the backbone of the consensus.
  • Note any DISAGREEMENTS or differences in approach, and weigh the evidence each gives.
  • Where analysts reach OPPOSITE conclusions about the same measurement, you must RESOLVE \
it, not file it. "Analyst 1 read the discontinuities as acquisition artefacts; Analyst 2 read \
them as biology" is a restatement, not a synthesis: both cannot hold, and a reader who is \
handed both learns only that the analysts differed. For each such point, do one of exactly \
two things:
      (a) resolve it — name the specific measurement, reported by one of the analysts, that \
decides it, and say which reading it supports and why; or
      (b) declare it UNRESOLVED — say plainly that the analysts' reports do not settle it, \
and name the measurement that would.
    Never present a contradiction as a mere difference in emphasis or approach, and never \
average two opposite readings into a hedge that asserts neither.
  • A conclusion that CONTRADICTS a measurement an analyst reported does not become \
defensible because another analyst shares it. Check each contested reading against the \
numbers quoted in the conclusions, including a neighbouring timepoint's value where one is \
given, and say when a reading is not consistent with them.
  • State the consensus conclusion clearly and directly, with the key quantitative evidence \
that multiple analysts support.
  • Be explicit about the confidence the consensus warrants and about any important caveats \
or open questions the analysts raised.
  • Do not invent results. Base the consensus only on what the analysts reported.
  • Some analysts arrive with an EVIDENCE NOT READ note listing measurements the run \
required them to consult and they did not. Treat that as a limit on their conclusion, not a \
disagreement to reconcile: two analysts converging on a point neither of them checked is \
agreement about an assumption, not evidence for it. Say so in the caveats, naming the \
unaddressed items, and do not let unanimity on such a point raise the stated confidence.
"""


@dataclass
class ConsensusResult:
    """Outcome of the meta-analysis."""

    consensus: Optional[str]
    synthesized: bool  # True if an LLM synthesis was run (vs. trivial pass-through)
    n_submitted: int
    n_total: int
    failure_reason: Optional[str] = None
    # Submitted analysts who left required measurements unread. Surfaced on the result so a
    # consensus built on incomplete reading is visible without opening the artifacts.
    n_with_unread_evidence: int = 0


def _evidence_note(evidence: Optional[dict]) -> str:
    """Render one analyst's unaddressed obligations, or "" when there are none to render.

    Kept factual and bounded: this is a record the environment produced, not a judgement
    about the analyst. The reviewer is told what was not read and left to weigh it.
    """
    if not evidence or not evidence.get("enforced"):
        return ""
    pending = evidence.get("undischarged") or []
    if not pending:
        return ""
    total = evidence.get("obligations_total")
    lines = [f"EVIDENCE NOT READ — {len(pending)} of {total} measurement(s) this run "
             "required were not consulted and are not addressed in the conclusion above:"]
    for item in pending[:8]:
        lines.append(f"  • {item.get('statement') or item.get('id')}")
    if len(pending) > 8:
        lines.append(f"  • [+{len(pending) - 8} more]")
    return "\n".join(lines)


def _build_user_message(
    question: str,
    conclusions: list[str],
    evidence: Optional[list[Optional[dict]]] = None,
) -> str:
    parts = [
        f"RESEARCH QUESTION:\n{question}\n",
        f"There are {len(conclusions)} independent analyst conclusions to synthesize.\n",
    ]
    notes = evidence or [None] * len(conclusions)
    for i, c in enumerate(conclusions, 1):
        parts.append(f"--- Analyst {i} conclusion ---\n{c}\n")
        note = _evidence_note(notes[i - 1] if i - 1 < len(notes) else None)
        if note:
            parts.append(note + "\n")
    tail = (
        "Now write the single consensus conclusion, following the guidelines. Begin with the "
        "headline conclusion, then the supporting evidence and agreement, then caveats."
    )
    if len(conclusions) > 1:
        # Named as a required section rather than left to the guidelines. A run whose analysts
        # read the same discontinuities as artefact and as biology produced a consensus with a
        # "Disagreements and Differences in Approach" heading under which both readings were
        # restated and neither was tested — the failure the guideline above describes, in a
        # section the model had invented for itself. A required heading that must contain a
        # resolution or an explicit "unresolved" is harder to satisfy vacuously.
        tail += (
            " Include a section headed CONTESTED POINTS listing every point on which the "
            "analysts reached opposite conclusions; for each, give either the measurement "
            "that resolves it (and which reading it supports) or an explicit statement that "
            "it is unresolved and what would settle it. If the analysts contradicted each "
            "other nowhere, write 'CONTESTED POINTS: none' — do not omit the section."
        )
    parts.append(tail)
    return "\n".join(parts)


def synthesize_consensus(
    provider: Provider,
    question: str,
    trajectory_results: list[dict],
) -> ConsensusResult:
    """Synthesize a consensus from per-trajectory results.

    ``trajectory_results`` items look like {"idx": int, "submitted": bool, "answer": str},
    optionally with an ``"evidence"`` dict from the trajectory's evidence-coverage record.
    Only submitted, non-empty answers are considered. With a single submitted answer no
    synthesis is needed (it is the consensus). With none, the consensus fails clearly.
    """
    n_total = len(trajectory_results)
    # Paired so an analyst's conclusion and its evidence record cannot drift out of step
    # when unsubmitted trajectories are filtered out.
    kept = [
        (r, (r.get("answer") or "").strip())
        for r in trajectory_results
        if r.get("submitted") and (r.get("answer") or "").strip()
    ]
    conclusions = [answer for _r, answer in kept]
    evidence = [r.get("evidence") for r, _answer in kept]
    n_gaps = sum(1 for e in evidence if _evidence_note(e))
    n_submitted = len(conclusions)

    if n_submitted == 0:
        return ConsensusResult(
            consensus=None,
            synthesized=False,
            n_submitted=0,
            n_total=n_total,
            failure_reason="No trajectory produced an answer, so no consensus could be formed.",
        )

    if n_submitted == 1:
        # Nothing to reconcile — the lone conclusion is the consensus.
        return ConsensusResult(
            consensus=conclusions[0],
            synthesized=False,
            n_submitted=1,
            n_total=n_total,
            n_with_unread_evidence=n_gaps,
        )

    messages = [
        Message(role="system", content=META_SYSTEM_PROMPT),
        Message(role="user",
                content=_build_user_message(question, conclusions, evidence)),
    ]
    try:
        text = provider.complete(messages)
    except NotImplementedError:
        # Provider can't synthesize; surface the individual conclusions rather than fail.
        joined = "\n\n".join(f"Analyst {i + 1}: {c}" for i, c in enumerate(conclusions))
        return ConsensusResult(
            consensus="(No synthesis available from this provider.) Individual conclusions:\n\n"
            + joined,
            synthesized=False,
            n_submitted=n_submitted,
            n_total=n_total,
            n_with_unread_evidence=n_gaps,
        )

    return ConsensusResult(
        consensus=(text or "").strip() or None,
        synthesized=True,
        n_submitted=n_submitted,
        n_total=n_total,
        failure_reason=None if (text or "").strip() else "Synthesis returned empty text.",
        n_with_unread_evidence=n_gaps,
    )
