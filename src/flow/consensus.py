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
  • State the consensus conclusion clearly and directly, with the key quantitative evidence \
that multiple analysts support.
  • Be explicit about the confidence the consensus warrants and about any important caveats \
or open questions the analysts raised.
  • Do not invent results. Base the consensus only on what the analysts reported.
"""


@dataclass
class ConsensusResult:
    """Outcome of the meta-analysis."""

    consensus: Optional[str]
    synthesized: bool  # True if an LLM synthesis was run (vs. trivial pass-through)
    n_submitted: int
    n_total: int
    failure_reason: Optional[str] = None


def _build_user_message(question: str, conclusions: list[str]) -> str:
    parts = [
        f"RESEARCH QUESTION:\n{question}\n",
        f"There are {len(conclusions)} independent analyst conclusions to synthesize.\n",
    ]
    for i, c in enumerate(conclusions, 1):
        parts.append(f"--- Analyst {i} conclusion ---\n{c}\n")
    parts.append(
        "Now write the single consensus conclusion, following the guidelines. Begin with the "
        "headline conclusion, then the supporting evidence and agreement, then caveats."
    )
    return "\n".join(parts)


def synthesize_consensus(
    provider: Provider,
    question: str,
    trajectory_results: list[dict],
) -> ConsensusResult:
    """Synthesize a consensus from per-trajectory results.

    ``trajectory_results`` items look like {"idx": int, "submitted": bool, "answer": str}.
    Only submitted, non-empty answers are considered. With a single submitted answer no
    synthesis is needed (it is the consensus). With none, the consensus fails clearly.
    """
    n_total = len(trajectory_results)
    conclusions = [
        (r.get("answer") or "").strip()
        for r in trajectory_results
        if r.get("submitted") and (r.get("answer") or "").strip()
    ]
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
        )

    messages = [
        Message(role="system", content=META_SYSTEM_PROMPT),
        Message(role="user", content=_build_user_message(question, conclusions)),
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
        )

    return ConsensusResult(
        consensus=(text or "").strip() or None,
        synthesized=True,
        n_submitted=n_submitted,
        n_total=n_total,
        failure_reason=None if (text or "").strip() else "Synthesis returned empty text.",
    )
