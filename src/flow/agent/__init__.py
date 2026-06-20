"""The ReAct agent: a carefully engineered loop that reasons, acts, and observes.

Maps to Finch's ReAct strategy. The agent is purely generative — the system prompt tells
it to derive every analysis step from the data and the question; FLOW supplies no
templates, gates, thresholds, or domain subroutines.
"""

from flow.agent.react import ReActAgent, AgentResult

__all__ = ["ReActAgent", "AgentResult"]
