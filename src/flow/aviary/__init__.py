"""Aviary-shaped core abstractions (minimal, dependency-light).

This package mirrors the interface of FutureHouse's ``fhaviary`` (Message, Tool,
ToolCall, Environment) so that Finch's environment/tool boundary is reproduced
faithfully *without* the heavy dependency tree. See ADR-001 in ``DESIGN.md``.

If you prefer the real framework, ``flow.env.notebook_env.NotebookEnvironment``
can be re-parented onto ``fhaviary.Environment`` with no changes to the agent loop,
because both expose the same ``reset()/step()/tools()`` contract used here.
"""

from flow.aviary.message import Message
from flow.aviary.tool import Tool, ToolCall
from flow.aviary.environment import Environment

__all__ = ["Message", "Tool", "ToolCall", "Environment"]
