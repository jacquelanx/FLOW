"""Deterministic offline provider for tests and demos.

CRITICAL: the MockProvider must NEVER perform real analysis. It only proves the loop
works (tool dispatch, observation handling, step budget, no-submit => fail). It emits a
single MINIMAL, GENERIC cell — list the dataset directory and print basic shapes — then
submits a generic, explicitly-placeholder answer. There is no domain logic here.
"""

from __future__ import annotations

from flow.aviary.message import Message
from flow.aviary.tool import Tool, ToolCall
from flow.providers.base import Provider

# A deliberately generic exploratory cell: it inspects whatever is mounted without
# assuming anything about the scientific domain or the analysis to perform.
_GENERIC_CELL = (
    "import os\n"
    "data_dir = os.environ.get('FLOW_DATA_DIR', FLOW_DATA_DIR if 'FLOW_DATA_DIR' in dir() else '/data')\n"
    "print('Files in dataset directory:')\n"
    "for name in sorted(os.listdir(data_dir)):\n"
    "    p = os.path.join(data_dir, name)\n"
    "    size = os.path.getsize(p) if os.path.isfile(p) else 0\n"
    "    print(f'  {name}  ({size} bytes)')\n"
)


class MockProvider(Provider):
    """Two-step deterministic agent: explore once, then submit a placeholder answer."""

    name = "mock"

    def __init__(self, model: str = "mock", max_explore_steps: int = 1):
        self.model = model
        self.max_explore_steps = max_explore_steps

    def _assistant_tool_calls(self, messages: list[Message]) -> int:
        return sum(1 for m in messages if m.role == "assistant" and m.tool_calls)

    def generate(self, messages: list[Message], tools: list[Tool]) -> ToolCall:
        prior = self._assistant_tool_calls(messages)
        if prior < self.max_explore_steps:
            return ToolCall(
                name="edit_cell",
                arguments={"index": "new", "source": _GENERIC_CELL, "execute": True},
                id=f"call_mock_{prior}",
            )
        return ToolCall(
            name="submit_answer",
            arguments={
                "answer": (
                    "[MOCK PROVIDER] This is a placeholder conclusion produced by the "
                    "offline mock provider, which performs no real analysis. It exists "
                    "only to exercise the agent loop end-to-end. Use a real provider "
                    "(ollama / gemini / groq / ...) for an actual data analysis."
                )
            },
            id=f"call_mock_submit_{prior}",
        )
