"""The ReAct agent loop.

Drives one trajectory: builds the conversation, asks the provider for one tool call,
executes it in the environment, appends the observation, and repeats until the agent
submits an answer or the step/time budget is exhausted. A callback streams each step so
the UI can show the agent reasoning live.

The loop is provider- and backend-agnostic: it works identically with the MockProvider +
InProcessKernel (tests) and a real provider + DockerKernel (production).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from flow.agent.prompts import SYSTEM_PROMPT, build_task_message
from flow.aviary.message import Message
from flow.env.notebook_env import NotebookEnvironment
from flow.providers.base import Provider


@dataclass
class StepRecord:
    """One ReAct step: the action taken and the observation returned."""

    step: int
    tool: str
    arguments: dict
    observation: str
    done: bool
    reward: float


@dataclass
class AgentResult:
    """Outcome of a full trajectory."""

    submitted: bool
    answer: Optional[str]
    steps: list[StepRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    failure_reason: Optional[str] = None


StepCallback = Callable[[StepRecord], None]


class ReActAgent:
    """A minimal, faithful ReAct loop over a NotebookEnvironment."""

    def __init__(
        self,
        provider: Provider,
        env: NotebookEnvironment,
        *,
        on_step: Optional[StepCallback] = None,
    ):
        self.provider = provider
        self.env = env
        self.on_step = on_step

    def run(self) -> AgentResult:
        obs, tools = self.env.reset()
        system_content = SYSTEM_PROMPT
        note = getattr(self.env, "first_run_note", "")
        if note:
            system_content += "\n\n" + note
        extra = getattr(self.env, "system_prompt_extra", "")
        if extra:
            # User-supplied guidance is added on top of FLOW's default agent prompt.
            system_content += "\n\n--- ADDITIONAL USER-PROVIDED GUIDANCE ---\n" + extra
        messages: list[Message] = [
            Message(role="system", content=system_content),
            Message(
                role="user",
                content=build_task_message(self.env.question, self.env.dataset_description),
            ),
            # The reset observation primes the agent with the live notebook state.
            Message(role="user", content=obs),
        ]
        steps: list[StepRecord] = []
        failure_reason: Optional[str] = None

        while True:
            try:
                call = self.provider.generate(messages, tools)
            except Exception as e:  # provider/transport/parse failure ends the run
                failure_reason = f"Provider error: {e}"
                break

            # Record the assistant's tool call in the conversation. Render once and reuse
            # so the assistant turn and its tool result agree on the id, and so any
            # provider-specific fields (e.g. Gemini's thought_signature) carried on the
            # raw object are replayed verbatim on the next request.
            rendered = call.to_openai_assistant_toolcall()
            call_id = rendered.get("id") or call.id or f"call_{call.name}"
            messages.append(
                Message(role="assistant", content="", tool_calls=[rendered])
            )

            observation, reward, done, info = self.env.step(call)

            messages.append(
                Message(
                    role="tool",
                    content=observation,
                    tool_call_id=call_id,
                    name=call.name,
                )
            )

            rec = StepRecord(
                step=info.get("step", len(steps) + 1),
                tool=call.name,
                arguments=call.arguments,
                observation=observation,
                done=done,
                reward=reward,
            )
            steps.append(rec)
            if self.on_step:
                self.on_step(rec)

            if done:
                if self.env.answer is None and info.get("budget_exhausted"):
                    failure_reason = "Step budget exhausted without submit_answer."
                elif self.env.answer is None and info.get("timeout"):
                    failure_reason = f"Timed out ({info['timeout']}) without submit_answer."
                break

        submitted = self.env.answer is not None
        if not submitted and failure_reason is None:
            failure_reason = "Trajectory ended without a submitted answer."

        return AgentResult(
            submitted=submitted,
            answer=self.env.answer,
            steps=steps,
            messages=messages,
            failure_reason=failure_reason if not submitted else None,
        )
