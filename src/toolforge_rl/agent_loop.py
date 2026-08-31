"""Protocol-consistent executable agent loop shared by data, RL, and evaluation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

from toolforge_rl.protocols.toolstar import (
    ActionKind,
    extract_final_answer,
    parse_next_action,
    render_result,
    validate_transcript,
)
from toolforge_rl.tools import LocalSearch, execute_python


GenerateFn = Callable[[str, tuple[str, ...]], str]


@dataclass(frozen=True)
class ToolEvent:
    index: int
    tool: str
    arguments: str
    response: str
    success: bool
    latency_seconds: float


@dataclass(frozen=True)
class EpisodeRollout:
    transcript: str
    final_answer: str | None
    tool_calls: tuple[ToolEvent, ...]
    schema_valid: bool
    execution_success: bool
    termination_reason: str
    repeated_call_count: int
    latency_seconds: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tool_call_count"] = len(self.tool_calls)
        return payload


class AgentLoop:
    def __init__(self, *, max_tool_calls: int = 3, max_turns: int = 4):
        self.max_tool_calls = max_tool_calls
        self.max_turns = max_turns

    def run(self, prompt: str, generate: GenerateFn, search: LocalSearch) -> EpisodeRollout:
        started = time.monotonic()
        transcript = ""
        events: list[ToolEvent] = []
        prior_calls: set[tuple[str, str]] = set()
        repeated = 0
        termination = "turn_limit"
        execution_success = True
        for _ in range(self.max_turns):
            delta = generate(prompt + transcript, ("</search>", "</python>", "</answer>"))
            transcript += delta
            action = parse_next_action(delta)
            if action.kind == ActionKind.FINAL:
                termination = "final"
                break
            if action.kind not in {ActionKind.SEARCH, ActionKind.PYTHON}:
                termination = action.kind.value
                break
            if len(events) >= self.max_tool_calls:
                termination = "tool_budget_exceeded"
                break
            call_key = (action.kind.value, action.content.strip())
            if call_key in prior_calls:
                repeated += 1
                termination = "repeated_call"
                break
            prior_calls.add(call_key)
            tool_started = time.monotonic()
            if action.kind == ActionKind.SEARCH:
                response = search.search(action.content)
                ok = not response.startswith("SEARCH_ERROR")
            else:
                result = execute_python(action.content)
                response, ok = result.output, result.ok
            transcript += render_result(response)
            events.append(
                ToolEvent(
                    len(events), action.kind.value, action.content, response, ok,
                    time.monotonic() - tool_started,
                )
            )
            if not ok:
                execution_success = False
        validation = validate_transcript(
            transcript, max_tool_calls=self.max_tool_calls, require_final=termination == "final"
        )
        return EpisodeRollout(
            transcript=transcript,
            final_answer=extract_final_answer(transcript),
            tool_calls=tuple(events),
            schema_valid=validation.valid and termination == "final",
            execution_success=execution_success and all(event.success for event in events),
            termination_reason=termination,
            repeated_call_count=repeated,
            latency_seconds=time.monotonic() - started,
        )
