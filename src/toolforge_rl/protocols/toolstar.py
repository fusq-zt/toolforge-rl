"""Deterministic adapter for the protocol used by Tool-Star-Qwen-3B.

The upstream checkpoint emits XML-like strings rather than function calls.  The
tags are ordinary multi-token strings in both the base and Tool-Star tokenizer.
This module therefore parses text only after a complete closing tag is seen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step "
    "with the help of the wikipedia search tool and python interpreter tool. "
    "Given a question, you need to first think about the reasoning process in the "
    "mind and then provide the answer. During thinking, you can invoke the "
    "wikipedia search tool to search and python interpreter tool to calculate the "
    "math problem for fact information about specific topics if needed. The "
    "reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags respectively, and the search query and result are "
    "enclosed within <search> </search> and <result> </result> tags respectively. "
    "For example, <think> This is the reasoning process. </think> <search> search "
    "query here </search> <result> search result here </result> <think> This is the "
    "reasoning process. </think> <python> python code here </python> <result> python "
    "interpreter result here </result> <think> This is the reasoning process. "
    "</think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
    "In the last part of the answer, the final exact answer is enclosed within "
    "\\boxed{} with latex format."
)


class ActionKind(StrEnum):
    PYTHON = "python_exec"
    SEARCH = "local_search"
    FINAL = "final"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    content: str = ""
    raw_tag: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProtocolValidation:
    valid: bool
    errors: tuple[str, ...]
    tool_calls: int
    final_answer: str | None


class ToolNameAdapter:
    """Map upstream tag names to the two frozen ToolForge tool names."""

    _MAPPING = {"search": "local_search", "python": "python_exec"}

    @classmethod
    def to_internal(cls, upstream_name: str) -> str:
        try:
            return cls._MAPPING[upstream_name.strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown upstream tool: {upstream_name}") from exc


_ACTION_TAGS = ("search", "python", "answer")
_PAIR_TAGS = ("think", "search", "python", "result", "answer")
_COMPLETE_RE = re.compile(
    r"<(search|python|answer)>\s*(.*?)\s*</\1>", re.IGNORECASE | re.DOTALL
)


def trim_at_first_stop(text: str, stops: tuple[str, ...] | list[str]) -> str:
    """Trim a decoded chunk at the first complete generation stop string."""

    ends = [position + len(stop) for stop in stops if (position := text.find(stop)) >= 0]
    return text[: min(ends)] if ends else text


def _last_complete(text: str) -> tuple[str, str, int] | None:
    matches = list(_COMPLETE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return match.group(1).lower(), match.group(2).strip(), match.end()


def parse_next_action(text: str) -> Action:
    """Parse the final complete action in an incremental generation chunk.

    A complete action is accepted only when it is the last non-whitespace text.
    This prevents executing an earlier example or a tag embedded in prose.
    """

    stripped = text.rstrip()
    last = _last_complete(stripped)
    if last is None:
        if any(f"<{tag}>" in stripped.lower() for tag in _ACTION_TAGS):
            return Action(ActionKind.INCOMPLETE, error="unclosed action tag")
        return Action(ActionKind.INCOMPLETE, error="no complete action tag")
    tag, content, end = last
    if end != len(stripped):
        return Action(ActionKind.INVALID, content, tag, "trailing text after action")
    if not content:
        return Action(ActionKind.INVALID, content, tag, "empty action body")
    kind = {
        "search": ActionKind.SEARCH,
        "python": ActionKind.PYTHON,
        "answer": ActionKind.FINAL,
    }[tag]
    return Action(kind, content, tag)


def render_result(result: object, *, max_chars: int = 4_000) -> str:
    """Render a bounded observation using the single upstream result tag."""

    value = str(result).replace("</result>", "&lt;/result&gt;")
    if len(value) > max_chars:
        value = value[: max_chars - 18] + "\n...[truncated]"
    return f"\n<result>\n{value}\n</result>\n"


def _last_boxed(text: str) -> str | None:
    """Return the last brace-balanced ``\\boxed{...}`` payload."""

    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
    return None


def extract_final_answer(text: str) -> str | None:
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if not answers:
        return None
    return _last_boxed(answers[-1]) or answers[-1].strip() or None


def validate_transcript(
    text: str, *, max_tool_calls: int = 3, require_final: bool = True
) -> ProtocolValidation:
    errors: list[str] = []
    lower = text.lower()
    for tag in _PAIR_TAGS:
        opens = lower.count(f"<{tag}>")
        closes = lower.count(f"</{tag}>")
        if opens != closes:
            errors.append(f"unbalanced {tag} tags: {opens} open/{closes} close")

    tokens = list(
        re.finditer(r"<(search|python|result|answer)>|</(search|python|result|answer)>", lower)
    )
    sequence: list[str] = []
    for token in tokens:
        if token.group(1):
            sequence.append(token.group(1))

    tool_calls = sequence.count("search") + sequence.count("python")
    if tool_calls > max_tool_calls:
        errors.append(f"tool budget exceeded: {tool_calls}>{max_tool_calls}")

    pending = False
    final_seen = False
    for item in sequence:
        if item in {"search", "python"}:
            if pending:
                errors.append("tool action not followed by result before next action")
            if final_seen:
                errors.append("tool action appears after final answer")
            pending = True
        elif item == "result":
            if not pending:
                errors.append("orphan result tag")
            pending = False
        elif item == "answer":
            if pending:
                errors.append("final answer appears before tool result")
            if final_seen:
                errors.append("multiple final answers")
            final_seen = True
    if pending:
        errors.append("last tool action has no result")
    answer = extract_final_answer(text)
    if require_final and answer is None:
        errors.append("missing complete final answer")
    return ProtocolValidation(not errors, tuple(errors), tool_calls, answer)
