"""Frozen v1 rewards shared by mining, GRPO, and evaluation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardInput:
    answer_correct: bool
    schema_valid: bool
    final_parsed: bool
    tool_call_count: int
    invalid_call_count: int = 0
    timeout_count: int = 0
    repeated_call_count: int = 0


@dataclass(frozen=True)
class RewardResult:
    total: float
    answer: float
    format: float
    invalid: float
    parse: float
    efficiency: float = 0.0
    minimum_correct_calls: int | None = None


def vanilla_reward(item: RewardInput) -> RewardResult:
    invalid_events = item.invalid_call_count + item.timeout_count + item.repeated_call_count
    invalid_penalty = -min(0.20 * invalid_events, 0.40)
    answer = 1.0 if item.answer_correct else 0.0
    format_reward = 0.05 if item.schema_valid else 0.0
    parse_penalty = 0.0 if item.final_parsed else -0.20
    total = answer + format_reward + invalid_penalty + parse_penalty
    return RewardResult(total, answer, format_reward, invalid_penalty, parse_penalty)


def efficient_group_rewards(
    group: list[RewardInput] | tuple[RewardInput, ...], *, efficiency_lambda: float = 0.15
) -> list[RewardResult]:
    if efficiency_lambda < 0:
        raise ValueError("efficiency_lambda must be non-negative")
    correct_calls = [item.tool_call_count for item in group if item.answer_correct]
    minimum = min(correct_calls) if correct_calls else None
    results = []
    for item in group:
        base = vanilla_reward(item)
        excess = 0 if minimum is None or not item.answer_correct else max(0, item.tool_call_count - minimum)
        penalty = -efficiency_lambda * excess
        results.append(
            RewardResult(
                total=base.total + penalty,
                answer=base.answer,
                format=base.format,
                invalid=base.invalid,
                parse=base.parse,
                efficiency=penalty,
                minimum_correct_calls=minimum,
            )
        )
    return results
