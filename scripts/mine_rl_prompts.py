#!/usr/bin/env python3
"""Mine isolated RL prompts from pass@4 SFT-policy rollouts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from toolforge_rl.rewards import RewardInput, vanilla_reward


TARGETS = {
    "direct_anchor": 160,
    "code_reasoning": 240,
    "retrieval_reasoning": 240,
    "retrieve_then_compute": 160,
}


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def reward_input(row: dict) -> RewardInput:
    return RewardInput(
        answer_correct=row["answer_correct"], schema_valid=row["schema_valid"],
        final_parsed=row["final_parsed"], tool_call_count=row["tool_call_count"],
        invalid_call_count=row.get("invalid_call_count", 0),
        repeated_call_count=row.get("repeated_call_count", 0),
        timeout_count=int(row.get("termination_reason") == "timeout"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw_rows = read_jsonl([args.raw])
    raw_by_id = {row["source_id"]: row for row in raw_rows}
    rollout_rows = read_jsonl(args.rollouts)
    by_source = defaultdict(list)
    rollout_ids = set()
    for row in rollout_rows:
        if row["rollout_id"] in rollout_ids:
            continue
        rollout_ids.add(row["rollout_id"])
        by_source[row["source_id"]].append(row)

    diagnostics = []
    eligible = defaultdict(list)
    excluded = Counter()
    for source_id, raw in raw_by_id.items():
        group = sorted(by_source.get(source_id, []), key=lambda row: row["rollout_index"])
        if len(group) != 4:
            excluded["incomplete_pass_at_4"] += 1
            continue
        correctness = [bool(row["answer_correct"]) for row in group]
        calls = [int(row["tool_call_count"]) for row in group]
        rewards = [vanilla_reward(reward_input(row)).total for row in group]
        reward_variance = statistics.pvariance(rewards)
        call_variance = statistics.pvariance(calls)
        if not any(correctness):
            kind = "all_wrong"
            excluded[kind] += 1
            continue
        if 0 < sum(correctness) < 4:
            kind, priority = "mixed_correctness", 0
        elif call_variance > 0:
            kind, priority = "all_correct_call_variance", 1
        elif reward_variance > 0:
            kind, priority = "reward_variance", 2
        else:
            kind = "constant_group"
            excluded[kind] += 1
            continue
        diag = {
            "source_id": source_id,
            "task_family": raw["task_family"],
            "selection_type": kind,
            "priority": priority,
            "pass_at_4": sum(correctness),
            "correctness_pattern": correctness,
            "tool_call_counts": calls,
            "vanilla_rewards": rewards,
            "reward_variance": reward_variance,
            "tool_call_variance": call_variance,
            "invalid_rate": sum(row.get("invalid_call_count", 0) > 0 for row in group) / 4,
            "truncation_rate": sum(row["termination_reason"] == "incomplete" for row in group) / 4,
        }
        diagnostics.append(diag)
        eligible[raw["task_family"]].append((priority, source_id, raw, diag))

    selected = []
    for family, target in TARGETS.items():
        family_rows = sorted(eligible[family], key=lambda item: (item[0], item[1]))
        for _, source_id, raw, diag in family_rows[:target]:
            selected.append({
                "source_dataset": raw["source_dataset"],
                "source_id": source_id,
                "task_family": family,
                "prompt": raw["prompt"],
                "documents": raw.get("documents", []),
                "reference_answer": raw["reference_answer"],
                "verifier_type": raw["verifier_type"],
                "aliases": raw.get("aliases", []),
                "tools": ["python_exec", "local_search"],
                "metadata": {
                    "selection_type": diag["selection_type"],
                    "pass_at_4": diag["pass_at_4"],
                    "reward_variance": diag["reward_variance"],
                    "tool_call_variance": diag["tool_call_variance"],
                },
            })
    selected.sort(key=lambda row: (row["task_family"], row["source_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_counts = Counter(row["task_family"] for row in selected)
    complete_groups = [group for group in by_source.values() if len(group) == 4]
    all_rollouts = [row for group in complete_groups for row in group]
    manifest = {
        "schema_version": 1,
        "status": "PASS" if selected else "FAIL",
        "candidate_prompts": len(raw_rows),
        "complete_pass_at_4_groups": len(complete_groups),
        "rollouts": len(all_rollouts),
        "final_selected": len(selected),
        "target_total": 800,
        "target_by_family": TARGETS,
        "selected_by_family": dict(selected_counts),
        "selection_types": dict(Counter(row["metadata"]["selection_type"] for row in selected)),
        "excluded": dict(excluded),
        "mean_pass_at_4": statistics.fmean(sum(row["answer_correct"] for row in group) for group in complete_groups) if complete_groups else None,
        "nonzero_reward_variance_ratio": sum(diag["reward_variance"] > 0 for diag in diagnostics) / len(complete_groups) if complete_groups else None,
        "nonzero_tool_call_variance_ratio": sum(diag["tool_call_variance"] > 0 for diag in diagnostics) / len(complete_groups) if complete_groups else None,
        "invalid_rate": sum(row.get("invalid_call_count", 0) > 0 for row in all_rollouts) / len(all_rollouts) if all_rollouts else None,
        "truncation_rate": sum(row["termination_reason"] == "incomplete" for row in all_rollouts) / len(all_rollouts) if all_rollouts else None,
        "quality_shortfall_not_padded": len(selected) < 800,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# RL Prompt Mining Report\n\n"
        f"Status: **{manifest['status']}**\n\n"
        f"- Isolated candidate prompts: {len(raw_rows)}\n"
        f"- Complete pass@4 groups: {len(complete_groups)}\n"
        f"- Selected prompts: {len(selected)}/800\n"
        f"- Mean correct rollouts per group: {manifest['mean_pass_at_4']:.3f}/4\n"
        f"- Non-zero reward variance groups: {manifest['nonzero_reward_variance_ratio']:.2%}\n"
        f"- Non-zero tool-call variance groups: {manifest['nonzero_tool_call_variance_ratio']:.2%}\n"
        f"- Invalid / truncation: {manifest['invalid_rate']:.2%} / {manifest['truncation_rate']:.2%}\n"
        f"- Family counts: {dict(selected_counts)}\n"
        f"- Selection types: {manifest['selection_types']}\n"
        f"- Excluded: {dict(excluded)}\n\n"
        "All-wrong and constant-policy groups are excluded. Shortfalls are reported and never padded. "
        "The output contains prompts, local environment, reference/verifier and mining metadata only; no gold trajectory.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
