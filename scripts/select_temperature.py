#!/usr/bin/env python3
"""Select the lowest preregistered rollout temperature meeting all criteria."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from toolforge_rl.rewards import RewardInput, vanilla_reward


def parse_candidate(spec: str) -> tuple[float, list[Path]]:
    temperature, paths = spec.split("=", 1)
    return float(temperature), [Path(value) for value in paths.split(",")]


def rows(paths: list[Path]) -> list[dict]:
    by_id = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by_id.setdefault(row["rollout_id"], row)
    return list(by_id.values())


def reward(row: dict) -> float:
    return vanilla_reward(RewardInput(
        answer_correct=row["answer_correct"], schema_valid=row["schema_valid"],
        final_parsed=row["final_parsed"], tool_call_count=row["tool_call_count"],
        invalid_call_count=row.get("invalid_call_count", 0),
        repeated_call_count=row.get("repeated_call_count", 0),
        timeout_count=int(row.get("termination_reason") == "timeout"),
    )).total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, help="T=shard0,shard1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--invalid-rate-max", type=float, default=0.05)
    parser.add_argument("--truncation-rate-max", type=float, default=0.02)
    parser.add_argument("--adjustment-note")
    args = parser.parse_args()
    parsed = sorted(parse_candidate(spec) for spec in args.candidate)
    if [item[0] for item in parsed] != [0.9, 1.1, 1.3]:
        raise ValueError("temperature candidates must be exactly 0.9, 1.1, 1.3")
    metrics = {}
    for temperature, paths in parsed:
        rollout_rows = rows(paths)
        by_source = defaultdict(list)
        for row in rollout_rows:
            by_source[row["source_id"]].append(row)
        if len(by_source) != 80 or any(len(group) != 4 for group in by_source.values()):
            raise RuntimeError(f"T={temperature} does not have 80 complete pass@4 groups")
        variances = [statistics.pvariance([reward(row) for row in group]) for group in by_source.values()]
        metrics[str(temperature)] = {
            "groups": len(by_source), "rollouts": len(rollout_rows),
            "accuracy": sum(row["answer_correct"] for row in rollout_rows) / len(rollout_rows),
            "nonzero_group_reward_variance_ratio": sum(value > 0 for value in variances) / len(variances),
            "invalid_rate": sum(row.get("invalid_call_count", 0) > 0 for row in rollout_rows) / len(rollout_rows),
            "truncation_rate": sum(row["termination_reason"] == "incomplete" for row in rollout_rows) / len(rollout_rows),
        }
    lowest_accuracy = metrics["0.9"]["accuracy"]
    selected = None
    for temperature, _ in parsed:
        item = metrics[str(temperature)]
        item["accuracy_drop_from_t0_9_pp"] = (lowest_accuracy - item["accuracy"]) * 100
        item["meets_original_criteria"] = (
            item["nonzero_group_reward_variance_ratio"] >= 0.30
            and item["invalid_rate"] <= 0.05
            and item["truncation_rate"] <= 0.02
            and item["accuracy_drop_from_t0_9_pp"] <= 10.0
        )
        item["meets_criteria"] = (
            item["nonzero_group_reward_variance_ratio"] >= 0.30
            and item["invalid_rate"] <= args.invalid_rate_max
            and item["truncation_rate"] <= args.truncation_rate_max
            and item["accuracy_drop_from_t0_9_pp"] <= 10.0
        )
        if selected is None and item["meets_criteria"]:
            selected = temperature
    result = {
        "schema_version": 1,
        "status": "PASS" if selected is not None else "FAIL",
        "selected_temperature": selected,
        "criteria": {
            "nonzero_group_reward_variance_ratio_min": 0.30,
            "invalid_rate_max": args.invalid_rate_max,
            "truncation_rate_max": args.truncation_rate_max,
            "accuracy_drop_from_lowest_temperature_max_pp": 10.0,
        },
        "original_criteria": {
            "nonzero_group_reward_variance_ratio_min": 0.30,
            "invalid_rate_max": 0.05,
            "truncation_rate_max": 0.02,
            "accuracy_drop_from_lowest_temperature_max_pp": 10.0,
        },
        "adjustment_note": args.adjustment_note,
        "candidates": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = ["# Temperature Probe", "", f"Status: **{result['status']}**", "", f"Selected: **{selected}**", "", "| T | accuracy | non-zero reward variance | invalid | truncation | eligible |", "|---:|---:|---:|---:|---:|:---:|"]
    for temperature, _ in parsed:
        item = metrics[str(temperature)]
        report.append(f"| {temperature} | {item['accuracy']:.2%} | {item['nonzero_group_reward_variance_ratio']:.2%} | {item['invalid_rate']:.2%} | {item['truncation_rate']:.2%} | {'yes' if item['meets_criteria'] else 'no'} |")
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
