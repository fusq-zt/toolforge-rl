#!/usr/bin/env python3
"""Summarize the preregistered three-seed Formal GRPO comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reduction(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for seed in args.seeds:
        vanilla = load(Path(f"runs/grpo_formal_vanilla_dev_seed{seed}/summary.json"))
        efficient = load(Path(f"runs/grpo_formal_efficient_dev_seed{seed}/summary.json"))
        train = load(Path(f"runs/grpo_formal_efficient_seed{seed}/summary.json"))
        v, e = vanilla["overall"], efficient["overall"]
        v_rtc = vanilla["by_family"]["retrieve_then_compute"]["accuracy"]
        e_rtc = efficient["by_family"]["retrieve_then_compute"]["accuracy"]
        item = {
            "seed": seed,
            "vanilla_accuracy": v["accuracy"],
            "efficient_accuracy": e["accuracy"],
            "accuracy_delta_pp": (e["accuracy"] - v["accuracy"]) * 100,
            "vanilla_average_calls": v["average_tool_calls"],
            "efficient_average_calls": e["average_tool_calls"],
            "average_calls_reduction": reduction(v["average_tool_calls"], e["average_tool_calls"]),
            "calls_per_correct_reduction": reduction(v["calls_per_correct"], e["calls_per_correct"]),
            "invalid_delta_pp": (e["invalid_rate"] - v["invalid_rate"]) * 100,
            "rtc_accuracy_delta_pp": (e_rtc - v_rtc) * 100,
            "efficient_relative_reward_group_ratio": train["relative_reward_group_ratio"],
        }
        item["continuation_checks"] = {
            "accuracy_within_3pp": item["accuracy_delta_pp"] >= -3.0,
            "average_calls_not_higher": item["average_calls_reduction"] >= 0,
            "invalid_rate_le_5pct": e["invalid_rate"] <= 0.05,
            "rtc_not_collapsed": item["rtc_accuracy_delta_pp"] >= -5.0 and e_rtc > 0,
            "relative_reward_groups_ge_25pct": item["efficient_relative_reward_group_ratio"] >= 0.25,
        }
        item["status"] = "PASS" if all(item["continuation_checks"].values()) else "MIXED"
        records.append(item)

    metrics = (
        "accuracy_delta_pp", "average_calls_reduction", "calls_per_correct_reduction",
        "invalid_delta_pp", "rtc_accuracy_delta_pp", "efficient_relative_reward_group_ratio",
    )
    aggregate = {
        name: {
            "mean": statistics.fmean(row[name] for row in records),
            "min": min(row[name] for row in records),
            "max": max(row[name] for row in records),
        }
        for name in metrics
    }
    pass_count = sum(row["status"] == "PASS" for row in records)
    result = {
        "schema_version": 1,
        "status": "PASS" if pass_count >= 2 and records[0]["status"] == "PASS" else "MIXED_REPLICATION",
        "primary_seed": 42,
        "seed_count": len(records),
        "passing_seeds": pass_count,
        "seeds": records,
        "aggregate": aggregate,
        "interpretation": "Seed 42 is the preregistered primary run; 123 and 2026 are trend replications added only after seed 42 passed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Formal GRPO Three-seed Replication", "", f"Status: **{result['status']}**", "",
        "| seed | accuracy Δ pp | avg-call reduction | calls/correct reduction | RTC Δ pp | invalid Δ pp | relative groups |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['seed']} | {row['accuracy_delta_pp']:+.2f} | {row['average_calls_reduction']:.2%} | "
            f"{row['calls_per_correct_reduction']:.2%} | {row['rtc_accuracy_delta_pp']:+.2f} | "
            f"{row['invalid_delta_pp']:+.2f} | {row['efficient_relative_reward_group_ratio']:.2%} |"
        )
    lines.extend(["", result["interpretation"]])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
