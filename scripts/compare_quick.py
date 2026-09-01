#!/usr/bin/env python3
"""Apply the preregistered continuation checks to Quick GRPO branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-eval", type=Path, required=True)
    parser.add_argument("--efficient-eval", type=Path, required=True)
    parser.add_argument("--efficient-train", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--title", default="Quick GRPO Comparison")
    args = parser.parse_args()
    vanilla = load(args.vanilla_eval)
    efficient = load(args.efficient_eval)
    training = load(args.efficient_train)
    v = vanilla["overall"]
    e = efficient["overall"]
    v_rtc = vanilla["by_family"]["retrieve_then_compute"]["accuracy"]
    e_rtc = efficient["by_family"]["retrieve_then_compute"]["accuracy"]
    checks = {
        "efficient_accuracy_ge_vanilla_minus_3pp": e["accuracy"] >= v["accuracy"] - 0.03,
        "efficient_average_calls_le_vanilla": e["average_tool_calls"] <= v["average_tool_calls"],
        "efficient_invalid_rate_le_5pct": e["invalid_rate"] <= 0.05,
        "retrieve_then_compute_not_collapsed": e_rtc >= v_rtc - 0.05 and e_rtc > 0,
        "relative_reward_groups_ge_25pct": training["relative_reward_group_ratio"] >= 0.25,
    }
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "efficiency_lambda": args.lambda_value,
        "checks": checks,
        "vanilla": {
            "accuracy": v["accuracy"], "average_tool_calls": v["average_tool_calls"],
            "invalid_rate": v["invalid_rate"], "rtc_accuracy": v_rtc,
        },
        "efficient": {
            "accuracy": e["accuracy"], "average_tool_calls": e["average_tool_calls"],
            "invalid_rate": e["invalid_rate"], "rtc_accuracy": e_rtc,
            "relative_reward_group_ratio": training["relative_reward_group_ratio"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = [f"# {args.title}", "", f"Status: **{result['status']}**", "", f"Efficiency lambda: {args.lambda_value}", ""]
    report.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    report.extend([
        "", "| branch | accuracy | average calls | invalid | RTC accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| Vanilla | {v['accuracy']:.2%} | {v['average_tool_calls']:.3f} | {v['invalid_rate']:.2%} | {v_rtc:.2%} |",
        f"| Efficient | {e['accuracy']:.2%} | {e['average_tool_calls']:.3f} | {e['invalid_rate']:.2%} | {e_rtc:.2%} |",
    ])
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
