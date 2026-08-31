#!/usr/bin/env python3
"""Summarize one or more real-tool rollout shards with Gate-compatible metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(paths: list[Path]) -> list[dict]:
    by_id = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    old = by_id.get(row["rollout_id"])
                    if old is not None and old["transcript"] != row["transcript"]:
                        raise RuntimeError(f"rollout collision: {row['rollout_id']}")
                    by_id.setdefault(row["rollout_id"], row)
    return list(by_id.values())


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(row["answer_correct"]) for row in rows)
    calls = [int(row["tool_call_count"]) for row in rows]
    tool_rows = [row for row in rows if row["tool_call_count"] > 0]
    direct = [row for row in rows if row["task_family"] == "direct_anchor"]
    retrieval = [row for row in rows if row["task_family"] in {"retrieval_reasoning", "retrieve_then_compute"}]
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "final_parse_rate": sum(bool(row["final_parsed"]) for row in rows) / n if n else None,
        "schema_valid_rate": sum(bool(row["schema_valid"]) for row in rows) / n if n else None,
        "tool_execution_success_rate": sum(bool(row["execution_success"]) for row in rows) / n if n else None,
        "multi_turn_completion_rate": (
            sum(row["final_parsed"] and row["schema_valid"] for row in tool_rows) / len(tool_rows)
            if tool_rows else None
        ),
        "infinite_loop_rate": sum(row["termination_reason"] == "turn_limit" for row in rows) / n if n else None,
        "average_tool_calls": statistics.fmean(calls) if calls else None,
        "calls_per_correct": sum(calls) / correct if correct else None,
        "direct_unnecessary_call_rate": (
            sum(row["tool_call_count"] > 0 for row in direct) / len(direct) if direct else None
        ),
        "retrieval_undercall_rate": (
            sum(row["tool_call_count"] == 0 for row in retrieval) / len(retrieval) if retrieval else None
        ),
        "repeated_call_rate": sum(row["repeated_call_count"] > 0 for row in rows) / n if n else None,
        "invalid_rate": sum(row["invalid_call_count"] > 0 for row in rows) / n if n else None,
        "truncation_rate": sum(row["termination_reason"] == "incomplete" for row in rows) / n if n else None,
        "generated_tokens_mean": statistics.fmean(row["generated_tokens"] for row in rows) if rows else None,
        "latency_mean_seconds": statistics.fmean(row["latency_seconds"] for row in rows) if rows else None,
        "latency_p95_seconds": percentile([row["latency_seconds"] for row in rows], 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--sft-gate", action="store_true")
    parser.add_argument("--sft-protocol-threshold", type=float, default=0.98)
    parser.add_argument("--sft-noncode-threshold", type=float)
    parser.add_argument("--sft-adjustment-note")
    args = parser.parse_args()
    rows = read_jsonl(args.inputs)
    by_family_rows = defaultdict(list)
    by_dataset_rows = defaultdict(list)
    by_difficulty_rows = defaultdict(list)
    by_suite_rows = defaultdict(list)
    for row in rows:
        by_family_rows[row["task_family"]].append(row)
        by_dataset_rows[row["source_dataset"]].append(row)
        by_difficulty_rows[row.get("difficulty", "unassigned")].append(row)
        by_suite_rows[row.get("evaluation_suite", "unspecified")].append(row)
    overall = metrics(rows)
    result = {
        "schema_version": 1,
        "label": args.label,
        "overall": overall,
        "by_family": {name: metrics(group) for name, group in sorted(by_family_rows.items())},
        "by_dataset": {name: metrics(group) for name, group in sorted(by_dataset_rows.items())},
        "by_difficulty": {name: metrics(group) for name, group in sorted(by_difficulty_rows.items())},
        "by_evaluation_suite": {name: metrics(group) for name, group in sorted(by_suite_rows.items())},
    }
    if args.sft_gate:
        baseline_accuracy = None
        if args.baseline_summary:
            baseline_accuracy = json.loads(args.baseline_summary.read_text(encoding="utf-8"))["overall"]["accuracy"]
        family_nonzero = all(group["accuracy"] > 0 for group in result["by_family"].values())
        rtc_accuracy = result["by_family"].get("retrieve_then_compute", {}).get("accuracy", 0)
        threshold_label = str(args.sft_protocol_threshold).replace(".", "_")
        checks = {
            f"final_parse_ge_{threshold_label}": overall["final_parse_rate"] >= args.sft_protocol_threshold,
            f"schema_valid_ge_{threshold_label}": overall["schema_valid_rate"] >= args.sft_protocol_threshold,
            "tool_execution_ge_0_95": overall["tool_execution_success_rate"] >= 0.95,
            "infinite_loop_zero": overall["infinite_loop_rate"] == 0,
            "all_families_nonzero_accuracy": family_nonzero,
            "retrieve_then_compute_ge_0_35": rtc_accuracy >= 0.35,
            "accuracy_above_base": baseline_accuracy is not None and overall["accuracy"] > baseline_accuracy,
        }
        if args.sft_noncode_threshold is not None:
            noncode = [
                group for name, group in result["by_family"].items()
                if name != "code_reasoning"
            ]
            noncode_label = str(args.sft_noncode_threshold).replace(".", "_")
            checks[f"noncode_parse_ge_{noncode_label}"] = all(
                group["final_parse_rate"] >= args.sft_noncode_threshold for group in noncode
            )
            checks[f"noncode_schema_ge_{noncode_label}"] = all(
                group["schema_valid_rate"] >= args.sft_noncode_threshold for group in noncode
            )
        result["sft_gate"] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "protocol_threshold_applied": args.sft_protocol_threshold,
            "original_protocol_threshold": 0.98,
            "strict_original_protocol_checks": {
                "final_parse_ge_0_98": overall["final_parse_rate"] >= 0.98,
                "schema_valid_ge_0_98": overall["schema_valid_rate"] >= 0.98,
            },
            "adjustment_note": args.sft_adjustment_note,
            "base_accuracy": baseline_accuracy,
            "accuracy_delta_pp": (overall["accuracy"] - baseline_accuracy) * 100 if baseline_accuracy is not None else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        lines = [f"# {args.label} rollout summary", "", f"Episodes: {overall['n']}", "", "| scope | accuracy | parse | schema | avg calls |", "|---|---:|---:|---:|---:|"]
        lines.append(f"| overall | {overall['accuracy']:.2%} | {overall['final_parse_rate']:.2%} | {overall['schema_valid_rate']:.2%} | {overall['average_tool_calls']:.3f} |")
        for name, group in result["by_family"].items():
            lines.append(f"| {name} | {group['accuracy']:.2%} | {group['final_parse_rate']:.2%} | {group['schema_valid_rate']:.2%} | {group['average_tool_calls']:.3f} |")
        if "sft_gate" in result:
            lines.extend(["", f"SFT Gate: **{result['sft_gate']['status']}**", ""])
            lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in result["sft_gate"]["checks"].items())
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
