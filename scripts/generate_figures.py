#!/usr/bin/env python3
"""Generate the nine preregistered static figures from real run artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_model(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finish(path: Path, title: str, ylabel: str | None = None) -> None:
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=summary.json")
    parser.add_argument("--vanilla-predictions", type=Path, required=True)
    parser.add_argument("--efficient-predictions", type=Path, required=True)
    parser.add_argument("--vanilla-train-metrics", type=Path, required=True)
    parser.add_argument("--efficient-train-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [(label, load_json(path)) for label, path in map(parse_model, args.model)]
    labels = [label for label, _ in models]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    plt.figure(figsize=(7, 5))
    for (label, summary), color in zip(models, colors):
        item = summary["overall"]
        plt.scatter(item["average_tool_calls"], item["accuracy"], label=label, color=color, s=55)
    plt.xlabel("Average tool calls")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=8)
    finish(args.output_dir / "01_accuracy_vs_average_calls.png", "Accuracy vs Average Tool Calls")

    plt.figure(figsize=(7, 5))
    for (label, summary), color in zip(models, colors):
        item = summary["overall"]
        plt.scatter(item["calls_per_correct"], item["accuracy"], label=label, color=color, s=55)
    plt.xlabel("Calls per correct answer")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=8)
    finish(args.output_dir / "02_accuracy_vs_calls_per_correct.png", "Accuracy vs Calls per Correct")

    families = ["direct_anchor", "code_reasoning", "retrieval_reasoning", "retrieve_then_compute"]
    x = np.arange(len(families))
    width = 0.8 / len(models)
    plt.figure(figsize=(10, 5))
    for index, ((label, summary), color) in enumerate(zip(models, colors)):
        values = [summary["by_family"].get(family, {}).get("accuracy", 0) for family in families]
        plt.bar(x - 0.4 + width / 2 + index * width, values, width, label=label, color=color)
    plt.xticks(x, ["Direct", "Code", "Retrieval", "Retrieve→Compute"])
    plt.legend(fontsize=7)
    finish(args.output_dir / "03_task_family_accuracy.png", "Accuracy by Task Family", "Accuracy")

    plt.figure(figsize=(8, 5))
    values = [summary["overall"]["direct_unnecessary_call_rate"] for _, summary in models]
    plt.bar(labels, values, color=colors)
    plt.xticks(rotation=25, ha="right")
    finish(args.output_dir / "04_direct_unnecessary_call_rate.png", "Direct Unnecessary Call Rate", "Rate")

    plt.figure(figsize=(8, 5))
    values = [summary["by_family"]["retrieve_then_compute"]["accuracy"] for _, summary in models]
    plt.bar(labels, values, color=colors)
    plt.xticks(rotation=25, ha="right")
    finish(args.output_dir / "05_retrieve_compute_completion.png", "Retrieve→Compute Completion", "Accuracy")

    vanilla_metrics = load_jsonl(args.vanilla_train_metrics)
    efficient_metrics = load_jsonl(args.efficient_train_metrics)
    plt.figure(figsize=(11, 4))
    for index, metric in enumerate(("mean_reward", "mean_kl", "token_entropy_proxy"), start=1):
        plt.subplot(1, 3, index)
        for name, rows in (("Vanilla", vanilla_metrics), ("Efficient", efficient_metrics)):
            plt.plot([row["processed_groups"] for row in rows], [row.get(metric, 0) for row in rows], label=name, alpha=0.8)
        plt.title(metric.replace("_", " "))
        plt.xlabel("Prompt groups")
        if index == 1:
            plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "06_reward_kl_entropy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, rows in (("Vanilla", vanilla_metrics), ("Efficient", efficient_metrics)):
        flags = np.array([row["reward_variance"] > 0 for row in rows], dtype=float)
        window = min(25, len(flags))
        rolling = np.convolve(flags, np.ones(window) / window, mode="valid") if window else flags
        plt.plot(np.arange(len(rolling)) + window, rolling, label=name)
    plt.xlabel("Prompt groups")
    plt.legend()
    finish(args.output_dir / "07_group_reward_variance_ratio.png", "Rolling Non-zero Group Reward Variance", "Ratio")

    vanilla = load_jsonl(args.vanilla_predictions)
    efficient = load_jsonl(args.efficient_predictions)
    bins = np.arange(-0.5, 4.5, 1)
    plt.figure(figsize=(7, 5))
    plt.hist([row["tool_call_count"] for row in vanilla], bins=bins, alpha=0.6, label="Vanilla")
    plt.hist([row["tool_call_count"] for row in efficient], bins=bins, alpha=0.6, label="Efficient")
    plt.xticks(range(4))
    plt.xlabel("Tool calls")
    plt.legend()
    finish(args.output_dir / "08_tool_call_distribution.png", "Tool Call Count Distribution", "Episodes")

    def failures(rows: list[dict]) -> Counter:
        counts = Counter()
        for row in rows:
            if row["answer_correct"]:
                counts["correct"] += 1
            elif not row["final_parsed"]:
                counts["parse_fail"] += 1
            elif row.get("invalid_call_count", 0):
                counts["invalid_call"] += 1
            elif row.get("repeated_call_count", 0):
                counts["repeated_call"] += 1
            elif row["termination_reason"] == "incomplete":
                counts["truncation"] += 1
            else:
                counts["wrong_answer"] += 1
        return counts
    categories = ["correct", "wrong_answer", "parse_fail", "invalid_call", "repeated_call", "truncation"]
    v_fail, e_fail = failures(vanilla), failures(efficient)
    x = np.arange(len(categories))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, [v_fail[name] for name in categories], 0.4, label="Vanilla")
    plt.bar(x + 0.2, [e_fail[name] for name in categories], 0.4, label="Efficient")
    plt.xticks(x, categories, rotation=25, ha="right")
    plt.legend()
    finish(args.output_dir / "09_failure_type_distribution.png", "Failure Type Distribution", "Episodes")

    manifest = {
        "schema_version": 1,
        "figures": sorted(path.name for path in args.output_dir.glob("*.png")),
        "note": "token_entropy_proxy is negative mean policy-token log probability, logged without an extra full-vocabulary entropy pass.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
