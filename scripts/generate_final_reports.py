#!/usr/bin/env python3
"""Render final result, failure-analysis, and resume reports from real artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            output.extend(json.loads(line) for line in handle if line.strip())
    return output


def failure_type(row: dict) -> str:
    if row["answer_correct"]:
        return "correct"
    if not row["final_parsed"]:
        return "parse_fail"
    if row.get("invalid_call_count", 0):
        return "invalid_call"
    if row.get("repeated_call_count", 0):
        return "repeated_call"
    if row["termination_reason"] == "incomplete":
        return "truncation"
    if row["task_family"] in {"retrieval_reasoning", "retrieve_then_compute"} and row["tool_call_count"] == 0:
        return "retrieval_undercall"
    return "wrong_answer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=summary.json")
    parser.add_argument("--vanilla-predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--efficient-predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    models = [(label, load_json(path)) for label, path in map(parse, args.model)]
    bootstrap = load_json(args.bootstrap)
    vanilla_rows = load_jsonl(args.vanilla_predictions)
    efficient_rows = load_jsonl(args.efficient_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_lines = [
        "# Final Evaluation Results", "",
        f"Frozen evaluation episodes: {models[0][1]['overall']['n']}", "",
        "| model | accuracy | parse | schema | avg calls | calls/correct | direct unnecessary | RTC accuracy | invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in models:
        overall = summary["overall"]
        rtc = summary["by_family"]["retrieve_then_compute"]["accuracy"]
        result_lines.append(
            f"| {label} | {overall['accuracy']:.2%} | {overall['final_parse_rate']:.2%} | "
            f"{overall['schema_valid_rate']:.2%} | {overall['average_tool_calls']:.3f} | "
            f"{overall['calls_per_correct']:.3f} | {overall['direct_unnecessary_call_rate']:.2%} | "
            f"{rtc:.2%} | {overall['invalid_rate']:.2%} |"
        )
    result_lines.extend([
        "", "## Preregistered Vanilla vs Efficient comparison", "",
        f"Outcome: **{bootstrap['status']}**", "",
    ])
    for name, value in bootstrap["point_estimates"].items():
        low, high = bootstrap["confidence_intervals_95"][name]
        result_lines.append(f"- {name}: {value:.4f} (paired-bootstrap 95% CI [{low:.4f}, {high:.4f}])")
    result_lines.extend(["", "No verifier, final test ID, or reward semantic was changed after results were observed."])
    (args.output_dir / "final_results.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    counts = {
        "Vanilla": Counter(failure_type(row) for row in vanilla_rows),
        "Efficient": Counter(failure_type(row) for row in efficient_rows),
    }
    failure_lines = ["# Failure Analysis", "", "| type | Vanilla | Efficient |", "|---|---:|---:|"]
    categories = sorted(set(counts["Vanilla"]) | set(counts["Efficient"]))
    for category in categories:
        failure_lines.append(f"| {category} | {counts['Vanilla'][category]} | {counts['Efficient'][category]} |")
    failure_lines.extend(["", "## Representative Efficient failures", ""])
    examples = [row for row in efficient_rows if not row["answer_correct"]][:12]
    for row in examples:
        prompt = " ".join(row["prompt"].split())[:180]
        failure_lines.append(
            f"- `{row['source_id']}` ({row['task_family']}, {failure_type(row)}, calls={row['tool_call_count']}): "
            f"{prompt}"
        )
    (args.output_dir / "failure_analysis.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    point = bootstrap["point_estimates"]
    calls_reduction = point["calls_per_correct_reduction"] * 100
    accuracy_delta = point["accuracy_delta_pp"]
    resume = f"""# Resume Bullets

## 中文

**ToolForge-RL｜数据驱动的小模型工具推理强化学习**

- 基于 Qwen2.5-3B-Instruct 构建 LoRA SFT→GRPO 后训练链路，完成教师轨迹采样、真实 Python/BM25 工具执行、程序化验证、质量过滤、难度分级和 pass@4 RL Prompt Mining。
- 设计 correctness-gated group-relative efficiency reward，在相同初始化与 rollout 预算下，相比 Vanilla GRPO 将 Calls per Correct 改变 **{calls_reduction:+.2f}%**，答案准确率变化 **{accuracy_delta:+.2f} pp**；实验结论按冻结成功标准判定为 **{bootstrap['status']}**。
- 实现受限 Python Sandbox、episode-local BM25、数学/QA Verifier、trajectory telemetry 与 {bootstrap['bootstrap_samples']} 次 episode-level paired bootstrap，并在冻结 Internal + Public benchmark 上分析工具过用、欠调用和多步完成率。

## English

**ToolForge-RL — Data-Centric Tool Reasoning with Efficient GRPO**

- Built a Qwen2.5-3B LoRA SFT→GRPO pipeline covering teacher trajectory sampling, real Python/BM25 execution, programmatic verification, quality filtering, difficulty labeling, and pass@4 RL prompt mining.
- Designed a correctness-gated group-relative efficiency reward; under identical initialization and rollout budgets, changed calls per correct by **{calls_reduction:+.2f}%** with **{accuracy_delta:+.2f} pp** accuracy change versus Vanilla GRPO. The frozen success criteria evaluate the outcome as **{bootstrap['status']}**.
- Implemented a restricted Python sandbox, episode-local BM25, math/QA verifiers, trajectory telemetry, and {bootstrap['bootstrap_samples']}-sample episode-level paired bootstrap over frozen Internal and Public benchmarks.
"""
    (args.output_dir / "resume_bullets.md").write_text(resume, encoding="utf-8")
    print(json.dumps({"reports": ["final_results.md", "failure_analysis.md", "resume_bullets.md"]}, indent=2))


if __name__ == "__main__":
    main()
