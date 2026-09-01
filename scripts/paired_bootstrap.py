#!/usr/bin/env python3
"""Episode-level paired bootstrap for the frozen Vanilla/Efficient comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["source_id"] in rows:
                        raise RuntimeError(f"duplicate source_id: {row['source_id']}")
                    rows[row["source_id"]] = row
    return rows


def ci(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def safe_reduction(baseline: float, improved: float) -> float:
    return (baseline - improved) / baseline if baseline else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla", nargs="+", type=Path, required=True)
    parser.add_argument("--efficient", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    vanilla, efficient = read(args.vanilla), read(args.efficient)
    if set(vanilla) != set(efficient):
        raise RuntimeError("Vanilla and Efficient source IDs differ")
    ids = sorted(vanilla)
    families = np.array([vanilla[source_id]["task_family"] for source_id in ids])
    v_correct = np.array([vanilla[source_id]["answer_correct"] for source_id in ids], dtype=float)
    e_correct = np.array([efficient[source_id]["answer_correct"] for source_id in ids], dtype=float)
    v_calls = np.array([vanilla[source_id]["tool_call_count"] for source_id in ids], dtype=float)
    e_calls = np.array([efficient[source_id]["tool_call_count"] for source_id in ids], dtype=float)
    v_invalid = np.array([vanilla[source_id].get("invalid_call_count", 0) > 0 for source_id in ids], dtype=float)
    e_invalid = np.array([efficient[source_id].get("invalid_call_count", 0) > 0 for source_id in ids], dtype=float)
    rng = np.random.default_rng(args.seed)

    points = {
        "accuracy_delta_pp": float((e_correct.mean() - v_correct.mean()) * 100),
        "average_calls_reduction": safe_reduction(float(v_calls.mean()), float(e_calls.mean())),
        "calls_per_correct_reduction": safe_reduction(
            float(v_calls.sum() / max(v_correct.sum(), 1)),
            float(e_calls.sum() / max(e_correct.sum(), 1)),
        ),
        "invalid_rate_delta_pp": float((e_invalid.mean() - v_invalid.mean()) * 100),
    }
    direct = families == "direct_anchor"
    rtc = families == "retrieve_then_compute"
    points["direct_unnecessary_call_reduction"] = safe_reduction(float(v_calls[direct].astype(bool).mean()), float(e_calls[direct].astype(bool).mean()))
    points["rtc_accuracy_delta_pp"] = float((e_correct[rtc].mean() - v_correct[rtc].mean()) * 100)

    bootstrap = {name: [] for name in points}
    n = len(ids)
    direct_indices = np.flatnonzero(direct)
    rtc_indices = np.flatnonzero(rtc)
    for _ in range(args.samples):
        index = rng.integers(0, n, n)
        vc, ec = v_correct[index], e_correct[index]
        vcall, ecall = v_calls[index], e_calls[index]
        bootstrap["accuracy_delta_pp"].append(float((ec.mean() - vc.mean()) * 100))
        bootstrap["average_calls_reduction"].append(safe_reduction(float(vcall.mean()), float(ecall.mean())))
        bootstrap["calls_per_correct_reduction"].append(safe_reduction(
            float(vcall.sum() / max(vc.sum(), 1)), float(ecall.sum() / max(ec.sum(), 1)),
        ))
        bootstrap["invalid_rate_delta_pp"].append(float((e_invalid[index].mean() - v_invalid[index].mean()) * 100))
        dindex = rng.choice(direct_indices, len(direct_indices), replace=True)
        bootstrap["direct_unnecessary_call_reduction"].append(safe_reduction(
            float(v_calls[dindex].astype(bool).mean()), float(e_calls[dindex].astype(bool).mean()),
        ))
        rindex = rng.choice(rtc_indices, len(rtc_indices), replace=True)
        bootstrap["rtc_accuracy_delta_pp"].append(float((e_correct[rindex].mean() - v_correct[rindex].mean()) * 100))

    checks = {
        "accuracy_drop_le_2pp": points["accuracy_delta_pp"] >= -2.0,
        "calls_or_calls_per_correct_reduction_ge_10pct": (
            points["average_calls_reduction"] >= 0.10 or points["calls_per_correct_reduction"] >= 0.10
        ),
        "direct_unnecessary_call_reduction_ge_15pct": points["direct_unnecessary_call_reduction"] >= 0.15,
        "rtc_accuracy_drop_le_3pp": points["rtc_accuracy_delta_pp"] >= -3.0,
        "invalid_rate_increase_le_2pp": points["invalid_rate_delta_pp"] <= 2.0,
    }
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "NEGATIVE_RESULT",
        "episodes": n,
        "bootstrap_samples": args.samples,
        "point_estimates": points,
        "confidence_intervals_95": {name: ci(values) for name, values in bootstrap.items()},
        "success_checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Paired Bootstrap: Vanilla vs Efficient", "", f"Outcome: **{result['status']}**", "", f"Episodes: {n}; resamples: {args.samples}", "", "| metric | estimate | 95% CI |", "|---|---:|---:|"]
    for name, value in points.items():
        low, high = result["confidence_intervals_95"][name]
        lines.append(f"| {name} | {value:.4f} | [{low:.4f}, {high:.4f}] |")
    lines.extend(["", "## Frozen success checks", ""])
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
