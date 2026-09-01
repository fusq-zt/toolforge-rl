#!/usr/bin/env python3
"""Verify the two-stage save/resume GRPO smoke run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    with (args.run_dir / "trajectories.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    terminations = Counter(row["termination_reason"] for row in rows)
    checkpoint_states = list((args.run_dir / "checkpoints").glob("step_*/state.json"))
    checks = {
        "64_prompt_groups": summary["processed_groups"] == 64,
        "256_rollouts": len(rows) == 256,
        "optimizer_steps_10_to_20": 10 <= summary["optimizer_steps"] <= 20,
        "reward_nonconstant": summary["relative_reward_groups"] > 0,
        "advantage_nonzero": summary["relative_reward_group_ratio"] > 0,
        "gradient_nonzero": summary["gradient_nonzero"],
        "checkpoint_saved": bool(checkpoint_states),
        "resume_completed": (args.run_dir / "final" / "state.json").exists(),
        "loop_is_hard_bounded": all(row["tool_call_count"] <= 3 for row in rows),
        "no_nan": summary["status"] == "PASS",
    }
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": summary,
        "termination_reasons": dict(terminations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# GRPO Smoke Report", "", f"Status: **{result['status']}**", ""]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
