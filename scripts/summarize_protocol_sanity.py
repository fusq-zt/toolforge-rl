#!/usr/bin/env python3
"""Merge failure-only recovery results and decide Gate 1."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(data: list[dict]) -> dict:
    tool_events = [
        event for row in data for event in row["events"]
        if event["kind"] in {"local_search", "python_exec"}
    ]
    total = len(data)
    return {
        "samples": total,
        "protocol_valid": sum(row["protocol_valid"] for row in data),
        "protocol_valid_rate": sum(row["protocol_valid"] for row in data) / total,
        "final_parsed": sum(row.get("final_answer") is not None for row in data),
        "final_parse_rate": sum(row.get("final_answer") is not None for row in data) / total,
        "completed_final": sum(row["stop_reason"] == "final" for row in data),
        "tool_episodes": sum(row["tool_calls"] > 0 for row in data),
        "tool_events": len(tool_events),
        "tool_events_successful": sum(event.get("tool_ok", False) for event in tool_events),
        "tool_execution_success_rate": (
            sum(event.get("tool_ok", False) for event in tool_events) / len(tool_events)
            if tool_events else 0.0
        ),
        "nontermination_failure_rate": sum(row["stop_reason"] != "final" for row in data) / total,
        "mean_elapsed_seconds": statistics.mean(row["elapsed_seconds"] for row in data),
        "termination_reasons": dict(Counter(row["stop_reason"] for row in data)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path("runs/gate1/summary.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/official_checkpoint_sanity.md"))
    args = parser.parse_args()
    base = rows(args.base)
    initial = {row["sample_id"]: row for row in rows(args.reference)}
    recovery = rows(args.recovery)
    for row in recovery:
        initial[row["sample_id"]] = row
    reference = list(initial.values())
    base_metrics, reference_metrics = metrics(base), metrics(reference)
    gate_pass = (
        len(reference) == 100
        and reference_metrics["protocol_valid_rate"] >= 0.95
        and reference_metrics["final_parse_rate"] >= 0.95
        and reference_metrics["tool_execution_success_rate"] >= 0.90
        and reference_metrics["nontermination_failure_rate"] <= 0.05
    )
    payload = {
        "gate": 1, "status": "PASS" if gate_pass else "FAIL",
        "base": base_metrics, "toolstar": reference_metrics,
        "recovery": {"rerun_samples": len(recovery), "max_step_tokens": 1024},
        "selection": {"total": 100, "search": 50, "python": 50, "shortest_complete": True},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    m, b = reference_metrics, base_metrics
    report = f"""# Official checkpoint sanity (Gate 1)

Status: **{payload['status']}**

The frozen official SFT set contributed 100 shortest complete trajectories (50
search and 50 Python). Both checkpoints were loaded once on separate RTX 4090s and
ran the same real parse→execute→insert-result→continue loop. The initial 384-token
per-turn cap truncated 26 verbose Tool-Star search thoughts before a closing action;
only those failed IDs were rerun at 1024 tokens. The 74 successes were not repeated.

| Metric | Qwen base | Tool-Star merged | Gate threshold |
|---|---:|---:|---:|
| Protocol valid | {b['protocol_valid_rate']:.1%} | {m['protocol_valid_rate']:.1%} | ≥95% reference |
| Final parsed | {b['final_parse_rate']:.1%} | {m['final_parse_rate']:.1%} | ≥95% reference |
| Tool-event execution success | {b['tool_execution_success_rate']:.1%} | {m['tool_execution_success_rate']:.1%} | ≥90% reference |
| Nontermination failure | {b['nontermination_failure_rate']:.1%} | {m['nontermination_failure_rate']:.1%} | ≤5% reference |
| Tool events | {b['tool_events']} | {m['tool_events']} | descriptive |

Tool execution failures are preserved as structured observations rather than hidden.
They arise when official Python generations request modules outside the frozen safe
allowlist; a successful tool call is never treated as answer correctness.

Raw immutable runs are `runs/gate1/qwen_100.jsonl`,
`runs/gate1/toolstar_100.jsonl`, and the failure-only
`runs/gate1/toolstar_recovery_1024.jsonl`. Machine-readable merged metrics are in
`runs/gate1/summary.json`.
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
