#!/usr/bin/env python3
"""Summarize the reusable 200-prompt pilot and create its 50-item review set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


REVIEW_TARGETS = {
    "retrieve_then_compute": 30,
    "code_reasoning": 7,
    "retrieval_reasoning": 7,
    "direct_anchor": 6,
}


def rows(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            output.extend(json.loads(line) for line in handle if line.strip())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, default=Path("data/examples/pilot_review_50.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/manifests/pilot_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_pilot_report.md"))
    parser.add_argument("--manual-reviewed", type=int, default=0)
    parser.add_argument("--manual-agreed", type=int, default=0)
    args = parser.parse_args()
    data = rows(args.inputs)
    by_source = defaultdict(list)
    for row in data:
        by_source[row["source_id"]].append(row)
    eligible = [row for row in data if row.get("quality_pass")]
    best = {}
    for row in eligible:
        old = best.get(row["source_id"])
        if old is None or (row["tool_call_count"], row["trajectory_tokens"], row["candidate_id"]) < (
            old["tool_call_count"], old["trajectory_tokens"], old["candidate_id"]
        ):
            best[row["source_id"]] = row
    review = []
    for family, target in REVIEW_TARGETS.items():
        family_rows = [row for row in best.values() if row["task_family"] == family]
        family_rows.sort(key=lambda row: hashlib.sha256(row["source_id"].encode()).hexdigest())
        review.extend(family_rows[:target])
    # If a hard family has fewer verified rows, preserve the shortfall visibly and
    # fill to 50 only for general inspection; do not claim the target was met.
    used = {row["source_id"] for row in review}
    remainder = sorted(
        (row for row in best.values() if row["source_id"] not in used),
        key=lambda row: hashlib.sha256(row["source_id"].encode()).hexdigest(),
    )
    review.extend(remainder[: max(0, 50 - len(review))])
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    with args.review_output.open("w", encoding="utf-8") as handle:
        for row in review:
            compact = {
                key: row.get(key) for key in [
                    "candidate_id", "source_id", "source_dataset", "task_family", "prompt",
                    "reference_answer", "final_answer", "normalized_prediction", "verifier_reason",
                    "tool_call_count", "tool_calls", "transcript",
                ]
            }
            handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
    manual_rate = args.manual_agreed / args.manual_reviewed if args.manual_reviewed else 0.0
    pilot_pass = (
        len(by_source) == 200 and len(data) == 600 and len(review) == 50
        and args.manual_reviewed == 50 and manual_rate >= 0.95
        and all(any(row["task_family"] == family for row in eligible) for family in REVIEW_TARGETS)
    )
    summary = {
        "status": "PASS" if pilot_pass else "PENDING_MANUAL_REVIEW",
        "raw_prompts": len(by_source),
        "candidates": len(data),
        "expected_candidates": len(by_source) * 3,
        "quality_pass": len(eligible),
        "sources_with_verified_candidate": len(best),
        "quality_pass_rate": len(eligible) / len(data) if data else 0,
        "by_family_candidates": dict(Counter(row["task_family"] for row in data)),
        "by_family_quality_pass": dict(Counter(row["task_family"] for row in eligible)),
        "by_hint_quality_pass": dict(Counter(row["sampling_hint"] for row in eligible)),
        "termination_reasons": dict(Counter(row["termination_reason"] for row in data)),
        "review_total": len(review),
        "review_by_family": dict(Counter(row["task_family"] for row in review)),
        "review_targets": REVIEW_TARGETS,
        "manual_reviewed": args.manual_reviewed,
        "manual_agreed_with_program_verifier": args.manual_agreed,
        "manual_program_agreement_rate": manual_rate,
        "pilot_reused_by_gate4": True,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# Gate 3 data pilot

Status: **{summary['status']}**

- Frozen source prompts: {summary['raw_prompts']} (target 200)
- Immutable candidates: {summary['candidates']} (target 600)
- Quality-pass trajectories: {summary['quality_pass']} ({summary['quality_pass_rate']:.1%})
- Sources with at least one verified trajectory: {summary['sources_with_verified_candidate']}
- Termination reasons: `{json.dumps(summary['termination_reasons'], ensure_ascii=False)}`
- Verified by family: `{json.dumps(summary['by_family_quality_pass'], ensure_ascii=False)}`
- Manual/program verifier agreement: {args.manual_agreed}/{args.manual_reviewed} ({manual_rate:.1%})

The 50-row review set is `data/examples/pilot_review_50.jsonl`. It attempts to
include 30 retrieve→compute examples now so these unchanged rows can count toward
the Gate 4 review of 100. Pilot candidate files remain the first immutable shards
of the full build; Gate 4 resumes around them instead of regenerating them.
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
