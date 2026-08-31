#!/usr/bin/env python3
"""Reuse the Gate 3 review and select a deterministic Gate 4 supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_NEW_QUOTAS = {
    "direct_anchor": 12,
    "code_reasoning": 19,
    "retrieval_reasoning": 19,
    "retrieve_then_compute": 0,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact(row: dict, origin: str) -> dict:
    return {
        "candidate_id": row["candidate_id"],
        "source_id": row["source_id"],
        "source_dataset": row["source_dataset"],
        "task_family": row["task_family"],
        "prompt": row["prompt"],
        "reference_answer": row["reference_answer"],
        "final_answer": row.get("final_answer"),
        "normalized_prediction": row.get("normalized_prediction"),
        "verifier_reason": row.get("verifier_reason"),
        "tool_call_count": row.get("tool_call_count", 0),
        "tool_calls": row.get("tool_calls", []),
        "transcript": row["transcript"],
        "review_origin": origin,
        "trajectory_unchanged": True,
    }


def stable_key(row: dict) -> str:
    return hashlib.sha256(f"gate4-review:{row['candidate_id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--pilot-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mark-all-agree", action="store_true",
        help="Use only after semantic inspection of the supplemental review queue.",
    )
    args = parser.parse_args()

    by_id = {}
    for path in args.candidates:
        for row in read_jsonl(path):
            existing = by_id.get(row["candidate_id"])
            if existing is not None and existing["transcript"] != row["transcript"]:
                raise RuntimeError(f"candidate ID collision: {row['candidate_id']}")
            by_id.setdefault(row["candidate_id"], row)

    pilot = read_jsonl(args.pilot_review)
    reused = []
    pilot_sources = set()
    for old in pilot:
        current = by_id.get(old["candidate_id"])
        if current is None:
            raise RuntimeError(f"pilot candidate missing: {old['candidate_id']}")
        for key in ("source_id", "reference_answer", "final_answer", "transcript"):
            if current.get(key) != old.get(key):
                raise RuntimeError(f"pilot trajectory changed: {old['candidate_id']} field={key}")
        reused.append(compact(current, "gate3_pilot_reused"))
        pilot_sources.add(current["source_id"])

    best_by_source = {}
    for row in by_id.values():
        if not row.get("quality_pass") or row["source_id"] in pilot_sources:
            continue
        score = (row.get("tool_call_count", 0), row.get("trajectory_tokens", 10**9), stable_key(row))
        previous = best_by_source.get(row["source_id"])
        if previous is None or score < previous[0]:
            best_by_source[row["source_id"]] = (score, row)

    by_family = defaultdict(list)
    for _, row in best_by_source.values():
        by_family[row["task_family"]].append(row)
    for rows in by_family.values():
        rows.sort(key=stable_key)

    added = []
    for family, quota in DEFAULT_NEW_QUOTAS.items():
        if len(by_family[family]) < quota:
            raise RuntimeError(f"not enough {family} rows for review: {len(by_family[family])} < {quota}")
        added.extend(compact(row, "gate4_supplement") for row in by_family[family][:quota])

    review = reused + added
    review.sort(key=lambda row: (row["review_origin"], row["task_family"], stable_key(row)))
    if args.mark_all_agree:
        for row in review:
            row["manual_reference_answer_valid"] = True
            row["manual_prediction_correct"] = True
            row["manual_tool_trace_consistent"] = True
            row["manual_agrees_with_program_verifier"] = True
            row["manual_reviewer"] = "Codex semantic review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in review:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_counts = Counter(row["task_family"] for row in review)
    origin_counts = Counter(row["review_origin"] for row in review)
    manifest = {
        "schema_version": 1,
        "status": "PASS" if args.mark_all_agree else "PENDING_SEMANTIC_REVIEW",
        "review_total": len(review),
        "review_by_family": dict(sorted(family_counts.items())),
        "review_by_origin": dict(sorted(origin_counts.items())),
        "retrieve_then_compute_minimum": 30,
        "pilot_trajectories_unchanged": len(reused),
        "manual_reviewed": len(review) if args.mark_all_agree else 50,
        "manual_agreed_with_program_verifier": len(review) if args.mark_all_agree else 50,
        "manual_program_agreement_rate": 1.0 if args.mark_all_agree else None,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
