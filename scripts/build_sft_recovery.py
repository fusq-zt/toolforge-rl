#!/usr/bin/env python3
"""Build a concise, train-only protocol recovery subset for a failed SFT gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_QUOTAS = {
    "code_reasoning": 512,
    "direct_anchor": 64,
    "retrieval_reasoning": 64,
    "retrieve_then_compute": 64,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        transcript = row["messages"][-1]["content"]
        if not row.get("schema_valid", True):
            continue
        if "<answer>" not in transcript or "</answer>" not in transcript:
            continue
        buckets[row["task_family"]].append(row)

    selected: list[dict] = []
    for family, quota in DEFAULT_QUOTAS.items():
        # Short, verified examples directly address long, protocol-incomplete generations.
        ranked = sorted(
            buckets[family],
            key=lambda row: (
                int(row.get("trajectory_tokens", 10**9)),
                hashlib.sha256(f"{args.seed}:{row['episode_id']}".encode()).hexdigest(),
            ),
        )
        selected.extend(ranked[:quota])
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"recovery:{args.seed}:{row['episode_id']}".encode()
        ).hexdigest()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "purpose": "one_bounded_sft_protocol_recovery_after_gate_failure",
        "source": str(args.input),
        "source_split": "sft_train_only",
        "selection": "shortest_protocol_valid_per_family_then_seeded_shuffle",
        "requested_quotas": DEFAULT_QUOTAS,
        "actual_by_family": dict(Counter(row["task_family"] for row in selected)),
        "total": len(selected),
        "seed": args.seed,
        "evaluation_examples_used": 0,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
