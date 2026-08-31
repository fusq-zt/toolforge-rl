#!/usr/bin/env python3
"""Select the Gate 3 pilot once; its outputs are reused by Gate 4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


QUOTAS = {"direct_anchor": 40, "code_reasoning": 60, "retrieval_reasoning": 60, "retrieve_then_compute": 40}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw_prompts/sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/raw_prompts/pilot_200.jsonl"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    counts = Counter()
    chosen = []
    for row in rows:
        family = row["task_family"]
        if counts[family] < QUOTAS[family]:
            chosen.append(row)
            counts[family] += 1
    if dict(counts) != QUOTAS:
        raise RuntimeError(f"pilot quotas not met: {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(chosen), "by_family": counts}, default=dict, indent=2))


if __name__ == "__main__":
    main()
