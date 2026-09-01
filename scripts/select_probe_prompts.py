#!/usr/bin/env python3
"""Freeze the stratified 80-prompt Dev subset used for temperature selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


QUOTAS = {
    "direct_anchor": 16,
    "code_reasoning": 24,
    "retrieval_reasoning": 24,
    "retrieve_then_compute": 16,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["task_family"]].append(row)
    selected = []
    for family, quota in QUOTAS.items():
        ordered = sorted(
            by_family[family],
            key=lambda row: hashlib.sha256(f"temperature-probe:{row['source_id']}".encode()).hexdigest(),
        )
        if len(ordered) < quota:
            raise RuntimeError(f"not enough {family} Dev rows")
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: row["source_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "total": len(selected),
        "by_family": dict(Counter(row["task_family"] for row in selected)),
        "source_ids": [row["source_id"] for row in selected],
        "selection": "sha256 deterministic stratified Dev subset",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
