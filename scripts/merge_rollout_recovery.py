#!/usr/bin/env python3
"""Merge targeted rollout recovery rows over an initial evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, nargs="+", required=True)
    parser.add_argument("--recovery", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", default="source_id")
    args = parser.parse_args()

    initial = read_jsonl(args.initial)
    recovery = read_jsonl(args.recovery)
    initial_keys = [row.get(args.key) for row in initial]
    recovery_keys = [row.get(args.key) for row in recovery]

    if any(key is None for key in initial_keys + recovery_keys):
        raise ValueError(f"missing merge key: {args.key}")
    if len(initial_keys) != len(set(initial_keys)):
        raise ValueError("initial inputs contain duplicate keys")
    if len(recovery_keys) != len(set(recovery_keys)):
        raise ValueError("recovery inputs contain duplicate keys")
    unknown = sorted(set(recovery_keys) - set(initial_keys))
    if unknown:
        raise ValueError(f"recovery contains {len(unknown)} unknown keys")

    recovered = dict(zip(recovery_keys, recovery, strict=True))
    merged = [recovered.get(row[args.key], row) for row in initial]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    replaced = sum(key in recovered for key in initial_keys)
    print(
        json.dumps(
            {
                "initial_rows": len(initial),
                "recovery_rows": len(recovery),
                "replaced_rows": replaced,
                "merged_rows": len(merged),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
