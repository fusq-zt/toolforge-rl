#!/usr/bin/env python3
"""Select original prompt rows corresponding to failed rollout records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", default="source_id")
    parser.add_argument(
        "--failure-field",
        default="schema_valid",
        help="Boolean rollout field; rows for which it is false are selected.",
    )
    parser.add_argument(
        "--termination-reason",
        help="Select only this termination reason (for example: incomplete).",
    )
    args = parser.parse_args()

    source = read_jsonl(args.source)
    rollouts = read_jsonl(args.rollouts)
    if args.termination_reason:
        failed_keys = {
            row[args.key]
            for row in rollouts
            if row.get("termination_reason") == args.termination_reason
        }
        selector = {"termination_reason": args.termination_reason}
    else:
        failed_keys = {
            row[args.key] for row in rollouts if not bool(row.get(args.failure_field))
        }
        selector = {"failure_field": args.failure_field}
    selected = [row for row in source if row.get(args.key) in failed_keys]

    selected_keys = {row.get(args.key) for row in selected}
    missing = sorted(failed_keys - selected_keys)
    if missing:
        raise ValueError(f"source is missing {len(missing)} failed keys")
    if len(selected) != len(selected_keys):
        raise ValueError("selected source contains duplicate keys")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                **selector,
                "failed_keys": len(failed_keys),
                "selected_rows": len(selected),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
