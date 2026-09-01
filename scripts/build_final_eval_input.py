#!/usr/bin/env python3
"""Combine frozen Internal and Public rows for one-load-per-model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for suite, path in (("internal_test", args.internal), ("public_heldout", args.public)):
        for row in read(path):
            item = dict(row)
            item["evaluation_suite"] = suite
            rows.append(item)
    source_ids = [row["source_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("duplicate source IDs in final evaluation input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "total": len(rows),
        "by_suite": dict(Counter(row["evaluation_suite"] for row in rows)),
        "by_family": dict(Counter(row["task_family"] for row in rows)),
        "sha256": digest,
        "purpose": "single model load followed by all frozen Internal and Public datasets",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
