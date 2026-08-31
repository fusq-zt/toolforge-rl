#!/usr/bin/env python3
"""Select a deterministic short/complete 100-example Tool-Star sanity set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from toolforge_rl.protocols.toolstar import validate_transcript


PAIR_RE = re.compile(
    r"<(search|python)>\s*(.*?)\s*</\1>\s*<result>\s*(.*?)\s*</result>",
    re.I | re.S,
)


def select(source: Path, output: Path, per_tool: int, seed: int) -> dict[str, int]:
    records = json.loads(source.read_text(encoding="utf-8"))
    candidates: dict[str, list[tuple[int, str, dict]]] = {"search": [], "python": []}
    for index, row in enumerate(records):
        trajectory = str(row.get("output", ""))
        instruction = str(row.get("instruction", "")).strip()
        if not instruction or len(instruction) > 1_200 or len(trajectory) > 6_000:
            continue
        pairs = [
            {"tool": m.group(1).lower(), "action": m.group(2).strip(), "result": m.group(3).strip()}
            for m in PAIR_RE.finditer(trajectory)
        ]
        tools = {pair["tool"] for pair in pairs}
        if len(tools) != 1 or not 1 <= len(pairs) <= 3:
            continue
        family = next(iter(tools))
        if not validate_transcript(trajectory, max_tool_calls=3).valid:
            continue
        digest = hashlib.sha256(f"{seed}:{index}:{instruction}".encode()).hexdigest()
        sample = {
            "sample_id": f"toolstar-sanity-{digest[:16]}",
            "source_index": index,
            "instruction": instruction,
            "input": str(row.get("input", "")),
            "reference_output": trajectory,
            "reference_pairs": pairs,
            "tool_family": family,
        }
        candidates[family].append((len(trajectory), digest, sample))

    chosen = []
    for family in ("search", "python"):
        # This is a wire-protocol check rather than an accuracy benchmark.  Prefer
        # the shortest complete official trajectories, breaking ties by a seeded
        # digest, so 100 real rollouts remain affordable and deterministic.
        selected = [sample for _, _, sample in sorted(candidates[family])[:per_tool]]
        if len(selected) != per_tool:
            raise RuntimeError(f"only {len(selected)} eligible {family} samples")
        chosen.extend(selected)
    chosen.sort(key=lambda row: row["sample_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"total": len(chosen), "search": per_tool, "python": per_tool}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-tool", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    print(json.dumps(select(args.source, args.output, args.per_tool, args.seed), indent=2))


if __name__ == "__main__":
    main()
