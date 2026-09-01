#!/usr/bin/env python3
"""Report targeted long-generation recovery without hiding initial failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for spec in args.run:
        label, raw_path = spec.split("=", 1)
        run_dir = Path(raw_path)
        config = json.loads((run_dir / "eval_manifest.json").read_text(encoding="utf-8"))
        final_rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").open(encoding="utf-8") if line.strip()]
        remaining = sum(row.get("termination_reason") == "incomplete" for row in final_rows)
        rows.append({
            "label": label,
            "initial_rows": config["initial_rows"],
            "recovery_rows": config["recovery_rows"],
            "final_rows": config["final_rows"],
            "remaining_incomplete": remaining,
        })
    result = {
        "schema_version": 1,
        "policy": "Evaluate all 2619 rows at 1024 tokens, retry only termination_reason=incomplete at 2048, preserve initial and recovery files.",
        "runs": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Final Evaluation Long-generation Recovery", "", result["policy"], "",
        "| model | initial | retried at 2048 | remaining incomplete | final |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['initial_rows']} | {row['recovery_rows']} | "
            f"{row['remaining_incomplete']} | {row['final_rows']} |"
        )
    lines.extend(["", "Initial 1024-token predictions are retained beside the merged final predictions; only genuinely incomplete episodes were regenerated."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
