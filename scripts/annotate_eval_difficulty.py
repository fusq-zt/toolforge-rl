#!/usr/bin/env python3
"""Freeze observable difficulty labels from the Base-policy evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label(row: dict) -> str:
    if row["task_family"] == "retrieve_then_compute":
        return "hard"
    if row["answer_correct"] and row["tool_call_count"] == 0:
        return "easy"
    if row["answer_correct"]:
        return "medium"
    return "hard"


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_prediction_pair(value: str) -> tuple[Path, Path]:
    try:
        source, output = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("prediction mapping must be INPUT=OUTPUT") from error
    return Path(source), Path(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_pair,
        default=[],
        metavar="INPUT=OUTPUT",
        help="Annotate an additional completed prediction file with the frozen Base labels.",
    )
    args = parser.parse_args()
    base = {}
    for path in args.base_predictions:
        for row in read(path):
            base[row["source_id"]] = row
    inputs = read(args.input)
    if {row["source_id"] for row in inputs} != set(base):
        raise RuntimeError("Base predictions do not match frozen evaluation input")
    labels = {source_id: label(row) for source_id, row in base.items()}
    annotated_inputs = []
    for row in inputs:
        item = dict(row)
        item["difficulty"] = labels[row["source_id"]]
        annotated_inputs.append(item)
    annotated_base = []
    for source_id in sorted(base):
        item = dict(base[source_id])
        item["difficulty"] = labels[source_id]
        annotated_base.append(item)
    write(args.output, annotated_inputs)
    write(args.base_output, annotated_base)
    annotated_prediction_files = []
    for prediction_input, prediction_output in args.prediction:
        prediction_rows = read(prediction_input)
        prediction_ids = {row["source_id"] for row in prediction_rows}
        if prediction_ids != set(labels):
            raise RuntimeError(
                f"Predictions in {prediction_input} do not match frozen evaluation input"
            )
        annotated = []
        for row in prediction_rows:
            item = dict(row)
            item["difficulty"] = labels[row["source_id"]]
            annotated.append(item)
        write(prediction_output, annotated)
        annotated_prediction_files.append(str(prediction_output))
    manifest = {
        "schema_version": 1,
        "total": len(labels),
        "by_difficulty": dict(Counter(labels.values())),
        "definition": {
            "easy": "Base greedy correct with zero calls",
            "medium": "Base greedy correct with one or more calls",
            "hard": "Base greedy wrong, or retrieve_then_compute",
        },
        "used_for_tuning": False,
        "annotated_prediction_files": annotated_prediction_files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
