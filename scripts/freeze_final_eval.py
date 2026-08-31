#!/usr/bin/env python3
"""Freeze public held-out IDs and run lightweight pre-generation contamination checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "hotpotqa/hotpot_qa": "1908d6afbbead072334abe2965f91bd2709910ab",
    "framolfese/2WikiMultihopQA": "fe713bfbd1afbca1a65246741a75890405d56a3a",
}


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower()).split())


def tokens(text: str) -> set[str]:
    return set(norm(text).split())


def ordered(rows: list[dict], seed: int, id_key: str) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row[id_key]}".encode()).hexdigest())


def read_many(root: Path, includes: tuple[str, ...]) -> list[dict]:
    paths = [p for p in root.rglob("*.parquet") if all(part in p.as_posix() for part in includes)]
    if not paths:
        raise RuntimeError(f"no parquet matching {includes} under {root}")
    return [row for path in sorted(paths) for row in pq.read_table(path).to_pylist()]


def documents(value) -> list[dict]:
    output = []
    if isinstance(value, dict):
        titles, sentences = value.get("title", []), value.get("sentences", [])
        value = list(zip(titles, sentences))
    for index, entry in enumerate(value or []):
        if isinstance(entry, dict):
            title, body = entry.get("title", f"doc-{index}"), entry.get("sentences", entry.get("text", ""))
        else:
            title, body = entry[0], entry[1]
        if isinstance(body, list):
            body = " ".join(map(str, body))
        output.append({"document_id": f"doc-{index}", "title": str(title), "text": str(body)})
    return output


def build(raw: Path, seed: int) -> list[dict]:
    heldout = []
    gsm = read_many(raw / "gsm8k", ("main", "test-"))
    for i, row in enumerate(gsm):
        heldout.append({
            "source_dataset": "openai/gsm8k", "source_id": f"gsm8k/test/{i}",
            "task_family": "code_reasoning", "prompt": row["question"],
            "reference_answer": str(row["answer"]).rsplit("####", 1)[-1].strip(),
            "verifier_type": "gsm8k", "documents": [], "split": "public_test",
        })
    with (raw / "math500" / "test.jsonl").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            heldout.append({
                "source_dataset": "HuggingFaceH4/MATH-500", "source_id": f"math500/test/{i}",
                "task_family": "code_reasoning", "prompt": row["problem"],
                "reference_answer": row["answer"], "verifier_type": "math",
                "documents": [], "split": "public_test",
            })
    hotpot = read_many(raw / "hotpot_qa", ("distractor", "validation-"))
    hotpot = ordered(
        [{**row, "_id_stable": str(row.get("id", row.get("_id", i)))} for i, row in enumerate(hotpot)],
        seed, "_id_stable",
    )[:200]
    for row in hotpot:
        heldout.append({
            "source_dataset": "hotpotqa/hotpot_qa", "source_id": f"hotpot/validation/{row['_id_stable']}",
            "task_family": "retrieval_reasoning", "prompt": row["question"], "reference_answer": row["answer"],
            "verifier_type": "hotpotqa", "documents": documents(row.get("context")), "split": "public_test",
        })
    wiki_rows = []
    for split in ("validation", "test"):
        rows = read_many(raw / "2wiki", (f"{split}-",))
        rows = ordered(
            [{**row, "_id_stable": str(row.get("id", row.get("_id", i)))} for i, row in enumerate(rows)],
            seed, "_id_stable",
        )[:100]
        wiki_rows.extend((split, row) for row in rows)
    for split, row in wiki_rows:
        heldout.append({
            "source_dataset": "framolfese/2WikiMultihopQA", "source_id": f"2wiki/{split}/{row['_id_stable']}",
            "task_family": "retrieval_reasoning", "prompt": row["question"], "reference_answer": row["answer"],
            "verifier_type": "2wiki", "documents": documents(row.get("context")), "split": "public_test",
        })
    return heldout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-prompts", type=Path, default=Path("data/raw_prompts/toolforge_raw_2500.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/public_heldout.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/final_eval_manifest.json"))
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    heldout = build(args.raw_root, args.seed)
    train = [json.loads(line) for line in args.train_prompts.open(encoding="utf-8")]
    train_exact = {norm(row["prompt"]): row["source_id"] for row in train}
    exact = [
        {"train_source_id": train_exact[norm(row["prompt"])], "eval_source_id": row["source_id"]}
        for row in heldout if norm(row["prompt"]) in train_exact
    ]
    high_overlap = []
    eval_token_rows = [(row["source_id"], tokens(row["prompt"])) for row in heldout]
    for row in train:
        a = tokens(row["prompt"])
        if len(a) < 5:
            continue
        for eval_id, b in eval_token_rows:
            if len(b) < 5:
                continue
            score = len(a & b) / len(a | b)
            if score >= 0.90:
                high_overlap.append({"train_source_id": row["source_id"], "eval_source_id": eval_id, "jaccard": round(score, 4)})
    if exact:
        raise RuntimeError(f"normalized exact train/eval contamination: {exact[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in heldout:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ids = [row["source_id"] for row in heldout]
    payload = {
        "schema_version": 1, "seed": args.seed, "frozen_before_training": True,
        "revisions": REVISIONS, "total": len(heldout),
        "counts": {
            "gsm8k_test": sum(row["source_dataset"] == "openai/gsm8k" for row in heldout),
            "math500": sum(row["source_dataset"] == "HuggingFaceH4/MATH-500" for row in heldout),
            "hotpot_validation": sum(row["source_dataset"] == "hotpotqa/hotpot_qa" for row in heldout),
            "2wiki_validation_test": sum(row["source_dataset"] == "framolfese/2WikiMultihopQA" for row in heldout),
        },
        "source_ids": ids,
        "source_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "normalized_exact_overlap_with_raw_2500": 0,
        "jaccard_ge_0_90_pairs": high_overlap,
    }
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("total", "counts", "normalized_exact_overlap_with_raw_2500")}, indent=2))
    print(f"jaccard_ge_0.90={len(high_overlap)}")


if __name__ == "__main__":
    main()
