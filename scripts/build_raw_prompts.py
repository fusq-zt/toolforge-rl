#!/usr/bin/env python3
"""Build, deduplicate, and source-isolate the frozen 2,500-prompt pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from toolforge_rl.data import RawPrompt


FAMILY_QUOTAS = {
    "direct_anchor": 500,
    "code_reasoning": 750,
    "retrieval_reasoning": 750,
    "retrieve_then_compute": 500,
}
SPLIT_QUOTAS = {
    "direct_anchor": {"sft": 220, "rl": 160, "dev": 40, "internal_test": 80},
    "code_reasoning": {"sft": 330, "rl": 240, "dev": 60, "internal_test": 120},
    "retrieval_reasoning": {"sft": 330, "rl": 240, "dev": 60, "internal_test": 120},
    "retrieve_then_compute": {"sft": 220, "rl": 160, "dev": 40, "internal_test": 80},
}


def normalized_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower()).split())


def stable_order(rows: list[dict], seed: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['source_id']}".encode()).hexdigest(),
    )


def read_parquet(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def find_many(root: Path, parts: tuple[str, ...]) -> list[Path]:
    candidates = [path for path in root.rglob("*.parquet") if all(p in path.as_posix() for p in parts)]
    if not candidates:
        raise RuntimeError(f"no parquet under {root} matching {parts}")
    return sorted(candidates)


def read_many(root: Path, parts: tuple[str, ...]) -> list[dict]:
    return [row for path in find_many(root, parts) for row in read_parquet(path)]


def math_boxed(solution: str) -> str:
    positions = [match.end() for match in re.finditer(r"\\boxed\s*\{", solution)]
    for start in reversed(positions):
        depth = 1
        for index in range(start, len(solution)):
            if solution[index] == "{":
                depth += 1
            elif solution[index] == "}":
                depth -= 1
                if depth == 0:
                    return solution[start:index].strip()
    return solution.strip()


def context_documents(value) -> list[dict[str, str]]:
    documents = []
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles") or []
        sentences = value.get("sentences") or value.get("sentence") or []
        for index, title in enumerate(titles):
            body = sentences[index] if index < len(sentences) else ""
            if isinstance(body, list):
                body = " ".join(map(str, body))
            documents.append({"document_id": f"doc-{index}", "title": str(title), "text": str(body)})
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            if isinstance(entry, dict):
                title = entry.get("title", f"doc-{index}")
                body = entry.get("sentences", entry.get("text", ""))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                title, body = entry[0], entry[1]
            else:
                title, body = f"doc-{index}", entry
            if isinstance(body, list):
                body = " ".join(map(str, body))
            documents.append({"document_id": f"doc-{index}", "title": str(title), "text": str(body)})
    return documents


def public_rows(raw_root: Path) -> dict[str, list[dict]]:
    gsm_train = read_many(raw_root / "gsm8k", ("main", "train-"))
    math_files = [p for p in (raw_root / "hendrycks_math").rglob("*.parquet") if "train" in p.name]
    math_train = [row for path in sorted(math_files) for row in read_parquet(path)]
    hotpot_train = read_many(raw_root / "hotpot_qa", ("distractor", "train-"))
    wiki_train = read_many(raw_root / "2wiki", ("train-",))

    gsm = []
    for index, row in enumerate(gsm_train):
        answer = str(row["answer"]).rsplit("####", 1)[-1].strip()
        gsm.append({
            "source_dataset": "openai/gsm8k", "source_id": f"gsm8k/train/{index}",
            "prompt": str(row["question"]).strip(), "reference_answer": answer,
            "verifier_type": "gsm8k", "documents": [],
        })
    math = []
    for index, row in enumerate(math_train):
        solution = str(row.get("solution", row.get("answer", "")))
        math.append({
            "source_dataset": "EleutherAI/hendrycks_math", "source_id": f"math/train/{index}",
            "prompt": str(row.get("problem", row.get("question", ""))).strip(),
            "reference_answer": math_boxed(solution), "verifier_type": "math", "documents": [],
        })
    retrieval = []
    for dataset, rows, prefix, verifier in [
        ("hotpotqa/hotpot_qa", hotpot_train, "hotpot/train", "hotpotqa"),
        ("framolfese/2WikiMultihopQA", wiki_train, "2wiki/train", "2wiki"),
    ]:
        for index, row in enumerate(rows):
            retrieval.append({
                "source_dataset": dataset, "source_id": f"{prefix}/{index}",
                "prompt": str(row["question"]).strip(), "reference_answer": str(row["answer"]).strip(),
                "verifier_type": verifier, "documents": context_documents(row.get("context", [])),
            })
    return {"gsm": gsm, "math": math, "retrieval": retrieval}


def synthetic_rtc(count: int, seed: int) -> list[dict]:
    rows = []
    operations = ("sum", "difference", "product", "weighted")
    for index in range(count):
        digest = hashlib.sha256(f"rtc:{seed}:{index}".encode()).digest()
        a = 10 + int.from_bytes(digest[:2], "big") % 491
        b = 5 + int.from_bytes(digest[2:4], "big") % 196
        op = operations[index % len(operations)]
        entity_a, entity_b = f"Neris-{index:04d}", f"Volda-{index:04d}"
        if op == "sum":
            question, answer = f"What is the sum of the indexed values for {entity_a} and {entity_b}?", a + b
        elif op == "difference":
            question, answer = f"How much larger is {entity_a}'s indexed value than {entity_b}'s?", a - b
        elif op == "product":
            question, answer = f"What is the product of the indexed values for {entity_a} and {entity_b}?", a * b
        else:
            question, answer = f"Compute twice {entity_a}'s indexed value plus {entity_b}'s indexed value.", 2 * a + b
        docs = [
            {"document_id": f"rtc-{index}-a", "title": f"Record {entity_a}", "text": f"The indexed value for {entity_a} is {a}."},
            {"document_id": f"rtc-{index}-b", "title": f"Record {entity_b}", "text": f"The indexed value for {entity_b} is {b}."},
            {"document_id": f"rtc-{index}-noise", "title": "Unrelated record", "text": f"The archived marker is {a+b+17}. It is not an indexed value."},
        ]
        rows.append({
            "source_dataset": "toolforge/synthetic_rtc_v1", "source_id": f"rtc/v1/{index}",
            "prompt": question, "reference_answer": str(answer),
            "verifier_type": "retrieve_then_compute", "documents": docs,
            "metadata": {"operation": op, "operands": [a, b]},
        })
    return rows


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = {}
    kept, removed = [], []
    for row in rows:
        key = normalized_text(row["prompt"])
        if not key or key in seen:
            removed.append({"source_id": row["source_id"], "duplicate_of": seen.get(key)})
            continue
        seen[key] = row["source_id"]
        kept.append(row)
    return kept, removed


def allocate(raw_root: Path, seed: int, excluded_source_ids: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    pools = public_rows(raw_root)
    excluded_source_ids = excluded_source_ids or set()
    pools = {
        name: [row for row in rows if row["source_id"] not in excluded_source_ids]
        for name, rows in pools.items()
    }
    gsm, gsm_dupes = deduplicate(stable_order(pools["gsm"], seed))
    math, math_dupes = deduplicate(stable_order(pools["math"], seed))
    retrieval, retrieval_dupes = deduplicate(stable_order(pools["retrieval"], seed))
    direct = gsm[:500]
    code = gsm[500:875] + math[:375]
    retrieval = retrieval[:750]
    rtc = synthetic_rtc(500, seed)
    family_rows = {
        "direct_anchor": direct,
        "code_reasoning": code,
        "retrieval_reasoning": retrieval,
        "retrieve_then_compute": rtc,
    }
    output = []
    for family, rows in family_rows.items():
        if len(rows) != FAMILY_QUOTAS[family]:
            raise RuntimeError(f"{family}: expected {FAMILY_QUOTAS[family]}, got {len(rows)}")
        cursor = 0
        for split, quota in SPLIT_QUOTAS[family].items():
            for row in rows[cursor: cursor + quota]:
                row = dict(row)
                row["task_family"] = family
                row["split"] = split
                output.append(RawPrompt(**row).to_dict())
            cursor += quota
    removed = gsm_dupes + math_dupes + retrieval_dupes
    return sorted(output, key=lambda x: (x["split"], x["task_family"], x["source_id"])), removed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_prompts"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--exclude-source-id", action="append", default=[])
    args = parser.parse_args()
    excluded = set(args.exclude_source_id)
    rows, duplicates = allocate(args.raw_root, args.seed, excluded)
    if len(rows) != 2500:
        raise RuntimeError(f"raw prompt target violated: {len(rows)}")
    write_jsonl(args.output_dir / "toolforge_raw_2500.jsonl", rows)
    for split in ("sft", "rl", "dev", "internal_test"):
        write_jsonl(args.output_dir / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "total": len(rows),
        "by_family": Counter(row["task_family"] for row in rows),
        "by_split": Counter(row["split"] for row in rows),
        "by_split_family": {
            split: Counter(row["task_family"] for row in rows if row["split"] == split)
            for split in ("sft", "rl", "dev", "internal_test")
        },
        "normalized_exact_duplicates_removed_before_generation": len(duplicates),
        "high_similarity_source_ids_removed_before_generation": sorted(excluded),
        "source_id_overlap": 0,
    }
    Path("data/manifests/raw_prompt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    Path("data/manifests/pre_generation_duplicates.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
