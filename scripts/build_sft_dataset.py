#!/usr/bin/env python3
"""Curate project-verified and pinned-official trajectories for LoRA SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from toolforge_rl.protocols.toolstar import OFFICIAL_SYSTEM_PROMPT, validate_transcript


TARGETS = {
    "direct_anchor": 800,
    "code_reasoning": 1200,
    "retrieval_reasoning": 1200,
    "retrieve_then_compute": 800,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_project(candidates: list[dict], alternative_fraction: float) -> list[dict]:
    groups = defaultdict(list)
    for row in candidates:
        if row.get("quality_pass"):
            groups[row["source_id"]].append(row)
    selected = []
    for source_id, rows in groups.items():
        rows.sort(key=lambda row: (row["tool_call_count"], row["trajectory_tokens"], row["candidate_id"]))
        family = rows[0]["task_family"]
        if family == "retrieve_then_compute" or len(rows) == 1:
            difficulty = "hard"
        elif any(row["tool_call_count"] == 0 for row in rows) and len(rows) >= 2:
            difficulty = "easy"
        else:
            difficulty = "medium"
        best = rows[0]
        best["sft_provenance"] = "project_teacher_real_tools_verified"
        best["difficulty"] = difficulty
        selected.append(best)
        alternate_key = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if alternate_key < alternative_fraction:
            for alternate in rows[1:]:
                if alternate["transcript"] != best["transcript"]:
                    alternate["sft_provenance"] = "project_teacher_real_tools_verified_alternative"
                    alternate["difficulty"] = difficulty
                    selected.append(alternate)
                    break
    return selected


def official_candidates(source: Path, tokenizer) -> list[dict]:
    data = json.loads(source.read_text(encoding="utf-8"))
    eligible = []
    for index, row in enumerate(data):
        transcript = str(row.get("output", ""))
        validation = validate_transcript(transcript, max_tool_calls=3)
        if not validation.valid:
            continue
        length = len(tokenizer.encode(transcript))
        if length > 3072:
            continue
        has_python, has_search = "<python>" in transcript, "<search>" in transcript
        if has_python and has_search:
            continue
        family = "code_reasoning" if has_python else "retrieval_reasoning" if has_search else "direct_anchor"
        instruction = str(row.get("instruction", "")).strip()
        if not instruction:
            continue
        if row.get("input"):
            instruction += "\n" + str(row["input"])
        digest = hashlib.sha256(f"official:{index}:{instruction}".encode()).hexdigest()
        eligible.append((length, digest, {
            "episode_id": f"official-{digest[:20]}",
            "source_dataset": "dongguanting/Tool-Star-SFT-54K",
            "source_id": f"toolstar_sft/train/{index}",
            "task_family": family,
            "difficulty": "official_reference",
            "prompt": instruction,
            "reference_answer": validation.final_answer,
            "final_answer": validation.final_answer,
            "verifier_type": "official_reference_only",
            "messages": [
                {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": transcript},
            ],
            "transcript": transcript,
            "tools": ["python_exec", "local_search"],
            "tool_calls": [],
            "tool_call_count": validation.tool_calls,
            "answer_correct": True,
            "schema_valid": True,
            "execution_success": True,
            "split": "sft",
            "sft_provenance": "pinned_official_reference_not_reexecuted",
            "generator_model": "dongguanting/Tool-Star-SFT-54K",
            "sampling_hint": "official_reference",
            "trajectory_tokens": length,
        }))
    return [row for _, _, row in sorted(eligible)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--output", type=Path, default=Path("data/sft/toolforge_sft.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/sft_manifest.json"))
    parser.add_argument("--official-max", type=int, default=1000)
    parser.add_argument("--alternative-fraction", type=float, default=0.125)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    candidates_by_id = {}
    for candidate_path in args.project_candidates:
        for row in read_jsonl(candidate_path):
            candidates_by_id.setdefault(row["candidate_id"], row)
    project = choose_project(list(candidates_by_id.values()), args.alternative_fraction)
    # Enforce family caps before adding official references.
    project.sort(key=lambda row: (row["task_family"], row["tool_call_count"], row["trajectory_tokens"], row["candidate_id"]))
    selected, counts = [], Counter()
    for row in project:
        family = row["task_family"]
        if counts[family] < TARGETS[family]:
            row["episode_id"] = "project-" + row["candidate_id"]
            row["difficulty"] = row.get("difficulty", "unassigned")
            row["tools"] = ["python_exec", "local_search"]
            row["generator_model"] = row.get("teacher_model", "dongguanting/Tool-Star-Qwen-3B")
            selected.append(row)
            counts[family] += 1
    official = official_candidates(args.official_source, tokenizer)
    official_by_family = defaultdict(list)
    for row in official:
        official_by_family[row["task_family"]].append(row)
    official_cursors = Counter()
    official_used = 0
    while official_used < args.official_max:
        available = [
            family for family in ("direct_anchor", "code_reasoning", "retrieval_reasoning")
            if counts[family] < TARGETS[family]
            and official_cursors[family] < len(official_by_family[family])
        ]
        if not available:
            break
        family = max(
            available,
            key=lambda name: ((TARGETS[name] - counts[name]) / TARGETS[name], name),
        )
        row = official_by_family[family][official_cursors[family]]
        official_cursors[family] += 1
        selected.append(row)
        counts[family] += 1
        official_used += 1
    selected.sort(key=lambda row: row["episode_id"])
    write_jsonl(args.output, selected)
    provenance = Counter(row["sft_provenance"] for row in selected)
    manifest = {
        "schema_version": 1,
        "target_name": "ToolForge-SFT-4K",
        "target_total": 4000,
        "actual_total": len(selected),
        "target_by_family": TARGETS,
        "actual_by_family": dict(counts),
        "provenance": dict(provenance),
        "official_cap": args.official_max,
        "official_used": official_used,
        "alternative_fraction": args.alternative_fraction,
        "candidate_files": [str(path) for path in args.project_candidates],
        "unique_candidate_count": len(candidates_by_id),
        "quality_shortfall_not_padded": len(selected) < 4000,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
