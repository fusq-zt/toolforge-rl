#!/usr/bin/env python3
"""Check that incremental observation insertion preserves prior token prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from toolforge_rl.protocols.toolstar import OFFICIAL_SYSTEM_PROMPT, render_result


CONTINUATIONS = [
    "<think>reason</think><search>alpha beta</search>" + render_result("evidence"),
    "<think>calculate</think><python>print(2 + 2)</python>" + render_result("4"),
    "<answer>\\boxed{4}</answer>",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/chat_prefix_check.json"))
    args = parser.parse_args()
    rows = []
    for model in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
        base = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
                {"role": "user", "content": "What is two plus two?"},
            ], tokenize=False, add_generation_prompt=True,
        )
        current = base
        passed = True
        checks = []
        for continuation in CONTINUATIONS:
            before = tokenizer.encode(current, add_special_tokens=False)
            after = tokenizer.encode(current + continuation, add_special_tokens=False)
            same = before == after[: len(before)]
            checks.append({"before_tokens": len(before), "after_tokens": len(after), "prefix_equal": same})
            passed &= same
            current += continuation
        rows.append({"model": model.as_posix(), "passed": passed, "checks": checks})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checks": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
