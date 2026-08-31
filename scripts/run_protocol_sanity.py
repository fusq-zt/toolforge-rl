#!/usr/bin/env python3
"""Run the frozen Tool-Star text protocol against 100 official SFT prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from toolforge_rl.protocols.toolstar import (
    OFFICIAL_SYSTEM_PROMPT,
    ActionKind,
    parse_next_action,
    render_result,
    trim_at_first_stop,
    validate_transcript,
)
from toolforge_rl.tools import LocalDocument, LocalSearch, execute_python


STOP_STRINGS = ["</search>", "</python>", "</answer>"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_one(model, tokenizer, sample: dict, max_step_tokens: int) -> dict:
    question = sample["instruction"]
    if sample.get("input"):
        question += "\n" + sample["input"]
    base = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    documents = [
        LocalDocument(f"reference-{i}", pair["result"], pair["action"])
        for i, pair in enumerate(sample["reference_pairs"])
        if pair["tool"] == "search"
    ]
    search = LocalSearch(documents)
    transcript = ""
    events = []
    total_new_tokens = 0
    started = time.monotonic()
    stop_reason = "turn_limit"
    for turn in range(4):
        encoded = tokenizer(base + transcript, return_tensors="pt").to(model.device)
        input_length = encoded.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_step_tokens,
                do_sample=False,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_ids = generated[0, input_length:]
        total_new_tokens += int(new_ids.numel())
        delta = trim_at_first_stop(tokenizer.decode(new_ids, skip_special_tokens=True), STOP_STRINGS)
        transcript += delta
        action = parse_next_action(delta)
        event = {"turn": turn, "kind": action.kind.value, "content": action.content[:1_000], "error": action.error}
        if action.kind == ActionKind.SEARCH:
            observation = search.search(action.content)
            transcript += render_result(observation)
            event["tool_ok"] = not observation.startswith("SEARCH_ERROR")
        elif action.kind == ActionKind.PYTHON:
            result = execute_python(action.content)
            transcript += render_result(result.output)
            event["tool_ok"] = result.ok
            event["tool_error_type"] = result.error_type
        elif action.kind == ActionKind.FINAL:
            stop_reason = "final"
            events.append(event)
            break
        else:
            stop_reason = action.kind.value
            events.append(event)
            break
        events.append(event)
    validation = validate_transcript(transcript, max_tool_calls=3)
    tool_events = [
        event for event in events if event["kind"] in {ActionKind.SEARCH.value, ActionKind.PYTHON.value}
    ]
    return {
        "sample_id": sample["sample_id"],
        "tool_family": sample["tool_family"],
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "new_tokens": total_new_tokens,
        "events": events,
        "transcript": transcript,
        "protocol_valid": validation.valid,
        "protocol_errors": validation.errors,
        "tool_calls": validation.tool_calls,
        "final_answer": validation.final_answer,
        "all_tool_calls_ok": bool(tool_events) and all(e.get("tool_ok", False) for e in tool_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--only-failed-from", type=Path,
        help="Run only sample IDs whose prior record lacked a valid final; avoids repeating successes.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    samples = read_jsonl(args.samples)
    if args.only_failed_from:
        prior = read_jsonl(args.only_failed_from)
        failed_ids = {
            row["sample_id"] for row in prior
            if row.get("stop_reason") != "final" or not row.get("protocol_valid", False)
        }
        samples = [sample for sample in samples if sample["sample_id"] in failed_ids]
    samples = samples[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            result = run_one(model, tokenizer, sample, args.max_step_tokens)
            result["model"] = args.model.name
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(samples)}] {sample['sample_id']} "
                f"stop={result['stop_reason']} valid={result['protocol_valid']} "
                f"tools_ok={result['all_tool_calls_ok']} sec={result['elapsed_seconds']}"
            )


if __name__ == "__main__":
    main()
