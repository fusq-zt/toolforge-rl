#!/usr/bin/env python3
"""Reusable batched real-tool rollouts for gates, mining, probes, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from toolforge_rl.protocols.toolstar import (
    OFFICIAL_SYSTEM_PROMPT, ActionKind, extract_final_answer, parse_next_action,
    render_result, trim_at_first_stop, validate_transcript,
)
from toolforge_rl.tools import LocalDocument, LocalSearch, execute_python
from toolforge_rl.verifiers import verify_answer


STOPS = ("</search>", "</python>", "</answer>")
SYSTEM = OFFICIAL_SYSTEM_PROMPT + (
    " In this experiment, <search> queries only the documents attached to this "
    "question; it is the local_search tool, not internet search."
)
EFFICIENT_SYSTEM_NOTE = (
    " Prefer a direct answer when it is already reliable. Use the fewest tool calls "
    "needed for correctness, but never trade away answer correctness to avoid a tool."
)


@dataclass
class State:
    raw: dict
    rollout_index: int
    rollout_id: str
    base_prompt: str
    transcript: str = ""
    events: list[dict] = field(default_factory=list)
    generated_tokens: int = 0
    termination_reason: str = "turn_limit"
    active: bool = True
    started: float = field(default_factory=time.monotonic)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_states(
    rows: list[dict], tokenizer, rollouts: int, run_label: str,
    system_prompt: str = SYSTEM,
) -> list[State]:
    output = []
    for raw in rows:
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": raw["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for index in range(rollouts):
            digest = hashlib.sha256(f"{run_label}:{raw['source_id']}:{index}".encode()).hexdigest()[:24]
            output.append(State(raw, index, digest, base))
    return output


def execute_action(state: State, action):
    started = time.monotonic()
    if action.kind == ActionKind.SEARCH:
        response = LocalSearch([LocalDocument(**doc) for doc in state.raw.get("documents", [])]).search(action.content)
        return response, not response.startswith("SEARCH_ERROR"), None, time.monotonic() - started
    result = execute_python(action.content)
    return result.output, result.ok, result.error_type, time.monotonic() - started


def generate_batch(model, tokenizer, states, max_tokens, do_sample, temperature, top_p):
    texts = [state.base_prompt + state.transcript for state in states]
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    width = encoded.input_ids.shape[1]
    kwargs = {}
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    with torch.inference_mode():
        generated = model.generate(
            **encoded, **kwargs, max_new_tokens=max_tokens, stop_strings=list(STOPS),
            tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
    for state, sequence in zip(states, generated):
        new_ids = sequence[width:]
        state.generated_tokens += int((new_ids != tokenizer.pad_token_id).sum().item())
        delta = trim_at_first_stop(tokenizer.decode(new_ids, skip_special_tokens=True), STOPS)
        state.transcript += delta
        yield state, parse_next_action(delta)


def run_cohort(model, tokenizer, states, batch_size, max_tokens, do_sample, temperature, top_p, workers):
    for _ in range(4):
        active = [state for state in states if state.active]
        if not active:
            break
        active.sort(key=lambda state: len(tokenizer.encode(state.base_prompt + state.transcript)))
        tool_jobs = []
        for offset in range(0, len(active), batch_size):
            for state, action in generate_batch(
                model, tokenizer, active[offset: offset + batch_size], max_tokens,
                do_sample, temperature, top_p,
            ):
                if action.kind == ActionKind.FINAL:
                    state.termination_reason, state.active = "final", False
                elif action.kind in {ActionKind.SEARCH, ActionKind.PYTHON}:
                    if len(state.events) >= 3:
                        state.termination_reason, state.active = "tool_budget_exceeded", False
                    elif any(e["tool"] == action.kind.value and e["arguments"] == action.content for e in state.events):
                        state.termination_reason, state.active = "repeated_call", False
                    else:
                        tool_jobs.append((state, action))
                else:
                    state.termination_reason, state.active = action.kind.value, False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute_action, state, action) for state, action in tool_jobs]
            for (state, action), future in zip(tool_jobs, futures):
                response, success, error_type, latency = future.result()
                state.transcript += render_result(response)
                state.events.append({
                    "tool": action.kind.value, "arguments": action.content, "response": response,
                    "success": success, "error_type": error_type, "latency_seconds": latency,
                })


def finalize(state: State, tokenizer, run_label: str) -> dict:
    answer = extract_final_answer(state.transcript)
    verification = verify_answer(
        state.raw["verifier_type"], answer or "", state.raw["reference_answer"],
        aliases=state.raw.get("aliases", []),
    )
    validation = validate_transcript(state.transcript, max_tool_calls=3)
    tools_ok = all(event["success"] for event in state.events)
    return {
        "rollout_id": state.rollout_id, "run_label": run_label,
        "rollout_index": state.rollout_index,
        "source_id": state.raw["source_id"], "source_dataset": state.raw["source_dataset"],
        "task_family": state.raw["task_family"], "prompt": state.raw["prompt"],
        "evaluation_suite": state.raw.get("evaluation_suite", "unspecified"),
        "difficulty": state.raw.get("difficulty", state.raw.get("metadata", {}).get("difficulty", "unassigned")),
        "reference_answer": state.raw["reference_answer"], "verifier_type": state.raw["verifier_type"],
        "final_answer": answer, "normalized_prediction": verification.normalized_prediction,
        "answer_correct": verification.correct, "verifier_reason": verification.reason,
        "schema_valid": validation.valid and state.termination_reason == "final",
        "execution_success": tools_ok, "tool_calls": state.events,
        "tool_call_count": len(state.events),
        "repeated_call_count": int(state.termination_reason == "repeated_call"),
        "invalid_call_count": int(state.termination_reason in {"invalid", "incomplete"}),
        "final_parsed": answer is not None,
        "generated_tokens": state.generated_tokens,
        "trajectory_tokens": len(tokenizer.encode(state.transcript)),
        "latency_seconds": time.monotonic() - state.started,
        "termination_reason": state.termination_reason,
        "transcript": state.transcript,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--rollouts-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cohort-size", type=int, default=64)
    parser.add_argument("--max-step-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--efficient-tool-prompt", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer_path = args.adapter if args.adapter else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    raw = read_jsonl(args.input)
    if args.limit:
        raw = raw[: args.limit]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    raw = raw[args.shard_index :: args.num_shards]
    system_prompt = SYSTEM + EFFICIENT_SYSTEM_NOTE if args.efficient_tool_prompt else SYSTEM
    states = make_states(raw, tokenizer, args.rollouts_per_prompt, args.run_label, system_prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        done = {row["rollout_id"] for row in read_jsonl(args.output)}
    pending = [state for state in states if state.rollout_id not in done]
    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.cohort_size):
            cohort = pending[offset: offset + args.cohort_size]
            run_cohort(
                model, tokenizer, cohort, args.batch_size, args.max_step_tokens,
                not args.greedy, args.temperature, args.top_p, args.workers,
            )
            records = [finalize(state, tokenizer, args.run_label) for state in cohort]
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"cohort {offset // args.cohort_size + 1}: n={len(records)} "
                f"correct={sum(r['answer_correct'] for r in records)} "
                f"valid={sum(r['schema_valid'] for r in records)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
