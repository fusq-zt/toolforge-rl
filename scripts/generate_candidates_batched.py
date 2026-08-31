#!/usr/bin/env python3
"""Length-bucketed multi-round Teacher agent generation with resumable cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from toolforge_rl.protocols.toolstar import (
    OFFICIAL_SYSTEM_PROMPT,
    ActionKind,
    extract_final_answer,
    parse_next_action,
    render_result,
    trim_at_first_stop,
    validate_transcript,
)
from toolforge_rl.tools import LocalDocument, LocalSearch, execute_python
from toolforge_rl.verifiers import verify_answer


STOPS = ("</search>", "</python>", "</answer>")
HINTS = {
    "no_hint": "",
    "tool_hint": "Tools are available when they help improve correctness.",
    "minimal_call_hint": "Use tools only when necessary and avoid redundant calls.",
}
LOCAL_PROTOCOL_NOTE = (
    " In this experiment, <search> queries only the documents attached to this "
    "question; it is the local_search tool, not internet search."
)


@dataclass
class State:
    raw: dict
    hint: str
    candidate_id: str
    base_prompt: str
    transcript: str = ""
    events: list[dict] = field(default_factory=list)
    generated_tokens: int = 0
    termination_reason: str = "turn_limit"
    started: float = field(default_factory=time.monotonic)
    active: bool = True


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_states(rows: list[dict], tokenizer) -> list[State]:
    states = []
    for row in rows:
        for hint_name, hint_text in HINTS.items():
            candidate_id = hashlib.sha256(f"{row['source_id']}:{hint_name}".encode()).hexdigest()[:24]
            user = row["prompt"] + (f"\n\n{hint_text}" if hint_text else "")
            base = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT + LOCAL_PROTOCOL_NOTE},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            states.append(State(row, hint_name, candidate_id, base))
    return states


def execute_state_action(state: State, action) -> tuple[str, bool, str | None, float]:
    started = time.monotonic()
    if action.kind == ActionKind.SEARCH:
        docs = [LocalDocument(**doc) for doc in state.raw.get("documents", [])]
        response = LocalSearch(docs).search(action.content)
        return response, not response.startswith("SEARCH_ERROR"), None, time.monotonic() - started
    result = execute_python(action.content)
    return result.output, result.ok, result.error_type, time.monotonic() - started


def generate_minibatch(model, tokenizer, states: list[State], max_step_tokens: int, temperature: float, top_p: float):
    texts = [state.base_prompt + state.transcript for state in states]
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_length = encoded.input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_step_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            stop_strings=list(STOPS),
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    for state, sequence in zip(states, generated):
        new_ids = sequence[input_length:]
        state.generated_tokens += int((new_ids != tokenizer.pad_token_id).sum().item())
        delta = trim_at_first_stop(tokenizer.decode(new_ids, skip_special_tokens=True), STOPS)
        state.transcript += delta
        action = parse_next_action(delta)
        yield state, action


def run_cohort(model, tokenizer, states: list[State], batch_size: int, max_step_tokens: int, temperature: float, top_p: float, workers: int):
    for _turn in range(4):
        active = [state for state in states if state.active]
        if not active:
            break
        active.sort(key=lambda state: len(tokenizer.encode(state.base_prompt + state.transcript)))
        pending_tools = []
        for start in range(0, len(active), batch_size):
            batch = active[start: start + batch_size]
            for state, action in generate_minibatch(model, tokenizer, batch, max_step_tokens, temperature, top_p):
                if action.kind == ActionKind.FINAL:
                    state.termination_reason, state.active = "final", False
                elif action.kind in {ActionKind.SEARCH, ActionKind.PYTHON}:
                    if len(state.events) >= 3:
                        state.termination_reason, state.active = "tool_budget_exceeded", False
                    elif any(e["tool"] == action.kind.value and e["arguments"] == action.content for e in state.events):
                        state.termination_reason, state.active = "repeated_call", False
                    else:
                        pending_tools.append((state, action))
                else:
                    state.termination_reason, state.active = action.kind.value, False
        if pending_tools:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(execute_state_action, state, action) for state, action in pending_tools]
                for (state, action), future in zip(pending_tools, futures):
                    response, ok, error_type, latency = future.result()
                    state.transcript += render_result(response)
                    state.events.append({
                        "tool": action.kind.value, "arguments": action.content, "response": response,
                        "success": ok, "error_type": error_type, "latency_seconds": latency,
                    })
    return states


def finalize(state: State, tokenizer, model_name: str, model_revision: str) -> dict:
    final = extract_final_answer(state.transcript)
    verification = verify_answer(
        state.raw["verifier_type"], final or "", state.raw["reference_answer"],
        aliases=state.raw.get("aliases", []),
    )
    validation = validate_transcript(state.transcript, max_tool_calls=3)
    all_tools_ok = all(event["success"] for event in state.events)
    token_length = len(tokenizer.encode(state.transcript))
    schema_valid = validation.valid and state.termination_reason == "final"
    quality_pass = (
        verification.correct and schema_valid and all_tools_ok and len(state.events) <= 3
        and state.termination_reason == "final" and token_length <= 4096
    )
    return {
        "candidate_id": state.candidate_id,
        **state.raw,
        "messages": [
            {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT + LOCAL_PROTOCOL_NOTE},
            {"role": "user", "content": state.raw["prompt"]},
            {"role": "assistant", "content": state.transcript},
        ],
        "transcript": state.transcript,
        "tool_calls": state.events,
        "tool_arguments": [event["arguments"] for event in state.events],
        "tool_responses": [event["response"] for event in state.events],
        "final_answer": final,
        "normalized_prediction": verification.normalized_prediction,
        "normalized_reference": verification.normalized_reference,
        "verifier_result": verification.correct,
        "verifier_reason": verification.reason,
        "schema_valid": schema_valid,
        "execution_success": all_tools_ok,
        "answer_correct": verification.correct,
        "tool_call_count": len(state.events),
        "repeated_call_count": int(state.termination_reason == "repeated_call"),
        "generated_tokens": state.generated_tokens,
        "trajectory_tokens": token_length,
        "latency": time.monotonic() - state.started,
        "termination_reason": state.termination_reason,
        "sampling_hint": state.hint,
        "teacher_model": model_name,
        "teacher_revision": model_revision,
        "quality_pass": quality_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Tool-Star-Qwen-3B"))
    parser.add_argument("--model-revision", default="2350a1d6ec6230d48e41f752babab730bad8aa92")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cohort-size", type=int, default=64)
    parser.add_argument("--max-step-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume-from", nargs="*", type=Path, default=[],
        help="Additional immutable candidate shards whose IDs must not be regenerated.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    ).eval()
    rows = jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    rows = rows[args.shard_index :: args.num_shards]
    states = make_states(rows, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for completed_path in [args.output, *args.resume_from]:
        if completed_path.exists():
            done.update(row["candidate_id"] for row in jsonl(completed_path))
    pending = [state for state in states if state.candidate_id not in done]
    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.cohort_size):
            cohort = pending[offset: offset + args.cohort_size]
            run_cohort(
                model, tokenizer, cohort, args.batch_size, args.max_step_tokens,
                args.temperature, args.top_p, args.workers,
            )
            records = [finalize(state, tokenizer, args.model.name, args.model_revision) for state in cohort]
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            passed = sum(record["quality_pass"] for record in records)
            print(f"cohort {offset // args.cohort_size + 1}: {len(records)} candidates, quality_pass={passed}", flush=True)


if __name__ == "__main__":
    main()
