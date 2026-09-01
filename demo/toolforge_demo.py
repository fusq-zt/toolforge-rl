#!/usr/bin/env python3
"""Run one local ToolForge episode and print the real tool trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rollout_agent_batched import finalize, make_states, run_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--documents", type=Path, help="JSON list of {doc_id,title,text}")
    parser.add_argument("--reference", default="")
    parser.add_argument("--verifier", default="qa")
    parser.add_argument("--output", type=Path, help="Optionally persist the demo trace as JSON.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--max-step-tokens",
        type=int,
        default=1024,
        help="Per-turn generation budget; 1024 avoids the observed 512-token long-reasoning truncation.",
    )
    args = parser.parse_args()
    documents = json.loads(args.documents.read_text(encoding="utf-8")) if args.documents else []
    tokenizer_path = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    raw = {
        "source_id": "demo/0", "source_dataset": "user_demo", "task_family": "demo",
        "prompt": args.prompt, "documents": documents,
        "reference_answer": args.reference, "verifier_type": args.verifier,
    }
    states = make_states([raw], tokenizer, 1, "demo")
    run_cohort(
        model, tokenizer, states, 1, args.max_step_tokens,
        True, args.temperature, 0.95, 4,
    )
    result = finalize(states[0], tokenizer, "demo")
    payload = {
        "final_answer": result["final_answer"],
        "answer_correct_when_reference_given": result["answer_correct"] if args.reference else None,
        "tool_calls": result["tool_calls"],
        "termination_reason": result["termination_reason"],
        "transcript": result["transcript"],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
