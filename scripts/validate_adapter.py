#!/usr/bin/env python3
"""Reload a saved LoRA adapter and perform a finite deterministic generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from toolforge_rl.protocols.toolstar import OFFICIAL_SYSTEM_PROMPT, extract_final_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
            {"role": "user", "content": "What is 2 + 3? Reply using the required final-answer protocol."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded, do_sample=False, max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    final_answer = extract_final_answer(completion)
    result = {
        "status": "PASS" if completion.strip() and final_answer is not None else "FAIL",
        "adapter": str(args.adapter),
        "completion": completion,
        "final_answer": final_answer,
        "loaded_trainable_parameters": trainable,
        "finite_parameters": all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
    }
    if not result["finite_parameters"]:
        result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
