#!/usr/bin/env python3
"""LoRA SFT with assistant-only loss over the shared Tool-Star rendering."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


class EpisodeDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, maximum: int | None):
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if maximum:
            rows = rows[:maximum]
        self.examples = []
        truncated = 0
        for row in rows:
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = tokenizer.encode(full, add_special_tokens=False)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                truncated += 1
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]
            if not any(label != -100 for label in labels):
                continue
            self.examples.append({"input_ids": input_ids, "labels": labels})
        self.stats = {"rows": len(rows), "usable": len(self.examples), "truncated": truncated}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, examples):
        width = max(len(example["input_ids"]) for example in examples)
        input_ids, labels, attention = [], [], []
        for example in examples:
            gap = width - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad] * gap)
            labels.append(example["labels"] + [-100] * gap)
            attention.append([1] * len(example["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-3B-Instruct"))
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional existing LoRA adapter to continue training in a bounded recovery run.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()
    dataset = EpisodeDataset(args.data, tokenizer, args.max_length, args.max_samples)
    if not dataset:
        raise RuntimeError("no usable SFT examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        (args.output_dir / "dataset_stats.json").write_text(json.dumps(dataset.stats, indent=2) + "\n")
        resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        (args.output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir / "adapter")
        history = trainer.state.log_history
        grad_norms = [float(row["grad_norm"]) for row in history if row.get("grad_norm") is not None]
        losses = [float(row["loss"]) for row in history if row.get("loss") is not None]
        metrics = {
            **result.metrics,
            **dataset.stats,
            "gradient_nonzero": any(math.isfinite(value) and value > 0 for value in grad_norms),
            "loss_finite": bool(losses) and all(math.isfinite(value) for value in losses),
            "adapter_saved": (args.output_dir / "adapter" / "adapter_model.safetensors").exists(),
            "log_history": history,
        }
        (args.output_dir / "train_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
