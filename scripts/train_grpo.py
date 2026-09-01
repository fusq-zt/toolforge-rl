#!/usr/bin/env python3
"""Minimal real-tool GRPO loop for the frozen ToolForge protocol and rewards."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from rollout_agent_batched import finalize, make_states, run_cohort
except ModuleNotFoundError:  # Allows importing this script as scripts.train_grpo in tests.
    from scripts.rollout_agent_batched import finalize, make_states, run_cohort
from toolforge_rl.rewards import RewardInput, efficient_group_rewards, vanilla_reward


RESULT_RE = re.compile(r"<result>.*?</result>\s*", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reward_input(row: dict) -> RewardInput:
    return RewardInput(
        answer_correct=row["answer_correct"], schema_valid=row["schema_valid"],
        final_parsed=row["final_parsed"], tool_call_count=row["tool_call_count"],
        invalid_call_count=row.get("invalid_call_count", 0),
        timeout_count=int(row.get("termination_reason") == "timeout"),
        repeated_call_count=row.get("repeated_call_count", 0),
    )


def encode_policy_tokens(tokenizer, base_prompt: str, transcript: str, maximum: int, device):
    text = base_prompt + transcript
    base_end = len(base_prompt)
    observation_ranges = [(base_end + match.start(), base_end + match.end()) for match in RESULT_RE.finditer(transcript)]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    policy_mask = []
    for start, end in offsets:
        generated = start >= base_end and end > start
        observation = any(start < obs_end and end > obs_start for obs_start, obs_end in observation_ranges)
        policy_mask.append(generated and not observation)
    if len(ids) > maximum:
        ids = ids[-maximum:]
        policy_mask = policy_mask[-maximum:]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.tensor(policy_mask[1:], dtype=torch.bool, device=device)
    if not mask.any():
        raise RuntimeError("trajectory has no policy tokens")
    return input_ids, mask


def selected_log_probs(model, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).reshape_as(targets)[0]
    return log_probs[mask]


def activate_default(model) -> None:
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".default." in name and "lora_" in name


def activate_reference(model) -> None:
    model.set_adapter("reference")
    for parameter in model.parameters():
        parameter.requires_grad = False


def save_checkpoint(model, tokenizer, optimizer, path: Path, processed_groups: int, optimizer_steps: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    activate_default(model)
    model.save_pretrained(path / "adapter", selected_adapters=["default"], safe_serialization=True)
    tokenizer.save_pretrained(path / "adapter")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    (path / "state.json").write_text(json.dumps({
        "processed_groups": processed_groups,
        "optimizer_steps": optimizer_steps,
    }, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("vanilla", "efficient"), required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-step-tokens", type=int, default=384)
    parser.add_argument("--max-sequence-length", type=int, default=2560)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--efficiency-lambda", type=float, default=0.15)
    parser.add_argument("--gradient-accumulation-groups", type=int, default=4)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.num_generations != 4:
        raise ValueError("the frozen experiment uses exactly four generations")
    set_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints",):
        (args.output_dir / name).mkdir(exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, local_files_only=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    initial_adapter = args.resume_from / "adapter" if args.resume_from else args.sft_adapter
    model = PeftModel.from_pretrained(base, initial_adapter, is_trainable=True)
    model.load_adapter(args.sft_adapter, adapter_name="reference", is_trainable=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    activate_default(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0,
    )
    processed_groups = 0
    optimizer_steps = 0
    if args.resume_from:
        state = json.loads((args.resume_from / "state.json").read_text(encoding="utf-8"))
        processed_groups = int(state["processed_groups"])
        optimizer_steps = int(state["optimizer_steps"])
        optimizer.load_state_dict(torch.load(args.resume_from / "optimizer.pt", map_location=model.device))

    prompts = read_jsonl(args.data)
    random.Random(args.seed).shuffle(prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    prompts = prompts[processed_groups:]
    trajectories_path = args.output_dir / "trajectories.jsonl"
    rewards_path = args.output_dir / "reward_components.jsonl"
    metrics_path = args.output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    relative_groups = 0
    total_groups = processed_groups

    for chunk_start in range(0, len(prompts), args.gradient_accumulation_groups):
        prompt_batch = prompts[chunk_start: chunk_start + args.gradient_accumulation_groups]
        activate_default(model)
        model.eval()
        model.config.use_cache = True
        states = make_states(prompt_batch, tokenizer, args.num_generations, args.run_label)
        run_cohort(
            model, tokenizer, states, args.batch_size, args.max_step_tokens,
            True, args.temperature, args.top_p, args.workers,
        )
        all_records = [finalize(state, tokenizer, args.run_label) for state in states]
        group_artifacts = []
        groups_in_batch = len(prompt_batch)
        for group_index, raw in enumerate(prompt_batch):
            start = group_index * args.num_generations
            stop = start + args.num_generations
            group_states = states[start:stop]
            records = all_records[start:stop]
            inputs = [reward_input(row) for row in records]
            if args.branch == "efficient":
                reward_results = efficient_group_rewards(inputs, efficiency_lambda=args.efficiency_lambda)
            else:
                reward_results = [vanilla_reward(item) for item in inputs]
            reward_values = torch.tensor([item.total for item in reward_results], dtype=torch.float32, device=model.device)
            reward_std = reward_values.std(unbiased=False)
            advantages = (reward_values - reward_values.mean()) / (reward_std + 1e-4)
            if reward_std.item() > 1e-8:
                relative_groups += 1

            episode_cache = []
            for state in group_states:
                input_ids, mask = encode_policy_tokens(
                    tokenizer, state.base_prompt, state.transcript,
                    args.max_sequence_length, model.device,
                )
                activate_reference(model)
                model.eval()
                with torch.no_grad():
                    reference_log_probs = selected_log_probs(model, input_ids, mask).detach()
                episode_cache.append((input_ids, mask, reference_log_probs))

            group_losses = []
            group_kls = []
            group_entropy_proxies = []
            for (input_ids, mask, reference_log_probs), advantage in zip(episode_cache, advantages):
                activate_default(model)
                model.train()
                model.config.use_cache = False
                new_log_probs = selected_log_probs(model, input_ids, mask)
                # One policy epoch per freshly collected group: the detached current
                # log-probabilities are the exact old-policy values for this update.
                old_log_probs = new_log_probs.detach()
                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantage
                reference_delta = reference_log_probs - new_log_probs
                kl = torch.exp(reference_delta) - reference_delta - 1
                loss = -(torch.minimum(unclipped, clipped) - args.beta * kl).mean()
                scaled = loss / (args.num_generations * groups_in_batch)
                scaled.backward()
                group_losses.append(float(loss.detach()))
                group_kls.append(float(kl.mean().detach()))
                group_entropy_proxies.append(float((-new_log_probs.mean()).detach()))
            group_artifacts.append((raw, records, reward_results, advantages, reward_values, group_losses, group_kls, group_entropy_proxies))

        activate_default(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0,
        ))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        total_groups = processed_groups + chunk_start + groups_in_batch
        for group_offset, artifact in enumerate(group_artifacts):
            raw, records, reward_results, advantages, reward_values, group_losses, group_kls, group_entropy_proxies = artifact
            for record, reward_result, advantage in zip(records, reward_results, advantages):
                append_jsonl(trajectories_path, record)
                append_jsonl(rewards_path, {
                    "rollout_id": record["rollout_id"], "source_id": record["source_id"],
                    "branch": args.branch, "reward_total": reward_result.total,
                    "reward_answer": reward_result.answer, "reward_format": reward_result.format,
                    "reward_invalid": reward_result.invalid, "reward_parse": reward_result.parse,
                    "reward_efficiency": reward_result.efficiency,
                    "minimum_correct_calls": reward_result.minimum_correct_calls,
                    "advantage": float(advantage),
                })
            group_metric = {
                "processed_groups": processed_groups + chunk_start + group_offset + 1,
                "optimizer_steps": optimizer_steps,
                "source_id": raw["source_id"], "branch": args.branch,
                "mean_reward": float(reward_values.mean()), "reward_variance": float(reward_values.var(unbiased=False)),
                "correct_rollouts": sum(row["answer_correct"] for row in records),
                "mean_tool_calls": sum(row["tool_call_count"] for row in records) / len(records),
                "mean_loss": sum(group_losses) / len(group_losses), "grad_norm": grad_norm,
                "mean_kl": sum(group_kls) / len(group_kls),
                "token_entropy_proxy": sum(group_entropy_proxies) / len(group_entropy_proxies),
            }
            append_jsonl(metrics_path, group_metric)
            print(json.dumps(group_metric), flush=True)
        if args.save_steps and optimizer_steps % args.save_steps == 0:
            save_checkpoint(
                model, tokenizer, optimizer,
                args.output_dir / "checkpoints" / f"step_{optimizer_steps:06d}",
                total_groups, optimizer_steps,
            )

    save_checkpoint(model, tokenizer, optimizer, args.output_dir / "final", total_groups, optimizer_steps)
    metric_rows = read_jsonl(metrics_path)
    summary = {
        "status": "PASS" if metric_rows and all(math.isfinite(row["mean_loss"]) for row in metric_rows) else "FAIL",
        "branch": args.branch,
        "processed_groups": total_groups,
        "rollouts": total_groups * args.num_generations,
        "optimizer_steps": optimizer_steps,
        "relative_reward_groups": sum(row["reward_variance"] > 0 for row in metric_rows),
        "relative_reward_group_ratio": sum(row["reward_variance"] > 0 for row in metric_rows) / len(metric_rows),
        "gradient_nonzero": any(row.get("grad_norm") is not None and row["grad_norm"] > 0 for row in metric_rows),
        "mean_reward": sum(row["mean_reward"] for row in metric_rows) / len(metric_rows),
        "mean_tool_calls": sum(row["mean_tool_calls"] for row in metric_rows) / len(metric_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
