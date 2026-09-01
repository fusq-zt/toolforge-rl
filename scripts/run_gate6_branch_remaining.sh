#!/usr/bin/env bash
# Finish the preregistered seed replications for one GRPO branch on one GPU.
# This deliberately reuses the frozen RL/Dev data and the existing SFT adapter.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {vanilla|efficient} GPU_INDEX WAIT_PID" >&2
  exit 2
fi

branch="$1"
gpu_index="$2"
wait_pid="$3"
if [[ "$branch" != "vanilla" && "$branch" != "efficient" ]]; then
  echo "invalid branch: $branch" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu_index"
export TOKENIZERS_PARALLELISM=false

while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 10
done

materialize() {
  local seed="$1"
  .venv/bin/python scripts/materialize_run_log.py \
    --run-dir "runs/grpo_formal_${branch}_seed${seed}"
}

train_seed_2026() {
  local run_dir="runs/grpo_formal_${branch}_seed2026"
  if [[ -f "$run_dir/final/adapter/adapter_config.json" ]]; then
    echo "seed 2026 ${branch}: existing final adapter reused"
    return
  fi
  .venv/bin/python scripts/train_grpo.py \
    --model models/Qwen2.5-3B-Instruct \
    --sft-adapter runs/sft_recovery_seed42/adapter \
    --data data/rl/toolforge_rl.jsonl \
    --output-dir "$run_dir" \
    --branch "$branch" \
    --run-label "grpo_formal_${branch}_seed2026" \
    --num-generations 4 \
    --temperature 0.9 \
    --top-p 0.95 \
    --max-step-tokens 768 \
    --max-sequence-length 2560 \
    --batch-size 16 \
    --workers 10 \
    --learning-rate 5e-6 \
    --beta 0.01 \
    --clip-epsilon 0.2 \
    --efficiency-lambda 0.15 \
    --gradient-accumulation-groups 4 \
    --seed 2026 \
    --save-steps 27
}

evaluate_seed() {
  local seed="$1"
  local eval_dir="runs/grpo_formal_${branch}_dev_seed${seed}"
  local predictions="$eval_dir/rollouts.jsonl"
  mkdir -p "$eval_dir"
  if [[ "$(wc -l < "$predictions" 2>/dev/null || echo 0)" -ne 200 ]]; then
    # rollout_agent_batched uses deterministic rollout IDs and resumes safely.
    # Reuse any already-complete cohorts rather than regenerating them.
    .venv/bin/python scripts/rollout_agent_batched.py \
      --input data/raw_prompts/dev.jsonl \
      --output "$predictions" \
      --model models/Qwen2.5-3B-Instruct \
      --adapter "runs/grpo_formal_${branch}_seed${seed}/final/adapter" \
      --run-label "grpo_formal_${branch}_dev_seed${seed}" \
      --rollouts-per-prompt 1 \
      --batch-size 16 \
      --cohort-size 32 \
      --max-step-tokens 1024 \
      --greedy \
      --workers 10 \
      --seed "$seed"
  fi
  .venv/bin/python scripts/summarize_rollouts.py \
    --inputs "$predictions" \
    --output "$eval_dir/summary.json" \
    --report "reports/grpo_formal_${branch}_dev_seed${seed}.md" \
    --label "Formal GRPO ${branch} Dev seed ${seed}"
}

materialize 123
train_seed_2026
materialize 2026
evaluate_seed 123
evaluate_seed 2026

touch "runs/.gate6_${branch}_replications.done"
echo "Gate 6 ${branch} seed replications complete"
