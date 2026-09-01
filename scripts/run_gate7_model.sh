#!/usr/bin/env bash
# Evaluate one frozen model configuration on all 2,619 Internal + Public rows.
# The initial 1,024-token pass is resumable. Only true token-incomplete rows are
# retried at 2,048 tokens, then deterministically merged over the raw pass.

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SLUG GPU_INDEX MODEL_PATH ADAPTER_PATH_OR_DASH EFFICIENT_PROMPT" >&2
  exit 2
fi

slug="$1"
gpu_index="$2"
model_path="$3"
adapter_path="$4"
efficient_prompt="$5"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu_index"
export TOKENIZERS_PARALLELISM=false

input="data/eval/final_all_2619.jsonl"
run_dir="runs/final_eval_${slug}"
initial="$run_dir/initial_1024.jsonl"
recovery_input="$run_dir/incomplete_inputs.jsonl"
recovery="$run_dir/recovery_2048.jsonl"
final="$run_dir/predictions.jsonl"
mkdir -p "$run_dir"

common=(
  --model "$model_path"
  --rollouts-per-prompt 1
  --greedy
  --workers 10
  --seed 42
)
if [[ "$adapter_path" != "-" ]]; then
  common+=(--adapter "$adapter_path")
fi
if [[ "$efficient_prompt" == "true" ]]; then
  common+=(--efficient-tool-prompt)
fi

initial_count=0
if [[ -f "$initial" ]]; then
  initial_count="$(wc -l < "$initial")"
fi
if [[ "$initial_count" -ne 2619 ]]; then
  if ! .venv/bin/python scripts/rollout_agent_batched.py \
      --input "$input" \
      --output "$initial" \
      --run-label "final_eval_${slug}_initial1024" \
      --batch-size 24 \
      --cohort-size 48 \
      --max-step-tokens 1024 \
      "${common[@]}"; then
    echo "batch 24 failed for $slug; resuming completed cohorts at safe batch 16" >&2
    .venv/bin/python scripts/rollout_agent_batched.py \
      --input "$input" \
      --output "$initial" \
      --run-label "final_eval_${slug}_initial1024" \
      --batch-size 16 \
      --cohort-size 32 \
      --max-step-tokens 1024 \
      "${common[@]}"
  fi
fi

.venv/bin/python scripts/select_rollout_failures.py \
  --source "$input" \
  --rollouts "$initial" \
  --output "$recovery_input" \
  --termination-reason incomplete

recovery_total="$(wc -l < "$recovery_input")"
if [[ "$recovery_total" -gt 0 ]]; then
  recovery_count=0
  if [[ -f "$recovery" ]]; then
    recovery_count="$(wc -l < "$recovery")"
  fi
  if [[ "$recovery_count" -ne "$recovery_total" ]]; then
    if ! .venv/bin/python scripts/rollout_agent_batched.py \
        --input "$recovery_input" \
        --output "$recovery" \
        --run-label "final_eval_${slug}_recovery2048" \
        --batch-size 12 \
        --cohort-size 24 \
        --max-step-tokens 2048 \
        "${common[@]}"; then
      echo "recovery batch 12 failed for $slug; resuming at safe batch 8" >&2
      .venv/bin/python scripts/rollout_agent_batched.py \
        --input "$recovery_input" \
        --output "$recovery" \
        --run-label "final_eval_${slug}_recovery2048" \
        --batch-size 8 \
        --cohort-size 16 \
        --max-step-tokens 2048 \
        "${common[@]}"
    fi
  fi
  .venv/bin/python scripts/merge_rollout_recovery.py \
    --initial "$initial" \
    --recovery "$recovery" \
    --output "$final"
else
  cp "$initial" "$final"
fi

final_count="$(wc -l < "$final")"
if [[ "$final_count" -ne 2619 ]]; then
  echo "final prediction count mismatch for $slug: $final_count" >&2
  exit 1
fi

cat > "$run_dir/eval_manifest.json" <<EOF
{
  "schema_version": 1,
  "slug": "$slug",
  "model": "$model_path",
  "adapter": "$adapter_path",
  "efficient_tool_prompt": $efficient_prompt,
  "input": "$input",
  "initial_max_step_tokens": 1024,
  "targeted_recovery_max_step_tokens": 2048,
  "targeted_recovery_selector": "termination_reason=incomplete",
  "initial_rows": 2619,
  "recovery_rows": $recovery_total,
  "final_rows": 2619
}
EOF

touch "$run_dir/.complete"
echo "Gate 7 model evaluation complete: $slug; recovered=$recovery_total"
