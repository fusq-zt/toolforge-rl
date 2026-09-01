#!/usr/bin/env bash
# Continue automatically from the two Gate 6 replication workers through Gate 7.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs

while [[ ! -f runs/.gate6_vanilla_replications.done || ! -f runs/.gate6_efficient_replications.done ]]; do
  sleep 20
done

.venv/bin/python scripts/summarize_formal_seeds.py \
  --seeds 42 123 2026 \
  --output data/manifests/grpo_formal_three_seed.json \
  --report reports/grpo_formal_three_seed.md
.venv/bin/python scripts/update_project_progress.py --gate 6

git add \
  PROGRESS.md experiments.csv \
  configs/grpo \
  data/manifests/rl_manifest.json \
  data/manifests/temperature_probe_80.json \
  data/manifests/temperature_selection.json \
  data/manifests/grpo_smoke_summary.json \
  data/manifests/grpo_quick_comparison.json \
  data/manifests/grpo_formal_comparison.json \
  data/manifests/grpo_formal_three_seed.json \
  reports/rl_prompt_mining_report.md \
  reports/temperature_probe_report.md \
  reports/grpo_smoke_report.md \
  reports/grpo_quick_* \
  reports/grpo_formal_* \
  scripts/compare_quick.py \
  scripts/materialize_run_log.py \
  scripts/mine_rl_prompts.py \
  scripts/rollout_agent_batched.py \
  scripts/run_gate6_branch_remaining.sh \
  scripts/select_probe_prompts.py \
  scripts/select_temperature.py \
  scripts/summarize_formal_seeds.py \
  scripts/summarize_grpo_smoke.py \
  scripts/train_grpo.py \
  scripts/update_project_progress.py
git commit -m "feat: complete Gate 6 GRPO comparison"

run_pair() {
  local wave="$1"
  shift
  local first_args=("${@:1:5}")
  local second_args=("${@:6:5}")
  bash scripts/run_gate7_model.sh "${first_args[@]}" > "runs/logs/gate7_${wave}_gpu0.log" 2>&1 &
  local first=$!
  bash scripts/run_gate7_model.sh "${second_args[@]}" > "runs/logs/gate7_${wave}_gpu1.log" 2>&1 &
  local second=$!
  wait "$first"
  wait "$second"
}

# One model per GPU; each model processes Internal and Public data in one load.
run_pair wave1 \
  base 0 models/Qwen2.5-3B-Instruct - false \
  base_efficient_prompt 1 models/Qwen2.5-3B-Instruct - true
run_pair wave2 \
  sft 0 models/Qwen2.5-3B-Instruct runs/sft_recovery_seed42/adapter false \
  vanilla 1 models/Qwen2.5-3B-Instruct runs/grpo_formal_vanilla_seed42/final/adapter false
run_pair wave3 \
  efficient 0 models/Qwen2.5-3B-Instruct runs/grpo_formal_efficient_seed42/final/adapter false \
  toolstar 1 models/Tool-Star-Qwen-3B - false

bash scripts/finalize_gate7.sh > runs/logs/gate7_finalize.log 2>&1
.venv/bin/python scripts/update_project_progress.py --gate 7

git add README.md PROGRESS.md experiments.csv configs data/manifests demo reports scripts
git commit -m "feat: complete Gate 7 final evaluation and delivery"
touch runs/.pipeline_complete
echo "ToolForge-RL Gate 0-7 pipeline complete"
