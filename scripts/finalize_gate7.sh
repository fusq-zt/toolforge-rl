#!/usr/bin/env bash
# CPU post-processing after all six frozen model evaluations finish.

set -euo pipefail
cd "$(dirname "$0")/.."

slugs=(base base_efficient_prompt sft vanilla efficient toolstar)
for slug in "${slugs[@]}"; do
  test -f "runs/final_eval_${slug}/.complete"
  test "$(wc -l < "runs/final_eval_${slug}/predictions.jsonl")" -eq 2619
done

.venv/bin/python scripts/annotate_eval_difficulty.py \
  --input data/eval/final_all_2619.jsonl \
  --base-predictions runs/final_eval_base/predictions.jsonl \
  --output data/eval/final_all_2619_annotated.jsonl \
  --base-output runs/final_eval_base/predictions_annotated.jsonl \
  --manifest data/manifests/final_eval_difficulty.json \
  --prediction runs/final_eval_base_efficient_prompt/predictions.jsonl=runs/final_eval_base_efficient_prompt/predictions_annotated.jsonl \
  --prediction runs/final_eval_sft/predictions.jsonl=runs/final_eval_sft/predictions_annotated.jsonl \
  --prediction runs/final_eval_vanilla/predictions.jsonl=runs/final_eval_vanilla/predictions_annotated.jsonl \
  --prediction runs/final_eval_efficient/predictions.jsonl=runs/final_eval_efficient/predictions_annotated.jsonl \
  --prediction runs/final_eval_toolstar/predictions.jsonl=runs/final_eval_toolstar/predictions_annotated.jsonl

labels=(
  "Qwen2.5-3B Base + tools"
  "Base + efficient tool prompt"
  "ToolForge SFT"
  "SFT → Vanilla GRPO"
  "SFT → Efficient GRPO"
  "Tool-Star-Qwen-3B (public reference)"
)
for index in "${!slugs[@]}"; do
  slug="${slugs[$index]}"
  label="${labels[$index]}"
  .venv/bin/python scripts/summarize_rollouts.py \
    --inputs "runs/final_eval_${slug}/predictions_annotated.jsonl" \
    --output "data/manifests/final_eval_${slug}_summary.json" \
    --report "reports/final_eval_${slug}.md" \
    --label "$label"
  cp "data/manifests/final_eval_${slug}_summary.json" "runs/final_eval_${slug}/summary.json"
done

.venv/bin/python scripts/paired_bootstrap.py \
  --vanilla runs/final_eval_vanilla/predictions_annotated.jsonl \
  --efficient runs/final_eval_efficient/predictions_annotated.jsonl \
  --output data/manifests/paired_bootstrap.json \
  --report reports/paired_bootstrap.md \
  --samples 5000 \
  --seed 20260901

.venv/bin/python scripts/summarize_final_recovery.py \
  --run 'Base=runs/final_eval_base' \
  --run 'Base efficient prompt=runs/final_eval_base_efficient_prompt' \
  --run 'SFT=runs/final_eval_sft' \
  --run 'Vanilla GRPO=runs/final_eval_vanilla' \
  --run 'Efficient GRPO=runs/final_eval_efficient' \
  --run 'Tool-Star reference=runs/final_eval_toolstar' \
  --output reports/final_evaluation_recovery.md \
  --manifest data/manifests/final_evaluation_recovery.json

.venv/bin/python scripts/generate_figures.py \
  --model 'Base=data/manifests/final_eval_base_summary.json' \
  --model 'Base+prompt=data/manifests/final_eval_base_efficient_prompt_summary.json' \
  --model 'SFT=data/manifests/final_eval_sft_summary.json' \
  --model 'Vanilla=data/manifests/final_eval_vanilla_summary.json' \
  --model 'Efficient=data/manifests/final_eval_efficient_summary.json' \
  --model 'Tool-Star ref=data/manifests/final_eval_toolstar_summary.json' \
  --vanilla-predictions runs/final_eval_vanilla/predictions_annotated.jsonl \
  --efficient-predictions runs/final_eval_efficient/predictions_annotated.jsonl \
  --vanilla-train-metrics runs/grpo_formal_vanilla_seed42/metrics.jsonl \
  --efficient-train-metrics runs/grpo_formal_efficient_seed42/metrics.jsonl \
  --output-dir reports/figures

.venv/bin/python scripts/generate_final_reports.py \
  --model 'Base=data/manifests/final_eval_base_summary.json' \
  --model 'Base+efficient prompt=data/manifests/final_eval_base_efficient_prompt_summary.json' \
  --model 'SFT=data/manifests/final_eval_sft_summary.json' \
  --model 'Vanilla GRPO=data/manifests/final_eval_vanilla_summary.json' \
  --model 'Efficient GRPO=data/manifests/final_eval_efficient_summary.json' \
  --model 'Tool-Star reference=data/manifests/final_eval_toolstar_summary.json' \
  --vanilla-predictions runs/final_eval_vanilla/predictions_annotated.jsonl \
  --efficient-predictions runs/final_eval_efficient/predictions_annotated.jsonl \
  --bootstrap data/manifests/paired_bootstrap.json \
  --output-dir reports

.venv/bin/python scripts/update_readme.py

CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo/toolforge_demo.py \
  'Compute the exact product 1,234,567 × 891.' \
  --model models/Qwen2.5-3B-Instruct \
  --adapter runs/grpo_formal_efficient_seed42/final/adapter \
  --reference 1099999197 \
  --verifier gsm8k \
  --max-step-tokens 1024 \
  --output runs/final_demo/trace.json

.venv/bin/python -m pytest -q
echo "Gate 7 post-processing complete"
