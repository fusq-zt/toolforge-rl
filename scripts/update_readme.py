#!/usr/bin/env python3
"""Write the final reproducible README from frozen manifests and real results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("README.md"))
    parser.add_argument("--data", type=Path, default=Path("data/manifests/full_data_summary.json"))
    parser.add_argument("--rl", type=Path, default=Path("data/manifests/rl_manifest.json"))
    parser.add_argument("--eval", type=Path, default=Path("data/manifests/final_all_2619.json"))
    parser.add_argument("--bootstrap", type=Path, default=Path("data/manifests/paired_bootstrap.json"))
    parser.add_argument("--vanilla", type=Path, default=Path("data/manifests/final_eval_vanilla_summary.json"))
    parser.add_argument("--efficient", type=Path, default=Path("data/manifests/final_eval_efficient_summary.json"))
    args = parser.parse_args()

    data, rl, evaluation = load(args.data), load(args.rl), load(args.eval)
    bootstrap = load(args.bootstrap)
    vanilla, efficient = load(args.vanilla), load(args.efficient)
    v, e = vanilla["overall"], efficient["overall"]
    points = bootstrap["point_estimates"]
    ci = bootstrap["confidence_intervals_95"]

    outcome = "met" if bootstrap["status"] == "PASS" else "did not meet"
    readme = f"""# ToolForge-RL

**Data-Centric Tool-Integrated Reasoning with Efficient GRPO**

ToolForge-RL is an end-to-end Qwen2.5-3B post-training experiment built from a clean server: public questions are converted into executable tool trajectories, verified with deterministic programs, used for LoRA SFT, mined with pass@4 for GRPO, and evaluated under a frozen Vanilla-versus-Efficient comparison.

Final status: **Gate 0–7 complete**. The preregistered success criteria **{outcome}** their joint threshold (`{bootstrap['status']}`). All reported numbers below are read from frozen artifacts; negative checks are retained rather than repaired post hoc.

## What is different from Tool-Star?

Tool-Star supplies the protocol reference, a public teacher/reference checkpoint, and up to 1,000 screened SFT examples. This project is not a full Tool-Star reproduction: it fixes a smaller two-tool scope (`python_exec`, episode-local BM25 `local_search`) and independently implements real execution, observation reinjection, programmatic verification, quality filtering, difficulty labeling, pass@4 prompt mining, LoRA SFT, and paired Vanilla/Efficient GRPO. `dongguanting/Tool-Star-Qwen-3B` is reported only as a public reference model, never as a project-trained result.

Qwen2.5-3B-Instruct is intentionally fixed: it is small enough for repeatable 24 GB GPU experiments while still exposing meaningful correctness/efficiency trade-offs. The project does not switch to Qwen3/7B, DPO, online search, vLLM, or a larger tool catalog.

## Data pipeline

```text
public train-only prompts
  → source-ID split and normalized exact dedup
  → batched teacher Agent Loop
  → real Python/BM25 execution and observation reinjection
  → deterministic answer verifier
  → filtering and difficulty signals
  → SFT trajectories / trajectory-free RL prompts
```

- Raw source prompts: **{data['raw_prompts']:,}** (the user-revised frozen target), with zero source-ID overlap among SFT/RL/Dev/Internal.
- Teacher candidates: **{data['unique_candidates']:,}**; verified: **{data['verified_candidates']:,}** ({pct(data['candidate_quality_pass_rate'])}).
- SFT examples: **{data['sft_actual_total']:,}** high-quality rows, including **1,076** project-generated, real-tool-verified trajectories and **1,000** pinned public reference rows. The original 4,000 target was not padded with low-quality data.
- RL mining: **{rl['candidate_prompts']:,}** prompts × 4 rollouts; **{rl['final_selected']:,}** prompts retained with policy/reward variation. The original 800 target was not padded (`non-zero reward variance={pct(rl['nonzero_reward_variance_ratio'])}`).
- Manual review: **{data['manual_review']['manual_reviewed']}** rows, including 30 retrieve→compute; program/human agreement **{pct(data['manual_review']['manual_program_agreement_rate'])}**.
- Frozen final evaluation: **{evaluation['total']:,}** episodes = {evaluation['by_suite']['internal_test']} Internal + {evaluation['by_suite']['public_heldout']:,} public held-out; SHA-256 `{evaluation['sha256']}`.

SFT contains successful message/tool trajectories. GRPO data deliberately contains only prompts, tools/environment, references, verifiers, and metadata—never a gold trajectory for the policy.

## Tools and verification

- `python_exec`: a fresh restricted subprocess with no network, shell, subprocess spawning, or arbitrary file I/O; bounded CPU, RAM, time, and output.
- `local_search`: deterministic BM25 over documents attached to the current episode only; no global or online retrieval.
- GSM8K numeric normalization, math equivalence (`math-verify` plus bounded symbolic fallback), normalized QA/alias matching, and deterministic retrieve→compute ground truth.

Real tool execution matters because a syntactically plausible `<python>` or `<search>` block is not evidence that the operation ran, the observation was genuine, or the final answer was correct.

## Training and reward

The SFT adapter is trained from `Qwen/Qwen2.5-3B-Instruct`, not from the teacher. Vanilla and Efficient GRPO then fork from the same frozen SFT adapter with identical prompts, sample order, temperature, rollout budget, optimizer, LoRA configuration, and seeds.

```text
R_vanilla = answer_reward + format_reward - invalid_penalty
R_efficient = R_vanilla - 0.15 × I(correct) × max(0, calls - min_correct_calls_in_group)
```

The efficiency term is group-relative and correctness-gated. An incorrect zero-call rollout therefore receives no cost advantage; only redundant calls among correct rollouts are penalized. Prompts are mined instead of randomly sampled so that groups contain correctness or tool-call variation and can produce a learning signal.

## Frozen final results

| branch | accuracy | avg calls | calls/correct | direct unnecessary | RTC accuracy | invalid |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla GRPO | {pct(v['accuracy'])} | {v['average_tool_calls']:.3f} | {v['calls_per_correct']:.3f} | {pct(v['direct_unnecessary_call_rate'])} | {pct(vanilla['by_family']['retrieve_then_compute']['accuracy'])} | {pct(v['invalid_rate'])} |
| Efficient GRPO | {pct(e['accuracy'])} | {e['average_tool_calls']:.3f} | {e['calls_per_correct']:.3f} | {pct(e['direct_unnecessary_call_rate'])} | {pct(efficient['by_family']['retrieve_then_compute']['accuracy'])} | {pct(e['invalid_rate'])} |

Paired episode-level bootstrap ({bootstrap['bootstrap_samples']:,} resamples):

- Accuracy delta: **{points['accuracy_delta_pp']:+.2f} pp**, 95% CI [{ci['accuracy_delta_pp'][0]:+.2f}, {ci['accuracy_delta_pp'][1]:+.2f}].
- Average-call reduction: **{points['average_calls_reduction']:.2%}**, 95% CI [{ci['average_calls_reduction'][0]:.2%}, {ci['average_calls_reduction'][1]:.2%}].
- Calls-per-correct reduction: **{points['calls_per_correct_reduction']:.2%}**, 95% CI [{ci['calls_per_correct_reduction'][0]:.2%}, {ci['calls_per_correct_reduction'][1]:.2%}].
- Direct unnecessary-call reduction: **{points['direct_unnecessary_call_reduction']:.2%}**.
- Retrieve→compute accuracy delta: **{points['rtc_accuracy_delta_pp']:+.2f} pp**; invalid-rate delta: **{points['invalid_rate_delta_pp']:+.2f} pp**.

The exact threshold outcome and every individual check are preserved in `reports/paired_bootstrap.md`. Per-model, per-family, per-dataset, per-difficulty, and Internal/Public results are in `reports/final_results.md`.

## Reproduce

Use a recent CUDA-capable PyTorch environment; do not overwrite a working system CUDA/PyTorch installation. The clean-server versions used here are frozen in `requirements.lock.txt`.

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
```

Download the pinned resources listed in `data/manifests/source_manifest.json`, then follow `PROGRESS.md`, `configs/`, and the stage entry points under `scripts/`. `scripts/run_remaining_pipeline.sh` is the resume-safe one-command Gate 6→7 orchestrator; `scripts/run_gate7_model.sh` preserves targeted long-output recovery. Model/data caches remain outside Git.

Once the frozen predictions and training logs are present, regenerate every final summary, bootstrap interval, figure, report, demo trace, and the full test result with one command:

```bash
make reproduce-final
```

Run the final adapter demo:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo/toolforge_demo.py \
  "Compute the exact product 1,234,567 × 891." \
  --adapter runs/grpo_formal_efficient_seed42/final/adapter \
  --reference 1099999197 --verifier gsm8k
```

## Figures and reports

The nine preregistered plots are under `reports/figures/`, including accuracy/cost trade-offs, family accuracy, direct overuse, retrieve→compute completion, reward/KL/entropy proxy, reward variance, call distribution, and failure types. Key reports include:

- `reports/data_card.md`, `reports/data_quality_report.md`, `reports/data_lineage_report.md`
- `reports/final_results.md`, `reports/paired_bootstrap.md`, `reports/failure_analysis.md`
- `reports/resume_bullets.md` (Chinese and English)

## Limitations

- Only a 3B model and two deterministic local tools are studied; conclusions need not transfer to larger or online agents.
- The user-revised 2,500-prompt pool yielded fewer quality-passing SFT/RL rows than the original aspirational quotas; no low-quality padding was used.
- SFT protocol recovery was bounded to one additional half epoch and the original stricter parse/schema checks remain recorded.
- The teacher and Tool-Star reference can share ecosystem biases with the student even though final IDs are frozen and train/test exact-overlap checks are clean.
- Generated-token latency is hardware- and batching-dependent; tool-call counts and correctness are the primary efficiency measures.
- Long-generation handling uses a 1,024-token main pass and retries only genuinely incomplete episodes at 2,048 tokens; both raw and recovered predictions are retained.

## License and attribution

Project code is under the repository license. Dataset/model licenses and pinned revisions are listed in `NOTICE`, `reports/source_assets.md`, and `data/manifests/source_manifest.json`. Third-party raw datasets and model weights are not redistributed.
"""
    args.output.write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": bootstrap["status"]}, indent=2))


if __name__ == "__main__":
    main()
