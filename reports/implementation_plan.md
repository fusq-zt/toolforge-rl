# Implementation Plan

The plan is deliberately small: prove each claim at the cheapest gate and stop on
a real failure instead of expanding infrastructure.

| Gate | Minimum work | Exit evidence |
|---|---|---|
| 0 Bootstrap | environment, storage, dependencies, frozen source/data/reward plans | three Gate 0 reports, lockfile, import smoke, initial commit |
| 1 Protocol | pin Tool-Star source/model; implement adapter; 100 short official samples | parse/final ≥95%, tool execution ≥90%, loop failures ≤5% |
| 2 Tools/verifiers | process-isolated Python, episode-local BM25, numeric/math/QA verifiers, loop | required unit/integration counts and report |
| 3 Data pilot | 200 source prompts, ≤3 candidates each, real tool loop, inspect 50 | verifier/filter statistics plus sample review decision |
| 4 Full data | build verified SFT, dev/internal, mine RL prompts from SFT pass@4 | leakage checks, data card, quality/lineage and mining reports |
| 5 SFT | 256 smoke, then 4K LoRA SFT, frozen Dev gate | parsing/execution gates and nonzero task-family performance |
| 6 GRPO | temperature probe, reward tests, smoke, quick pair, then formal pair | nonzero group variance and fair Vanilla/Efficient configs |
| 7 Eval | frozen internal then public held-out, paired bootstrap, figures/report/demo | preregistered metrics, immutable predictions, honest result |

## Execution order

1. Finish Gate 0 and commit.
2. Run Gate 1 protocol sanity before writing data-generation logic that depends on
   unknown Tool-Star tokens.
3. Build only `python_exec`, `local_search`, their verifiers, and one agent loop.
4. Run the 200-prompt pilot before any 4K generation.
5. Train SFT only after data gates; run GRPO only after SFT and reward gates.
6. Start with seed 42. Additional seeds are conditional on observing the main trend.

## Resource policy

- CPU tasks may parallelize within the 22-core cgroup allocation.
- GPU-heavy phases are scheduled, not overlapped; both GPUs may participate in one
  job after a single-GPU memory smoke proves configuration safety.
- Keep `use_vllm=false`; do not install vLLM, DeepSpeed, veRL, OpenRLHF, or
  bitsandbytes for the first complete run.

## Change control

Changing the main model, adding 7B/Qwen3/DPO/online search, altering final items or
verifiers, weakening data gates, or forcing GRPO after an SFT failure requires a
stop-and-report decision. Small reversible engineering choices within the frozen
scope are made locally and recorded.

