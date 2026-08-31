# Evaluation Preregistration

Frozen before model/data downloads and before training.

## Main hypothesis

Against Vanilla GRPO under identical initialization, prompts, sampling/training
budget, and seed, Efficient GRPO will reduce tool calls without materially reducing
final-answer accuracy.

Primary success rule:

- accuracy change ≥ -2 percentage points; and
- average calls or calls per correct improves by ≥10%; and
- direct unnecessary calls improve by ≥15%; and
- retrieve→compute accuracy change ≥ -3pp; and
- invalid-rate increase ≤2pp.

A failure is reported as a negative result; final items/verifiers are unchanged.

## Model matrix

1. Qwen2.5-3B base + tools;
2. base + efficient-use prompt;
3. ToolForge SFT;
4. SFT→Vanilla GRPO;
5. SFT→Efficient GRPO;
6. public Tool-Star-Qwen-3B reference (clearly external).

## Evaluation sets and opening order

After training choices freeze: Internal 400 → GSM8K test → MATH-500 → frozen
HotpotQA validation 200 → frozen 2Wiki validation/test 200. IDs are written before
training. Optional external generalization is outside the primary claim.

## Metrics

Primary: final-answer accuracy, average calls, calls per correct, direct unnecessary
call rate, retrieve→compute completion/accuracy, invalid rate.

Secondary: final parse, schema validity, tool execution, multi-turn completion,
retrieval under-call, repeated-call, generated tokens, end-to-end/P95 latency,
truncation, reward/KL/entropy, group reward variance, and failure type. Report by
task family, dataset, and observed difficulty.

## Fair comparison

Vanilla and Efficient branches start from the identical frozen SFT checkpoint and
share RL prompt order, seed, temperature/top-p, generations, completion limits,
max calls, optimizer/LoRA/KL settings, steps, and compute allocation. Only the
efficiency penalty differs. Run sequentially. Seed 42 is primary; seeds 123/2026
are added only if the main trend exists and budget permits.

## Tuning isolation

- Temperature: fixed Dev 80, four rollouts, T=0.9/1.1/1.3; select lowest meeting
  ≥30% nonzero reward variance, ≤5% invalid, ≤2% truncation, and ≤10pp accuracy
  loss versus lowest T.
- Lambda: initial 0.15; at most one Dev comparison among 0.10/0.15/0.25.
- Quick continuation: Efficient accuracy ≥ Vanilla−3pp, calls no higher, invalid
  ≤5%, no retrieve→compute collapse, and ≥25% informative groups.
- Internal and public final sets never tune temperature/lambda/configuration.

## Statistical analysis

Use episode-level paired differences for models evaluated on the same items.
Generate 10,000 paired bootstrap resamples with seed 42 and percentile 95% CIs for
accuracy and efficiency deltas. Dataset-level results are reported separately
before any pooled summary. No significance threshold is used to rewrite the success
rule, and missing/failed episodes count according to preregistered failure handling.

## Reporting integrity

All predictions, trajectories, configs, source revisions, reward components, and
failure runs remain available. Public checkpoint numbers are references, not
project achievements. Resume bullets are filled only from verified final summaries.

