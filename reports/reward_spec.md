# Reward Specification

Reward version: `v1`. The final answer verifier used for evaluation is the same
correctness source used by reward; tool success alone gives no correctness credit.

## Per-rollout telemetry

Before reward, record final-parse status, verifier result/reason, every parsed call,
schema validity, execution status, timeout/repetition, total call count, generated
tokens, and termination reason. Reward code consumes this structured telemetry,
not model claims.

## Vanilla

For rollout `i`:

`R_vanilla_i = answer_i + format_i - invalid_i - final_parse_i`

- `answer_i = 1.00` iff the programmatic final verifier is correct, otherwise 0;
- `format_i = 0.05` iff a final is parseable and every emitted tool call has a
  valid known-tool schema (zero calls is valid), otherwise 0;
- `invalid_i = min(0.20 × invalid_call_count, 0.40)`;
- `final_parse_i = 0.20` iff no valid final can be extracted.

An invalid call is an unknown tool, malformed arguments/marker, policy-blocked
execution, runtime failure, or timeout. Repeated valid calls are counted as calls
but are not assigned a second ad-hoc penalty in v1. A truncation or max-call exit
without final receives the final-parse penalty. Correctness is impossible when no
final is parseable.

## Efficient group-relative cost

Cost is only `C_i = tool_call_count_i`, including repeated or failed emitted calls.
For one prompt's rollout group:

`C*_q = min(C_j for correct j)` when at least one rollout is correct.

`E_i = max(0, C_i - C*_q)`.

`R_efficient_i = R_vanilla_i - λ × I(correct_i) × E_i`.

Initial `λ=0.15`. Exactly one Dev-only comparison may evaluate 0.10/0.15/0.25.
If no rollout is correct, all efficiency penalties are zero. Wrong zero-call
rollouts never gain a cost bonus. Only correct trajectories are ranked by excess
calls, and all rollout groups are computed within a single prompt.

## Worked groups

1. Correct calls `[0,1,3]`: `C*=0`; efficiency penalties `[0,.15,.45]`.
2. Correct calls `[1,1,2]`, wrong zero-call: `C*=1`; correct penalties
   `[0,0,.15]`, wrong penalty `0` and no answer reward.
3. All wrong: no `C*`; efficiency penalties all `0`.
4. Only one correct with 3 calls: `C*=3`; no efficiency penalty—the group offers
   no evidence of a cheaper correct policy.

## Invariants

- Given identical format/invalid state, correct beats wrong.
- Among correct rollouts, lower calls never receive lower Efficient reward.
- Cheapest correct Efficient reward equals its Vanilla reward.
- Wrong reward is identical between Vanilla and Efficient.
- Permuting rollout order does not change mapped rewards.
- Adding a cheaper correct rollout may reduce other correct rewards but never wrong
  rewards; this is expected group-relative behavior.

## Implementation controls

- One shared tested `score_vanilla` implementation is called by both branches.
- Efficient adds one separately logged `efficiency_penalty` component.
- Training logs every component and group `C*` so totals can be recomputed.
- At least 40 pre-RL tests in `reports/reward_test_plan.md` must pass.
- Reward/version, verifier/version, lambda, and telemetry schema are frozen in each
  run config. Final/Internal data can never choose or alter them.

