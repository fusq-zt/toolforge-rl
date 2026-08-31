# Data Pipeline Design

## Research unit

`ToolForgeEpisode` is the single format used by generation, SFT rendering, RL
environment construction, and evaluation. Required fields are episode/source IDs,
task family/difficulty, prompt, tool schema, messages, reference/final answers,
verifier type/result, tool calls/responses/counts, generator/revision/hint, split,
termination, latency, and token counts.

Raw candidates are append-only. Normalized/selected datasets are rebuildable from
raw records plus manifests.

## Ten small stages

1. Build a train-only raw source pool and a separately frozen final manifest.
2. Render episode-local tools/context with one protocol implementation.
3. Generate up to three teacher candidates (`no_hint`, `tool_hint`,
   `minimal_call_hint`).
4. Execute real calls, inject observations, finish generation, and verify final.
5. Normalize telemetry and filter quality failures.
6. Label difficulty from observed base/teacher outcomes, calls, and length.
7. Select SFT targets from correct verified trajectories.
8. Freeze SFT, run pass@4 on an isolated RL candidate pool, and mine prompts with
   correctness or correct-call-count variance.
9. Materialize split datasets and prove source-group disjointness.
10. Generate count, quality, lineage, failure, and license reports.

## Four task families

- `direct_anchor`: correct zero-call behavior is explicitly represented.
- `code_reasoning`: GSM8K/MATH train; direct or Python paths may both be correct.
- `retrieval_reasoning`: HotpotQA/2Wiki train with episode-local context/BM25.
- `retrieve_then_compute`: deterministic train-context or nonce documents whose
  facts must be retrieved before a programmatically verified calculation.

There is no `Python-required`, `Search-required`, gold tool path, or minimum static
cost label. Selection is based on final correctness, valid execution, and observed
trajectory quality.

## Source pool and candidates

User-approved scope revision (2026-09-01): target at least **2,500** unique source
prompts, allocated by task family at roughly 500 direct anchors, 750 code reasoning,
750 retrieval reasoning, and 500 retrieve→compute. GSM8K/MATH and HotpotQA/2Wiki
are balanced within their respective families as availability/quality permits. A
retrieve→compute item may derive facts from a train-side parent context but receives
a new unique episode/source ID and remains grouped with every parent derivative.

The downstream SFT-4K target and all quality gates remain aspirations, not reasons
to duplicate prompts or relax filters. Under the 2,500-source budget, if the allowed
10–15% alternative-trajectory policy and verified yield cannot support 4,000 SFT
records (including ≤1,000 Tool-Star records and the desired project-generated
majority), Gate 4 reports the achievable high-quality count and proceeds only under
the existing honest-shortfall rule.

Each source generates at most three real-loop candidates. Keep all raw outcomes;
verified eligibility requires correct/parseable final, valid schema, successful
calls, ≤3 calls, no consecutive identical call, no timeout/truncation/loop, and
preferred length ≤3072 (hard maximum 4096).

## SFT selection

Target 4,000: 800 direct, 1,200 code, 1,200 retrieval, 800 retrieve→compute.
At most 1,000 come from Tool-Star-SFT-54K and at least 3,000 from this executable
pipeline. Prefer fewer-call correct trajectories; retain an alternative correct
path for only 10–15% of prompts. Never lower quality filters to meet a quota.

## Dev/internal and RL

- Dev 200: used for SFT gate, temperature, lambda, and quick GRPO.
- Internal test 400: opened only after all training/tuning choices freeze.
- RL candidates 1,500–2,500: SFT pass@4 telemetry.
- RL 800 target: 160/240/240/160 across the four families; policy sees prompt,
  tools/environment, reference/verifier metadata—not a gold trajectory.

Prefer pass@4 groups with correctness variation or multiple correct call counts.
Exclude consistently impossible and persistently zero-variance items.

## Lightweight quality and leakage audit

- group by canonical source/parent context before any candidate generation;
- exact source-ID set intersection must be empty across SFT/RL/Dev/Internal/Final;
- normalized exact question comparison;
- token-set Jaccard flag at a preregistered 0.85 threshold, manually adjudicated;
- 100 verified SFT records reviewed, including ≥30 retrieve→compute; program/manual
  agreement target ≥95%.

No MinHash cluster service, audit blockchain, or LLM judge is added.
