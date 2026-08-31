# Progress

## Gate 0 — Server Bootstrap

- [x] Attachment and server screenshot read in full.
- [x] Fresh server/home state checked; no prior ToolForge-RL project found.
- [x] Repository initialized on the expanded user storage volume.
- [x] OS, cgroup, GPU, CUDA, Torch, BF16, RAM, CPU, disk, and connectivity audited.
- [x] Direct dependencies installed in `.venv` and locked.
- [x] Source IDs and immutable revisions preregistered; no assets downloaded.
- [x] First-round data, reward, split, tool, protocol, and evaluation designs written.
- [x] Environment smoke and first-round artifact validation pass.
- [x] Gate 0 final commit prepared (commit hash recorded after commit).

Decision: **PASS** at 2026-09-01T03:12:19+08:00. Gate 1 may start; no training has started.

## Approved scope revision

- 2026-09-01: raw source-prompt target changed from ≥4,500 to **≥2,500** by the
  user. Four task families, split isolation, verifiers, and quality gates are
  unchanged; downstream SFT shortfalls must be reported rather than padded.

## Gate 1 — Tool-Star Protocol Sanity

- [x] Pinned Tool-Star source, Qwen base, Tool-Star checkpoint, and official SFT data downloaded once to persistent storage.
- [x] Frozen upstream tag protocol and both identical tokenizers audited.
- [x] Deterministic official set frozen: 100 shortest complete rows (50 search, 50 Python).
- [x] Base and Tool-Star real parse→execute→observation→continue loops completed.
- [x] Only 26 initial truncations were recovered at a larger per-turn limit; 74 successes were not rerun.

Reference checkpoint metrics: protocol valid 99%, final parse 100%, tool-event
execution 96.15%, nontermination 1%. Decision: **PASS** at
2026-09-01T04:15:41+08:00.

## Gate 2 — Tools, Verifiers, Agent Loop, Reward

- [x] `python_exec` and episode-local deterministic BM25 `local_search` implemented.
- [x] Numeric, symbolic math, and extractive QA verifiers implemented without an LLM judge.
- [x] Shared protocol agent loop, three-call budget, repeat detection, and observation escaping implemented.
- [x] Vanilla and correctness-gated group-relative efficiency rewards implemented.
- [x] 170/170 remote pytest items passed: Python 36, search 23, loop 23, reward 42, protocol/verifier 46.
- [x] Both tokenizer history-prefix checks passed after action/result insertion.

Decision: **PASS** at 2026-09-01T04:15:41+08:00.

## Gate 3 — Reusable Data Pilot

- [x] Public train/test assets downloaded at frozen revisions.
- [x] Exactly 2,500 train-side source prompts deduplicated before generation and source-isolated as SFT 1,100 / RL 800 / Dev 200 / Internal 400.
- [x] Final public evaluation IDs frozen before generation: GSM8K 1,319, MATH500 500, Hotpot 200, 2Wiki 200.
- [x] One Jaccard-0.92 near duplicate removed and replaced; exact and ≥0.90 overlap are now zero.
- [x] Pilot 200 frozen with 40/60/60/40 family allocation.
- [ ] Generate 600 real-loop candidates in two persistent, resumable GPU shards.
- [ ] Review 50 verified trajectories; reuse unchanged rows in Gate 4 review.

Execution optimization: the Pilot is the first immutable part of the full build;
Gate 4 resumes its shard files instead of regenerating it. Teacher generation is
length-bucketed and batched by round, with CPU-parallel tool execution.
