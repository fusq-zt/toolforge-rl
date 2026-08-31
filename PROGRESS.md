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
- [x] Generated 600 real-loop candidates in two persistent, resumable GPU shards; 468 passed all quality filters (78%).
- [x] Reviewed 50 verified trajectories (30 retrieve→compute); manual/program agreement 50/50.

Execution optimization: the Pilot is the first immutable part of the full build;
Gate 4 resumes its shard files instead of regenerating it. Teacher generation is
length-bucketed and batched by round, with CPU-parallel tool execution.

Decision: **PASS** at 2026-09-01T04:32:00+08:00. The 600 immutable candidates
and 50 review decisions are carried forward into Gate 4 without regeneration.

## Gate 4 — Full Data Build (SFT-ready phase)

- [x] Completed 3,300/3,300 Teacher candidates for the 1,100 frozen SFT sources; the 600 Pilot rows were reused by candidate ID.
- [x] 2,517 candidates passed correctness, protocol, execution, termination, and length filters (76.27%); 984/1,100 sources have a verified trajectory.
- [x] Candidate IDs are unique with zero content collisions; raw shards remain append-only.
- [x] Curated 2,076 SFT rows: 1,076 project-executed/verified trajectories plus the capped 1,000 provenance-labelled official references.
- [x] The 4,000-row SFT target is unattainable under the user-revised 2,500-prompt budget, split isolation, 12.5% alternative cap, and 1,000 official cap; the shortfall is reported without padding or relaxed filters.
- [x] Reused all 50 unchanged Pilot review rows and reviewed 50 new rows; manual/program agreement is 100/100 and retrieve→compute coverage is 30.
- [x] Dev 200, Internal 400, RL candidate 800, and Public held-out 2,219 remain source-isolated; exact and Jaccard ≥0.90 held-out overlap remain zero.
- [x] All 2,076 SFT examples load at max length 3,072 with zero truncation.

Dependency adjustment: the specification asks Gate 4 to mine RL prompts with a
"frozen SFT checkpoint", which only exists after Gate 5. Gate 4 therefore freezes
the isolated 800-prompt RL candidate pool now; pass@4 mining runs immediately
after Gate 5 and before any GRPO step. No prompt moves between splits.

Decision: **PASS_FOR_SFT** at 2026-09-01T05:25:44+08:00. Quality thresholds are
unchanged; the only shortfall is the mathematically constrained dataset quantity.
