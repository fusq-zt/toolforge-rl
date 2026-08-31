# ToolForge-RL Data Card

Status: **PASS_FOR_SFT**

## Frozen source prompt pool

- Raw prompts: 2500 (user-revised target: 2,500)
- SFT / RL / Dev / Internal: 1100 / 800 / 200 / 400
- Source-ID overlap across splits: 0
- Public held-out examples frozen before generation: 2219
- Exact held-out overlap: 0; Jaccard >= 0.90 pairs: 0

## Teacher candidates

- Unique candidates: 3300 (600 Pilot candidates reused; no regeneration)
- Verified candidates: 2517 (76.27%) across 984 source prompts
- Candidate-ID content collisions: 0
- Verified token length mean / median / max: 607.5 / 485.0 / 2016

| family | verified candidates |
|---|---:|
| direct_anchor | 483 |
| code_reasoning | 753 |
| retrieval_reasoning | 825 |
| retrieve_then_compute | 456 |

## Curated SFT

- Actual: 2076; requested target: 4,000
- Project-executed trajectory quality errors: 0
- Shortfall is reported rather than padded after the raw-prompt budget was revised from 4,500+ to 2,500.

| family | curated trajectories |
|---|---:|
| direct_anchor | 464 |
| code_reasoning | 696 |
| retrieval_reasoning | 696 |
| retrieve_then_compute | 220 |

Pinned official Tool-Star references are provenance-labelled and are not represented as project-executed trajectories.

## Review and limitations

- Semantic review: 100/100; verifier agreement: 100.00%
- Retrieve-then-compute reviewed: 30
- No unique-correct-tool labels, online APIs, LLM judge, or relaxed verifier were used.
