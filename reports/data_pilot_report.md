# Gate 3 data pilot

Status: **PASS**

- Frozen source prompts: 200 (target 200)
- Immutable candidates: 600 (target 600)
- Quality-pass trajectories: 468 (78.0%)
- Sources with at least one verified trajectory: 183
- Termination reasons: `{"final": 577, "incomplete": 22, "repeated_call": 1}`
- Verified by family: `{"code_reasoning": 141, "direct_anchor": 91, "retrieval_reasoning": 150, "retrieve_then_compute": 86}`
- Manual/program verifier agreement: 50/50 (100.0%)

The 50-row review set is `data/examples/pilot_review_50.jsonl`. It attempts to
include 30 retrieve→compute examples now so these unchanged rows can count toward
the Gate 4 review of 100. Pilot candidate files remain the first immutable shards
of the full build; Gate 4 resumes around them instead of regenerating them.
