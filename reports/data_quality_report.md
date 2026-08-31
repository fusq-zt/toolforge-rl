# Gate 4 Data Quality Report

Status: **PASS_FOR_SFT**

## Checks

- Candidate count: 3300/3,300 (1,100 sources x 3); Gate 3 Pilot rows are byte-for-byte reused.
- Verified candidate count: 2517; source coverage: 984/1,100.
- All curated project trajectories correct + schema-valid + execution-successful: PASS.
- SFT episode IDs unique and project sources confined to the frozen SFT split: PASS.
- Candidate ID collision check: PASS.
- Frozen split source overlap: PASS.
- Final held-out contamination checks: PASS.
- Manual/program verifier agreement: 100.00% (required >=95%).
- Retrieve-then-compute review count: 30 (required >=30).

## Honest quota handling

The user revised the independent raw-prompt budget to 2,500. With frozen source isolation, one preferred trajectory per SFT source, alternatives on only 12.5% of sources, and at most 1,000 pinned official references, 4,000 SFT rows are not attainable without padding or breaking the curation rules. The actual verified count is retained; quality thresholds are unchanged.

## Model-dependent RL mining

The 800 isolated RL candidate prompts are frozen now. The written specification also requires pass@4 mining with the frozen SFT checkpoint, which does not exist until Gate 5. Therefore mining is scheduled immediately after Gate 5 and before any GRPO work. This is a dependency-order correction, not a skipped Gate.
