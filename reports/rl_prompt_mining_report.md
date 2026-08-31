# RL Prompt Mining Report

Status: **FROZEN CANDIDATE POOL; PASS@4 PENDING FROZEN SFT**

- Isolated candidate prompts: 800
- Family targets already frozen: direct 160, code 240, retrieval 240, retrieve-then-compute 160.
- No gold trajectory is stored with the future policy input.
- pass@4 correctness, reward variance, tool-call variance, invalid and truncation metrics will be populated immediately after Gate 5.

This sequencing resolves an internal dependency in the experiment specification: Gate 4 asks for an RL-mined set while also requiring that mining use the Gate 5 frozen SFT checkpoint. No prompt is regenerated or moved between splits.
