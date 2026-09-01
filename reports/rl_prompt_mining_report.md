# RL Prompt Mining Report

Status: **PASS**

- Isolated candidate prompts: 800
- Complete pass@4 groups: 800
- Selected prompts: 432/800
- Mean correct rollouts per group: 2.716/4
- Non-zero reward variance groups: 40.62%
- Non-zero tool-call variance groups: 37.38%
- Invalid / truncation: 5.12% / 5.06%
- Family counts: {'code_reasoning': 131, 'direct_anchor': 56, 'retrieval_reasoning': 144, 'retrieve_then_compute': 101}
- Selection types: {'mixed_correctness': 312, 'all_correct_call_variance': 118, 'reward_variance': 2}
- Excluded: {'all_wrong': 114, 'constant_group': 254}

All-wrong and constant-policy groups are excluded. Shortfalls are reported and never padded. The output contains prompts, local environment, reference/verifier and mining metadata only; no gold trajectory.
