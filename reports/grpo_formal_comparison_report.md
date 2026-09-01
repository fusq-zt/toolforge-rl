# Formal GRPO Dev Comparison

Status: **PASS**

Efficiency lambda: 0.15

- efficient_accuracy_ge_vanilla_minus_3pp: PASS
- efficient_average_calls_le_vanilla: PASS
- efficient_invalid_rate_le_5pct: PASS
- retrieve_then_compute_not_collapsed: PASS
- relative_reward_groups_ge_25pct: PASS

| branch | accuracy | average calls | invalid | RTC accuracy |
|---|---:|---:|---:|---:|
| Vanilla | 80.50% | 1.055 | 2.50% | 100.00% |
| Efficient | 79.00% | 0.910 | 1.00% | 97.50% |
