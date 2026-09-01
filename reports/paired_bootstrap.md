# Paired Bootstrap: Vanilla vs Efficient

Outcome: **PASS**

Episodes: 2619; resamples: 5000

| metric | estimate | 95% CI |
|---|---:|---:|
| accuracy_delta_pp | -0.0382 | [-1.2218, 1.1837] |
| average_calls_reduction | 0.2188 | [0.1991, 0.2393] |
| calls_per_correct_reduction | 0.2184 | [0.1941, 0.2430] |
| invalid_rate_delta_pp | 0.1527 | [-0.0764, 0.3818] |
| direct_unnecessary_call_reduction | 0.2222 | [0.1299, 0.3200] |
| rtc_accuracy_delta_pp | 0.0000 | [0.0000, 0.0000] |

## Frozen success checks

- accuracy_drop_le_2pp: PASS
- calls_or_calls_per_correct_reduction_ge_10pct: PASS
- direct_unnecessary_call_reduction_ge_15pct: PASS
- rtc_accuracy_drop_le_3pp: PASS
- invalid_rate_increase_le_2pp: PASS
