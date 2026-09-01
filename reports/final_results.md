# Final Evaluation Results

Frozen evaluation episodes: 2619

| model | accuracy | parse | schema | avg calls | calls/correct | direct unnecessary | RTC accuracy | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 29.90% | 55.67% | 50.10% | 1.208 | 4.042 | 78.75% | 12.50% | 22.30% |
| Base+efficient prompt | 23.86% | 60.44% | 50.74% | 1.433 | 6.005 | 80.00% | 22.50% | 13.86% |
| SFT | 69.15% | 96.11% | 95.65% | 1.026 | 1.484 | 93.75% | 80.00% | 0.80% |
| Vanilla GRPO | 70.98% | 97.52% | 97.48% | 0.958 | 1.350 | 90.00% | 90.00% | 0.19% |
| Efficient GRPO | 70.94% | 98.70% | 98.70% | 0.748 | 1.055 | 70.00% | 90.00% | 0.34% |
| Tool-Star reference | 72.39% | 98.21% | 83.39% | 1.224 | 1.691 | 100.00% | 62.50% | 0.38% |

## Preregistered Vanilla vs Efficient comparison

Outcome: **PASS**

- accuracy_delta_pp: -0.0382 (paired-bootstrap 95% CI [-1.2218, 1.1837])
- average_calls_reduction: 0.2188 (paired-bootstrap 95% CI [0.1991, 0.2393])
- calls_per_correct_reduction: 0.2184 (paired-bootstrap 95% CI [0.1941, 0.2430])
- invalid_rate_delta_pp: 0.1527 (paired-bootstrap 95% CI [-0.0764, 0.3818])
- direct_unnecessary_call_reduction: 0.2222 (paired-bootstrap 95% CI [0.1299, 0.3200])
- rtc_accuracy_delta_pp: 0.0000 (paired-bootstrap 95% CI [0.0000, 0.0000])

No verifier, final test ID, or reward semantic was changed after results were observed.
