# Final Evaluation Long-generation Recovery

Evaluate all 2619 rows at 1024 tokens, retry only termination_reason=incomplete at 2048, preserve initial and recovery files.

| model | initial | retried at 2048 | remaining incomplete | final |
|---|---:|---:|---:|---:|
| Base | 2619 | 745 | 582 | 2619 |
| Base efficient prompt | 2619 | 527 | 363 | 2619 |
| SFT | 2619 | 78 | 21 | 2619 |
| Vanilla GRPO | 2619 | 40 | 5 | 2619 |
| Efficient GRPO | 2619 | 52 | 9 | 2619 |
| Tool-Star reference | 2619 | 70 | 10 | 2619 |

Initial 1024-token predictions are retained beside the merged final predictions; only genuinely incomplete episodes were regenerated.
