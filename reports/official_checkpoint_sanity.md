# Official checkpoint sanity (Gate 1)

Status: **PASS**

The frozen official SFT set contributed 100 shortest complete trajectories (50
search and 50 Python). Both checkpoints were loaded once on separate RTX 4090s and
ran the same real parse→execute→insert-result→continue loop. The initial 384-token
per-turn cap truncated 26 verbose Tool-Star search thoughts before a closing action;
only those failed IDs were rerun at 1024 tokens. The 74 successes were not repeated.

| Metric | Qwen base | Tool-Star merged | Gate threshold |
|---|---:|---:|---:|
| Protocol valid | 77.0% | 99.0% | ≥95% reference |
| Final parsed | 81.0% | 100.0% | ≥95% reference |
| Tool-event execution success | 88.9% | 96.2% | ≥90% reference |
| Nontermination failure | 19.0% | 1.0% | ≤5% reference |
| Tool events | 135 | 104 | descriptive |

Tool execution failures are preserved as structured observations rather than hidden.
They arise when official Python generations request modules outside the frozen safe
allowlist; a successful tool call is never treated as answer correctness.

Raw immutable runs are `runs/gate1/qwen_100.jsonl`,
`runs/gate1/toolstar_100.jsonl`, and the failure-only
`runs/gate1/toolstar_recovery_1024.jsonl`. Machine-readable merged metrics are in
`runs/gate1/summary.json`.
