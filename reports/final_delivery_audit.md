# Final Delivery Audit

Audit scope: the 24 mandatory deliverables in the frozen project prompt, after Gate 7 completion. This is a lightweight existence-and-result audit, not a separate compliance system.

| # | Deliverable | Evidence | Status |
|---:|---|---|---|
| 1 | Complete Git repository | `.git`, Gate commits, clean final tree | PASS |
| 2 | Clean-server installation instructions | `README.md`, Gate 0 reports | PASS |
| 3 | Requirements lock | `requirements.lock.txt` | PASS |
| 4 | Source revisions | `data/manifests/source_manifest.json` and per-run `source_revisions.json` | PASS |
| 5 | Tool-Star protocol adapter | `src/toolforge_rl/protocols/toolstar.py` | PASS |
| 6 | Python Sandbox | `src/toolforge_rl/tools/python_exec.py` | PASS |
| 7 | Local Search | `src/toolforge_rl/tools/local_search.py` | PASS |
| 8 | Verifiers | `src/toolforge_rl/verifiers/answers.py` | PASS |
| 9 | Complete data pipeline | raw build, batched teacher loop, SFT builder, RL mining scripts | PASS |
| 10 | ToolForge SFT dataset | `data/sft/toolforge_sft.jsonl` | PASS_ADJUSTED: 2,076 quality-passing rows; the original 4,000 target was not padded after the user changed raw prompts to 2,500 |
| 11 | ToolForge RL dataset | `data/rl/toolforge_rl.jsonl` | PASS_ADJUSTED: 432 variation-bearing prompts selected from the frozen 800-prompt candidate pool; no zero-variance padding |
| 12 | Data card and lineage | `reports/data_card.md`, `reports/data_quality_report.md`, `reports/data_lineage_report.md` | PASS |
| 13 | SFT adapter | `runs/sft_recovery_seed42/adapter/adapter_model.safetensors` | PASS |
| 14 | Vanilla GRPO adapter | `runs/grpo_formal_vanilla_seed42/final/adapter/adapter_model.safetensors` | PASS |
| 15 | Efficient GRPO adapter | `runs/grpo_formal_efficient_seed42/final/adapter/adapter_model.safetensors` | PASS |
| 16 | All benchmark results | six `data/manifests/final_eval_*_summary.json` files and per-model reports | PASS |
| 17 | Trajectories | six final prediction/trajectory sets, each 2,619 rows with unique source IDs | PASS |
| 18 | Paired bootstrap | `reports/paired_bootstrap.md`, 5,000 paired resamples | PASS |
| 19 | Figures | nine separate PNG files in `reports/figures/` | PASS |
| 20 | Failure analysis | `reports/failure_analysis.md` | PASS |
| 21 | Demo | `demo/toolforge_demo.py`, published `demo/example_trace.json` copy of the generated `runs/final_demo/trace.json` | PASS |
| 22 | README | `README.md` | PASS |
| 23 | Chinese and English resume text | `reports/resume_bullets.md` | PASS |
| 24 | One-command reproduction entry | `scripts/run_remaining_pipeline.sh`; `make reproduce-final` regenerates final statistics/reports/demo/tests from frozen outputs | PASS |

## Frozen artifact checks

- Raw prompts: 2,500 rows; SHA-256 `46fe7743199e304e2e5033e44bf8408a32a69db715bf6f9ce0a6b05ba2ef44165`.
- SFT data: 2,076 rows; SHA-256 `eebfcf69d0bfbbb680e9aa9a4985a03e88898b1bfdf6475b6c4df48b9e3ee6bb7`.
- RL data: 432 rows; SHA-256 `e15472a898c9fb510ee4e967e060fb97424e00363f4ed42c76f7c52d15ec6b075`.
- Final evaluation: 2,619 rows; SHA-256 `1ac3d2aaaad1f4a15727363cca766965ffca8b0a38468e87202505b103fc9995`.
- All six model matrices contain exactly 2,619 final rows, 2,619 unique non-null source IDs, and a retained initial/recovery trace.
- Final test suite: 170/170 passed.
- No verifier, reward semantic, or frozen final-evaluation ID was changed after results were observed.

Overall decision: **24/24 delivered; project success criteria PASS**. The two quantity adjustments are explicitly recorded and do not weaken verification or leak evaluation data into training.
