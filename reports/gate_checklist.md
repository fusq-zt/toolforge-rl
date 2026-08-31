# Gate 0–7 Execution Checklist

Every gate ends with reports, `PROGRESS.md`/`experiments.csv` updates, tests, an
explicit PASS/FAIL, and a Git commit. FAIL does not authorize skipping ahead.

## Gate 0 — Server Bootstrap

- [x] Verify new-server state; create dedicated repository on user storage.
- [x] Audit OS, cgroup CPU/RAM, disk, GPU topology/VRAM, driver/CUDA/Torch/BF16.
- [x] Preregister source IDs, immutable revisions, license/redistribution policy.
- [x] Write first-round protocol/tool/data/split/reward/evaluation designs.
- [x] Install direct dependencies without replacing system Torch.
- [x] Generate `requirements.lock.txt`; run import/CUDA and artifact smokes.
- [x] Finalize Gate 0 reports/status and commit.

## Gate 1 — Tool-Star Protocol Sanity

- [ ] Acquire pinned source and two model snapshots; verify revisions/sizes.
- [ ] Audit actual prompt, markers, tokens, template, parser, observation injection.
- [ ] Implement protocol adapter and prefix-consistency tests.
- [ ] Freeze 100 official SFT sanity IDs; run base and reference real loops.
- [ ] Meet reference parse/execution/termination thresholds; report and commit.

## Gate 2 — Tools and Verifiers

- [ ] Implement process-isolated `python_exec`; ≥30 tests.
- [ ] Implement episode-local deterministic BM25 `local_search`; ≥20 tests.
- [ ] Implement GSM8K/numeric, MATH, QA, and retrieve→compute verifiers.
- [ ] Implement ToolNameAdapter and real observation agent loop; ≥20 integration tests.
- [ ] Complete ≥40 executable reward tests; report and commit.

## Gate 3 — Data Pilot

- [ ] Freeze 200 train-side source prompts across all four families.
- [ ] Generate ≤3 real-loop teacher candidates per source.
- [ ] Verify/filter without overwriting raw failures.
- [ ] Manually inspect 50, including multi-step/retrieve→compute examples.
- [ ] Decide PASS/FAIL from correctness/schema/execution/failure evidence; commit.

## Gate 4 — Full Data Build

- [ ] Build ≥2,500 grouped raw prompts (user-revised scope) and up to three candidates each.
- [ ] Produce verified SFT target (4K if quality allows; report actual shortfall).
- [ ] Freeze Dev 200 and Internal 400 with source isolation.
- [ ] Run SFT pass@4 over 1,500–2,500 RL candidates; mine RL-800.
- [ ] Prove source/final disjointness and lightweight similarity audit.
- [ ] Review 100 verified SFT including ≥30 retrieve→compute; agreement ≥95%.
- [ ] Generate data card, quality, lineage, and RL mining reports; commit.

## Gate 5 — SFT

- [ ] Run 256-trajectory/20–30-step smoke; verify BF16/gradient/save-reload/no OOM.
- [ ] Train Qwen2.5-3B LoRA from base, not teacher.
- [ ] Evaluate frozen Dev 200 and meet parse/schema/execution/loop/family gates.
- [ ] If eligible, use at most one prescribed retrieve→compute recovery.
- [ ] Freeze SFT checkpoint or honest negative result; report and commit.

## Gate 6 — GRPO

- [ ] Temperature probe T=0.9/1.1/1.3 on fixed Dev 80×4.
- [ ] Confirm ≥40 reward tests and reward recomputation.
- [ ] Run 64-prompt smoke with nonconstant rewards/advantages and save-resume.
- [ ] Run fair 256-prompt Vanilla/Efficient quick pair.
- [ ] If quick passes, run sequential formal RL-800 pair, seed 42 first.
- [ ] Add seeds only after main trend and budget justify them; report and commit.

## Gate 7 — Final Evaluation

- [ ] Freeze every config; open Internal 400 once.
- [ ] Evaluate model matrix on frozen GSM8K/MATH-500/Hotpot/2Wiki IDs.
- [ ] Compute all preregistered metrics and paired bootstrap 95% CIs.
- [ ] Produce nine separate figures and failure analysis with failed runs retained.
- [ ] Complete Demo, full README, reproducibility command, Chinese/English resume bullets.
- [ ] Verify final archive excludes base/teacher weights, `.venv`, and HF cache.
- [ ] Final PASS or honest negative conclusion; commit and deliver.
