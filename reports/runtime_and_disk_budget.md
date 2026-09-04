# Runtime and Disk Budget

## Authoritative capacity

- cgroup compute: 22 CPU cores, 120 GiB RAM, no cgroup swap.
- accelerators: 2 × RTX 4090, 24,564 MiB each, no NVLink.
- project volume: 150GB free after expansion.
- root overlay: 30GB free but excluded from the project budget.

The earlier 50GB project-volume observation is superseded by the verified 150GB
expanded capacity.

## Disk reservation

| Class | Budget | Control |
|---|---:|---|
| repository, venv, wheel cache | 5GB | system Torch reused; purge pip cache after lock if needed |
| two pinned BF16 model snapshots | 16GB | one copy each; no merged base checkpoints |
| public dataset snapshots/cache | 8GB | train/test manifests separate; no duplicate conversions |
| candidates/verified/SFT/RL/eval data | 18GB | JSONL/Parquet, never silently overwrite raw candidates |
| LoRA adapters and optimizer checkpoints | 22GB | retain milestone/failure checkpoints, not every step |
| run telemetry, predictions, plots | 12GB | compress finished JSONL logs only after checksums |
| temporary generation/training scratch | 20GB | one GPU phase at a time; clean only reproducible scratch |
| safety reserve | 35GB | downloads stop before free space drops below 25GB |
| **Total planned** | **136GB** | leaves about 14GB beyond reservations |

Use `HF_HOME=<project-root>/.cache/huggingface` and keep the hub
snapshot cache on this volume. Model and dataset working paths should refer to the
same snapshots rather than copy them.

## Compute plan

| Phase | Expected order-of-magnitude | Hardware use |
|---|---|---|
| Gate 0 dependencies/reports | <1 hour excluding network | CPU only |
| Protocol sanity | 1–3 hours after assets exist | one GPU, then release |
| Tools/verifiers | 3–6 hours | CPU |
| 200-prompt data pilot | 3–6 hours | teacher GPU generation serialized |
| Full data | 6–12 hours | CPU preparation + scheduled teacher generation |
| SFT smoke/full | 2–5 hours, remeasure after smoke | one or two GPUs as proven |
| GRPO probes/quick/formal pair | 12–30 hours | Vanilla/Efficient sequential |
| Eval/report | 4–8 hours | scheduled inference + CPU analysis |

These are planning ranges, not results. Actual elapsed time and peak memory must
come from run logs.

## Memory policy

- Start all smokes with per-device batch 1, BF16, gradient checkpointing, and the
  frozen sequence lengths.
- Do not assume two 24GB cards satisfy a single-card 48GB allocation.
- If SFT or GRPO OOMs, first adjust accumulation/distribution or measured sequence
  batching; do not change model size or research question.
- The 120GiB host-memory limit is sufficient for the planned data pipeline, but
  data loaders must stream and avoid copying full corpora per worker.

## Gate decision

Disk is **PASS** after expansion. Large downloads are permitted only after the
remaining Gate 0 dependency/import checks pass and only at preregistered revisions.

