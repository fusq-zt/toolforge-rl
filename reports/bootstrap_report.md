# Bootstrap Report

## Scope

This report covers only Gate 0. The server was treated as new: no repository,
virtual environment, data, models, checkpoints, adapters, or prior logs were
assumed. Hidden home files supplied by the image were not treated as project
artifacts.

## Decisions made

- Repository: `<project-volume>/toolforge-rl`, on the expanded 150GB writable
  user volume.
- Branch: `main`.
- Python: `.venv --system-site-packages` to retain verified NVIDIA PyTorch.
- Git identity: repository-local neutral identity; no personal credential added.
- Asset cache: under the same 150GB volume; do not duplicate models in root cache.
- External assets: IDs and immutable revisions preregistered, but none downloaded
  in this first round.
- Scope remains Qwen2.5-3B, two local tools, LoRA SFT, then Vanilla/Efficient GRPO.

## Gate 0 checklist

| Requirement | Evidence | Status |
|---|---|---|
| New location and Git repository | empty directory initialized on `main` | PASS |
| OS/CPU/RAM audit | `reports/environment_report.md` with cgroup limits | PASS |
| GPU/driver/CUDA/BF16 audit | two 4090s, Torch CUDA available, BF16 true | PASS |
| Disk threshold | 150GB free on project volume after user expansion | PASS |
| Isolated environment | `.venv` created with system site packages | PASS |
| Direct dependencies | resolved compatible set; `pip check` and import smoke passed | PASS |
| Human-readable requirements | `requirements.txt` | PASS |
| Resolved lock | original Gate 0 server freeze contained 323 entries; this public export provides portable direct pins in `requirements.lock.txt` | PASS |
| Source asset plan | 10 assets pinned; license/usage decisions recorded | PASS |
| No premature download/training | repository and reports only | PASS |

## Non-blocking risks

1. Direct GitHub/Hugging Face access times out. PyPI and the HF mirror work; Gate
   1 must record any mirror/local-transfer method and verify revisions.
2. Two 4090s have no NVLink. Do not make performance claims based on pooled VRAM.
3. The NVIDIA Torch build is an alpha/vendor build. It passed CUDA/BF16 probing;
   compatibility is accepted provisionally and must pass the project import smoke.

## Resolved environment evidence

Installed in `.venv`: Transformers 4.52.4, Datasets 3.6.0, PEFT 0.15.2,
TRL 0.17.0, Accelerate 1.7.0, Hugging Face Hub 0.33.5, SentencePiece
0.2.2, rank-bm25 0.2.2, and math-verify 0.8.0. `packaging==24.2`
keeps compatibility with the NVIDIA image's DALI constraint.

`pip check` reported no broken requirements. The environment smoke imported all
direct dependencies and reconfirmed Torch 2.7.0a0+cu128, two CUDA devices, and
BF16 support. The artifact validator confirmed 14 reports, 20 episode examples,
40 reward tests, and 10 pinned assets. Project volume usage was 202MB of 150GB.

## Result

**PASS** at 2026-09-01T03:12:19+08:00. Gate 1 protocol work may begin. Model/data
training remains prohibited until their later gates.
