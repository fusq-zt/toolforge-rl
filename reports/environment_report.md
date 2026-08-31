# Environment Report

Audit time: 2026-09-01 02:34–02:41 CST  
Project path: `/root/shared-nvme/toolforge-rl`  
Method: read-only probes before any package, model, or dataset installation.

## Effective allocation

| Item | Observed | Interpretation |
|---|---|---|
| OS | Ubuntu 24.04.1 LTS, kernel 5.15.0-78 | Container image on an older host kernel; acceptable |
| CPU | AMD EPYC 7402 visible; cgroup `cpu.max=2200000 100000` | **22 CPU cores effective**, matching instance contract; ignore 96 host threads shown by `lscpu` |
| RAM | cgroup `memory.max=128849018880` | **120 GiB effective**, matching contract; ignore 503 GiB host-visible total |
| Swap | cgroup swap limit 0 | Treat as no usable swap; avoid memory overcommit |
| GPU | 2 × NVIDIA GeForce RTX 4090 | 24,564 MiB each; 24 GiB per process/device, not pooled 48 GiB |
| GPU topology | GPU0↔GPU1 = `SYS` | No NVLink; cross-GPU traffic traverses host/NUMA interconnect |
| Driver | 550.54.14 | Working |
| `nvidia-smi` CUDA | 12.4 | Driver capability display |
| NVCC | 12.8.93 | Toolkit compiler installed |
| Python | 3.12.3 at `/usr/bin/python` | Supported by selected dependency ranges |
| PyTorch | `2.7.0a0+7c8ec84dab.nv25.03` | NVIDIA image build; retain rather than replace |
| Torch CUDA / cuDNN | 12.8 / 90800 | CUDA runtime successfully initialized |
| CUDA available | true, two devices | PASS |
| BF16 | supported | PASS |
| Root overlay | 30GB available | Keep for image/runtime only |
| User storage | 150GB available at `/root/shared-nvme` after expansion | PASS: above the 100GB recommendation |

At audit time both GPUs were idle (1 MiB reported per device, no processes).

## Python environment

The image already contains a broad system environment including NumPy 1.26.4,
Pandas 2.2.3, SciPy 1.15.2, Matplotlib 3.10.1, SymPy 1.13.1,
pytest 8.1.1, psutil 7.0.0, Rich 13.9.4, Safetensors 0.5.3, CUDA
libraries, and FlashAttention 2.7.3. The first audit found these project packages
missing: Transformers, Datasets, PEFT, TRL, Accelerate, Hugging Face Hub,
SentencePiece, rank-bm25, and math-verify.

Decision:

1. Create `.venv --system-site-packages` to preserve the working Torch/CUDA stack.
2. Install only direct missing/required dependencies from `requirements.txt`.
   Apply `constraints.txt` so the venv's `packaging` remains compatible with the
   NVIDIA image's DALI installation.
3. Do not import or depend on preinstalled FlashAttention in Gate 0; its presence
   does not change the frozen first implementation.
4. Freeze the resolved environment only after the import/CUDA smoke passes.

## Connectivity

| Endpoint | Result | Action |
|---|---|---|
| DNS for GitHub/Hugging Face/PyPI | resolves | DNS is not the blocker |
| `github.com` direct HTTPS | timed out | query/transfer from an external connected host if still unavailable at Gate 1 |
| `huggingface.co` direct HTTPS | timed out | do not retry large downloads against the official endpoint |
| `pypi.org` | HTTP 200 | use for the small dependency install |
| `hf-mirror.com` | HTTP 200 | allowed transport mirror; canonical repository IDs and immutable revisions remain authoritative |
| proxy variables | none found | no hidden proxy assumed |

The mirror is a transport workaround, not a source substitution. Every downloaded
asset must be checked against the preregistered repository ID/revision and recorded
in the manifest.

## Audit commands

The evidence came from `uname -a`, `/etc/os-release`, `lscpu`, cgroup v2 limits,
`nvidia-smi`, `nvidia-smi topo -m`, the required Torch probe, `df -hT`, `free -h`,
`lsblk`, targeted package metadata, DNS resolution, and short HTTPS probes. No
training, model loading, dataset loading, or old project file was used.

## Resolved project packages

The isolated install completed with Transformers 4.52.4, Datasets 3.6.0, PEFT
0.15.2, TRL 0.17.0, Accelerate 1.7.0, Hugging Face Hub 0.33.5,
SentencePiece 0.2.2, rank-bm25 0.2.2, and math-verify 0.8.0.
`packaging==24.2` satisfies the preinstalled NVIDIA DALI constraint. `pip check`
returned “No broken requirements found.” The project smoke imported every direct
dependency and reconfirmed CUDA/BF16/two devices.

## Gate 0 environment decision

**PASS with one explicit constraint:** a single 4090 has 24GB, so all memory
smokes must be proven on this hardware before fixing formal batch sizes. The
model, task, and research question remain unchanged. Multi-GPU training may use
both devices, but memory is not described as a single 48GB pool.
