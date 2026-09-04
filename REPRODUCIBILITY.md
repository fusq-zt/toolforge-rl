# Reproducibility guide

This repository is the lightweight, GitHub-ready record of the ToolForge-RL
experiment. It includes the implementation, frozen configurations and manifests,
small review samples, final reports, and all publication figures. It does not
redistribute model weights, adapters, public datasets, full predictions, caches,
or training logs.

## 1. Lightweight CPU verification

The protocol parser, deterministic tools, answer verifiers, reward functions,
and Agent Loop have unit tests that do not load a language model.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pytest==8.1.1 rank-bm25==0.2.2 math-verify==0.8.0 sympy==1.13.1
python -m pip install -e . --no-deps
python -m pytest -q
```

The same check runs in `.github/workflows/ci.yml` on every GitHub push and pull
request. The Python execution tool uses Linux resource limits, so its complete
test suite is intended for Linux rather than native Windows.

## 2. Recreate the publication figures

Install the plotting dependencies, then regenerate the figure pack solely from
the included frozen JSON manifests:

```bash
python -m pip install numpy==1.26.4 matplotlib==3.10.1
python scripts/generate_paper_figures.py \
  --manifest-dir data/manifests \
  --output-dir reports/figures_paper
```

The numbered PNG and vector PDF outputs are described in
`reports/figures_paper/README.md` and `reports/figures_paper/FIGURE_GUIDE_zh.md`.

## 3. Recreate final reports from materialized predictions

If the omitted prediction JSONL files and run logs have been materialized under
the paths recorded by the scripts, run:

```bash
make reproduce-final
```

This recomputes summaries, paired bootstrap intervals, the preregistered figure
set, final reports, the demo trace, and unit tests. It does not rerun training or
model inference.

## 4. Full training and evaluation

The original environment was Ubuntu 24.04 with Python 3.12 and two 24 GB RTX
4090 GPUs. Preserve a working system PyTorch/CUDA installation and install the
project dependencies into a virtual environment:

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements.lock.txt
python -m pip install -e . --no-deps
```

Download the exact model and dataset revisions in
`data/manifests/source_manifest.json`. Then follow `PROGRESS.md` and the staged
entry points in `scripts/`. Large artifacts are deliberately ignored by Git;
their expected locations are documented in `.gitignore` and the reports.

## Included versus excluded

| Included in Git | Intentionally excluded |
|---|---|
| Source, tests, configs, and scripts | Model checkpoints and adapters |
| Frozen manifests and small review samples | Raw/public datasets and generated candidate corpora |
| Aggregate results and statistical reports | Full prediction JSONL and training logs |
| PNG and vector-PDF figures | Virtual environments and package/model caches |

This separation keeps the repository small and license-conscious while retaining
the evidence needed to inspect the claimed results and regenerate every chart.
