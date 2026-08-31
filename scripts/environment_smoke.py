"""Minimal Gate 0 runtime smoke; intentionally does not download assets."""

from __future__ import annotations

import importlib
import json
import os
import sys


MODULES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
    "rank_bm25",
    "sympy",
    "math_verify",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "psutil",
    "pytest",
    "rich",
    "yaml",
]


def main() -> int:
    imported: dict[str, str] = {}
    for name in MODULES:
        module = importlib.import_module(name)
        imported[name] = str(getattr(module, "__version__", "import-ok"))

    import torch

    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "imports": imported,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "official-default"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    required = report["cuda_available"] and report["bf16_supported"] and report["device_count"] == 2
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())

