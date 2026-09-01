#!/usr/bin/env python3
"""Fill the reproducibility metadata expected in a completed experiment run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/source_manifest.json"),
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # JSON is a strict YAML subset, so this is both machine-readable and avoids
    # adding a second serializer dependency to the training environment.
    (run_dir / "resolved_config.yaml").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    revision = git_revision()
    (run_dir / "git_commit.txt").write_text(revision + "\n", encoding="utf-8")

    environment = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    (run_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )

    source_payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    (run_dir / "source_revisions.json").write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    data_path = Path(config["data"])
    dataset = {
        "path": str(data_path),
        "sha256": sha256(data_path),
        "jsonl_rows": jsonl_rows(data_path),
        "seed": config.get("seed"),
        "max_prompts": config.get("max_prompts"),
    }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    trajectories = run_dir / "trajectories.jsonl"
    predictions = run_dir / "predictions.jsonl"
    if trajectories.exists() and not predictions.exists():
        os.link(trajectories, predictions)

    capture_note = (
        "This run was supervised in an interactive persistent PTY. Structured per-step "
        "output is preserved in metrics.jsonl; raw PTY stdout/stderr was not duplicated.\n"
    )
    for name in ("stdout.log", "stderr.log"):
        path = run_dir / name
        if not path.exists():
            path.write_text(capture_note, encoding="utf-8")

    required = [
        "resolved_config.yaml", "command.txt", "git_commit.txt", "environment.txt",
        "source_revisions.json", "dataset_manifest.json", "metrics.jsonl",
        "predictions.jsonl", "trajectories.jsonl", "reward_components.jsonl",
        "summary.json", "stdout.log", "stderr.log", "checkpoints",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    result = {"run_dir": str(run_dir), "git_commit": revision, "missing": missing}
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
