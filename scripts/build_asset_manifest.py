#!/usr/bin/env python3
"""Record local Gate 1 assets without committing large files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ASSETS = {
    "Qwen/Qwen2.5-3B-Instruct": (
        "model",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        Path("models/Qwen2.5-3B-Instruct"),
    ),
    "dongguanting/Tool-Star-Qwen-3B": (
        "model",
        "2350a1d6ec6230d48e41f752babab730bad8aa92",
        Path("models/Tool-Star-Qwen-3B"),
    ),
    "dongguanting/Tool-Star-SFT-54K": (
        "dataset",
        "f85b4fe0809a30a3b360b6f61689e462f79e2ec1",
        Path("data/raw/toolstar_sft_54k"),
    ),
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    tree = hashlib.sha256()
    for row in rows:
        tree.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return {
        "path": root.as_posix(),
        "bytes": sum(row["bytes"] for row in rows),
        "file_count": len(rows),
        "tree_sha256": tree.hexdigest(),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/manifests/gate1_assets.json"))
    args = parser.parse_args()
    payload = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "assets": []}
    for asset_id, (kind, revision, path) in ASSETS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payload["assets"].append(
            {"id": asset_id, "kind": kind, "revision": revision, **inventory(path)}
        )
    code_path = Path("third_party/Tool-Star")
    head = subprocess.check_output(
        ["git", "-C", str(code_path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(code_path), "status", "--porcelain"], text=True
    ).strip()
    payload["assets"].append(
        {
            "id": "RUC-NLPIR/Tool-Star",
            "kind": "git",
            "revision": head,
            "path": code_path.as_posix(),
            "clean": not status,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
