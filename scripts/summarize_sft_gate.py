#!/usr/bin/env python3
"""Consolidate SFT Smoke, full training, adapter reload, and Dev Gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-train", type=Path, required=True)
    parser.add_argument("--smoke-reload", type=Path, required=True)
    parser.add_argument("--full-train", type=Path, required=True)
    parser.add_argument("--full-reload", type=Path, required=True)
    parser.add_argument("--recovery-train", type=Path)
    parser.add_argument("--recovery-reload", type=Path)
    parser.add_argument("--base-dev", type=Path, required=True)
    parser.add_argument("--sft-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    smoke, smoke_reload = load(args.smoke_train), load(args.smoke_reload)
    full, full_reload = load(args.full_train), load(args.full_reload)
    recovery = load(args.recovery_train) if args.recovery_train else None
    recovery_reload = load(args.recovery_reload) if args.recovery_reload else None
    base, sft = load(args.base_dev), load(args.sft_dev)
    checks = {
        "smoke_gradient_nonzero": smoke["gradient_nonzero"],
        "smoke_loss_finite": smoke["loss_finite"],
        "smoke_adapter_saved": smoke["adapter_saved"],
        "smoke_adapter_reload": smoke_reload["status"] == "PASS",
        "smoke_truncation_le_10pct": smoke["truncated"] / smoke["usable"] <= 0.10,
        "full_gradient_nonzero": full["gradient_nonzero"],
        "full_loss_finite": full["loss_finite"],
        "full_adapter_reload": full_reload["status"] == "PASS",
        "dev_sft_gate": sft.get("sft_gate", {}).get("status") == "PASS",
    }
    if recovery is not None:
        checks.update({
            "recovery_gradient_nonzero": recovery["gradient_nonzero"],
            "recovery_loss_finite": recovery["loss_finite"],
            "recovery_zero_truncation": recovery["truncated"] == 0,
            "recovery_adapter_reload": recovery_reload is not None and recovery_reload["status"] == "PASS",
        })
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "smoke": {key: smoke.get(key) for key in ("rows", "usable", "truncated", "train_loss", "train_runtime")},
        "full": {key: full.get(key) for key in ("rows", "usable", "truncated", "train_loss", "train_runtime")},
        "bounded_recovery": (
            {key: recovery.get(key) for key in ("rows", "usable", "truncated", "train_loss", "train_runtime", "epoch")}
            if recovery is not None else None
        ),
        "base_dev_accuracy": base["overall"]["accuracy"],
        "sft_dev": sft,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = sft.get("sft_gate", {})
    lines = [
        "# Gate 5 SFT Report", "", f"Status: **{result['status']}**", "",
        "## Training", "",
        f"- Smoke: {smoke['usable']} usable rows, {smoke['truncated']} truncated, loss {smoke.get('train_loss'):.4f}.",
        f"- Full: {full['usable']} usable rows, {full['truncated']} truncated, one epoch, loss {full.get('train_loss'):.4f}.",
        *(
            [f"- Bounded protocol recovery: {recovery['usable']} train-only rows, {recovery['truncated']} truncated, {recovery.get('epoch', 0):.1f} epoch, loss {recovery.get('train_loss'):.4f}."]
            if recovery is not None else []
        ),
        f"- Base → SFT Dev accuracy: {base['overall']['accuracy']:.2%} → {sft['overall']['accuracy']:.2%} ({gate.get('accuracy_delta_pp', 0):+.2f} pp).",
        "", "## Gate checks", "",
    ]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
