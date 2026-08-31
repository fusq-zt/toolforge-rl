"""Check that the frozen first-round audit is complete and internally countable."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    "bootstrap_report.md",
    "environment_report.md",
    "implementation_plan.md",
    "source_assets.md",
    "toolstar_protocol_sanity_plan.md",
    "tool_and_verifier_spec.md",
    "data_pipeline_design.md",
    "dataset_split_plan.md",
    "reward_spec.md",
    "evaluation_preregistration.md",
    "runtime_and_disk_budget.md",
    "episode_examples.md",
    "reward_test_plan.md",
    "gate_checklist.md",
]


def main() -> int:
    missing = [name for name in REPORTS if not (ROOT / "reports" / name).is_file()]
    if missing:
        raise SystemExit(f"missing first-round reports: {missing}")

    episodes = (ROOT / "reports" / "episode_examples.md").read_text(encoding="utf-8")
    reward_tests = (ROOT / "reports" / "reward_test_plan.md").read_text(encoding="utf-8")
    episode_ids = re.findall(r"^### E(\d{2})\b", episodes, flags=re.MULTILINE)
    test_ids = re.findall(r"^\| R(\d{2}) \|", reward_tests, flags=re.MULTILINE)
    assert episode_ids == [f"{i:02d}" for i in range(1, 21)], episode_ids
    assert test_ids == [f"{i:02d}" for i in range(1, 41)], test_ids

    manifest = json.loads((ROOT / "data" / "manifests" / "source_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["assets"]) == 10
    assert all(item["revision"] and item["revision"] != "main" for item in manifest["assets"])
    print("first-round audit artifacts: PASS (14 reports, 20 episodes, 40 reward tests, 10 pinned assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

