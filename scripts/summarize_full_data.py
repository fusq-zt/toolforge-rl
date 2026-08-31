#!/usr/bin/env python3
"""Build concise Gate 4 manifests and reports from frozen, reusable artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


FAMILIES = ("direct_anchor", "code_reasoning", "retrieval_reasoning", "retrieve_then_compute")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(counts: dict[str, int]) -> str:
    return "\n".join(f"| {family} | {counts.get(family, 0)} |" for family in FAMILIES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_prompts"))
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--sft-manifest", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/manifests/raw_prompt_manifest.json"))
    parser.add_argument("--final-eval-manifest", type=Path, default=Path("data/manifests/final_eval_manifest.json"))
    parser.add_argument("--output-manifest", type=Path, default=Path("data/manifests/full_data_summary.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    raw_manifest = json.loads(args.raw_manifest.read_text(encoding="utf-8"))
    final_eval = json.loads(args.final_eval_manifest.read_text(encoding="utf-8"))
    sft_manifest = json.loads(args.sft_manifest.read_text(encoding="utf-8"))
    review_manifest = json.loads(args.review_manifest.read_text(encoding="utf-8"))
    splits = {name: read_jsonl(args.raw_dir / f"{name}.jsonl") for name in ("sft", "rl", "dev", "internal_test")}

    source_sets = {name: {row["source_id"] for row in rows} for name, rows in splits.items()}
    pair_overlaps = {}
    names = list(source_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pair_overlaps[f"{left}__{right}"] = len(source_sets[left] & source_sets[right])

    candidates_by_id = {}
    candidate_collisions = 0
    for path in args.candidates:
        for row in read_jsonl(path):
            old = candidates_by_id.get(row["candidate_id"])
            if old is not None and old["transcript"] != row["transcript"]:
                candidate_collisions += 1
            candidates_by_id.setdefault(row["candidate_id"], row)
    candidates = list(candidates_by_id.values())
    verified = [row for row in candidates if row.get("quality_pass")]
    verified_sources = {row["source_id"] for row in verified}
    candidate_by_family = Counter(row["task_family"] for row in candidates)
    verified_by_family = Counter(row["task_family"] for row in verified)
    verified_by_hint = Counter(row["sampling_hint"] for row in verified)
    termination = Counter(row["termination_reason"] for row in candidates)
    lengths = [row["trajectory_tokens"] for row in verified]

    sft = read_jsonl(args.sft)
    sft_by_family = Counter(row["task_family"] for row in sft)
    sft_by_provenance = Counter(row["sft_provenance"] for row in sft)
    project_sft = [row for row in sft if row["sft_provenance"].startswith("project_teacher")]
    duplicate_episode_ids = len(sft) - len({row["episode_id"] for row in sft})
    project_sources_outside_sft_split = len({row["source_id"] for row in project_sft} - source_sets["sft"])
    project_quality_errors = sum(
        not (row.get("answer_correct") and row.get("schema_valid") and row.get("execution_success"))
        for row in project_sft
    )

    pass_rate = len(verified) / len(candidates) if candidates else 0.0
    summary = {
        "schema_version": 1,
        "status": "PASS_FOR_SFT" if (
            len(candidates) == 3300
            and candidate_collisions == 0
            and not any(pair_overlaps.values())
            and review_manifest.get("status") == "PASS"
            and project_quality_errors == 0
            and duplicate_episode_ids == 0
            and project_sources_outside_sft_split == 0
        ) else "FAIL",
        "raw_prompts": raw_manifest["total"],
        "raw_by_split": raw_manifest["by_split"],
        "source_split_overlap": pair_overlaps,
        "final_eval_frozen_before_training": final_eval["frozen_before_training"],
        "final_eval_total": final_eval["total"],
        "final_eval_normalized_exact_overlap": final_eval["normalized_exact_overlap_with_raw_2500"],
        "final_eval_jaccard_ge_0_90_pairs": final_eval["jaccard_ge_0_90_pairs"],
        "unique_candidates": len(candidates),
        "candidate_collisions": candidate_collisions,
        "verified_candidates": len(verified),
        "verified_sources": len(verified_sources),
        "candidate_quality_pass_rate": pass_rate,
        "candidate_by_family": dict(candidate_by_family),
        "verified_by_family": dict(verified_by_family),
        "verified_by_hint": dict(verified_by_hint),
        "termination_reasons": dict(termination),
        "verified_trajectory_tokens": {
            "mean": statistics.fmean(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "sft_actual_total": len(sft),
        "sft_target_total": 4000,
        "sft_by_family": dict(sft_by_family),
        "sft_by_provenance": dict(sft_by_provenance),
        "project_sft_quality_errors": project_quality_errors,
        "duplicate_sft_episode_ids": duplicate_episode_ids,
        "project_sources_outside_sft_split": project_sources_outside_sft_split,
        "quality_shortfall_not_padded": len(sft) < 4000,
        "manual_review": review_manifest,
        "rl_candidate_pool": len(splits["rl"]),
        "rl_mining_status": "DEFERRED_UNTIL_FROZEN_SFT_CHECKPOINT",
        "gate_sequence_adjustment": "Gate 4 freezes the isolated RL pool; pass@4 mining runs immediately after Gate 5 because the specification requires the frozen SFT checkpoint.",
    }
    dump(args.output_manifest, summary)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    data_card = f"""# ToolForge-RL Data Card

Status: **{summary['status']}**

## Frozen source prompt pool

- Raw prompts: {summary['raw_prompts']} (user-revised target: 2,500)
- SFT / RL / Dev / Internal: {len(splits['sft'])} / {len(splits['rl'])} / {len(splits['dev'])} / {len(splits['internal_test'])}
- Source-ID overlap across splits: {sum(pair_overlaps.values())}
- Public held-out examples frozen before generation: {summary['final_eval_total']}
- Exact held-out overlap: {summary['final_eval_normalized_exact_overlap']}; Jaccard >= 0.90 pairs: {len(summary['final_eval_jaccard_ge_0_90_pairs'])}

## Teacher candidates

- Unique candidates: {len(candidates)} (600 Pilot candidates reused; no regeneration)
- Verified candidates: {len(verified)} ({pass_rate:.2%}) across {len(verified_sources)} source prompts
- Candidate-ID content collisions: {candidate_collisions}
- Verified token length mean / median / max: {summary['verified_trajectory_tokens']['mean']:.1f} / {summary['verified_trajectory_tokens']['median']:.1f} / {summary['verified_trajectory_tokens']['max']}

| family | verified candidates |
|---|---:|
{table(verified_by_family)}

## Curated SFT

- Actual: {len(sft)}; requested target: 4,000
- Project-executed trajectory quality errors: {project_quality_errors}
- Shortfall is reported rather than padded after the raw-prompt budget was revised from 4,500+ to 2,500.

| family | curated trajectories |
|---|---:|
{table(sft_by_family)}

Pinned official Tool-Star references are provenance-labelled and are not represented as project-executed trajectories.

## Review and limitations

- Semantic review: {review_manifest.get('manual_reviewed', 0)}/100; verifier agreement: {review_manifest.get('manual_program_agreement_rate', 0):.2%}
- Retrieve-then-compute reviewed: {review_manifest.get('review_by_family', {}).get('retrieve_then_compute', 0)}
- No unique-correct-tool labels, online APIs, LLM judge, or relaxed verifier were used.
"""
    (args.reports_dir / "data_card.md").write_text(data_card, encoding="utf-8")

    quality = f"""# Gate 4 Data Quality Report

Status: **{summary['status']}**

## Checks

- Candidate count: {len(candidates)}/3,300 (1,100 sources x 3); Gate 3 Pilot rows are byte-for-byte reused.
- Verified candidate count: {len(verified)}; source coverage: {len(verified_sources)}/1,100.
- All curated project trajectories correct + schema-valid + execution-successful: {'PASS' if project_quality_errors == 0 else 'FAIL'}.
- SFT episode IDs unique and project sources confined to the frozen SFT split: {'PASS' if duplicate_episode_ids == 0 and project_sources_outside_sft_split == 0 else 'FAIL'}.
- Candidate ID collision check: {'PASS' if candidate_collisions == 0 else 'FAIL'}.
- Frozen split source overlap: {'PASS' if not any(pair_overlaps.values()) else 'FAIL'}.
- Final held-out contamination checks: {'PASS' if summary['final_eval_normalized_exact_overlap'] == 0 and not summary['final_eval_jaccard_ge_0_90_pairs'] else 'FAIL'}.
- Manual/program verifier agreement: {review_manifest.get('manual_program_agreement_rate', 0):.2%} (required >=95%).
- Retrieve-then-compute review count: {review_manifest.get('review_by_family', {}).get('retrieve_then_compute', 0)} (required >=30).

## Honest quota handling

The user revised the independent raw-prompt budget to 2,500. With frozen source isolation, one preferred trajectory per SFT source, alternatives on only 12.5% of sources, and at most 1,000 pinned official references, 4,000 SFT rows are not attainable without padding or breaking the curation rules. The actual verified count is retained; quality thresholds are unchanged.

## Model-dependent RL mining

The 800 isolated RL candidate prompts are frozen now. The written specification also requires pass@4 mining with the frozen SFT checkpoint, which does not exist until Gate 5. Therefore mining is scheduled immediately after Gate 5 and before any GRPO work. This is a dependency-order correction, not a skipped Gate.
"""
    (args.reports_dir / "data_quality_report.md").write_text(quality, encoding="utf-8")

    lineage = f"""# Data Lineage Report

```text
Pinned public train assets (fixed revisions)
  -> normalized exact dedup (25 removed)
  -> source-ID split before generation (SFT 1100 / RL 800 / Dev 200 / Internal 400)
  -> held-out exact + light Jaccard audit (one MATH train item replaced)
  -> Tool-Star teacher, three sampling hints, real local tools
  -> programmatic answer verifier + protocol/execution filter
  -> {len(verified)} verified candidates
  -> minimum-call selection + 12.5% deterministic alternative allowance
  -> {len(project_sft)} project trajectories
  -> <=1000 provenance-labelled official Tool-Star references
  -> {len(sft)} frozen SFT trajectories
```

The 600 Pilot candidates and 50 Pilot review rows are reused unchanged. Raw candidates are append-only and retained for failure analysis. Public held-out IDs were frozen before trajectory generation and never enter Teacher generation, SFT selection, or RL mining.
"""
    (args.reports_dir / "data_lineage_report.md").write_text(lineage, encoding="utf-8")

    rl_report = f"""# RL Prompt Mining Report

Status: **FROZEN CANDIDATE POOL; PASS@4 PENDING FROZEN SFT**

- Isolated candidate prompts: {len(splits['rl'])}
- Family targets already frozen: direct 160, code 240, retrieval 240, retrieve-then-compute 160.
- No gold trajectory is stored with the future policy input.
- pass@4 correctness, reward variance, tool-call variance, invalid and truncation metrics will be populated immediately after Gate 5.

This sequencing resolves an internal dependency in the experiment specification: Gate 4 asks for an RL-mined set while also requiring that mining use the Gate 5 frozen SFT checkpoint. No prompt is regenerated or moved between splits.
"""
    (args.reports_dir / "rl_prompt_mining_report.md").write_text(rl_report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
