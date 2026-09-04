# Report index

The reports preserve both the final evidence and the Gate-by-Gate research
record. Final claims should be read from the core reports below; intermediate
reports document preregistered decisions, failed checks, and bounded recoveries.

## Core results

- `final_results.md` — complete model, dataset, task-family, and difficulty results
- `paired_bootstrap.md` — paired uncertainty intervals and final threshold decision
- `failure_analysis.md` — error and invalid-output analysis
- `final_delivery_audit.md` — final artifact and scope audit
- `data_card.md`, `data_quality_report.md`, `data_lineage_report.md` — data provenance and quality
- `reward_spec.md` — Vanilla and Efficient GRPO reward definitions
- `source_assets.md` — pinned upstream revisions, licenses, and redistribution policy

## Figures

- `figures/` — nine preregistered result plots
- `figures_paper/` — publication/PPT pack with 15 analytical figures, a gallery,
  PNG exports, and vector PDFs

## Gate and recovery evidence

The remaining Markdown files are the chronological Gate 0–7 audit trail. In
particular, `gate_checklist.md` is a frozen Gate 0 preregistration artifact; its
unchecked boxes are not the final completion state. Final status is recorded in
the repository `PROGRESS.md` and `experiments.csv`.

Some report-regeneration commands require large predictions or run logs that are
not included in Git. See the root `REPRODUCIBILITY.md` for exact scope.
