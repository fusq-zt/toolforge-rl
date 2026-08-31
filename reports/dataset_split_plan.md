# Dataset Split Plan

Seed: 42. Splitting occurs at source-group level before teacher generation.

## Public source policy

| Dataset | Training-side use | Final use | Forbidden use |
|---|---|---|---|
| GSM8K | `train` | full `test` | final items in teacher/SFT/RL/tuning |
| MATH | `train` only | none (MATH-500 is final) | any known MATH-500 item in train pipeline |
| MATH-500 | none | all 500 | any generation or tuning |
| HotpotQA distractor | `train` | fixed validation 200 | final IDs/context in generation/tuning |
| 2Wiki | `train` | fixed 200 from validation/test | final IDs/context in generation/tuning |
| Tool-Star-SFT-54K | Gate 1 sanity + filtered SFT ≤1,000 | none | replacing project-generated majority |
| Multi-Tool-RL-10K | schema/format inspection only | none | direct RL training substitute |

## Group key

`source_group_id = canonical_dataset + canonical_source_id` for public items. A
retrieve→compute derivative adds its generator version and nonce ID but retains a
`parent_group_id`; all derivatives of one parent are assigned together. Duplicate
Tool-Star rows group by normalized prompt hash.

## Allocation order

1. Download only pinned metadata/files.
2. Freeze public final IDs in `final_eval_manifest.json`.
3. Remove final groups from every training-side pool.
4. Hash `(seed, source_group_id)` and deterministically assign SFT-candidate,
   RL-candidate, Dev, or Internal groups to family quotas.
5. Only then generate candidates.

Dev and Internal each contain all four families; planned distributions follow the
same 20/30/30/20 ratio: Dev 40/60/60/40, Internal 80/120/120/80.

## Independence rules

- SFT candidates, RL candidates, Dev, and Internal are pairwise source-group
  disjoint.
- Final is disjoint by both source ID and normalized prompt.
- Teacher hints are candidate-generation metadata and never enter RL prompts.
- Internal/final predictions remain unopened until configuration freeze; failed
  runs do not authorize replacing IDs.

## Leakage checks and response

Run source-ID intersection, normalized exact matching, then token Jaccard ≥0.85.
Exact overlaps are removed from train-side data. High-Jaccard flags are manually
reviewed; ambiguous items are removed from train-side pools, never from final.
Counts and decisions are written to the lineage report. This is deliberately
lighter than MinHash/fuzzy clustering and adequate for this research pilot.

