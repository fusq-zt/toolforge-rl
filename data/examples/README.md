# Review examples and attribution

These files are small audit samples, not replacements for the upstream datasets.
They contain prompts, references, tool observations, and review annotations used
to inspect the ToolForge-RL data pipeline.

| File | Contents | Source and terms |
|---|---|---|
| `pilot_review_50.jsonl` | Gate 3 pilot subset, also present in the full review | Mixed sources listed per row |
| `full_review_100.jsonl` | Gate 4 structured semantic-review sample | Project synthetic rows plus GSM8K (MIT), HotpotQA (CC BY-SA 4.0), and 2WikiMultihopQA (upstream Apache-2.0 attribution) |
| `toolstar_official_sanity_100.jsonl` | Protocol sanity sample from the public Tool-Star SFT corpus | Tool-Star project/data attribution; MIT as recorded in the frozen source manifest |

The project MIT license applies to ToolForge-RL code and original annotations. It
does not replace the licenses of incorporated upstream text. Dataset/model IDs,
revisions, licenses, and known redistribution constraints are frozen in
`../manifests/source_manifest.json`, `../../NOTICE`, and
`../../reports/source_assets.md`.

The `manual_*` field names are historical schema names. Rows identify the actual
reviewer as `Codex semantic review`; they must not be interpreted as evidence of
independent human annotation.
