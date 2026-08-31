# Tool-Star Protocol Sanity Plan

Gate 1 exists to discover the protocol from pinned artifacts, not to guess it.

## Inputs

- code revision: `RUC-NLPIR/Tool-Star@df08f67a89b27feda425306cfe892d65f6569f9a`;
- reference checkpoint: `dongguanting/Tool-Star-Qwen-3B@2350a1d6...`;
- base checkpoint: `Qwen/Qwen2.5-3B-Instruct@aa8e7253...`;
- 100 short, complete records selected from pinned Tool-Star-SFT-54K by a fixed
  seed after excluding malformed/truncated rows.

## Audit before generation

Record exact values and source file/line for:

1. system prompt and tool schemas;
2. tool-call start/end tokens and JSON grammar;
3. tool-response/observation rendering;
4. final-answer marker/extraction;
5. tokenizer added/special tokens;
6. chat template and generation prompt behavior;
7. official parser and loop termination logic;
8. official tool names and mapping to `python_exec`/`local_search`.

No marker written in this plan is treated as authoritative. The implementation
will expose only `parse_action`, `render_tool_result`, `extract_final_answer`, and
`validate_protocol`; discovered upstream syntax remains behind the adapter.

## Deterministic 100-record selection

- Load only the pinned `train` split.
- Keep records with a nonempty instruction/input/output and a complete final.
- Calculate a stable key from upstream row index and revision.
- Sort by rendered token length, take a seeded sample from the shorter 50% while
  preserving examples with zero, one, and multiple calls.
- Save selected upstream IDs and hashes in `data/manifests/protocol_sanity.json`.

This selection is protocol evidence only and is not a final benchmark.

## Two-model loop

For each selected record, run the same rendered prompt through the base and
reference checkpoints. A real loop must parse an action, map/execute the local
stubbed-equivalent tool, inject the actual observation, and continue until final
or a fixed stop. Never accept a model-authored tool response as observation.

Capture per model/record:

- final parse;
- tool-call parse;
- schema validity;
- execution success;
- continuation after observation;
- number/repetition of calls;
- termination reason and token count.

## Exit criteria

The reference checkpoint must satisfy:

- protocol parse ≥95%;
- final parse ≥95%;
- tool execution success ≥90%;
- loop/nontermination failure ≤5%.

Failure blocks data generation and training. Fix order is tokenizer/special tokens,
prompt/chat template, observation injection, then parser. The base model is a
comparison and is not required to meet reference thresholds.

## Outputs

- `src/toolforge/protocols/toolstar_protocol.py`;
- focused protocol/unit tests including prefix consistency;
- `reports/toolstar_protocol_audit.md`;
- `reports/official_checkpoint_sanity.md`;
- immutable raw predictions and a summary JSON under a Gate 1 run directory.

