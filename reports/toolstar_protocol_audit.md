# Tool-Star protocol audit (Gate 1)

Status: implemented against frozen upstream commit
`df08f67a89b27feda425306cfe892d65f6569f9a` and both downloaded tokenizers.

## Wire format

Tool-Star uses an XML-like text protocol, not JSON schema or model-native function
calling. Its upstream system prompt defines these ordinary strings:

| Meaning | Model text | ToolForge internal name |
|---|---|---|
| private reasoning | `<think>...</think>` | not dispatched |
| retrieval request | `<search>...</search>` | `local_search` |
| Python request | `<python>...</python>` | `python_exec` |
| either tool observation | `<result>...</result>` | observation |
| final response | `<answer>...</answer>` | final |

The exact prompt is frozen in `toolforge_rl.protocols.toolstar.OFFICIAL_SYSTEM_PROMPT`.
Upstream generation stops after `</search>` or `</python>`, executes the requested
tool, appends a `<result>` block directly to the same assistant stream, and resumes
generation. ToolForge also stops on `</answer>` and imposes the preregistered total
budget of three calls across both tools.

## Tokenizer evidence

`Qwen2.5-3B-Instruct` and `Tool-Star-Qwen-3B` both report vocabulary size 151,643,
tokenizer length 151,665, the same Qwen chat template, and no tool tags in
`additional_special_tokens`. Every opening/closing tool tag encodes to multiple
ordinary tokens (for example `<search>` is three tokens). Therefore parsing is
triggered only by a complete closing string; token-ID assumptions are forbidden.

## Upstream behavior and deliberate adapter choices

- Upstream source contains similar but duplicated parsers in evaluation and reward
  code. ToolForge has one deterministic parser and one validator.
- A complete action is accepted only when its closing tag is the last non-whitespace
  text in the generated chunk. Earlier examples embedded in prose are not executed.
- Observations are length-bounded and any literal `</result>` inside tool output is
  escaped before insertion.
- The official prompt requests a final LaTeX `\boxed{}` answer. The extractor uses
  the last complete `<answer>` and a brace-balanced last box; it retains a raw-answer
  fallback for exact-string QA while the format reward can still penalize the absent
  box.
- Upstream allows three Python and three search calls independently. ToolForge v1
  intentionally uses the stricter preregistered total budget of three.
- `python_exec` and `local_search` are the only dispatchable tools. The word
  “wikipedia” remains in the frozen checkpoint prompt for compatibility, but search
  is redirected to episode-local evidence and never accesses the network.

## Official SFT sanity set

The pinned `Tool-Star-SFT-54K` file contains 53,971 rows with `instruction`, `input`,
and `output`. `scripts/select_official_sanity.py` deterministically selects 100 short,
complete, protocol-valid rows: 50 search trajectories and 50 Python trajectories,
each with one to three recorded calls. Selection is shortest-first with seeded digest
tie-breaking because this is an execution check rather than an accuracy benchmark.
The generated JSONL retains source indices and
reference action/result pairs so the sanity runner can build a closed local corpus.

The base-vs-Tool-Star generation comparison is written after the two-GPU run to
`reports/toolstar_protocol_sanity_results.md`; raw rollouts remain under `runs/gate1/`.
