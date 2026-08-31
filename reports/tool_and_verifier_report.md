# Gate 2 tool and verifier report

Status: PASS

## Implemented boundary

Only two tools are exposed:

- `python_exec`: a fresh `python -I -S` subprocess per call, AST/module policy,
  no file/network/process modules, private-attribute blocking, 3-second timeout,
  CPU/RAM limits, bounded stdout, and structured failures;
- `local_search`: episode-local BM25 only, deterministic input-order tie breaking,
  top-k 3 by default (hard cap 5), title/snippet output, and bounded observations.

The upstream names `python` and `search` are mapped to these tools without any
checkpoint modification. Tool output is inserted through the single upstream
`<result>` tag after escaping closing-tag injection.

The shared executable `AgentLoop` enforces three total calls, detects repeated
identical calls, stops on incomplete/invalid tags, records real execution and
latency, and validates the final transcript. Data generation, later rollout mining,
and evaluation use the same protocol functions.

## Verifiers

- GSM8K and synthetic retrieve-then-compute use normalized numeric comparison with
  commas/currency/units removed, fraction support, and `1e-6` absolute/relative
  tolerance.
- MATH/MATH500 use `math-verify` first and deterministic numeric fallback only when
  symbolic parsing is unavailable.
- HotpotQA/2Wiki use lowercase, punctuation/article/whitespace normalization and
  exact normalized reference/alias matching.
- No LLM judge is used by training, reward, or primary evaluation.

Every result records prediction, normalized prediction/reference, boolean decision,
and a machine-readable reason.

## Test evidence

Remote server run: 170/170 pytest items passed.

| Area | Collected items | Required | Result |
|---|---:|---:|---|
| Python sandbox | 36 | 30 | PASS |
| Local search | 23 | 20 | PASS |
| Agent loop | 23 | 20 | PASS |
| Reward | 42 | 40 | PASS |
| Protocol + verifier | 46 | focused coverage | PASS |

Safety cases include blocked `os`, `sys`, socket, subprocess, file IO, eval/exec,
private attributes and globals; success cases cover all allowlisted calculation
modules. Agent-loop tests cover direct, search, Python, search→Python, real
observation insertion, budget, repetition, failure, truncation, and final parsing.

`scripts/check_chat_prefix.py` separately checks both downloaded tokenizers: adding
tool actions and real observations must leave every already-tokenized history prefix
unchanged. Its machine-readable output is `data/manifests/chat_prefix_check.json`.
