# Tool and Verifier Specification

Only two local tools and deterministic programmatic verifiers are in scope.

## `python_exec(code: str) -> str`

Each call runs in a fresh subprocess with no carried state. The parent validates
source syntax, applies a small import/AST policy, launches with a minimal
environment, and enforces:

- no network/socket, shell, subprocess, dynamic `eval`/`exec`, or file I/O;
- allowlisted standard modules: `math`, `statistics`, `fractions`, `decimal`,
  `itertools`, and `collections`;
- 3-second wall timeout plus CPU and address-space limits where supported;
- bounded source, stdout/stderr, and return length;
- deterministic locale/hash seed;
- structured status (`ok`, `policy_error`, `runtime_error`, `timeout`,
  `output_limit`) separated from text output.

SymPy is not exposed as arbitrary import in version 1. If a verifier needs it,
the verifier calls a controlled host-side equivalence function.

Minimum evidence: 30 tests spanning normal arithmetic/statistics/collections,
syntax/runtime errors, all prohibited capabilities, timeout, memory/output bounds,
Unicode, and clean-state isolation.

## `local_search(query: str, top_k: int = 3) -> str`

The search index is built per episode from that episode's frozen context only.
BM25 tokenization, document order, tie-breaking, snippet extraction, top-k, and
output bounds are deterministic and versioned. Results contain document IDs,
titles, and snippets but never the reference answer field.

Minimum evidence: 20 tests for ranking, ties, case/punctuation, empty/long query,
top-k limits, repeated terms, Unicode, output bounds, episode isolation, and
identical-run determinism.

## Tool adapter and agent loop

`ToolNameAdapter` maps discovered upstream names to the two internal names; it
does not alter model weights. The loop permits at most three calls:

`render → generate → parse → validate → execute → inject actual observation → continue`.

Unknown tools, invalid schemas, timeouts, repeated identical calls, truncation,
missing final, and max-call exit are explicit termination/telemetry states.
Minimum evidence: 20 integration tests, including mixed search→compute paths.

## Verifiers

### GSM8K/numeric

Extract final only; strip presentation commas/currency/units; parse integer,
decimal, fraction, and percent when the reference permits; compare exact rational
where possible and use a documented small tolerance only for floats.

### MATH

Use `math-verify` first. A controlled SymPy fallback handles safe equivalent
expressions. Test boxed answers, fractions, radicals, polynomials, intervals,
sets, and equivalent forms. Parser failure is not a semantic match.

### HotpotQA/2Wiki

Lowercase, normalize Unicode/punctuation/articles/whitespace, then exact alias or
normalized match. No LLM judge and no fuzzy semantic credit.

### Retrieve→compute

The deterministic generator stores the calculation expression and numeric ground
truth separately from the prompt/context. The normal numeric verifier scores final;
lineage checks confirm all input facts came from the episode context.

Every verification record stores raw/normalized prediction, reference, boolean,
reason code, and verifier version. Tool execution success never substitutes for
answer correctness.

## Security posture

This is a research sandbox, not a hardened multi-tenant service. The design uses
process isolation and tests sufficient to run generated snippets on this dedicated
server, without adding containers, kernel modules, or a complex policy engine.

