# Twenty Proposed Episode Examples

These are synthetic design fixtures, not training/evaluation records and not evidence
of model performance. They exercise the schema without imposing a unique gold tool
path. Final correctness is programmatically checked.

### E01 — direct_anchor / arithmetic fact

- Prompt: “What is 9 × 7?”
- Episode context: none; both tools advertised.
- Reference/verifier: `63`, numeric exact.
- Plausible correct paths: direct final; Python is accepted but less efficient.
- Intended split: generated train-side fixture.

### E02 — direct_anchor / short comparison

- Prompt: “Which is larger, 0.8 or 3/4?”
- Context: none.
- Reference/verifier: `0.8`, normalized exact alias.
- Paths: direct or Python; zero-call correctness is preferred only by selection/reward.
- Intended split: train-side fixture.

### E03 — direct_anchor / language fact in prompt

- Prompt: “The note says the access code is BLUE-17. What is the access code?”
- Context: fact is in the prompt, not the search corpus.
- Reference/verifier: `BLUE-17`, normalized exact.
- Paths: direct; unnecessary calls remain legal but measurable.
- Intended split: train-side nonce.

### E04 — direct_anchor / list observation

- Prompt: “In `[pear, apple, plum]`, which item appears second?”
- Context: none.
- Reference/verifier: `apple`, normalized exact.
- Paths: direct or Python list indexing.
- Intended split: train-side fixture.

### E05 — direct_anchor / unit-preserving answer

- Prompt: “A timer shows 12 seconds and then 3 more seconds pass. How many seconds?”
- Context: none.
- Reference/verifier: `15`, numeric with optional unit.
- Paths: direct or Python.
- Intended split: train-side fixture.

### E06 — code_reasoning / multi-step word arithmetic

- Prompt: “A lab runs 18 batches with 24 samples each and discards 17 samples. How many remain?”
- Context: none.
- Reference/verifier: `415`, numeric exact.
- Paths: direct reasoning or `python_exec("18*24-17")`.
- Intended source: deterministic train-side analogue of GSM8K, not a held-out item.

### E07 — code_reasoning / fraction

- Prompt: “Compute `7/12 + 5/18` and give a simplified fraction.”
- Context: none.
- Reference/verifier: `31/36`, rational exact.
- Paths: direct fraction work or allowlisted `fractions` in Python.
- Intended split: train-side synthetic.

### E08 — code_reasoning / combinatorics

- Prompt: “How many unordered pairs can be chosen from 15 distinct sensors?”
- Context: none.
- Reference/verifier: `105`, numeric exact.
- Paths: direct formula or Python.
- Intended split: train-side synthetic.

### E09 — code_reasoning / descriptive statistics

- Prompt: “Find the median of `[4, 11, 6, 9, 3, 8]`.”
- Context: none.
- Reference/verifier: `7`, numeric exact.
- Paths: direct sort or `statistics.median`.
- Intended split: train-side synthetic.

### E10 — code_reasoning / equivalent algebra

- Prompt: “Solve `3(x-2)=21` for x.”
- Context: none.
- Reference/verifier: `x=9`, math equivalence.
- Paths: direct algebra or numeric Python.
- Intended source: train-side MATH-style fixture.

### E11 — retrieval_reasoning / single fact

- Prompt: “What instrument does Mira Vale play?”
- Episode documents: `D1: Mira Vale — Mira Vale performs on the cello.`;
  `D2: Rowan Pike — Rowan Pike builds violins.`
- Reference/verifier: `cello`, normalized exact.
- Paths: local search then final; a correct answer from prior knowledge would still
  score, but nonce names prevent that shortcut.
- Intended split: train-side nonce.

### E12 — retrieval_reasoning / two-hop person→city→river

- Prompt: “Which river crosses the city where Tal Ren was born?”
- Documents: `D1: Tal Ren was born in Norwick.`; `D2: Norwick is crossed by the River Elan.`;
  two distractors about other people/cities.
- Reference/verifier: `River Elan`, aliases `{Elan, River Elan}`.
- Plausible calls: one or two search queries; no prescribed query string.
- Intended split: train-side nonce.

### E13 — retrieval_reasoning / comparison

- Prompt: “Were the founders of Kestrel Labs and Amber Works born in the same country?”
- Documents: founder and birthplace facts for both companies plus distractors.
- Reference/verifier: `yes`, normalized aliases.
- Paths: one broad query or two focused searches.
- Intended split: train-side synthetic.

### E14 — retrieval_reasoning / title disambiguation

- Prompt: “Who directed the film *Silent Meridian*?”
- Documents: one film record naming director Ana Borel; another book with the same title;
  unrelated film snippets.
- Reference/verifier: `Ana Borel`, normalized exact.
- Paths: search query should disambiguate, but only final correctness is scored.
- Intended split: train-side nonce.

### E15 — retrieval_reasoning / chronology

- Prompt: “Which opened first, the Lark Museum or the North Archive?”
- Documents: Lark Museum opened 1987; North Archive opened 1992; distractors.
- Reference/verifier: `Lark Museum`, normalized exact.
- Paths: one/two search calls; Python is legal but unnecessary.
- Intended split: train-side nonce.

### E16 — retrieve_then_compute / difference

- Prompt: “How many more visitors did Orin Park receive in June than May?”
- Documents: Orin Park May visitors `12,480`; June visitors `15,205`; distractors for other parks.
- Reference/verifier: `2725`, numeric exact.
- Plausible path: search to obtain values, then direct subtraction or Python.
- Lineage assertion: both operands must exist in this episode's documents.
- Intended split: train-side deterministic generator.

### E17 — retrieve_then_compute / percentage

- Prompt: “What percentage of the 240 Aurora units were returned if the report lists 18 returns?”
- Documents: separate snippets for shipped units and returns; distractor quarter.
- Reference/verifier: `7.5%`, numeric percent.
- Paths: search then direct/Python; no unique-tool label.
- Intended split: train-side deterministic generator.

### E18 — retrieve_then_compute / weighted total

- Prompt: “Using the catalog quantities and prices, what is the combined cost of 7 Vela pins and 4 Nox clips?”
- Documents: Vela pin `$3.25`; Nox clip `$1.80`; unrelated product prices.
- Reference/verifier: `$29.95`, decimal/currency normalization.
- Paths: retrieve both prices, then direct or Python decimal arithmetic.
- Intended split: train-side nonce catalog.

### E19 — retrieve_then_compute / date gap

- Prompt: “How many years after the Alder Hall opened did the Beryl Annex open?”
- Documents: Alder Hall opened `1968`; Beryl Annex opened `1985`; distractors.
- Reference/verifier: `17`, numeric exact.
- Paths: search then subtract; extra calls counted but not statically forbidden.
- Intended split: train-side nonce.

### E20 — retrieve_then_compute / ratio

- Prompt: “The two reports give Sol team 84 points and Lumen team 63 points. Give the simplified Sol:Lumen ratio.”
- Documents: separate team report snippets with season nonce; distractor seasons.
- Reference/verifier: `4:3`, rational ratio normalization.
- Paths: retrieve season-matched values, simplify directly or via Python.
- Intended split: train-side deterministic generator.

## Fixture audit

- family counts: 5/5/5/5;
- no online API or business-operation tool;
- no example says a tool is uniquely required;
- nonce retrieval facts prevent memorized shortcuts without contaminating public final sets;
- all references are deterministically verifiable;
- none is drawn from a frozen public test item.

