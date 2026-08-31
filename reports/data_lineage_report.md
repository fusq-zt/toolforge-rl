# Data Lineage Report

```text
Pinned public train assets (fixed revisions)
  -> normalized exact dedup (25 removed)
  -> source-ID split before generation (SFT 1100 / RL 800 / Dev 200 / Internal 400)
  -> held-out exact + light Jaccard audit (one MATH train item replaced)
  -> Tool-Star teacher, three sampling hints, real local tools
  -> programmatic answer verifier + protocol/execution filter
  -> 2517 verified candidates
  -> minimum-call selection + 12.5% deterministic alternative allowance
  -> 1076 project trajectories
  -> <=1000 provenance-labelled official Tool-Star references
  -> 2076 frozen SFT trajectories
```

The 600 Pilot candidates and 50 Pilot review rows are reused unchanged. Raw candidates are append-only and retained for failure analysis. Public held-out IDs were frozen before trajectory generation and never enter Teacher generation, SFT selection, or RL mining.
