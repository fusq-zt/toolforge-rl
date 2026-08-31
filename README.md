# ToolForge-RL

Data-centric tool reasoning with Qwen2.5-3B-Instruct, executable trajectories, LoRA SFT, and a fair Vanilla-vs-Efficient GRPO comparison.

Status: **Gate 0 first-round audit**. No model or dataset has been downloaded and no training has started.

Frozen first implementation:

- model: `Qwen/Qwen2.5-3B-Instruct`;
- reference/teacher: `dongguanting/Tool-Star-Qwen-3B`;
- tools: deterministic `python_exec` and episode-local BM25 `local_search`;
- task families: direct, code reasoning, retrieval reasoning, retrieval-then-compute;
- main claim: correctness-gated group-relative efficiency reward can reduce redundant calls without materially reducing answer accuracy.

The authoritative first-round decisions are in `reports/`. Tool-Star is an external reference, not this project's result.

