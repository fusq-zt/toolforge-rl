# Resume Bullets

## 中文

**ToolForge-RL｜数据驱动的小模型工具推理强化学习**

- 基于 Qwen2.5-3B-Instruct 构建 LoRA SFT→GRPO 后训练链路，完成教师轨迹采样、真实 Python/BM25 工具执行、程序化验证、质量过滤、难度分级和 pass@4 RL Prompt Mining。
- 设计 correctness-gated group-relative efficiency reward，在相同初始化与 rollout 预算下，相比 Vanilla GRPO 将 Calls per Correct **降低 21.84%**，答案准确率变化 **-0.04 pp**；实验结论按冻结成功标准判定为 **PASS**。
- 实现受限 Python Sandbox、episode-local BM25、数学/QA Verifier、trajectory telemetry 与 5000 次 episode-level paired bootstrap，并在冻结 Internal + Public benchmark 上分析工具过用、欠调用和多步完成率。

## English

**ToolForge-RL — Data-Centric Tool Reasoning with Efficient GRPO**

- Built a Qwen2.5-3B LoRA SFT→GRPO pipeline covering teacher trajectory sampling, real Python/BM25 execution, programmatic verification, quality filtering, difficulty labeling, and pass@4 RL prompt mining.
- Designed a correctness-gated group-relative efficiency reward; under identical initialization and rollout budgets, **reduced calls per correct by 21.84%** with **-0.04 pp** accuracy change versus Vanilla GRPO. The frozen success criteria evaluate the outcome as **PASS**.
- Implemented a restricted Python sandbox, episode-local BM25, math/QA verifiers, trajectory telemetry, and 5000-sample episode-level paired bootstrap over frozen Internal and Public benchmarks.
