# ToolForge-RL 新增图表使用建议

本目录包含 15 张新增分析图和 1 张总览图；每张均同时提供 260-dpi PNG 与矢量 PDF。

- `00_paper_figure_gallery`：全部新增图的快速总览。
- `11_efficiency_pareto_frontier`：最适合展示核心结论，直接呈现 Efficient GRPO 在几乎不损失准确率时降低工具调用。
- `16_paired_bootstrap_forest`：论文结果章节的统计主图，包含 5,000 次 paired bootstrap 的点估计与 95% CI。
- `23_experiment_storyboard`：适合汇报/PPT，一页覆盖 Accuracy–Cost、效率收益、任务族表现和数据规模。
- `12_task_family_accuracy_heatmap`、`13_dataset_accuracy_heatmap`、`22_difficulty_accuracy_heatmap`：适合分层结果和失败分析。
- `17_three_seed_robustness`：展示三个 seed 的准确率、效率和 Retrieve→Compute 稳定性。
- `18_data_construction_flow`、`19_rl_prompt_mining_composition`：适合方法章节，解释数据流水线和 RL Prompt Mining。
- `20_targeted_recovery_outcomes`：解释 1,024→2,048 token 的定向恢复效果。
- `24_temperature_probe`：解释为何选择 T=0.9。

所有数值直接读取冻结的 `data/manifests/*.json`，没有重新推理、修改测试集或手工填写结果。
