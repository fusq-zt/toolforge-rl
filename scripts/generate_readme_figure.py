#!/usr/bin/env python3
"""根据已保存的评测结果生成中文首页主图，无需加载模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#8192AA", "#148B80"]
INK = "#23364D"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures_readme")
    args = parser.parse_args()

    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei"]
    font = next((name for name in fonts if name in available), None)
    if font is None:
        raise SystemExit("请先安装中文字体，例如 Noto Sans CJK SC，再生成图表。")
    plt.rcParams.update({
        "font.family": font, "font.size": 12, "axes.unicode_minus": False,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": "#66758A", "ytick.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42,
    })

    def read(name: str) -> dict:
        return json.loads((args.manifest_dir / name).read_text(encoding="utf-8"))

    vanilla = read("final_eval_vanilla_summary.json")["overall"]
    efficient = read("final_eval_efficient_summary.json")["overall"]
    paired = read("paired_bootstrap.json")
    if not vanilla["n"] == efficient["n"] == paired["episodes"]:
        raise ValueError("两组评测和统计报告的样本数不一致。")
    delta = 100 * (efficient["accuracy"] - vanilla["accuracy"])
    reduction = 100 * (1 - efficient["average_tool_calls"] / vanilla["average_tool_calls"])
    accuracy_ci = paired["confidence_intervals_95"]["accuracy_delta_pp"]
    calls_ci = [100 * x for x in paired["confidence_intervals_95"]["average_calls_reduction"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.25, top=0.65, wspace=0.55)
    fig.text(0.055, 0.92, "准确率接近，工具调用减少约 22%", fontsize=23, weight="bold")
    fig.text(0.055, 0.85, f"ToolForge-RL  ·  同一 SFT 起点  ·  {vanilla['n']:,} 道固定评测题", fontsize=11, color="#66758A")

    panels = [
        ("答案准确率", "越高越好", "accuracy", 100, 100, "{:.2f}%", f"差异 {delta:+.2f} 个百分点"),
        ("平均工具调用次数", "越低越好", "average_tool_calls", 1, 1.2, "{:.3f}", f"减少 {reduction:.2f}%"),
    ]
    for ax, (title, direction, key, scale, limit, fmt, note) in zip(axes, panels):
        values = [vanilla[key] * scale, efficient[key] * scale]
        ax.barh([1, 0], values, height=0.43, color=COLORS, zorder=3)
        ax.set_yticks([1, 0], ["标准 GRPO", "高效 GRPO"])
        ax.set_xlim(0, limit)
        ax.set_ylim(-0.65, 1.65)
        ax.set_title(title, loc="left", pad=24, fontsize=15, weight="bold")
        ax.text(1, 1.12, direction, transform=ax.transAxes, ha="right", fontsize=10, color="#66758A")
        ax.grid(axis="x", color="#E8EDF2", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, value in zip([1, 0], values):
            ax.text(value + limit * 0.035, y, fmt.format(value), va="center", fontsize=13, weight="bold")
        ax.text(0, -0.34, note, transform=ax.transAxes, fontsize=13, weight="bold", color=COLORS[1])
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axes[1].set_xticks([0, 0.3, 0.6, 0.9, 1.2])
    fig.text(0.055, 0.055,
             f"95% 置信区间：准确率差异 [{accuracy_ci[0]:+.2f}, {accuracy_ci[1]:+.2f}] 个百分点；"
             f"调用降幅 [{calls_ci[0]:.2f}%, {calls_ci[1]:.2f}%]。",
             fontsize=10, color="#66758A")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        path = args.output_dir / f"main_results_zh.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
