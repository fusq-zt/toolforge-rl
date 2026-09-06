#!/usr/bin/env python3
"""根据已保存的评测结果生成中文效率与准确率散点图，无需加载模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
INK = "#253142"
MODEL_SPECS = [
    ("base", "基础模型", "#7A8793", "o"),
    ("base_efficient_prompt", "基础模型＋效率提示", "#A7B0B8", "s"),
    ("sft", "SFT", "#4C78A8", "D"),
    ("vanilla", "标准 GRPO", "#F28E2B", "^"),
    ("efficient", "高效 GRPO", "#2A9D8F", "*"),
    ("toolstar", "Tool-Star（参考）", "#8064A2", "P"),
]


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

    models = {
        slug: read(f"final_eval_{slug}_summary.json")["overall"]
        for slug, *_ in MODEL_SPECS
    }
    vanilla, efficient = models["vanilla"], models["efficient"]
    paired = read("paired_bootstrap.json")
    if any(model["n"] != paired["episodes"] for model in models.values()):
        raise ValueError("模型评测和统计报告的样本数不一致。")
    delta = 100 * (efficient["accuracy"] - vanilla["accuracy"])
    reduction = 100 * (1 - efficient["calls_per_correct"] / vanilla["calls_per_correct"])

    fig, ax = plt.subplots(figsize=(11.5, 8.1))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.125, top=0.91)
    label_positions = {
        "efficient": (0.23, 73.6), "vanilla": (1.15, 76.1),
        "sft": (2.18, 66.7), "toolstar": (2.18, 74.2),
    }
    for slug, label, color, marker in MODEL_SPECS:
        result = models[slug]
        point = (result["calls_per_correct"], result["accuracy"] * 100)
        ax.scatter(*point, s=85 + 850 * result["invalid_rate"],
                   color=color, marker=marker, edgecolor="white", linewidth=1.1, zorder=4)
        if slug in label_positions:
            ax.annotate(label, point, xytext=label_positions[slug], fontsize=12,
                        va="center", arrowprops={"arrowstyle": "-", "color": "#94A3B8", "linewidth": 1})
        else:
            offset = (-12, 13) if slug == "base_efficient_prompt" else (12, -2)
            ax.annotate(label, point, xytext=offset, textcoords="offset points",
                        ha="right" if slug == "base_efficient_prompt" else "left", va="center", fontsize=12)
    ax.add_patch(FancyArrowPatch(
        (vanilla["calls_per_correct"], vanilla["accuracy"] * 100),
        (efficient["calls_per_correct"], efficient["accuracy"] * 100),
        arrowstyle="-|>", mutation_scale=17, linewidth=2, color="#2A9D8F",
        connectionstyle="arc3,rad=-0.16", zorder=3,
    ))
    ax.text(0.35, 63.7, f"调用成本降低 {reduction:.2f}%\n准确率变化 {delta:+.2f} 个百分点",
            color="#26796F", fontsize=12, weight="bold", linespacing=1.6)
    ax.set_title("工具调用效率与准确率", loc="left", fontsize=21, weight="bold", pad=14)
    ax.set_xlabel("每答对一题的工具调用成本（越低越好）", labelpad=12)
    ax.set_ylabel("答案准确率（%，越高越好）", labelpad=12)
    ax.set_xlim(-0.2, 6.7)
    ax.set_ylim(18, 78.5)
    ax.set_xticks(range(7))
    ax.set_yticks(range(20, 80, 10))
    ax.grid(color="#E5E7EB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")
        ax.spines[side].set_linewidth(0.9)
    ax.text(0.99, 0.025, "标记越大，无效输出率越高", transform=ax.transAxes,
            ha="right", va="bottom", color="#6B7280", fontsize=10)
    fig.text(0.085, 0.025, f"固定评测：{vanilla['n']:,} 题  ·  调用成本 = 总调用次数 ÷ 答对题数",
             fontsize=10, color="#66758A")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        path = args.output_dir / f"main_results_zh.{extension}"
        fig.savefig(path, dpi=260, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
