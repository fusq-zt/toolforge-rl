#!/usr/bin/env python3
"""Generate publication-ready ToolForge-RL figures from frozen JSON manifests.

The script intentionally depends only on matplotlib and numpy. It does not load
models or regenerate predictions, and writes both high-resolution PNG and vector
PDF versions of every figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MODEL_SPECS = [
    ("Base", "base", "#7A8793", "o"),
    ("Base + prompt", "base_efficient_prompt", "#A7B0B8", "s"),
    ("SFT", "sft", "#4C78A8", "D"),
    ("Vanilla GRPO", "vanilla", "#F28E2B", "^"),
    ("Efficient GRPO", "efficient", "#2A9D8F", "*"),
    ("Tool-Star ref.", "toolstar", "#8064A2", "P"),
]

FAMILY_LABELS = {
    "direct_anchor": "Direct",
    "code_reasoning": "Code",
    "retrieval_reasoning": "Retrieval",
    "retrieve_then_compute": "Retrieve→Compute",
}

DATASET_LABELS = {
    "openai/gsm8k": "GSM8K",
    "EleutherAI/hendrycks_math": "MATH train",
    "HuggingFaceH4/MATH-500": "MATH-500",
    "hotpotqa/hotpot_qa": "HotpotQA",
    "framolfese/2WikiMultihopQA": "2Wiki",
    "toolforge/synthetic_rtc_v1": "Synthetic RTC",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#C9D1D9",
            "axes.linewidth": 0.8,
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "text.color": "#1F2937",
            "axes.labelcolor": "#374151",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111827",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png.name, pdf.name]


def annotate_hbars(ax: plt.Axes, bars, fmt: str, offset: float = 0.01) -> None:
    xmax = ax.get_xlim()[1]
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + xmax * offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color="#1F2937",
        )


def model_overview(models: list[dict], out: Path) -> list[str]:
    labels = [m["short"] for m in models]
    colors = [m["color"] for m in models]
    y = np.arange(len(models))[::-1]
    metrics = [
        ("Accuracy", [m["overall"]["accuracy"] * 100 for m in models], "{:.1f}%", (0, 80)),
        ("Average tool calls", [m["overall"]["average_tool_calls"] for m in models], "{:.2f}", (0, 1.65)),
        ("Calls per correct", [m["overall"]["calls_per_correct"] for m in models], "{:.2f}", (0, 6.8)),
        ("Invalid rate", [m["overall"]["invalid_rate"] * 100 for m in models], "{:.1f}%", (0, 26)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), constrained_layout=True)
    fig.suptitle("Final model matrix: quality and tool efficiency", fontsize=17, fontweight="semibold")
    for idx, (ax, (title, values, fmt, xlim)) in enumerate(zip(axes.flat, metrics)):
        bars = ax.barh(y, values, color=colors, height=0.62, zorder=3)
        ax.set_yticks(y, labels if idx % 2 == 0 else [""] * len(labels))
        ax.set_xlim(*xlim)
        ax.set_title(title, loc="left")
        clean_axis(ax)
        annotate_hbars(ax, bars, fmt)
        panel_label(ax, chr(ord("A") + idx))
    return save_figure(fig, out, "10_model_matrix_overview")


def pareto_frontier(models: list[dict], out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.5, 6.6), constrained_layout=True)
    cluster_label_positions = {
        "efficient": (0.28, 73.5, "left"),
        "vanilla": (1.25, 75.2, "left"),
        "sft": (2.18, 66.4, "left"),
        "toolstar": (2.18, 74.0, "left"),
    }
    for model in models:
        o = model["overall"]
        size = 85 + 850 * o["invalid_rate"]
        ax.scatter(
            o["calls_per_correct"],
            o["accuracy"] * 100,
            s=size,
            color=model["color"],
            marker=model["marker"],
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
        )
        if model["slug"] in cluster_label_positions:
            tx, ty, align = cluster_label_positions[model["slug"]]
            ax.annotate(
                model["short"],
                (o["calls_per_correct"], o["accuracy"] * 100),
                xytext=(tx, ty),
                textcoords="data",
                fontsize=9.5,
                ha=align,
                va="center",
                arrowprops=dict(arrowstyle="-", color="#94A3B8", linewidth=0.9),
            )
        else:
            ax.annotate(
                model["short"],
                (o["calls_per_correct"], o["accuracy"] * 100),
                xytext=(10, -2),
                textcoords="offset points",
                fontsize=9.5,
                ha="left",
                va="center",
            )
    vanilla = next(m for m in models if m["slug"] == "vanilla")
    efficient = next(m for m in models if m["slug"] == "efficient")
    va, ef = vanilla["overall"], efficient["overall"]
    ax.add_patch(
        FancyArrowPatch(
            (va["calls_per_correct"], va["accuracy"] * 100),
            (ef["calls_per_correct"], ef["accuracy"] * 100),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.8,
            color="#2A9D8F",
            connectionstyle="arc3,rad=-0.16",
            zorder=3,
        )
    )
    ax.text(0.30, 65.3, "−21.84% calls/correct\n−0.04 pp accuracy", color="#26796F", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Calls per correct answer  ← lower is better")
    ax.set_ylabel("Final answer accuracy (%)  → higher is better")
    ax.set_title("Efficiency–accuracy Pareto view", loc="left", fontsize=16)
    ax.set_xlim(-0.2, 6.7)
    ax.set_ylim(18, 77)
    clean_axis(ax, "both")
    ax.text(0.99, 0.02, "Marker size encodes invalid rate", transform=ax.transAxes, ha="right", va="bottom", color="#6B7280", fontsize=9)
    return save_figure(fig, out, "11_efficiency_pareto_frontier")


def heatmap(
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    stem: str,
    out: Path,
    cmap: str = "YlGnBu",
    vmin: float = 0,
    vmax: float = 100,
) -> list[str]:
    width = max(9.5, 1.2 * len(col_labels) + 4.0)
    height = max(5.2, 0.62 * len(row_labels) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="x", rotation=22)
    ax.set_title(title, loc="left", fontsize=16, pad=14)
    threshold = (vmin + vmax) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value > threshold else "#111827", fontsize=9, fontweight="semibold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.025)
    cbar.set_label("Accuracy (%)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save_figure(fig, out, stem)


def family_heatmap(models: list[dict], out: Path) -> list[str]:
    keys = list(FAMILY_LABELS)
    values = np.array([[m["summary"]["by_family"][k]["accuracy"] * 100 for k in keys] for m in models])
    return heatmap(values, [m["short"] for m in models], [FAMILY_LABELS[k] for k in keys], "Accuracy by task family", "12_task_family_accuracy_heatmap", out)


def dataset_heatmap(models: list[dict], out: Path) -> list[str]:
    keys = list(DATASET_LABELS)
    values = np.array([[m["summary"]["by_dataset"][k]["accuracy"] * 100 for k in keys] for m in models])
    return heatmap(values, [m["short"] for m in models], [DATASET_LABELS[k] for k in keys], "Accuracy by frozen dataset", "13_dataset_accuracy_heatmap", out)


def protocol_heatmap(models: list[dict], out: Path) -> list[str]:
    metrics = [
        ("final_parse_rate", "Final parse"),
        ("schema_valid_rate", "Schema valid"),
        ("tool_execution_success_rate", "Tool execution"),
        ("multi_turn_completion_rate", "Multi-turn"),
        ("invalid_rate", "1 − invalid"),
        ("truncation_rate", "1 − truncation"),
    ]
    rows = []
    for m in models:
        row = []
        for key, _ in metrics:
            value = m["overall"][key]
            if key in {"invalid_rate", "truncation_rate"}:
                value = 1 - value
            row.append(value * 100)
        rows.append(row)
    return heatmap(np.array(rows), [m["short"] for m in models], [label for _, label in metrics], "Protocol reliability (higher is better)", "14_protocol_reliability_heatmap", out, cmap="BuGn", vmin=45, vmax=100)


def suite_dumbbell(models: list[dict], out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    y = np.arange(len(models))[::-1]
    for yi, m in zip(y, models):
        internal = m["summary"]["by_evaluation_suite"]["internal_test"]["accuracy"] * 100
        public = m["summary"]["by_evaluation_suite"]["public_heldout"]["accuracy"] * 100
        ax.plot([internal, public], [yi, yi], color="#CBD5E1", linewidth=4, solid_capstyle="round", zorder=1)
        ax.scatter(internal, yi, color="#4C78A8", s=80, marker="o", edgecolor="white", zorder=3, label="Internal 400" if yi == y[0] else None)
        ax.scatter(public, yi, color="#2A9D8F", s=90, marker="D", edgecolor="white", zorder=3, label="Public 2,219" if yi == y[0] else None)
        ax.text(internal - 1.0, yi + 0.19, f"{internal:.1f}", ha="right", va="bottom", fontsize=9)
        ax.text(public + 1.0, yi - 0.19, f"{public:.1f}", ha="left", va="top", fontsize=9)
    ax.set_yticks(y, [m["short"] for m in models])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Internal vs public held-out generalization", loc="left", fontsize=16)
    ax.legend(loc="lower right", ncols=2)
    clean_axis(ax)
    return save_figure(fig, out, "15_internal_vs_public_accuracy")


def bootstrap_forest(bootstrap: dict, out: Path) -> list[str]:
    specs = [
        ("Accuracy Δ (pp)", "accuracy_delta_pp", 1.0),
        ("Average calls reduction", "average_calls_reduction", 100.0),
        ("Calls/correct reduction", "calls_per_correct_reduction", 100.0),
        ("Invalid-rate Δ (pp)", "invalid_rate_delta_pp", 1.0),
        ("Direct overuse reduction", "direct_unnecessary_call_reduction", 100.0),
        ("RTC accuracy Δ (pp)", "rtc_accuracy_delta_pp", 1.0),
    ]
    points = bootstrap["point_estimates"]
    cis = bootstrap["confidence_intervals_95"]
    y = np.arange(len(specs))[::-1]
    fig, ax = plt.subplots(figsize=(10.3, 6.1), constrained_layout=True)
    for yi, (label, key, scale) in zip(y, specs):
        point = points[key] * scale
        low, high = [x * scale for x in cis[key]]
        color = "#2A9D8F" if "reduction" in label.lower() else "#4C78A8"
        ax.plot([low, high], [yi, yi], color=color, linewidth=3, solid_capstyle="round")
        ax.scatter(point, yi, s=85, color=color, edgecolor="white", linewidth=1, zorder=3)
        ax.text(high + 0.8, yi, f"{point:+.2f}  [{low:+.2f}, {high:+.2f}]", va="center", fontsize=9)
    ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--")
    ax.set_yticks(y, [x[0] for x in specs])
    ax.set_xlabel("Effect size in native percentage units (5,000 paired resamples)")
    ax.set_xlim(-4, 38)
    ax.set_title("Paired-bootstrap effects: Efficient − Vanilla", loc="left", fontsize=16)
    clean_axis(ax)
    ax.text(0.99, 0.02, "Positive reduction = fewer calls; Δ metrics retain their sign", transform=ax.transAxes, ha="right", color="#6B7280", fontsize=9)
    return save_figure(fig, out, "16_paired_bootstrap_forest")


def seed_robustness(three_seed: dict, out: Path) -> list[str]:
    seeds = three_seed["seeds"]
    x = np.arange(len(seeds))
    labels = [str(s["seed"]) + (" · primary" if s["seed"] == three_seed["primary_seed"] else "") for s in seeds]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    colors = ["#2A9D8F" if s["status"] == "PASS" else "#E9A23B" for s in seeds]

    vals = [s["accuracy_delta_pp"] for s in seeds]
    axes[0].bar(x, vals, color=colors, width=0.62)
    axes[0].axhline(-2, color="#D1495B", linestyle="--", linewidth=1.2, label="−2 pp gate")
    axes[0].set_title("Accuracy delta")
    axes[0].set_ylabel("Efficient − Vanilla (pp)")
    axes[0].legend(loc="lower right")

    avg = [s["average_calls_reduction"] * 100 for s in seeds]
    cpc = [s["calls_per_correct_reduction"] * 100 for s in seeds]
    width = 0.34
    axes[1].bar(x - width / 2, avg, width, color="#4C78A8", label="Average calls")
    axes[1].bar(x + width / 2, cpc, width, color="#2A9D8F", label="Calls/correct")
    axes[1].axhline(10, color="#6B7280", linestyle="--", linewidth=1.2, label="10% gate")
    axes[1].set_title("Tool-efficiency reduction")
    axes[1].set_ylabel("Reduction (%)")
    axes[1].legend(loc="upper right")

    rtc = [s["rtc_accuracy_delta_pp"] for s in seeds]
    axes[2].bar(x, rtc, color=colors, width=0.62)
    axes[2].axhline(-3, color="#D1495B", linestyle="--", linewidth=1.2, label="−3 pp gate")
    axes[2].set_title("Retrieve→Compute delta")
    axes[2].set_ylabel("Efficient − Vanilla (pp)")
    axes[2].legend(loc="lower right")

    for i, ax in enumerate(axes):
        ax.set_xticks(x, labels, rotation=15)
        clean_axis(ax, "y")
        panel_label(ax, chr(ord("A") + i))
    fig.suptitle("Formal GRPO robustness across seeds", fontsize=17, fontweight="semibold")
    return save_figure(fig, out, "17_three_seed_robustness")


def data_flow(full: dict, rl: dict, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(14.5, 6.6), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("From public prompts to verified SFT and variation-bearing RL data", loc="left", fontsize=17, pad=10)

    nodes = [
        (0.4, 4.35, 2.2, 1.25, "Raw prompts", f"{full['raw_prompts']:,}\nsource-isolated", "#DCEAF7"),
        (3.1, 4.35, 2.2, 1.25, "Teacher candidates", f"{full['unique_candidates']:,}\nreal Agent Loop", "#D9F0EA"),
        (5.8, 4.35, 2.2, 1.25, "Verified", f"{full['verified_candidates']:,}\n{full['candidate_quality_pass_rate']:.1%} pass", "#CDECE3"),
        (8.5, 4.35, 2.2, 1.25, "SFT rows", f"{full['sft_actual_total']:,}\nno quality padding", "#C7DDF4"),
        (3.1, 1.25, 2.2, 1.25, "RL candidates", f"{rl['candidate_prompts']:,}\nfrozen prompts", "#FCE7C3"),
        (5.8, 1.25, 2.2, 1.25, "Pass@4 rollouts", f"{rl['rollouts']:,}\n4 per prompt", "#F9D7A4"),
        (8.5, 1.25, 2.2, 1.25, "RL selected", f"{rl['final_selected']:,}\nvariation-bearing", "#F4C982"),
    ]
    for x, y, w, h, title, value, color in nodes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=0, facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.16, y + h - 0.28, title, fontsize=11, fontweight="semibold", va="top")
        ax.text(x + 0.16, y + 0.22, value, fontsize=13, va="bottom", color="#1F2937")
    for a, b in [((2.62, 4.98), (3.06, 4.98)), ((5.32, 4.98), (5.76, 4.98)), ((8.02, 4.98), (8.46, 4.98)), ((5.32, 1.88), (5.76, 1.88)), ((8.02, 1.88), (8.46, 1.88))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, color="#64748B", linewidth=1.4))
    ax.add_patch(FancyArrowPatch((1.5, 4.32), (3.1, 2.5), arrowstyle="-|>", mutation_scale=15, color="#64748B", linewidth=1.4, connectionstyle="arc3,rad=0.12"))
    split_text = "Frozen split: SFT 1,100 · RL 800 · Dev 200 · Internal 400\nFinal public held-out 2,219 frozen before training · overlap 0"
    ax.text(11.1, 4.95, split_text, va="center", fontsize=10.5, color="#475569", bbox=dict(boxstyle="round,pad=0.45", facecolor="#F3F4F6", edgecolor="none"))
    ax.text(11.1, 1.9, f"RL signal: {rl['nonzero_reward_variance_ratio']:.1%} groups with reward variance\n{rl['final_selected']}/{rl['candidate_prompts']} prompts retained", va="center", fontsize=10.5, color="#475569", bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF4DF", edgecolor="none"))
    return save_figure(fig, out, "18_data_construction_flow")


def rl_mining(rl: dict, out: Path) -> list[str]:
    family_order = list(FAMILY_LABELS)
    selected = np.array([rl["selected_by_family"][k] for k in family_order])
    target = np.array([rl["target_by_family"][k] for k in family_order])
    y = np.arange(len(family_order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    axes[0].barh(y, target, color="#E5E7EB", height=0.68, label="Candidate quota")
    bars = axes[0].barh(y, selected, color="#2A9D8F", height=0.42, label="Selected")
    axes[0].set_yticks(y, [FAMILY_LABELS[k] for k in family_order])
    axes[0].set_xlabel("Prompts")
    axes[0].set_title("Selected prompts by family", loc="left")
    axes[0].legend(loc="lower right")
    clean_axis(axes[0])
    annotate_hbars(axes[0], bars, "{:.0f}", 0.015)
    panel_label(axes[0], "A")

    labels = ["Mixed correctness", "All-correct call variance", "Reward variance", "All wrong", "Constant group"]
    values = [
        rl["selection_types"]["mixed_correctness"],
        rl["selection_types"]["all_correct_call_variance"],
        rl["selection_types"]["reward_variance"],
        rl["excluded"]["all_wrong"],
        rl["excluded"]["constant_group"],
    ]
    colors = ["#2A9D8F", "#69B3A2", "#A8D8CD", "#E9A23B", "#C9CED6"]
    bars2 = axes[1].barh(np.arange(len(labels))[::-1], values, color=colors, height=0.62)
    axes[1].set_yticks(np.arange(len(labels))[::-1], labels)
    axes[1].set_xlabel("Prompt groups")
    axes[1].set_title("Why prompts were retained or excluded", loc="left")
    clean_axis(axes[1])
    annotate_hbars(axes[1], bars2, "{:.0f}", 0.015)
    panel_label(axes[1], "B")
    fig.suptitle("RL prompt mining: select learning signal, not quota filler", fontsize=17, fontweight="semibold")
    return save_figure(fig, out, "19_rl_prompt_mining_composition")


def recovery_chart(recovery: dict, out: Path) -> list[str]:
    runs = recovery["runs"]
    labels = [r["label"].replace(" reference", " ref.") for r in runs]
    recovered = np.array([r["recovery_rows"] - r["remaining_incomplete"] for r in runs])
    remaining = np.array([r["remaining_incomplete"] for r in runs])
    y = np.arange(len(runs))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    ax.barh(y, recovered, color="#2A9D8F", height=0.62, label="Completed by 2,048-token retry")
    ax.barh(y, remaining, left=recovered, color="#E7B454", height=0.62, label="Still incomplete after one retry")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Episodes initially incomplete at 1,024 tokens")
    ax.set_title("Targeted long-output recovery", loc="left", fontsize=16)
    clean_axis(ax)
    ax.legend(loc="lower right")
    for yi, r, ok in zip(y, runs, recovered):
        rate = ok / r["recovery_rows"] if r["recovery_rows"] else 0
        ax.text(r["recovery_rows"] + 10, yi, f"{rate:.0%} recovered", va="center", fontsize=9)
    ax.set_xlim(0, max(r["recovery_rows"] for r in runs) * 1.18)
    return save_figure(fig, out, "20_targeted_recovery_outcomes")


def sft_progress(sft: dict, out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)
    acc = [sft["base_dev_accuracy"] * 100, sft["sft_dev"]["overall"]["accuracy"] * 100]
    bars = axes[0].bar(["Base", "SFT + bounded\nrecovery"], acc, color=["#A7B0B8", "#4C78A8"], width=0.58)
    axes[0].set_ylabel("Dev accuracy (%)")
    axes[0].set_ylim(0, 85)
    axes[0].set_title("Dev capability gain")
    axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in acc], padding=4, fontsize=10)

    phases = ["Smoke", "Full 1 epoch", "Recovery 0.5"]
    losses = [sft["smoke"]["train_loss"], sft["full"]["train_loss"], sft["bounded_recovery"]["train_loss"]]
    bars2 = axes[1].bar(phases, losses, color=["#A7B0B8", "#4C78A8", "#2A9D8F"], width=0.6)
    axes[1].set_ylabel("Train loss")
    axes[1].set_title("Bounded training stages")
    axes[1].bar_label(bars2, fmt="%.3f", padding=4, fontsize=9)
    axes[1].tick_params(axis="x", rotation=15)

    family_order = list(FAMILY_LABELS)
    fam_acc = [sft["sft_dev"]["by_family"][k]["accuracy"] * 100 for k in family_order]
    bars3 = axes[2].barh(np.arange(4)[::-1], fam_acc, color="#2A9D8F", height=0.6)
    axes[2].set_yticks(np.arange(4)[::-1], [FAMILY_LABELS[k] for k in family_order])
    axes[2].set_xlabel("Accuracy (%)")
    axes[2].set_xlim(0, 100)
    axes[2].set_title("Final SFT Dev by family")
    annotate_hbars(axes[2], bars3, "{:.1f}%", 0.01)

    for i, ax in enumerate(axes):
        clean_axis(ax, "y")
        panel_label(ax, chr(ord("A") + i))
    fig.suptitle("SFT: one main epoch plus one bounded protocol recovery", fontsize=17, fontweight="semibold")
    return save_figure(fig, out, "21_sft_training_and_dev_gain")


def difficulty_heatmap(models: list[dict], out: Path) -> list[str]:
    keys = ["easy", "medium", "hard"]
    values = np.array([[m["summary"]["by_difficulty"][k]["accuracy"] * 100 for k in keys] for m in models])
    return heatmap(values, [m["short"] for m in models], [x.title() for x in keys], "Accuracy by empirical difficulty", "22_difficulty_accuracy_heatmap", out)


def final_storyboard(models: list[dict], bootstrap: dict, full: dict, out: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
    ax = axes[0, 0]
    for m in models:
        o = m["overall"]
        ax.scatter(o["average_tool_calls"], o["accuracy"] * 100, s=105, color=m["color"], marker=m["marker"], edgecolor="white", linewidth=1)
        ax.annotate(m["short"], (o["average_tool_calls"], o["accuracy"] * 100), xytext=(5, 5), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("Average tool calls")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy–cost frontier", loc="left")
    clean_axis(ax, "both")
    panel_label(ax, "A")

    ax = axes[0, 1]
    names = ["Avg calls", "Calls/correct", "Direct overuse"]
    vals = [bootstrap["point_estimates"]["average_calls_reduction"] * 100, bootstrap["point_estimates"]["calls_per_correct_reduction"] * 100, bootstrap["point_estimates"]["direct_unnecessary_call_reduction"] * 100]
    bars = ax.barh(np.arange(3)[::-1], vals, color=["#4C78A8", "#2A9D8F", "#69B3A2"], height=0.58)
    ax.axvline(10, color="#6B7280", linestyle="--", linewidth=1.1, label="10% efficiency gate")
    ax.set_yticks(np.arange(3)[::-1], names)
    ax.set_xlabel("Reduction (%)")
    ax.set_xlim(0, 28)
    ax.set_title("Efficient GRPO effect", loc="left")
    ax.legend(loc="lower right")
    annotate_hbars(ax, bars, "{:.1f}%", 0.012)
    clean_axis(ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    vanilla = next(m for m in models if m["slug"] == "vanilla")
    efficient = next(m for m in models if m["slug"] == "efficient")
    fams = list(FAMILY_LABELS)
    x = np.arange(len(fams))
    width = 0.36
    v = [vanilla["summary"]["by_family"][k]["accuracy"] * 100 for k in fams]
    e = [efficient["summary"]["by_family"][k]["accuracy"] * 100 for k in fams]
    ax.bar(x - width / 2, v, width, color="#F28E2B", label="Vanilla")
    ax.bar(x + width / 2, e, width, color="#2A9D8F", label="Efficient")
    ax.set_xticks(x, [FAMILY_LABELS[k] for k in fams], rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy preserved across task families", loc="left")
    ax.legend(loc="lower right")
    clean_axis(ax, "y")
    panel_label(ax, "C")

    ax = axes[1, 1]
    counts = [full["raw_prompts"], full["unique_candidates"], full["verified_candidates"], full["sft_actual_total"]]
    labels = ["Raw prompts", "Candidates", "Verified", "SFT rows"]
    colors = ["#A7B0B8", "#4C78A8", "#2A9D8F", "#8064A2"]
    bars = ax.bar(labels, counts, color=colors, width=0.62)
    ax.bar_label(bars, labels=[f"{x:,}" for x in counts], padding=4, fontsize=9)
    ax.set_ylabel("Rows")
    ax.set_title("Data-centric construction scale", loc="left")
    ax.tick_params(axis="x", rotation=15)
    clean_axis(ax, "y")
    panel_label(ax, "D")

    fig.suptitle("ToolForge-RL experiment at a glance", fontsize=18, fontweight="semibold")
    return save_figure(fig, out, "23_experiment_storyboard")


def temperature_probe(temp: dict, out: Path) -> list[str]:
    candidates = temp["candidates"]
    ts = np.array(sorted(float(k) for k in candidates))
    acc = np.array([candidates[str(t)]["accuracy"] * 100 for t in ts])
    variance = np.array([candidates[str(t)]["nonzero_group_reward_variance_ratio"] * 100 for t in ts])
    invalid = np.array([candidates[str(t)]["invalid_rate"] * 100 for t in ts])
    trunc = np.array([candidates[str(t)]["truncation_rate"] * 100 for t in ts])
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    for ax in axes:
        ax.axvline(temp["selected_temperature"], color="#2A9D8F", linewidth=6, alpha=0.12)
        ax.set_xticks(ts)
        ax.set_xlabel("Sampling temperature")
        clean_axis(ax, "y")
    axes[0].plot(ts, acc, color="#4C78A8", marker="o", linewidth=2.2)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Rollout accuracy")
    axes[1].plot(ts, variance, color="#8064A2", marker="D", linewidth=2.2)
    axes[1].axhline(30, color="#6B7280", linestyle="--", linewidth=1.1)
    axes[1].set_ylabel("Groups with reward variance (%)")
    axes[1].set_title("Learning-signal diversity")
    axes[2].plot(ts, invalid, color="#D1495B", marker="o", linewidth=2.2, label="Invalid")
    axes[2].plot(ts, trunc, color="#E9A23B", marker="s", linewidth=2.2, label="Truncation")
    axes[2].axhline(10, color="#6B7280", linestyle="--", linewidth=1.1, label="Applied 10% ceiling")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_title("Rollout failure rates")
    axes[2].legend(loc="lower right")
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord("A") + i))
    fig.suptitle("Temperature probe: T=0.9 balances accuracy and reward variance", fontsize=17, fontweight="semibold")
    return save_figure(fig, out, "24_temperature_probe")


def make_gallery(output_dir: Path, pngs: list[str]) -> list[str]:
    selected = [x for x in pngs if not x.startswith("00_")]
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes).flat
    for ax, name in zip(axes, selected):
        ax.imshow(mpimg.imread(output_dir / name))
        ax.set_title(name.removesuffix(".png").replace("_", " "), fontsize=10, loc="left")
        ax.axis("off")
    for ax in list(axes)[len(selected):]:
        ax.axis("off")
    fig.suptitle("ToolForge-RL paper figure gallery", fontsize=18, fontweight="semibold")
    return save_figure(fig, output_dir, "00_paper_figure_gallery")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()

    models = []
    for short, slug, color, marker in MODEL_SPECS:
        summary = load_json(args.manifest_dir / f"final_eval_{slug}_summary.json")
        models.append({"short": short, "slug": slug, "color": color, "marker": marker, "summary": summary, "overall": summary["overall"]})
    bootstrap = load_json(args.manifest_dir / "paired_bootstrap.json")
    three_seed = load_json(args.manifest_dir / "grpo_formal_three_seed.json")
    full = load_json(args.manifest_dir / "full_data_summary.json")
    rl = load_json(args.manifest_dir / "rl_manifest.json")
    recovery = load_json(args.manifest_dir / "final_evaluation_recovery.json")
    sft = load_json(args.manifest_dir / "sft_gate_summary.json")
    temp = load_json(args.manifest_dir / "temperature_selection.json")

    generated: list[str] = []
    for maker in [
        lambda: model_overview(models, args.output_dir),
        lambda: pareto_frontier(models, args.output_dir),
        lambda: family_heatmap(models, args.output_dir),
        lambda: dataset_heatmap(models, args.output_dir),
        lambda: protocol_heatmap(models, args.output_dir),
        lambda: suite_dumbbell(models, args.output_dir),
        lambda: bootstrap_forest(bootstrap, args.output_dir),
        lambda: seed_robustness(three_seed, args.output_dir),
        lambda: data_flow(full, rl, args.output_dir),
        lambda: rl_mining(rl, args.output_dir),
        lambda: recovery_chart(recovery, args.output_dir),
        lambda: sft_progress(sft, args.output_dir),
        lambda: difficulty_heatmap(models, args.output_dir),
        lambda: final_storyboard(models, bootstrap, full, args.output_dir),
        lambda: temperature_probe(temp, args.output_dir),
    ]:
        generated.extend(maker())
    pngs = [x for x in generated if x.endswith(".png")]
    generated.extend(make_gallery(args.output_dir, pngs))

    manifest = {
        "schema_version": 1,
        "source": "Frozen ToolForge-RL Gate 7 manifests",
        "manifest_dir": str(args.manifest_dir),
        "figure_count": len([x for x in generated if x.endswith(".png")]),
        "files": generated,
        "formats": ["PNG 260 dpi", "vector PDF"],
        "notes": [
            "No model inference or prediction regeneration was performed.",
            "All values are read from frozen JSON manifests.",
            "Model colors are held constant across figures.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# ToolForge-RL Paper Figures

This directory contains publication/PPT-oriented views derived only from the frozen Gate 7 JSON manifests. Each numbered chart is available as a high-resolution PNG and a vector PDF. `00_paper_figure_gallery.png` is a visual index; use the numbered files for papers and slides.

Regenerate with:

```bash
python scripts/generate_paper_figures.py --manifest-dir data/manifests --output-dir reports/figures_paper
```
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
