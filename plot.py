#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("Agg")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
    }
)

df = pd.read_csv("./results.csv")

for col in ["Accuracy", "Precision", "Recall", "F1"]:
    df[col] = df[col].str.rstrip("%").astype(float) / 100

model_names = {
    "qwen-3": "Qwen 3",
    "deepseek-v3.2": "DeepSeek V3.2",
    "kimi-k2": "Kimi K2",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-haiku": "Claude Haiku",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5-mini": "GPT-5 Mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-sonnet": "Claude Sonnet",
    "gpt-5.2": "GPT-5.2",
    "o3": "o3",
}

task_names = {
    "hearsay": "Hearsay",
    "method_application": "Method Application",
    "eligibility_nli": "Clinical Trial Eligibility",
}

mode_names = {
    "few-shot": "Few-Shot",
    "cot": "CoT",
    "sd-no-comp": "SD",
    "sd-direct-no-comp": "SD-Direct",
}

df["Model_Display"] = df["Model"].map(model_names)
df["Task_Display"] = df["Task"].map(task_names)
df["Mode_Display"] = df["Mode"].map(mode_names)

colors = {
    "few-shot": "#4575b4",
    "cot": "#74add1",
    "sd-no-comp": "#d73027",
    "sd-direct-no-comp": "#fdae61",
}


def create_improvement_chart():
    fig, ax = plt.subplots(figsize=(3.5, 5.5))

    tasks = ["hearsay", "method_application", "eligibility_nli"]
    models = list(model_names.keys())

    y_positions = []
    improvements = []
    baseline_vals = []
    sd_vals = []
    model_labels = []
    task_labels = []

    y_pos = 0
    for task in tasks:
        task_data = df[df["Task"] == task]
        for model in models:
            baseline = task_data[
                (task_data["Model"] == model) & (task_data["Mode"] == "few-shot")
            ]["F1"].values
            sd = task_data[
                (task_data["Model"] == model) & (task_data["Mode"] == "sd-no-comp")
            ]["F1"].values

            if len(baseline) > 0 and len(sd) > 0:
                y_positions.append(y_pos)
                baseline_vals.append(baseline[0])
                sd_vals.append(sd[0])
                improvements.append(sd[0] - baseline[0])
                model_labels.append(model_names[model])
                task_labels.append(task)
                y_pos += 1
        y_pos += 0.5

    for i in range(len(y_positions)):
        color = "#2ca02c" if improvements[i] > 0 else "#d62728"
        alpha = 0.7

        ax.plot(
            [baseline_vals[i], sd_vals[i]],
            [y_positions[i], y_positions[i]],
            color=color,
            alpha=alpha,
            linewidth=1.5,
            zorder=1,
        )

        ax.scatter(
            baseline_vals[i],
            y_positions[i],
            color=colors["few-shot"],
            s=30,
            zorder=2,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.scatter(
            sd_vals[i],
            y_positions[i],
            color=colors["sd-no-comp"],
            s=30,
            zorder=2,
            edgecolors="black",
            linewidth=0.3,
        )

        # +/- marker so direction survives grayscale print (reviewer request)
        sign = "+" if improvements[i] > 0 else "−"
        ax.text(
            max(baseline_vals[i], sd_vals[i]) + 0.018,
            y_positions[i],
            sign,
            color=color,
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=3,
        )

    task_boundaries = [10.5, 21.5]
    for boundary in task_boundaries:
        ax.axhline(y=boundary, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{model_labels[i]}" for i in range(len(y_positions))], fontsize=7
    )

    ax.text(
        1.02,
        5,
        "Hearsay",
        transform=ax.get_yaxis_transform(),
        fontsize=6,
        va="center",
        rotation=270,
    )
    ax.text(
        1.02,
        16,
        "Method Application",
        transform=ax.get_yaxis_transform(),
        fontsize=6,
        va="center",
        rotation=270,
    )
    ax.text(
        1.02,
        27.5,
        "Clinical Trial Eligibility",
        transform=ax.get_yaxis_transform(),
        fontsize=6,
        va="center",
        rotation=270,
    )

    ax.set_xlabel("F1 Score", fontsize=6)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Few-Shot → Structured Decomposition", fontweight="bold", fontsize=8)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["few-shot"],
            markersize=6,
            label="Few-Shot",
            markeredgecolor="black",
            markeredgewidth=0.3,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["sd-no-comp"],
            markersize=6,
            label="Struct. Decomp.",
            markeredgecolor="black",
            markeredgewidth=0.3,
        ),
        plt.Line2D([0], [0], color="#2ca02c", label="Improvement (+)", linewidth=1.5),
        plt.Line2D([0], [0], color="#d62728", label="Regression (−)", linewidth=1.5),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=6,
        frameon=True,
        columnspacing=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("./fig_improvement.png", format="png", dpi=300)
    plt.close()


def create_precision_recall():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)

    tasks = ["hearsay", "method_application", "eligibility_nli"]
    modes = ["few-shot", "cot", "sd-no-comp", "sd-direct-no-comp"]

    markers = {"few-shot": "o", "cot": "s", "sd-no-comp": "^", "sd-direct-no-comp": "D"}

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = df[df["Task"] == task]

        for f1 in [0.5, 0.6, 0.7, 0.8, 0.9]:
            recall_range = np.linspace(f1 / 2, 1.0, 100)
            precision_range = (f1 * recall_range) / (2 * recall_range - f1)
            valid = (precision_range > 0) & (precision_range <= 1)
            ax.plot(
                recall_range[valid],
                precision_range[valid],
                color="gray",
                alpha=0.3,
                linewidth=0.5,
                linestyle="--",
            )
            if f1 >= 0.6:
                ax.text(
                    0.99,
                    f1 + 0.02,
                    f"F1={f1}",
                    fontsize=6,
                    color="gray",
                    ha="right",
                    alpha=0.7,
                )

        for mode in modes:
            mode_data = task_data[task_data["Mode"] == mode]
            ax.scatter(
                mode_data["Recall"],
                mode_data["Precision"],
                c=colors[mode],
                marker=markers[mode],
                s=25,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.3,
                label=mode_names[mode],
            )

        ax.set_xlabel("Recall")
        ax.set_xlim(0.25, 1.05)
        ax.set_ylim(0.45, 1.05)
        ax.set_title(task_names[task].replace("\n", " "), fontweight="bold", fontsize=9)

        if idx == 0:
            ax.set_ylabel("Precision")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        fontsize=7,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("./fig_precision_recall.png", format="png", dpi=300)
    plt.close()


if __name__ == "__main__":
    create_improvement_chart()
    create_precision_recall()
