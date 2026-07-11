#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

MODE_MAPPING = {
    "few-shot": "FS",
    "cot": "CoT",
    "sd-no-comp": "SD",  # SD without complementary predicates
    "sd": "SD-C",  # SD with complementary predicates
    "sd-direct-no-comp": "SD-Direct",  # ablation without comp
    "sd-direct": "SD-Direct-C",  # ablation with comp
}

TASK_MAPPING = {
    "hearsay": "Hearsay",
    "method_application": "Method Application",
    "eligibility_nli": "Clinical Trial Eligibility",
}

MODEL_MAPPING = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5.2": "GPT-5.2",
    "o3": "o3",
    "claude-haiku": "Claude 4.5 Haiku",
    "claude-sonnet": "Claude 4.5 Sonnet",
    "deepseek-v3.2": "DeepSeek v3.2",
    "kimi-k2": "Kimi K2",
    "qwen-3": "Qwen 3",
}


def parse_percentage(val: str) -> float:
    if isinstance(val, str) and val.endswith("%"):
        return float(val.rstrip("%"))
    return float(val)


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        df[col] = df[col].apply(parse_percentage)

    df["Model"] = df["Model"].str.lower().str.strip()

    mask = (
        (df["Model"] == "claude-sonnet")
        & (df["Task"] == "hearsay")
        & (df["Mode"] == "sd-no-comp")
        & (df["F1"].round(1) == 79.5)
    )
    df.loc[mask, "Mode"] = "sd-direct-no-comp"

    dups = df.groupby(["Model", "Task", "Mode"]).size()
    if (dups > 1).any():
        print("Warning: Remaining duplicates found, taking first occurrence")
        df = df.drop_duplicates(subset=["Model", "Task", "Mode"], keep="first")

    return df


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    t_stat, p_val = stats.ttest_rel(x, y)
    d = cohens_d(x, y)
    df = len(x) - 1
    return t_stat, p_val, d, df


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def get_metric_by_mode(df: pd.DataFrame, mode: str, metric: str = "F1") -> pd.DataFrame:
    subset = df[df["Mode"] == mode][["Model", "Task", metric]].copy()
    subset["Model_Task"] = subset["Model"] + "_" + subset["Task"]
    return subset.set_index("Model_Task")[metric]


def get_metric_by_mode_and_task(
    df: pd.DataFrame, mode: str, task: str, metric: str = "F1"
) -> pd.Series:
    subset = df[(df["Mode"] == mode) & (df["Task"] == task)]
    return subset.set_index("Model")[metric]


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def compare_conditions(
    df: pd.DataFrame,
    mode1: str,
    mode2: str,
    metric: str = "F1",
    label1: str | None = None,
    label2: str | None = None,
):
    label1 = label1 or MODE_MAPPING.get(mode1, mode1)
    label2 = label2 or MODE_MAPPING.get(mode2, mode2)

    data1 = get_metric_by_mode(df, mode1, metric)
    data2 = get_metric_by_mode(df, mode2, metric)

    common_idx = data1.index.intersection(data2.index)
    x = data1.loc[common_idx].values
    y = data2.loc[common_idx].values

    t_stat, p_val, d, dof = paired_ttest(x, y)
    w_stat, w_p = stats.wilcoxon(x, y)
    diff = x - y
    mean_diff = np.mean(diff)
    ci_low, ci_high = stats.t.interval(0.95, dof, loc=mean_diff, scale=stats.sem(diff))

    print(f"\n{label1} vs {label2} ({metric}):")
    print(f"  {label1} mean: {np.mean(x):.1f}%")
    print(f"  {label2} mean: {np.mean(y):.1f}%")
    print(
        f"  Difference: {mean_diff:+.1f} percentage points, "
        f"95% CI [{ci_low:+.1f}, {ci_high:+.1f}]"
    )
    print(f"  t({dof}) = {t_stat:.2f}, p {fmt_p(p_val)}, Cohen's d = {d:.2f}")
    print(f"  Wilcoxon signed-rank: W = {w_stat:.1f}, p {fmt_p(w_p)}")
    print(f"  N comparisons: {len(common_idx)}")

    return t_stat, p_val, d, mean_diff


def task_level_analysis(df: pd.DataFrame, mode1: str, mode2: str):
    print("\nTask-Level Breakdown:")
    print("-" * 50)

    results = []
    for task in df["Task"].unique():
        data1 = get_metric_by_mode_and_task(df, mode1, task, "F1")
        data2 = get_metric_by_mode_and_task(df, mode2, task, "F1")

        common_models = data1.index.intersection(data2.index)
        x = data1.loc[common_models].values
        y = data2.loc[common_models].values

        t_stat, p_val = stats.ttest_rel(x, y)
        mean_diff = np.mean(x) - np.mean(y)

        n_improved = np.sum(x > y)
        n_total = len(common_models)

        task_name = TASK_MAPPING.get(task, task)
        print(f"  {task_name}:")
        print(
            f"    Δ F1: {mean_diff:+.1f}pp, "
            f"t({n_total - 1}) = {t_stat:.2f}, p {fmt_p(p_val)}"
        )
        print(
            f"    {MODE_MAPPING.get(mode1, mode1)} > {MODE_MAPPING.get(mode2, mode2)}: {n_improved}/{n_total} models"
        )

        results.append(
            {
                "task": task_name,
                "delta": mean_diff,
                "p_value": p_val,
                "improved": n_improved,
                "total": n_total,
            }
        )

    return results


def compute_aggregate_metrics(df: pd.DataFrame):
    print_separator("AGGREGATE METRICS")

    modes = ["few-shot", "cot", "sd-no-comp", "sd", "sd-direct-no-comp", "sd-direct"]

    print("\nMethod       | Acc.   | Prec.  | Recall | F1")
    print("-" * 50)

    for mode in modes:
        subset = df[df["Mode"] == mode]
        acc = subset["Accuracy"].mean()
        prec = subset["Precision"].mean()
        rec = subset["Recall"].mean()
        f1 = subset["F1"].mean()
        label = MODE_MAPPING.get(mode, mode)
        print(f"{label:12} | {acc:5.1f}% | {prec:5.1f}% | {rec:5.1f}% | {f1:.1f}%")


def main():
    print("Loading results.csv...")
    df = load_and_preprocess("results.csv")

    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['Model'].nunique()} unique")
    print(f"Tasks: {df['Task'].unique().tolist()}")
    print(f"Modes: {df['Mode'].unique().tolist()}")

    compute_aggregate_metrics(df)
    print_separator("MAIN STATISTICAL COMPARISONS")
    print("\n--- Primary Comparison: SD vs Few-Shot ---")
    compare_conditions(df, "sd-no-comp", "few-shot", "F1")
    print("\n--- SD vs Chain-of-Thought ---")
    compare_conditions(df, "sd-no-comp", "cot", "F1")
    print_separator("ABLATION STUDY: Symbolic Verification")
    print("\n--- SD vs SD-Direct (Symbolic Verification Contribution) ---")
    compare_conditions(df, "sd-no-comp", "sd-direct-no-comp", "F1", "SD", "SD-Direct")
    print("\nTask-Level Ablation (SD vs SD-Direct):")
    print("-" * 50)
    for task in df["Task"].unique():
        sd_data = get_metric_by_mode_and_task(df, "sd-no-comp", task, "F1")
        sd_direct_data = get_metric_by_mode_and_task(
            df, "sd-direct-no-comp", task, "F1"
        )

        common = sd_data.index.intersection(sd_direct_data.index)
        diff = sd_data.loc[common].mean() - sd_direct_data.loc[common].mean()

        task_name = TASK_MAPPING.get(task, task)
        print(
            f"  {task_name}: SD={sd_data.mean():.1f}%, SD-Direct={sd_direct_data.mean():.1f}%, Δ={diff:+.1f}pp"
        )

    print_separator("COMPLEMENTARY PREDICATES EFFECT")
    print("\n--- SD (no comp) vs SD-C (with comp) ---")
    compare_conditions(df, "sd-no-comp", "sd", "F1", "SD", "SD-C")
    print("\nTask-Level Complementary Predicates Effect:")
    print("-" * 50)
    for task in df["Task"].unique():
        sd_data = get_metric_by_mode_and_task(df, "sd-no-comp", task, "F1")
        sd_c_data = get_metric_by_mode_and_task(df, "sd", task, "F1")

        common = sd_data.index.intersection(sd_c_data.index)
        diff = sd_data.loc[common].mean() - sd_c_data.loc[common].mean()

        task_name = TASK_MAPPING.get(task, task)
        print(
            f"  {task_name}: SD={sd_data.mean():.1f}%, SD-C={sd_c_data.mean():.1f}%, Δ={diff:+.1f}pp"
        )

    print_separator("PRECISION-RECALL CHARACTERISTICS")

    print("\n--- Recall Comparison: SD vs Few-Shot ---")
    compare_conditions(df, "sd-no-comp", "few-shot", "Recall", "SD", "Few-Shot")

    print("\n--- Precision Comparison: SD vs Few-Shot ---")
    compare_conditions(df, "sd-no-comp", "few-shot", "Precision", "SD", "Few-Shot")

    print_separator("TASK-LEVEL ANALYSIS")

    task_level_analysis(df, "sd-no-comp", "few-shot")

    print_separator("MODEL PERFORMANCE SUMMARY")

    print("\nModels with SD > Few-Shot (by task):")
    print("-" * 50)

    for task in df["Task"].unique():
        sd_data = get_metric_by_mode_and_task(df, "sd-no-comp", task, "F1")
        fs_data = get_metric_by_mode_and_task(df, "few-shot", task, "F1")

        common = sd_data.index.intersection(fs_data.index)
        improvements = sd_data.loc[common] - fs_data.loc[common]

        n_improved = (improvements > 0).sum()
        n_total = len(common)
        avg_improvement = improvements.mean()

        task_name = TASK_MAPPING.get(task, task)
        print(
            f"  {task_name}: {n_improved}/{n_total} models improved, avg Δ = {avg_improvement:+.1f}pp"
        )

    print("\nModel-Level Average Improvement (SD - Few-Shot, F1):")
    print("-" * 50)

    model_improvements = []
    for model in df["Model"].unique():
        sd_data = df[(df["Model"] == model) & (df["Mode"] == "sd-no-comp")]["F1"].mean()
        fs_data = df[(df["Model"] == model) & (df["Mode"] == "few-shot")]["F1"].mean()
        diff = sd_data - fs_data
        model_improvements.append((MODEL_MAPPING.get(model, model), diff, sd_data))

    model_improvements.sort(key=lambda x: x[1], reverse=True)

    for model, diff, sd_avg in model_improvements:
        print(f"  {model:20}: {diff:+.1f}pp (SD avg: {sd_avg:.1f}%)")

    n_positive = sum(1 for _, d, _ in model_improvements if d > 0)
    print(
        f"\n  Total: {n_positive}/{len(model_improvements)} models showed improvement"
    )

    print_separator("F1 SCORES BY MODEL-TASK-CONDITION")

    print("\nHearsay:")
    print(f"{'Model':<20} | {'FS':>6} | {'CoT':>6} | {'SD':>6} | {'SD-C':>6}")
    print("-" * 55)
    for model in sorted(df["Model"].unique()):
        row = []
        for mode in ["few-shot", "cot", "sd-no-comp", "sd"]:
            val = df[
                (df["Model"] == model)
                & (df["Task"] == "hearsay")
                & (df["Mode"] == mode)
            ]["F1"]
            row.append(f"{val.iloc[0]:5.1f}" if len(val) > 0 else "  N/A")
        model_name = MODEL_MAPPING.get(model, model)
        print(
            f"{model_name:<20} | {row[0]:>6} | {row[1]:>6} | {row[2]:>6} | {row[3]:>6}"
        )

    print("\nMethod Application:")
    print(f"{'Model':<20} | {'FS':>6} | {'CoT':>6} | {'SD':>6} | {'SD-C':>6}")
    print("-" * 55)
    for model in sorted(df["Model"].unique()):
        row = []
        for mode in ["few-shot", "cot", "sd-no-comp", "sd"]:
            val = df[
                (df["Model"] == model)
                & (df["Task"] == "method_application")
                & (df["Mode"] == mode)
            ]["F1"]
            row.append(f"{val.iloc[0]:5.1f}" if len(val) > 0 else "  N/A")
        model_name = MODEL_MAPPING.get(model, model)
        print(
            f"{model_name:<20} | {row[0]:>6} | {row[1]:>6} | {row[2]:>6} | {row[3]:>6}"
        )

    print("\nClinical Trial Eligibility:")
    print(f"{'Model':<20} | {'FS':>6} | {'CoT':>6} | {'SD':>6} | {'SD-C':>6}")
    print("-" * 55)
    for model in sorted(df["Model"].unique()):
        row = []
        for mode in ["few-shot", "cot", "sd-no-comp", "sd"]:
            val = df[
                (df["Model"] == model)
                & (df["Task"] == "eligibility_nli")
                & (df["Mode"] == mode)
            ]["F1"]
            row.append(f"{val.iloc[0]:5.1f}" if len(val) > 0 else "  N/A")
        model_name = MODEL_MAPPING.get(model, model)
        print(
            f"{model_name:<20} | {row[0]:>6} | {row[1]:>6} | {row[2]:>6} | {row[3]:>6}"
        )


if __name__ == "__main__":
    main()
