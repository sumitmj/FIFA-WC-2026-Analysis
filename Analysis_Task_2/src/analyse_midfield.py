#!/usr/bin/env python3
"""Objective 1 analysis of 2026 World Cup midfield tackle rates."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


SEED = 2026
SAMPLE_PER_STRATUM = 50
REPETITIONS = 10_000
STAGES = ["Group-stage exit", "Knockout stage"]
COLORS = {STAGES[0]: "#F59E0B", STAGES[1]: "#0F9D8A"}
NAVY = "#14213D"
RED = "#C1121F"
MUTED = "#64748B"
GRID = "#D8E0E8"

SOURCE_COLUMNS = [
    "player",
    "position",
    "team",
    "age",
    "minutes_90s",
    "fouls",
    "fouled",
    "interceptions",
    "tackles_won",
    "stage",
    "team_90s",
]
NUMERIC_COLUMNS = [
    "age",
    "minutes_90s",
    "fouls",
    "fouled",
    "interceptions",
    "tackles_won",
    "team_90s",
]


def load_and_prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame[SOURCE_COLUMNS].copy()
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    frame[["player", "position", "team", "stage"]] = frame[
        ["player", "position", "team", "stage"]
    ].apply(lambda column: column.astype(str).str.strip())

    derived_stage = np.where(frame["team_90s"] > 3, STAGES[1], STAGES[0])
    if not np.array_equal(frame["stage"].to_numpy(), derived_stage):
        raise ValueError("Stage labels do not match the squad-90s rule")

    frame["minutes"] = (frame["minutes_90s"] * 90).round().astype(int)
    frame["tackles_won_per90"] = frame["tackles_won"] / frame["minutes_90s"]
    frame["interceptions_per90"] = frame["interceptions"] / frame["minutes_90s"]
    frame["fouls_per90"] = frame["fouls"] / frame["minutes_90s"]
    frame["fouled_per90"] = frame["fouled"] / frame["minutes_90s"]
    frame["stage"] = pd.Categorical(frame["stage"], categories=STAGES, ordered=True)
    frame = frame.sort_values(["stage", "player", "team"]).reset_index(drop=True)
    validate_population(frame)
    return frame


def validate_population(frame: pd.DataFrame) -> None:
    counts = {stage: int((frame["stage"] == stage).sum()) for stage in STAGES}
    rules = {
        "eligible rows": len(frame) == 240,
        "stage counts": counts == {STAGES[0]: 60, STAGES[1]: 180},
        "48 teams": frame["team"].nunique() == 48,
        "unique player-team rows": not frame.duplicated(["player", "team"]).any(),
        "no missing source values": not frame[SOURCE_COLUMNS].isna().any().any(),
        "midfield positions": frame["position"].str.contains("MF", regex=False).all(),
        "minimum 180 minutes": frame["minutes_90s"].ge(2).all(),
        "finite outcome": np.isfinite(frame["tackles_won_per90"]).all(),
    }
    failed = [name for name, passed in rules.items() if not passed]
    if failed:
        raise ValueError(f"Population validation failed: {failed}")


def build_data_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("eligible_rows", len(frame), 240),
        ("unique_player_team_rows", frame[["player", "team"]].drop_duplicates().shape[0], 240),
        ("missing_primary_outcomes", frame["tackles_won_per90"].isna().sum(), 0),
        ("minimum_minutes", frame["minutes"].min(), 180),
        ("group_exit_population", (frame["stage"] == STAGES[0]).sum(), 60),
        ("knockout_population", (frame["stage"] == STAGES[1]).sum(), 180),
        ("national_teams", frame["team"].nunique(), 48),
    ]
    result = pd.DataFrame(rows, columns=["check", "observed", "expected"])
    result["passed"] = result["observed"] == result["expected"]
    return result


def draw_stratified_sample(
    population: pd.DataFrame,
    seed: int = SEED,
    sample_per_stratum: int = SAMPLE_PER_STRATUM,
) -> pd.DataFrame:
    rng = random.Random(seed)
    pieces = []
    for stage in STAGES:
        stratum = population.loc[population["stage"] == stage].sort_values(["player", "team"])
        if sample_per_stratum > len(stratum):
            raise ValueError(f"Cannot sample {sample_per_stratum} players from {stage}")
        pieces.append(population.loc[rng.sample(list(stratum.index), sample_per_stratum)])

    sample = pd.concat(pieces).sort_values(["stage", "player", "team"]).reset_index(drop=True)
    population_sizes = population["stage"].value_counts().to_dict()
    sample["sample_id"] = [f"WC26-MF-S{number:03d}" for number in range(1, len(sample) + 1)]
    sample["selection_probability"] = sample["stage"].map(
        {stage: sample_per_stratum / population_sizes[stage] for stage in STAGES}
    ).astype(float)
    sample["sample_weight"] = 1 / sample["selection_probability"]
    return sample


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    standard_error = float(stats.sem(array))
    margin = float(stats.t.ppf(0.975, len(array) - 1)) * standard_error
    return mean, mean - margin, mean + margin, standard_error


def build_descriptive_statistics(
    sample: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        group = sample.loc[sample["stage"] == stage]
        values = group["tackles_won_per90"]
        mean, low, high, standard_error = mean_ci(values)
        population_n = int((population["stage"] == stage).sum())
        rows.append(
            {
                "stage": stage,
                "population_N": population_n,
                "sample_n": len(group),
                "sampling_fraction": len(group) / population_n,
                "mean": mean,
                "mean_standard_error": standard_error,
                "mean_95ci_low": low,
                "mean_95ci_high": high,
                "median": values.median(),
                "standard_deviation": values.std(ddof=1),
                "variance": values.var(ddof=1),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "minimum": values.min(),
                "maximum": values.max(),
                "skewness": stats.skew(values, bias=False),
                "sample_minutes": int(round(group["minutes_90s"].sum() * 90)),
                "aggregate_tackles_per90": group["tackles_won"].sum()
                / group["minutes_90s"].sum(),
            }
        )
    return pd.DataFrame(rows)


def welch_test(
    knockout_values: pd.Series | np.ndarray,
    exit_values: pd.Series | np.ndarray,
) -> dict[str, float]:
    knockout = np.asarray(knockout_values, dtype=float)
    exits = np.asarray(exit_values, dtype=float)
    n1, n0 = len(knockout), len(exits)
    variance1, variance0 = knockout.var(ddof=1), exits.var(ddof=1)
    component1, component0 = variance1 / n1, variance0 / n0
    standard_error = math.sqrt(component1 + component0)
    degrees_freedom = (component1 + component0) ** 2 / (
        component1**2 / (n1 - 1) + component0**2 / (n0 - 1)
    )
    difference = float(knockout.mean() - exits.mean())
    margin = float(stats.t.ppf(0.975, degrees_freedom)) * standard_error
    test = stats.ttest_ind(knockout, exits, equal_var=False)
    pooled_sd = math.sqrt(
        ((n1 - 1) * variance1 + (n0 - 1) * variance0) / (n1 + n0 - 2)
    )
    cohens_d = difference / pooled_sd
    correction = 1 - 3 / (4 * (n1 + n0 - 2) - 1)
    return {
        "difference": difference,
        "standard_error": standard_error,
        "ci_low": difference - margin,
        "ci_high": difference + margin,
        "t_statistic": float(test.statistic),
        "degrees_freedom": degrees_freedom,
        "p_value_two_sided": float(test.pvalue),
        "cohens_d": cohens_d,
        "hedges_g": correction * cohens_d,
    }


def robustness_checks(
    knockout_values: pd.Series,
    exit_values: pd.Series,
    seed: int = SEED,
    repetitions: int = REPETITIONS,
) -> dict[str, float]:
    knockout = knockout_values.to_numpy(dtype=float)
    exits = exit_values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed + 1)
    bootstrap_difference = (
        rng.choice(knockout, (repetitions, len(knockout)), replace=True).mean(axis=1)
        - rng.choice(exits, (repetitions, len(exits)), replace=True).mean(axis=1)
    )
    bootstrap_low, bootstrap_high = np.quantile(bootstrap_difference, [0.025, 0.975])

    observed = abs(knockout.mean() - exits.mean())
    pooled = np.concatenate([knockout, exits])
    permutation_rng = np.random.default_rng(seed + 2)
    extreme = 0
    for _ in range(repetitions):
        shuffled = permutation_rng.permutation(pooled)
        difference = abs(shuffled[: len(knockout)].mean() - shuffled[len(knockout) :].mean())
        extreme += difference >= observed
    return {
        "bootstrap_ci_low": float(bootstrap_low),
        "bootstrap_ci_high": float(bootstrap_high),
        "permutation_p_two_sided": (extreme + 1) / (repetitions + 1),
    }


def build_assumption_tests(sample: pd.DataFrame) -> pd.DataFrame:
    arrays = {
        stage: sample.loc[sample["stage"] == stage, "tackles_won_per90"].to_numpy()
        for stage in STAGES
    }
    rows = []
    for stage, values in arrays.items():
        result = stats.shapiro(values)
        rows.append(("Shapiro-Wilk normality", stage, result.statistic, result.pvalue))
    variance_result = stats.levene(arrays[STAGES[0]], arrays[STAGES[1]], center="median")
    rows.append(
        ("Brown-Forsythe variance check", "Both groups", variance_result.statistic, variance_result.pvalue)
    )
    return pd.DataFrame(rows, columns=["diagnostic", "group", "statistic", "p_value"])


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sample_design(population: pd.DataFrame, sample: pd.DataFrame, path: Path) -> None:
    population_counts = [int((population["stage"] == stage).sum()) for stage in STAGES]
    sample_counts = [int((sample["stage"] == stage).sum()) for stage in STAGES]
    positions = np.arange(2)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(positions - 0.18, population_counts, 0.36, color=NAVY, label="Population")
    bars2 = ax.bar(
        positions + 0.18,
        sample_counts,
        0.36,
        color=[COLORS[stage] for stage in STAGES],
        label="Sample",
    )
    ax.bar_label(bars1, padding=4, fontweight="bold")
    ax.bar_label(bars2, padding=4, fontweight="bold")
    for index, (population_n, sample_n) in enumerate(zip(population_counts, sample_counts)):
        ax.text(
            index,
            population_n + 14,
            f"Sampled {sample_n / population_n:.1%} | weight {population_n / sample_n:.1f}",
            ha="center",
        )
    ax.set_xticks(positions, STAGES)
    ax.set_ylabel("Eligible midfielders")
    ax.set_ylim(0, 220)
    fig.suptitle("Stratified sampling balances the comparison", x=0.08, y=0.98, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.08, 0.925, "Population N = 240 | Sample n = 100 | Seed = 2026", color=MUTED)
    ax.legend(frameon=False)
    clean_axis(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_png(fig, path)


def plot_distribution(sample: pd.DataFrame, descriptives: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sns.boxplot(
        data=sample,
        x="stage",
        y="tackles_won_per90",
        order=STAGES,
        hue="stage",
        palette=COLORS,
        width=0.45,
        legend=False,
        ax=ax,
    )
    np.random.seed(SEED)
    sns.stripplot(
        data=sample,
        x="stage",
        y="tackles_won_per90",
        order=STAGES,
        hue="stage",
        palette=COLORS,
        jitter=0.18,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
        ax=ax,
    )
    for index, stage in enumerate(STAGES):
        row = descriptives.loc[descriptives["stage"] == stage].iloc[0]
        mean, low, high = row["mean"], row["mean_95ci_low"], row["mean_95ci_high"]
        ax.errorbar(index + 0.29, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=RED, capsize=6)
        ax.text(
            index,
            3.25,
            f"n = 50 | mean {mean:.2f}\n95% CI {low:.2f} to {high:.2f} | median {row['median']:.2f}",
            ha="center",
            va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID},
        )
    ax.set_xlabel("")
    ax.set_ylabel("Tackles won per 90 minutes")
    ax.set_ylim(-0.1, 3.55)
    fig.suptitle("The two midfield distributions overlap substantially", x=0.08, y=0.98, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.08, 0.925, "Dots are players; boxes show quartiles; red marks show means and 95% CIs", color=MUTED)
    clean_axis(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_png(fig, path)


def plot_primary_result(descriptives: pd.DataFrame, test: dict[str, float], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={"width_ratios": [1, 1.25]})
    for index, stage in enumerate(STAGES):
        row = descriptives.loc[descriptives["stage"] == stage].iloc[0]
        mean, low, high = row["mean"], row["mean_95ci_low"], row["mean_95ci_high"]
        axes[0].errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o", color=COLORS[stage], capsize=7, linewidth=2.8)
        axes[0].text(high + 0.03, index, f"{mean:.2f} [{low:.2f}, {high:.2f}]", va="center")
    axes[0].set_yticks(range(2), STAGES)
    axes[0].set_xlabel("Mean tackles won per 90 (95% CI)")
    axes[0].set_xlim(0.45, 1.5)
    axes[0].set_title("Group means", loc="left", fontweight="bold")
    clean_axis(axes[0], "x")

    axes[1].axvline(0, color=MUTED, linestyle="--")
    difference, low, high = test["difference"], test["ci_low"], test["ci_high"]
    axes[1].errorbar(difference, 0, xerr=[[difference - low], [high - difference]], fmt="o", color=RED, capsize=8, linewidth=3)
    axes[1].set_yticks([0], ["Knockout - exit"])
    axes[1].set_xlabel("Difference in mean tackles won per 90")
    axes[1].set_xlim(-0.35, 0.58)
    axes[1].set_title("Primary effect estimate", loc="left", fontweight="bold")
    axes[1].text(
        0.03,
        0.86,
        f"Difference {difference:+.2f}\n95% CI {low:+.2f} to {high:+.2f}\nWelch t({test['degrees_freedom']:.2f}) = {test['t_statistic']:.2f}\np = {test['p_value_two_sided']:.3f} | Hedges' g = {test['hedges_g']:.2f}",
        transform=axes[1].transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8FAFC", "edgecolor": GRID},
    )
    axes[1].text(0.03, 0.16, "The interval crosses zero", transform=axes[1].transAxes, color=RED, fontweight="bold")
    clean_axis(axes[1], "x")
    fig.suptitle("Estimation and hypothesis testing tell the same story", x=0.06, ha="left", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_png(fig, path)


def build_inferential_results(
    descriptives: pd.DataFrame,
    test: dict[str, float],
    robustness: dict[str, float],
    assumptions: pd.DataFrame,
) -> pd.DataFrame:
    exit_row = descriptives.loc[descriptives["stage"] == STAGES[0]].iloc[0]
    knockout_row = descriptives.loc[descriptives["stage"] == STAGES[1]].iloc[0]
    values = [
        ("sample_seed", SEED),
        ("eligible_population_n", 240),
        ("sample_n", 100),
        ("group_exit_sample_mean", exit_row["mean"]),
        ("group_exit_mean_ci_low", exit_row["mean_95ci_low"]),
        ("group_exit_mean_ci_high", exit_row["mean_95ci_high"]),
        ("knockout_sample_mean", knockout_row["mean"]),
        ("knockout_mean_ci_low", knockout_row["mean_95ci_low"]),
        ("knockout_mean_ci_high", knockout_row["mean_95ci_high"]),
        ("mean_difference_knockout_minus_exit", test["difference"]),
        ("difference_ci_low", test["ci_low"]),
        ("difference_ci_high", test["ci_high"]),
        ("welch_t", test["t_statistic"]),
        ("welch_df", test["degrees_freedom"]),
        ("welch_p_two_sided", test["p_value_two_sided"]),
        ("cohens_d", test["cohens_d"]),
        ("hedges_g", test["hedges_g"]),
        ("bootstrap_difference_ci_low", robustness["bootstrap_ci_low"]),
        ("bootstrap_difference_ci_high", robustness["bootstrap_ci_high"]),
        ("permutation_p_two_sided", robustness["permutation_p_two_sided"]),
    ]
    for _, row in assumptions.iterrows():
        name = f"{row['diagnostic']}_{row['group']}_p".lower().replace("-", "_").replace(" ", "_")
        values.append((name, row["p_value"]))
    return pd.DataFrame(values, columns=["metric", "value"])


def run_analysis(
    root: Path,
    input_path: Path,
    seed: int = SEED,
    sample_per_stratum: int = SAMPLE_PER_STRATUM,
) -> dict[str, float]:
    population = load_and_prepare(input_path)
    sample = draw_stratified_sample(population, seed, sample_per_stratum)
    descriptives = build_descriptive_statistics(sample, population)
    assumptions = build_assumption_tests(sample)
    quality = build_data_quality_summary(population)
    knockout = sample.loc[sample["stage"] == STAGES[1], "tackles_won_per90"]
    exits = sample.loc[sample["stage"] == STAGES[0], "tackles_won_per90"]
    test = welch_test(knockout, exits)
    robustness = robustness_checks(knockout, exits, seed)
    inferential = build_inferential_results(descriptives, test, robustness, assumptions)

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    population.to_csv(processed / "midfield_population.csv", index=False)
    sample.to_csv(processed / "midfield_sample_seed2026.csv", index=False)
    descriptives.to_csv(processed / "descriptive_statistics.csv", index=False)
    inferential.to_csv(processed / "inferential_results.csv", index=False)
    assumptions.to_csv(processed / "assumption_tests.csv", index=False)
    quality.to_csv(processed / "data_quality_summary.csv", index=False)
    sample_design = descriptives[["stage", "population_N", "sample_n", "sampling_fraction"]].copy()
    sample_design["sample_weight"] = 1 / sample_design["sampling_fraction"]
    sample_design.to_csv(processed / "sample_design.csv", index=False)

    figure_dir = root / "figures"
    sns.set_theme(style="whitegrid")
    plot_sample_design(population, sample, figure_dir / "01_sample_design.png")
    plot_distribution(sample, descriptives, figure_dir / "02_tackles_distribution.png")
    plot_primary_result(descriptives, test, figure_dir / "03_mean_difference_ci.png")

    summary = {
        "population_n": len(population),
        "sample_n": len(sample),
        "group_exit_mean": float(descriptives.loc[descriptives["stage"] == STAGES[0], "mean"].iloc[0]),
        "knockout_mean": float(descriptives.loc[descriptives["stage"] == STAGES[1], "mean"].iloc[0]),
        **test,
        **robustness,
    }
    (processed / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--input", type=Path, default=root / "data/raw/fbref_2026_midfield_population.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_analysis(args.root, args.input, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
