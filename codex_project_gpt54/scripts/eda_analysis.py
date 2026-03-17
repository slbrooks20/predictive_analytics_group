from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "outputs" / "benchmark" / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASET_PATH = BASE_DIR / "dataset" / "hour.csv"
CLEANED_DATA_PATH = BASE_DIR / "outputs" / "cleaned_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DOCS_DIR = OUTPUTS_DIR / "docs"
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
EXPERIMENT_LOG_PATH = BENCHMARK_DIR / "experiment_log.txt"
EDA_TABLES_PATH = BENCHMARK_DIR / "eda_tables.csv"
EDA_SUMMARY_PATH = DOCS_DIR / "eda_summary.txt"

TARGET_COLUMN = "cnt"
REQUIRED_PLOT_FILES = [
    "target_distribution.png",
    "demand_time_series.png",
    "heatmap_hour_weekday.png",
    "demand_vs_temp.png",
    "demand_vs_hum.png",
    "demand_vs_windspeed.png",
    "correlation_matrix.png",
]


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def remove_unrequested_plots() -> None:
    for plot_path in FIGURES_DIR.glob("*.png"):
        if plot_path.name not in REQUIRED_PLOT_FILES:
            plot_path.unlink()
            log_step(f"Removed unrequested plot file: {plot_path.name}")


def log_step(message: str) -> None:
    print(message)
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def load_data() -> tuple[pd.DataFrame, Path]:
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH), CLEANED_DATA_PATH
    return pd.read_csv(DATASET_PATH), DATASET_PATH


def save_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df[TARGET_COLUMN], bins=40, color="#2f6f8f", edgecolor="white")
    plt.title("Distribution of Hourly Demand")
    plt.xlabel("cnt")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def save_demand_time_series(df: pd.DataFrame) -> None:
    daily_demand = df.groupby("dteday", as_index=False)[TARGET_COLUMN].sum()
    daily_demand["dteday"] = pd.to_datetime(daily_demand["dteday"])
    plt.figure(figsize=(12, 6))
    plt.plot(daily_demand["dteday"], daily_demand[TARGET_COLUMN], color="#0d3b66", linewidth=1.5)
    plt.title("Daily Total Demand Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total daily cnt")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demand_time_series.png", dpi=150)
    plt.close()


def save_heatmap_hour_weekday(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(values=TARGET_COLUMN, index="weekday", columns="hr", aggfunc="mean")
    plt.figure(figsize=(12, 5))
    image = plt.imshow(pivot.values, aspect="auto", cmap="YlGnBu", origin="upper")
    plt.colorbar(image, label="Mean cnt")
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.title("Mean Demand by Weekday and Hour")
    plt.xlabel("Hour")
    plt.ylabel("Weekday")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heatmap_hour_weekday.png", dpi=150)
    plt.close()


def save_scatter_with_trend(df: pd.DataFrame, feature: str, output_name: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature], df[TARGET_COLUMN], alpha=0.15, s=12, color="#2f6f8f")
    valid = df[[feature, TARGET_COLUMN]].dropna()
    if len(valid) > 1 and valid[feature].nunique() > 1:
        coeffs = np.polyfit(valid[feature], valid[TARGET_COLUMN], deg=1)
        x_vals = np.linspace(valid[feature].min(), valid[feature].max(), 100)
        y_vals = np.polyval(coeffs, x_vals)
        plt.plot(x_vals, y_vals, color="#d1495b", linewidth=2)
    plt.title(f"Demand vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("cnt")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=150)
    plt.close()


def save_correlation_matrix(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    image = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image)
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_directories([FIGURES_DIR, METRICS_DIR, DOCS_DIR, BENCHMARK_DIR])
    remove_unrequested_plots()

    df, source_path = load_data()
    log_step(f"Task 2 EDA started with random seed {RANDOM_SEED}.")
    log_step(f"Loaded EDA dataset from {source_path} with shape {df.shape}.")
    log_step(f"Target variable identified: {TARGET_COLUMN}")

    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    log_step(f"Total missing values in EDA dataset: {missing_total}")
    log_step(f"Duplicate rows in EDA dataset: {duplicate_rows}")

    target_stats = df[TARGET_COLUMN].describe()
    q1 = float(df[TARGET_COLUMN].quantile(0.25))
    q3 = float(df[TARGET_COLUMN].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((df[TARGET_COLUMN] < lower_bound) | (df[TARGET_COLUMN] > upper_bound)).sum())
    outlier_share = outlier_count / len(df)
    log_step(
        "Target distribution stats:\n" + target_stats.to_string()
    )
    log_step(
        f"Target outlier check using IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}] found {outlier_count} rows ({outlier_share:.2%})."
    )

    feature_correlations = (
        df.select_dtypes(include=[np.number]).corr(numeric_only=True)[TARGET_COLUMN].drop(labels=[TARGET_COLUMN], errors="ignore")
    )
    top_relationships = feature_correlations.abs().sort_values(ascending=False).head(5)
    log_step("Top absolute numeric feature correlations with cnt:\n" + top_relationships.to_string())

    mean_demand_by_hour = df.groupby("hr", as_index=False)[TARGET_COLUMN].mean()
    mean_demand_by_weekday = df.groupby("weekday", as_index=False)[TARGET_COLUMN].mean()
    eda_tables = pd.concat(
        [
            pd.DataFrame({
                "table": "mean_demand_by_hour",
                "group": mean_demand_by_hour["hr"],
                "mean_cnt": mean_demand_by_hour[TARGET_COLUMN],
            }),
            pd.DataFrame({
                "table": "mean_demand_by_weekday",
                "group": mean_demand_by_weekday["weekday"],
                "mean_cnt": mean_demand_by_weekday[TARGET_COLUMN],
            }),
        ],
        ignore_index=True,
    )
    eda_tables.to_csv(EDA_TABLES_PATH, index=False)
    log_step(f"Saved EDA tables to {EDA_TABLES_PATH}.")

    save_target_distribution(df)
    save_demand_time_series(df)
    save_heatmap_hour_weekday(df)
    save_scatter_with_trend(df, "temp", "demand_vs_temp.png")
    save_scatter_with_trend(df, "hum", "demand_vs_hum.png")
    save_scatter_with_trend(df, "windspeed", "demand_vs_windspeed.png")
    save_correlation_matrix(df)
    log_step("Saved requested EDA figures to outputs/figures.")

    peak_hour = int(mean_demand_by_hour.loc[mean_demand_by_hour[TARGET_COLUMN].idxmax(), "hr"])
    low_hour = int(mean_demand_by_hour.loc[mean_demand_by_hour[TARGET_COLUMN].idxmin(), "hr"])
    peak_weekday = int(mean_demand_by_weekday.loc[mean_demand_by_weekday[TARGET_COLUMN].idxmax(), "weekday"])
    rare_regime_text = "No rare regime columns identified."
    if "weathersit" in df.columns:
        weather_counts = df["weathersit"].value_counts(dropna=False).sort_index()
        rare_weather = weather_counts[weather_counts <= max(10, int(0.01 * len(df)))]
        if rare_weather.empty:
            rare_regime_text = "Weather categories are present without extremely rare levels under the current threshold."
        else:
            rare_regime_text = "Rare weather regimes: " + ", ".join(
                [f"weathersit={idx} ({count} rows)" for idx, count in rare_weather.items()]
            )

    leakage_notes = []
    if {"casual", "registered"}.issubset(df.columns):
        leakage_notes.append("Potential leakage remains because casual and registered are still present.")
    else:
        leakage_notes.append("No direct leakage columns remain from Task 1; casual and registered are absent.")
    if missing_total == 0 and duplicate_rows == 0:
        leakage_notes.append("No remaining missingness or duplicate-row concerns were observed.")
    else:
        leakage_notes.append(
            f"Data quality follow-up: missing_total={missing_total}, duplicate_rows={duplicate_rows}."
        )

    summary_lines = [
        f"Demand pattern over time: daily demand varies substantially over the calendar span, with stronger average demand at hour {peak_hour} and weaker average demand at hour {low_hour}; weekday {peak_weekday} has the highest mean demand in the weekly profile.",
        f"Any rare regimes: {rare_regime_text}",
        f"Likely modelling risks: the target distribution is right-skewed with {outlier_count} IQR-based high-end or low-end outliers ({outlier_share:.2%} of rows), and demand levels vary sharply by hour, which can make errors uneven across low-demand and peak-demand periods.",
        f"Any remaining leakage or data quality concerns: {' '.join(leakage_notes)}",
    ]
    EDA_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log_step(f"Saved EDA summary to {EDA_SUMMARY_PATH}.")

    print("\nEDA summary")
    for line in summary_lines:
        print(f"- {line}")

    missing_plots = [name for name in REQUIRED_PLOT_FILES if not (FIGURES_DIR / name).exists()]
    if missing_plots:
        raise FileNotFoundError(f"Missing required plot files: {missing_plots}")


if __name__ == "__main__":
    main()
