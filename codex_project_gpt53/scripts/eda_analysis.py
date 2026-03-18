from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42


def write_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_pipeline() -> None:
    np.random.seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parents[1]
    outputs_root = project_root / "outputs"
    figures_dir = outputs_root / "figures"
    benchmark_dir = outputs_root / "benchmark"
    docs_dir = outputs_root / "docs"
    for path in (figures_dir, benchmark_dir, docs_dir):
        path.mkdir(parents=True, exist_ok=True)

    log_path = benchmark_dir / "experiment_log.txt"
    write_log(log_path, "Started EDA pipeline.")
    write_log(log_path, f"Random seed set to {RANDOM_SEED}.")

    cleaned_path = outputs_root / "cleaned_data.csv"
    raw_path = project_root / "dataset" / "hour.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        source_path = cleaned_path
    else:
        df = pd.read_csv(raw_path)
        source_path = raw_path
    write_log(log_path, f"Loaded EDA dataset from {source_path} with shape {df.shape}.")

    if "cnt" not in df.columns:
        raise ValueError("Target column 'cnt' is missing from the dataset.")
    target = "cnt"

    missing_total = int(df.isna().sum().sum())
    write_log(log_path, f"Missing values total: {missing_total}")

    q1 = float(df[target].quantile(0.25))
    q3 = float(df[target].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((df[target] < lower) | (df[target] > upper)).sum())
    outlier_pct = float(outlier_count / len(df) * 100.0)
    write_log(log_path, f"Target outliers by IQR rule: {outlier_count} rows ({outlier_pct:.2f}%).")

    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

    if {"dteday", "hr"}.issubset(df.columns):
        df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
        ts_df = df.sort_values("datetime")
    elif "dteday" in df.columns:
        ts_df = df.sort_values("dteday")
    else:
        ts_df = df.copy()

    plt.figure(figsize=(10, 6))
    plt.hist(df[target], bins=40, color="#1f77b4", alpha=0.85)
    plt.title("Target Distribution: cnt")
    plt.xlabel("cnt")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    if "datetime" in ts_df.columns:
        plt.plot(ts_df["datetime"], ts_df[target], color="#2ca02c", linewidth=0.8)
        plt.xlabel("Datetime")
    elif "dteday" in ts_df.columns:
        daily = ts_df.groupby("dteday", as_index=False)[target].mean()
        plt.plot(daily["dteday"], daily[target], color="#2ca02c", linewidth=1.0)
        plt.xlabel("Date")
    else:
        plt.plot(ts_df.index, ts_df[target], color="#2ca02c", linewidth=0.8)
        plt.xlabel("Row index")
    plt.title("Demand Over Time")
    plt.ylabel("cnt")
    plt.tight_layout()
    plt.savefig(figures_dir / "demand_time_series.png", dpi=150)
    plt.close()

    if {"hr", "weekday"}.issubset(df.columns):
        heat = df.pivot_table(index="weekday", columns="hr", values=target, aggfunc="mean")
    else:
        heat = pd.DataFrame(np.nan, index=range(7), columns=range(24))
    plt.figure(figsize=(12, 6))
    plt.imshow(heat.values, aspect="auto", cmap="YlGnBu", origin="lower")
    plt.colorbar(label="Mean cnt")
    plt.xticks(ticks=np.arange(heat.shape[1]), labels=heat.columns)
    plt.yticks(ticks=np.arange(heat.shape[0]), labels=heat.index)
    plt.title("Mean cnt by Weekday and Hour")
    plt.xlabel("Hour")
    plt.ylabel("Weekday")
    plt.tight_layout()
    plt.savefig(figures_dir / "heatmap_hour_weekday.png", dpi=150)
    plt.close()

    def plot_scatter(feature: str, filename: str, color: str) -> None:
        plt.figure(figsize=(8, 6))
        if feature in df.columns:
            plt.scatter(df[feature], df[target], s=10, alpha=0.35, c=color, edgecolors="none")
        else:
            plt.text(0.5, 0.5, f"'{feature}' not available", ha="center", va="center")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        plt.title(f"Demand vs {feature}")
        plt.xlabel(feature)
        plt.ylabel("cnt")
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150)
        plt.close()

    plot_scatter("temp", "demand_vs_temp.png", "#d62728")
    plot_scatter("hum", "demand_vs_hum.png", "#9467bd")
    plot_scatter("windspeed", "demand_vs_windspeed.png", "#ff7f0e")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="Correlation")
    plt.xticks(ticks=np.arange(corr.shape[1]), labels=corr.columns, rotation=90)
    plt.yticks(ticks=np.arange(corr.shape[0]), labels=corr.index)
    plt.title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    write_log(log_path, "Saved all required EDA plots.")

    hour_table = (
        df.groupby("hr", as_index=False)[target].mean().rename(columns={target: "mean_cnt"})
        if "hr" in df.columns
        else pd.DataFrame(columns=["hr", "mean_cnt"])
    )
    hour_table["table"] = "mean_demand_by_hour"
    hour_table = hour_table[["table", "hr", "mean_cnt"]]

    weekday_table = (
        df.groupby("weekday", as_index=False)[target].mean().rename(columns={target: "mean_cnt"})
        if "weekday" in df.columns
        else pd.DataFrame(columns=["weekday", "mean_cnt"])
    )
    weekday_table["table"] = "mean_demand_by_weekday"
    weekday_table = weekday_table[["table", "weekday", "mean_cnt"]]

    eda_tables = pd.concat([hour_table, weekday_table], ignore_index=True, sort=False)
    eda_tables_path = benchmark_dir / "eda_tables.csv"
    eda_tables.to_csv(eda_tables_path, index=False)
    write_log(log_path, f"Saved EDA aggregate tables to {eda_tables_path}.")

    rare_regimes = "No weather regime column available."
    if "weathersit" in df.columns:
        weather_share = df["weathersit"].value_counts(normalize=True).sort_index()
        rare = weather_share[weather_share < 0.05]
        if rare.empty:
            rare_regimes = "No weather category below 5% frequency."
        else:
            pairs = [f"weathersit={idx}: {val:.2%}" for idx, val in rare.items()]
            rare_regimes = "Rare weather categories (<5%): " + ", ".join(pairs)

    leakage_note = "No obvious leakage columns (casual/registered) present in working dataset."
    if {"casual", "registered", "cnt"}.issubset(df.columns):
        identity = bool((df["casual"] + df["registered"] == df["cnt"]).all())
        if identity:
            leakage_note = "Potential leakage remains: casual + registered exactly reconstructs cnt."
        else:
            leakage_note = "casual/registered exist but do not perfectly reconstruct cnt."

    if "hr" in df.columns:
        peak_hour = int(df.groupby("hr")[target].mean().idxmax())
        trough_hour = int(df.groupby("hr")[target].mean().idxmin())
        time_pattern = (
            f"Hourly demand is structured, with highest average demand around hour {peak_hour} "
            f"and lowest around hour {trough_hour}."
        )
    else:
        time_pattern = "Hourly demand pattern could not be computed because 'hr' is unavailable."

    risk_parts = [
        f"Target outliers by IQR are {outlier_count} rows ({outlier_pct:.2f}%), which can skew RMSE.",
        f"Missingness is {missing_total} values across the dataset.",
        "Strong daily/weekly seasonality suggests models need temporal signal handling without overfitting.",
    ]
    risks_text = " ".join(risk_parts)

    summary_lines = [
        "EDA Summary",
        f"Demand pattern over time: {time_pattern}",
        f"Rare regimes: {rare_regimes}",
        f"Likely modelling risks: {risks_text}",
        f"Remaining leakage or data quality concerns: {leakage_note}",
    ]
    summary_path = docs_dir / "eda_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_log(log_path, f"Saved EDA summary to {summary_path}.")

    print("\nEDA completed. Key outputs:")
    print(f"- Figures saved to: {figures_dir}")
    print(f"- EDA tables: {eda_tables_path}")
    print(f"- Summary: {summary_path}")
    write_log(log_path, "EDA pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
