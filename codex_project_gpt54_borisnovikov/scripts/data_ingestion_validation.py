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
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DOCS_DIR = OUTPUTS_DIR / "docs"
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
CLEANED_DATA_PATH = OUTPUTS_DIR / "cleaned_data.csv"
VALIDATION_REPORT_PATH = BENCHMARK_DIR / "data_validation_report.csv"
EXPERIMENT_LOG_PATH = BENCHMARK_DIR / "experiment_log.txt"
CORRELATION_REPORT_PATH = METRICS_DIR / "feature_target_correlations.csv"
SUMMARY_PATH = DOCS_DIR / "data_summary.txt"


CATEGORICAL_CANDIDATES = {
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
}
COUNT_LIKE_COLUMNS = {"casual", "registered", "cnt"}


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    EXPERIMENT_LOG_PATH.write_text("", encoding="utf-8")


def log_step(message: str) -> None:
    print(message)
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def identify_variable_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [col for col in df.columns if col in CATEGORICAL_CANDIDATES or df[col].dtype == "object"]
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return numeric_columns, categorical_columns


def check_count_like_columns(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for column in columns:
        if column not in df.columns:
            continue
        negative_count = int((df[column] < 0).sum())
        non_integer_like_count = int((df[column] % 1 != 0).sum()) if pd.api.types.is_numeric_dtype(df[column]) else len(df)
        if negative_count == 0 and non_integer_like_count == 0:
            results[column] = "ok"
        else:
            results[column] = (
                f"negative_values={negative_count}; non_integer_like_values={non_integer_like_count}"
            )
    return results


def save_missing_values_figure(missing_values: pd.Series) -> None:
    plt.figure(figsize=(12, 6))
    missing_values.sort_values(ascending=False).plot(kind="bar", color="#2f6f8f")
    plt.title("Missing Values Per Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Values")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values_per_column.png", dpi=150)
    plt.close()


def save_correlation_figure(correlations: pd.Series) -> None:
    plot_series = correlations.drop(labels=["cnt"], errors="ignore").sort_values(key=np.abs, ascending=False)
    plt.figure(figsize=(12, 7))
    colors = plt.cm.Blues_r(np.linspace(0.25, 0.85, len(plot_series)))
    plt.barh(plot_series.index, plot_series.values, color=colors)
    plt.title("Feature Correlation With cnt")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_target_correlations.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_directories([FIGURES_DIR, METRICS_DIR, DOCS_DIR, BENCHMARK_DIR])
    reset_log()

    log_step(f"Random seed set to {RANDOM_SEED}.")
    log_step(f"Loading dataset from {DATASET_PATH}.")
    df = pd.read_csv(DATASET_PATH)

    log_step(f"Dataset shape: {df.shape}")
    log_step("Column names: " + ", ".join(df.columns.tolist()))
    log_step("Data types:\n" + df.dtypes.to_string())

    numeric_columns, categorical_columns = identify_variable_types(df)
    log_step("Numeric variables: " + ", ".join(numeric_columns))
    log_step("Categorical variables: " + ", ".join(categorical_columns))

    missing_values = df.isna().sum().sort_index()
    total_missing_values = int(missing_values.sum())
    log_step("Missing values per column:\n" + missing_values.to_string())

    duplicate_rows = int(df.duplicated().sum())
    log_step(f"Duplicate rows: {duplicate_rows}")

    leakage_identity_detected = False
    leakage_columns_removed: list[str] = []
    if {"casual", "registered", "cnt"}.issubset(df.columns):
        leakage_identity_detected = bool((df["casual"] + df["registered"] == df["cnt"]).all())
        log_step(f"Leakage identity casual + registered = cnt: {leakage_identity_detected}")
        if leakage_identity_detected:
            leakage_columns_removed = ["casual", "registered"]
            df = df.drop(columns=leakage_columns_removed)
            log_step("Removed leakage columns: casual, registered")
    else:
        log_step("Leakage validation skipped because casual, registered, or cnt is missing.")

    target_nonnegative_check = bool((df["cnt"] >= 0).all())
    log_step(f"Target non-negative check: {target_nonnegative_check}")

    count_like_checks = check_count_like_columns(pd.read_csv(DATASET_PATH), COUNT_LIKE_COLUMNS)
    for column, result in count_like_checks.items():
        log_step(f"Count-like field check for {column}: {result}")

    numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    correlations = df[numeric_features].corr(numeric_only=True)["cnt"].sort_values(ascending=False)
    feature_correlations = correlations.drop(labels=["cnt"], errors="ignore")
    high_corr_features = feature_correlations[feature_correlations.abs() > 0.95]
    max_feature_target_corr = float(feature_correlations.abs().max()) if not feature_correlations.empty else float("nan")
    log_step("Feature correlations with cnt:\n" + feature_correlations.to_string())
    log_step(
        "Features with absolute correlation > 0.95: "
        + (", ".join(high_corr_features.index.tolist()) if not high_corr_features.empty else "None")
    )

    current_numeric_columns = [col for col in numeric_columns if col in df.columns]
    current_categorical_columns = [col for col in categorical_columns if col in df.columns]
    missing_numeric_columns = df[current_numeric_columns].columns[df[current_numeric_columns].isna().any()].tolist()
    missing_categorical_columns = (
        df[current_categorical_columns].columns[df[current_categorical_columns].isna().any()].tolist()
    )
    if missing_numeric_columns or missing_categorical_columns:
        log_step("Handling missing values using median for numeric and mode for categorical columns.")
        for column in missing_numeric_columns:
            df[column] = df[column].fillna(df[column].median())
        for column in missing_categorical_columns:
            mode_series = df[column].mode(dropna=True)
            if not mode_series.empty:
                df[column] = df[column].fillna(mode_series.iloc[0])
    else:
        log_step("No missing values detected. No imputation required.")

    df.to_csv(CLEANED_DATA_PATH, index=False)
    log_step(f"Saved cleaned dataset to {CLEANED_DATA_PATH}.")

    correlation_report = feature_correlations.rename("correlation_with_cnt").reset_index()
    correlation_report.columns = ["feature", "correlation_with_cnt"]
    correlation_report.to_csv(CORRELATION_REPORT_PATH, index=False)
    log_step(f"Saved correlation report to {CORRELATION_REPORT_PATH}.")

    save_missing_values_figure(missing_values)
    save_correlation_figure(correlations)
    log_step(f"Saved figures to {FIGURES_DIR}.")

    validation_rows = [
        {"check": "missing_values_total", "value": total_missing_values},
        {"check": "duplicate_rows", "value": duplicate_rows},
        {"check": "leakage_identity_detected", "value": leakage_identity_detected},
        {
            "check": "leakage_columns_removed",
            "value": ",".join(leakage_columns_removed) if leakage_columns_removed else "None",
        },
        {"check": "max_feature_target_corr", "value": max_feature_target_corr},
        {"check": "target_nonnegative_check", "value": target_nonnegative_check},
    ]
    for column, result in count_like_checks.items():
        validation_rows.append({"check": f"count_like_check_{column}", "value": result})
    if not high_corr_features.empty:
        validation_rows.append(
            {
                "check": "high_corr_features_abs_gt_0_95",
                "value": ",".join(high_corr_features.index.tolist()),
            }
        )
    else:
        validation_rows.append({"check": "high_corr_features_abs_gt_0_95", "value": "None"})

    validation_report = pd.DataFrame(validation_rows)
    validation_report.to_csv(VALIDATION_REPORT_PATH, index=False)
    log_step(f"Saved validation report to {VALIDATION_REPORT_PATH}.")

    summary_lines = [
        f"Dataset shape: {df.shape}",
        f"Total missing values: {total_missing_values}",
        f"Duplicate rows: {duplicate_rows}",
        f"Leakage identity detected: {leakage_identity_detected}",
        "Leakage columns removed: " + (", ".join(leakage_columns_removed) if leakage_columns_removed else "None"),
        f"Maximum absolute feature-target correlation: {max_feature_target_corr:.6f}",
        f"Target non-negative check passed: {target_nonnegative_check}",
        "High-correlation features (abs > 0.95): "
        + (", ".join(high_corr_features.index.tolist()) if not high_corr_features.empty else "None"),
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nSummary of findings and changes")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
