from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "outputs" / "benchmark" / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASET_PATH = BASE_DIR / "dataset" / "hour.csv"
CLEANED_DATA_PATH = BASE_DIR / "outputs" / "cleaned_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
DOCS_DIR = OUTPUTS_DIR / "docs"
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
EXPERIMENT_LOG_PATH = BENCHMARK_DIR / "experiment_log.txt"
TRAIN_PATH = OUTPUTS_DIR / "train.csv"
VAL_PATH = OUTPUTS_DIR / "val.csv"
TEST_PATH = OUTPUTS_DIR / "test.csv"
PREPROCESSING_REPORT_PATH = BENCHMARK_DIR / "preprocessing_report.csv"
BASELINE_RESULTS_PATH = METRICS_DIR / "baseline_model_results.csv"
BASELINE_SUMMARY_PATH = DOCS_DIR / "baseline_model_summary.txt"
ACTUAL_VS_PREDICTED_PATH = FIGURES_DIR / "actual_vs_predicted.png"
RESIDUAL_DISTRIBUTION_PATH = FIGURES_DIR / "residual_distribution.png"

TARGET_COLUMN = "cnt"
TIME_DATE_COLUMN = "dteday"
TIME_HOUR_COLUMN = "hr"
F0_FEATURES = ["hr", "weekday", "workingday", "season", "mnth", "yr", "weathersit", "temp", "atemp", "hum", "windspeed"]
F1_FEATURES = F0_FEATURES + ["sin_hour", "cos_hour", "sin_month", "cos_month"]


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def log_step(message: str) -> None:
    print(message)
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def load_data() -> tuple[pd.DataFrame, Path]:
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH), CLEANED_DATA_PATH
    return pd.read_csv(DATASET_PATH), DATASET_PATH


def add_time_and_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result[TIME_DATE_COLUMN]) + pd.to_timedelta(result[TIME_HOUR_COLUMN], unit="h")
    result["sin_hour"] = np.sin(2 * np.pi * result["hr"] / 24.0)
    result["cos_hour"] = np.cos(2 * np.pi * result["hr"] / 24.0)
    result["sin_month"] = np.sin(2 * np.pi * (result["mnth"] - 1) / 12.0)
    result["cos_month"] = np.cos(2 * np.pi * (result["mnth"] - 1) / 12.0)
    return result


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()
    return train_df, val_df, test_df


def verify_chronological_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    train_to_val = train_df["timestamp"].max() < val_df["timestamp"].min()
    val_to_test = val_df["timestamp"].max() < test_df["timestamp"].min()
    chronological_pass = bool(train_to_val and val_to_test)
    if not chronological_pass:
        raise ValueError("Chronological split verification failed.")
    return chronological_pass


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    log_step(f"Saved splits to {TRAIN_PATH}, {VAL_PATH}, and {TEST_PATH}.")


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression(n_jobs=1)),
        ]
    )


def train_and_evaluate(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], feature_set_name: str) -> tuple[dict[str, object], np.ndarray, Pipeline]:
    X_train = train_df[features].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_val = val_df[features].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    pipeline = build_pipeline()
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - start_time
    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    result = {
        "model": "LinearRegression",
        "feature_set": feature_set_name,
        "split": "validation",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "training_time_seconds": float(training_time_seconds),
    }
    return result, predictions, pipeline


def save_model(model: Pipeline, feature_set_name: str) -> None:
    model_path = MODELS_DIR / f"linear_regression_{feature_set_name}.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)
    log_step(f"Saved model artifact to {model_path}.")


def save_actual_vs_predicted_plot(y_true: pd.Series, predictions_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    y_true_values = y_true.to_numpy()
    global_min = min(float(y_true_values.min()), *(float(pred.min()) for pred in predictions_map.values()))
    global_max = max(float(y_true_values.max()), *(float(pred.max()) for pred in predictions_map.values()))

    for axis, (feature_set_name, preds) in zip(axes, predictions_map.items()):
        axis.scatter(y_true_values, preds, alpha=0.18, s=12, color="#2f6f8f")
        axis.plot([global_min, global_max], [global_min, global_max], color="#d1495b", linewidth=2)
        axis.set_title(f"Actual vs Predicted ({feature_set_name})")
        axis.set_xlabel("Actual cnt")
        axis.set_ylabel("Predicted cnt")

    plt.tight_layout()
    plt.savefig(ACTUAL_VS_PREDICTED_PATH, dpi=150)
    plt.close(fig)


def save_residual_distribution_plot(y_true: pd.Series, predictions_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, (feature_set_name, preds) in zip(axes, predictions_map.items()):
        residuals = y_true.to_numpy() - preds
        axis.hist(residuals, bins=40, color="#6c9a8b", edgecolor="white")
        axis.axvline(0.0, color="#d1495b", linewidth=2)
        axis.set_title(f"Residual Distribution ({feature_set_name})")
        axis.set_xlabel("Residual")
        axis.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(RESIDUAL_DISTRIBUTION_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_directories([FIGURES_DIR, METRICS_DIR, MODELS_DIR, DOCS_DIR, BENCHMARK_DIR])

    df, source_path = load_data()
    log_step(f"Task 3 baseline modelling started with random seed {RANDOM_SEED}.")
    log_step(f"Loaded modelling dataset from {source_path} with shape {df.shape}.")

    missing_required_columns = [col for col in [TIME_DATE_COLUMN, TIME_HOUR_COLUMN, TARGET_COLUMN, *F0_FEATURES] if col not in df.columns]
    if missing_required_columns:
        raise KeyError(f"Missing required columns for baseline modelling: {missing_required_columns}")

    prepared_df = add_time_and_cyclical_features(df)
    train_df, val_df, test_df = chronological_split(prepared_df)
    chronological_pass = verify_chronological_split(train_df, val_df, test_df)
    log_step(
        "Chronological split verified: "
        f"max(train_time)={train_df['timestamp'].max()}, min(val_time)={val_df['timestamp'].min()}, "
        f"max(val_time)={val_df['timestamp'].max()}, min(test_time)={test_df['timestamp'].min()}"
    )

    save_splits(train_df, val_df, test_df)

    preprocessing_report = pd.DataFrame(
        [
            {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "chronological_split_pass": chronological_pass,
                "feature_set_F0_defined": True,
                "feature_set_F1_defined": True,
            }
        ]
    )
    preprocessing_report.to_csv(PREPROCESSING_REPORT_PATH, index=False)
    log_step(f"Saved preprocessing report to {PREPROCESSING_REPORT_PATH}.")

    results: list[dict[str, object]] = []
    predictions_map: dict[str, np.ndarray] = {}
    model_summary_lines: list[str] = []

    for feature_set_name, features in (("F0", F0_FEATURES), ("F1", F1_FEATURES)):
        result, predictions, model = train_and_evaluate(train_df, val_df, features, feature_set_name)
        results.append(result)
        predictions_map[feature_set_name] = predictions
        save_model(model, feature_set_name)
        log_step(
            f"Validation results for {feature_set_name}: MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}, training_time_seconds={result['training_time_seconds']:.6f}"
        )
        model_summary_lines.append(
            f"{feature_set_name}: MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}, training_time_seconds={result['training_time_seconds']:.6f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(BASELINE_RESULTS_PATH, index=False)
    log_step(f"Saved baseline results to {BASELINE_RESULTS_PATH}.")

    save_actual_vs_predicted_plot(val_df[TARGET_COLUMN], predictions_map)
    save_residual_distribution_plot(val_df[TARGET_COLUMN], predictions_map)
    log_step(f"Saved diagnostic plots to {ACTUAL_VS_PREDICTED_PATH} and {RESIDUAL_DISTRIBUTION_PATH}.")

    best_row = results_df.sort_values(["RMSE", "MAE"], ascending=[True, True]).iloc[0]
    summary_lines = [
        f"Chronological split rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}.",
        "Baseline validation results:",
        *model_summary_lines,
        f"Best validation feature set by RMSE: {best_row['feature_set']} (MAE={best_row['MAE']:.4f}, RMSE={best_row['RMSE']:.4f}).",
        "Test set was held out and not used for model selection or evaluation.",
    ]
    BASELINE_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log_step(f"Saved baseline summary to {BASELINE_SUMMARY_PATH}.")

    print("\nBaseline summary")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
