from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


RANDOM_SEED = 42


def write_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["sin_hour"] = np.sin(2 * np.pi * result["hr"] / 24.0)
    result["cos_hour"] = np.cos(2 * np.pi * result["hr"] / 24.0)
    result["sin_month"] = np.sin(2 * np.pi * result["mnth"] / 12.0)
    result["cos_month"] = np.cos(2 * np.pi * result["mnth"] / 12.0)
    return result


def train_and_evaluate_linear_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[LinearRegression, np.ndarray, float, float, float]:
    x_train = train_df[features]
    y_train = train_df[target]
    x_val = val_df[features]
    y_val = val_df[target]

    model = LinearRegression(n_jobs=1)

    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    training_time_seconds = time.perf_counter() - start_time

    val_pred = model.predict(x_val)
    mae = float(mean_absolute_error(y_val, val_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    return model, val_pred, mae, rmse, float(training_time_seconds)


def run_pipeline() -> None:
    np.random.seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parents[1]
    outputs_root = project_root / "outputs"

    figures_dir = outputs_root / "figures"
    metrics_dir = outputs_root / "metrics"
    models_dir = outputs_root / "models"
    benchmark_dir = outputs_root / "benchmark"
    docs_dir = outputs_root / "docs"

    for path in (figures_dir, metrics_dir, models_dir, benchmark_dir, docs_dir):
        path.mkdir(parents=True, exist_ok=True)

    log_path = benchmark_dir / "experiment_log.txt"
    write_log(log_path, "Started Task 3 baseline modeling pipeline.")
    write_log(log_path, f"Random seed set to {RANDOM_SEED}; stochastic n_jobs fixed to 1.")

    cleaned_data_path = outputs_root / "cleaned_data.csv"
    raw_data_path = project_root / "dataset" / "hour.csv"
    if cleaned_data_path.exists():
        df = pd.read_csv(cleaned_data_path)
        source = cleaned_data_path
    else:
        df = pd.read_csv(raw_data_path)
        source = raw_data_path
    write_log(log_path, f"Loaded modeling dataset from {source} with shape {df.shape}.")

    required_columns = [
        "hr", "weekday", "workingday", "season", "mnth", "yr", "weathersit",
        "temp", "atemp", "hum", "windspeed", "cnt", "dteday",
    ]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
    if df["dteday"].isna().any():
        raise ValueError("Found invalid dates in dteday.")
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    sort_cols = ["datetime"]
    if "instant" in df.columns:
        sort_cols.append("instant")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    write_log(log_path, "Sorted rows chronologically by datetime.")

    n_rows = len(df)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Invalid chronological split boundaries.")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_max = train_df["datetime"].max()
    val_min = val_df["datetime"].min()
    val_max = val_df["datetime"].max()
    test_min = test_df["datetime"].min()

    chronological_split_pass = bool((train_max < val_min) and (val_max < test_min))
    if not chronological_split_pass:
        raise ValueError(
            "Chronological split failed: max(train_time) < min(validation_time) and "
            "max(validation_time) < min(test_time) must both hold."
        )
    write_log(
        log_path,
        "Chronological assertions passed: "
        f"max(train)={train_max}, min(val)={val_min}, max(val)={val_max}, min(test)={test_min}.",
    )

    train_df = add_cyclical_features(train_df)
    val_df = add_cyclical_features(val_df)
    test_df = add_cyclical_features(test_df)

    f0_features = [
        "hr", "weekday", "workingday", "season", "mnth", "yr", "weathersit",
        "temp", "atemp", "hum", "windspeed",
    ]
    f1_features = f0_features + ["sin_hour", "cos_hour", "sin_month", "cos_month"]

    feature_set_f0_defined = bool(set(f0_features).issubset(train_df.columns))
    feature_set_f1_defined = bool(set(f1_features).issubset(train_df.columns))
    if not feature_set_f0_defined or not feature_set_f1_defined:
        raise ValueError("Feature set definition failed.")

    train_df.to_csv(outputs_root / "train.csv", index=False)
    val_df.to_csv(outputs_root / "val.csv", index=False)
    test_df.to_csv(outputs_root / "test.csv", index=False)
    write_log(log_path, "Saved chronological splits to outputs/train.csv, outputs/val.csv, outputs/test.csv.")

    preprocessing_report = pd.DataFrame(
        [
            {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "chronological_split_pass": chronological_split_pass,
                "feature_set_F0_defined": feature_set_f0_defined,
                "feature_set_F1_defined": feature_set_f1_defined,
            }
        ]
    )
    preprocessing_path = benchmark_dir / "preprocessing_report.csv"
    preprocessing_report.to_csv(preprocessing_path, index=False)
    write_log(log_path, f"Saved preprocessing report to {preprocessing_path}.")

    target = "cnt"
    results: list[dict[str, object]] = []
    prediction_cache: dict[str, np.ndarray] = {}
    models: dict[str, LinearRegression] = {}

    for feature_set_name, features in (("F0", f0_features), ("F1", f1_features)):
        model, val_pred, mae, rmse, train_sec = train_and_evaluate_linear_regression(
            train_df=train_df,
            val_df=val_df,
            features=features,
            target=target,
        )
        models[feature_set_name] = model
        prediction_cache[feature_set_name] = val_pred
        results.append(
            {
                "model": "LinearRegression",
                "feature_set": feature_set_name,
                "split": "validation",
                "MAE": mae,
                "RMSE": rmse,
                "training_time_seconds": train_sec,
            }
        )
        write_log(
            log_path,
            f"Trained LinearRegression on {feature_set_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, "
            f"training_time_seconds={train_sec:.6f}.",
        )

        model_path = models_dir / f"linear_regression_{feature_set_name}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        write_log(log_path, f"Saved model artifact to {model_path}.")

    results_df = pd.DataFrame(results)
    baseline_results_path = metrics_dir / "baseline_model_results.csv"
    results_df.to_csv(baseline_results_path, index=False)
    write_log(log_path, f"Saved baseline metrics to {baseline_results_path}.")

    best_feature_set = str(results_df.sort_values(["RMSE", "MAE"], ascending=True).iloc[0]["feature_set"])
    y_val = val_df[target].to_numpy()
    y_pred = prediction_cache[best_feature_set]
    residuals = y_val - y_pred
    write_log(log_path, f"Diagnostic plots generated from best validation feature set: {best_feature_set}.")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_pred, s=10, alpha=0.35, c="#1f77b4", edgecolors="none")
    line_min = min(float(y_val.min()), float(y_pred.min()))
    line_max = max(float(y_val.max()), float(y_pred.max()))
    plt.plot([line_min, line_max], [line_min, line_max], color="red", linewidth=1.5)
    plt.xlabel("Actual cnt (validation)")
    plt.ylabel("Predicted cnt (validation)")
    plt.title(f"Actual vs Predicted ({best_feature_set})")
    plt.tight_layout()
    actual_vs_pred_path = figures_dir / "actual_vs_predicted.png"
    plt.savefig(actual_vs_pred_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40, color="#2ca02c", alpha=0.85)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution ({best_feature_set})")
    plt.tight_layout()
    residual_hist_path = figures_dir / "residual_distribution.png"
    plt.savefig(residual_hist_path, dpi=150)
    plt.close()
    write_log(log_path, f"Saved diagnostics to {actual_vs_pred_path} and {residual_hist_path}.")

    summary_path = docs_dir / "baseline_summary.txt"
    summary_lines = [
        "Baseline Model Summary",
        "Model: LinearRegression",
        "Evaluation split: validation",
        f"Feature sets evaluated: F0, F1",
        f"Best feature set by RMSE: {best_feature_set}",
        f"Best MAE: {results_df.loc[results_df['feature_set'] == best_feature_set, 'MAE'].iloc[0]:.4f}",
        f"Best RMSE: {results_df.loc[results_df['feature_set'] == best_feature_set, 'RMSE'].iloc[0]:.4f}",
        "Test set was not used for model selection or evaluation.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_log(log_path, f"Saved baseline summary to {summary_path}.")
    write_log(log_path, "Task 3 baseline modeling pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
