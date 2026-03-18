from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "outputs" / "benchmark" / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASELINE_RESULTS_PATH = BASE_DIR / "outputs" / "metrics" / "baseline_model_results.csv"
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
ALL_RESULTS_PATH = METRICS_DIR / "all_results.csv"
TUNING_RESULTS_PATH = METRICS_DIR / "tuning_results.csv"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
FINAL_MODEL_RESULTS_PATH = METRICS_DIR / "final_model_results.csv"
VALIDATION_CURVE_GB_PATH = FIGURES_DIR / "validation_curve_gb.png"
FINAL_SUMMARY_PATH = DOCS_DIR / "model_improvement_summary.txt"
RESIDUAL_DISTRIBUTION_PATH = FIGURES_DIR / "residual_distribution.png"
MAE_BY_HOUR_PATH = FIGURES_DIR / "mae_by_hour.png"
MAE_BY_WEEKDAY_PATH = FIGURES_DIR / "mae_by_weekday.png"
RESIDUAL_VS_TEMPERATURE_PATH = FIGURES_DIR / "residual_vs_temperature.png"
ROLLING_MAE_OVER_TIME_PATH = FIGURES_DIR / "rolling_mae_over_time.png"

TARGET_COLUMN = "cnt"
F0_FEATURES = ["hr", "weekday", "workingday", "season", "mnth", "yr", "weathersit", "temp", "atemp", "hum", "windspeed"]
F1_FEATURES = F0_FEATURES + ["sin_hour", "cos_hour", "sin_month", "cos_month"]
FEATURE_SETS = {"F0": F0_FEATURES, "F1": F1_FEATURES}


RIDGE_CONFIGS = [
    {"alpha": 0.1},
    {"alpha": 1.0},
    {"alpha": 10.0},
    {"alpha": 100.0},
]
RF_CONFIGS = [
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 4},
    {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 4},
]
GB_CONFIGS = [
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 2},
    {"n_estimators": 300, "learning_rate": 0.10, "max_depth": 3},
    {"n_estimators": 400, "learning_rate": 0.10, "max_depth": 3},
]
MLP_CONFIGS = [
    {"hidden_layer_sizes": (64,), "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128,), "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (64, 32), "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128,), "alpha": 0.0010, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (64,), "alpha": 0.0010, "learning_rate_init": 0.010},
]
MODEL_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "Ridge": RIDGE_CONFIGS,
    "RandomForestRegressor": RF_CONFIGS,
    "GradientBoostingRegressor": GB_CONFIGS,
    "MLPRegressor": MLP_CONFIGS,
}


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def log_step(message: str) -> None:
    print(message)
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not VAL_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Expected outputs/train.csv, outputs/val.csv, and outputs/test.csv from Task 3.")
    return load_split(TRAIN_PATH), load_split(VAL_PATH), load_split(TEST_PATH)


def build_estimator(model_name: str, config: dict[str, Any]):
    if model_name == "Ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=config["alpha"])),
            ]
        )
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
    if model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=RANDOM_SEED,
        )
    if model_name == "MLPRegressor":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=config["hidden_layer_sizes"],
                        alpha=config["alpha"],
                        learning_rate_init=config["learning_rate_init"],
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(model_name: str, config: dict[str, Any], feature_set_name: str, features: list[str], train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict[str, Any], Any, np.ndarray]:
    estimator = build_estimator(model_name, config)
    X_train = train_df[features].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_val = val_df[features].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - start_time
    val_predictions = estimator.predict(X_val)

    mae = mean_absolute_error(y_val, val_predictions)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    row = {
        "model": model_name,
        "feature_set": feature_set_name,
        "split": "validation",
        "params": repr(config),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "training_time_seconds": float(training_time_seconds),
    }
    return row, estimator, val_predictions


def select_best_rows(tuning_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = tuning_df.sort_values(["model", "feature_set", "MAE", "RMSE", "training_time_seconds"], ascending=[True, True, True, True, True])
    best_df = sorted_df.groupby(["model", "feature_set"], as_index=False).first()
    return best_df[["model", "feature_set", "split", "MAE", "RMSE", "training_time_seconds"]]


def save_validation_curve_gb(gb_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    for feature_set_name in sorted(gb_df["feature_set"].unique()):
        subset = gb_df[gb_df["feature_set"] == feature_set_name].copy()
        summary = subset.groupby("n_estimators", as_index=False)["MAE"].min().sort_values("n_estimators")
        plt.plot(summary["n_estimators"], summary["MAE"], marker="o", linewidth=2, label=feature_set_name)
    plt.title("Gradient Boosting Validation MAE vs n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("Validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VALIDATION_CURVE_GB_PATH, dpi=150)
    plt.close()


def retrain_best_model(best_row: pd.Series, train_val_df: pd.DataFrame):
    model_name = str(best_row["model"])
    feature_set_name = str(best_row["feature_set"])
    params = eval(best_row["params"], {"__builtins__": {}}, {})
    features = FEATURE_SETS[feature_set_name]
    estimator = build_estimator(model_name, params)
    estimator.fit(train_val_df[features], train_val_df[TARGET_COLUMN])
    return estimator, features, params


def save_final_artifacts(test_df: pd.DataFrame, predictions: np.ndarray, model_name: str, feature_set_name: str) -> None:
    residuals = test_df[TARGET_COLUMN].to_numpy() - predictions

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, color="#6c9a8b", edgecolor="white")
    plt.axvline(0.0, color="#d1495b", linewidth=2)
    plt.title("Residual Distribution on Test Set")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESIDUAL_DISTRIBUTION_PATH, dpi=150)
    plt.close()

    eval_df = test_df.copy()
    eval_df["prediction"] = predictions
    eval_df["absolute_error"] = np.abs(eval_df[TARGET_COLUMN] - eval_df["prediction"])
    eval_df["residual"] = eval_df[TARGET_COLUMN] - eval_df["prediction"]

    mae_by_hour = eval_df.groupby("hr", as_index=False)["absolute_error"].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(mae_by_hour["hr"], mae_by_hour["absolute_error"], color="#2f6f8f")
    plt.title(f"Test MAE by Hour ({model_name}, {feature_set_name})")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(MAE_BY_HOUR_PATH, dpi=150)
    plt.close()

    mae_by_weekday = eval_df.groupby("weekday", as_index=False)["absolute_error"].mean()
    plt.figure(figsize=(9, 5))
    plt.bar(mae_by_weekday["weekday"], mae_by_weekday["absolute_error"], color="#0d3b66")
    plt.title(f"Test MAE by Weekday ({model_name}, {feature_set_name})")
    plt.xlabel("Weekday")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(MAE_BY_WEEKDAY_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(eval_df["temp"], eval_df["residual"], alpha=0.18, s=14, color="#7a9e7e")
    plt.axhline(0.0, color="#d1495b", linewidth=2)
    plt.title(f"Residual vs Temperature ({model_name}, {feature_set_name})")
    plt.xlabel("temp")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(RESIDUAL_VS_TEMPERATURE_PATH, dpi=150)
    plt.close()

    rolling_mae = eval_df.sort_values("timestamp").set_index("timestamp")["absolute_error"].rolling(window=168, min_periods=24).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_mae.index, rolling_mae.values, color="#2f6f8f", linewidth=1.5)
    plt.title(f"Rolling Test MAE Over Time ({model_name}, {feature_set_name})")
    plt.xlabel("Time")
    plt.ylabel("Rolling MAE (168-hour window)")
    plt.tight_layout()
    plt.savefig(ROLLING_MAE_OVER_TIME_PATH, dpi=150)
    plt.close()


def main() -> None:
    ensure_directories([FIGURES_DIR, METRICS_DIR, MODELS_DIR, DOCS_DIR, BENCHMARK_DIR])

    train_df, val_df, test_df = load_splits()
    log_step(f"Task 4 model improvement started with random seed {RANDOM_SEED}.")
    log_step(f"Loaded train/val/test shapes: {train_df.shape}, {val_df.shape}, {test_df.shape}.")

    tuning_rows: list[dict[str, Any]] = []

    for model_name, config_list in MODEL_CONFIGS.items():
        log_step(f"Evaluating model family: {model_name}")
        for feature_set_name, features in FEATURE_SETS.items():
            for config in config_list:
                row, _, _ = evaluate_model(model_name, config, feature_set_name, features, train_df, val_df)
                if model_name == "GradientBoostingRegressor":
                    row["n_estimators"] = config["n_estimators"]
                tuning_rows.append(row)
                log_step(
                    f"{model_name} on {feature_set_name} with {config}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, training_time_seconds={row['training_time_seconds']:.6f}"
                )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(TUNING_RESULTS_PATH, index=False)
    log_step(f"Saved tuning results to {TUNING_RESULTS_PATH}.")

    all_results_df = select_best_rows(tuning_df)
    all_results_df.to_csv(ALL_RESULTS_PATH, index=False)
    log_step(f"Saved best validation results per model/feature set to {ALL_RESULTS_PATH}.")

    gb_df = tuning_df[tuning_df["model"] == "GradientBoostingRegressor"].copy()
    save_validation_curve_gb(gb_df)
    log_step(f"Saved Gradient Boosting validation curve to {VALIDATION_CURVE_GB_PATH}.")

    best_validation_row = tuning_df.sort_values(["MAE", "RMSE", "training_time_seconds"], ascending=[True, True, True]).iloc[0]
    log_step(
        f"Best validation model: {best_validation_row['model']} with {best_validation_row['feature_set']} and params {best_validation_row['params']} (MAE={best_validation_row['MAE']:.4f}, RMSE={best_validation_row['RMSE']:.4f})."
    )

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model, final_features, final_params = retrain_best_model(best_validation_row, train_val_df)
    with FINAL_MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "model_name": str(best_validation_row["model"]),
                "feature_set": str(best_validation_row["feature_set"]),
                "params": final_params,
                "features": final_features,
                "estimator": final_model,
            },
            model_file,
        )
    log_step(f"Saved final model to {FINAL_MODEL_PATH}.")

    test_predictions = final_model.predict(test_df[final_features])
    test_mae = mean_absolute_error(test_df[TARGET_COLUMN], test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], test_predictions))
    final_results_df = pd.DataFrame(
        [
            {
                "model": str(best_validation_row["model"]),
                "feature_set": str(best_validation_row["feature_set"]),
                "split": "test",
                "MAE": float(test_mae),
                "RMSE": float(test_rmse),
            }
        ]
    )
    final_results_df.to_csv(FINAL_MODEL_RESULTS_PATH, index=False)
    log_step(f"Saved final test results to {FINAL_MODEL_RESULTS_PATH}.")

    save_final_artifacts(test_df, test_predictions, str(best_validation_row["model"]), str(best_validation_row["feature_set"]))
    log_step("Saved final evaluation plots to outputs/figures.")

    baseline_note = "Baseline results file not found."
    if BASELINE_RESULTS_PATH.exists():
        baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
        baseline_best = baseline_df.sort_values(["MAE", "RMSE"], ascending=[True, True]).iloc[0]
        mae_delta = float(baseline_best["MAE"] - best_validation_row["MAE"])
        rmse_delta = float(baseline_best["RMSE"] - best_validation_row["RMSE"])
        baseline_note = (
            f"Best baseline validation result was {baseline_best['model']} {baseline_best['feature_set']} with MAE={baseline_best['MAE']:.4f} and RMSE={baseline_best['RMSE']:.4f}. "
            f"Selected model changed validation MAE by {mae_delta:.4f} and RMSE by {rmse_delta:.4f}."
        )

    summary_lines = [
        f"Validation winner: {best_validation_row['model']} on {best_validation_row['feature_set']} with params {best_validation_row['params']}.",
        f"Validation metrics: MAE={best_validation_row['MAE']:.4f}, RMSE={best_validation_row['RMSE']:.4f}.",
        f"Final test metrics: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}.",
        baseline_note,
        "Model selection used validation data only. Test data was used once for final evaluation after selection.",
    ]
    FINAL_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log_step(f"Saved model improvement summary to {FINAL_SUMMARY_PATH}.")

    print("\nModel improvement summary")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
