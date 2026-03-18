from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import time
import ast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor


RANDOM_SEED = 42
N_JOBS = 1


def write_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "sin_hour" not in result.columns:
        result["sin_hour"] = np.sin(2 * np.pi * result["hr"] / 24.0)
    if "cos_hour" not in result.columns:
        result["cos_hour"] = np.cos(2 * np.pi * result["hr"] / 24.0)
    if "sin_month" not in result.columns:
        result["sin_month"] = np.sin(2 * np.pi * result["mnth"] / 12.0)
    if "cos_month" not in result.columns:
        result["cos_month"] = np.cos(2 * np.pi * result["mnth"] / 12.0)
    return result


def make_model(model_name: str, params: dict[str, object]):
    if model_name == "Ridge":
        return Ridge(**params)
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    if model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**params)
    if model_name == "MLPRegressor":
        return MLPRegressor(**params)
    raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(
    model_name: str,
    params: dict[str, object],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    target: str = "cnt",
) -> tuple[object, np.ndarray, float, float, float]:
    model = make_model(model_name, params)
    x_train = train_df[features]
    y_train = train_df[target]
    x_eval = eval_df[features]
    y_eval = eval_df[target]

    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_sec = float(time.perf_counter() - start)

    pred = model.predict(x_eval)
    mae = float(mean_absolute_error(y_eval, pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, pred)))
    return model, pred, mae, rmse, train_sec


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
    write_log(log_path, "Started Task 4 model improvement pipeline.")
    write_log(log_path, f"Random seed fixed at {RANDOM_SEED}; n_jobs fixed at {N_JOBS}.")

    train_path = outputs_root / "train.csv"
    val_path = outputs_root / "val.csv"
    test_path = outputs_root / "test.csv"
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError("Expected outputs/train.csv, outputs/val.csv, outputs/test.csv from Task 3.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    train_df = add_cyclical_features(train_df)
    val_df = add_cyclical_features(val_df)
    test_df = add_cyclical_features(test_df)
    write_log(
        log_path,
        f"Loaded splits: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}.",
    )

    f0 = ["hr", "weekday", "workingday", "season", "mnth", "yr", "weathersit", "temp", "atemp", "hum", "windspeed"]
    f1 = f0 + ["sin_hour", "cos_hour", "sin_month", "cos_month"]
    feature_sets = {"F0": f0, "F1": f1}
    target = "cnt"

    defaults = {
        "Ridge": {"alpha": 1.0},
        "RandomForestRegressor": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "random_state": RANDOM_SEED, "n_jobs": N_JOBS},
        "GradientBoostingRegressor": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": RANDOM_SEED},
        "MLPRegressor": {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.001, "learning_rate_init": 0.001, "max_iter": 400, "early_stopping": True, "random_state": RANDOM_SEED},
    }

    all_results: list[dict[str, object]] = []
    for model_name, base_params in defaults.items():
        for fs_name, features in feature_sets.items():
            _, _, mae, rmse, train_sec = evaluate_model(
                model_name=model_name,
                params=base_params,
                train_df=train_df,
                eval_df=val_df,
                features=features,
                target=target,
            )
            all_results.append(
                {
                    "model": model_name,
                    "feature_set": fs_name,
                    "split": "validation",
                    "MAE": mae,
                    "RMSE": rmse,
                    "training_time_seconds": train_sec,
                }
            )
            write_log(log_path, f"Comparison {model_name} on {fs_name}: MAE={mae:.4f}, RMSE={rmse:.4f}.")

    all_results_df = pd.DataFrame(all_results)
    all_results_path = metrics_dir / "all_results.csv"
    all_results_df.to_csv(all_results_path, index=False)
    write_log(log_path, f"Saved comparison results to {all_results_path}.")

    tuning_rows: list[dict[str, object]] = []
    best_by_tuning: dict[str, object] | None = None

    def record_tuning(model_name: str, fs_name: str, params: dict[str, object], mae: float, rmse: float, train_sec: float) -> None:
        nonlocal best_by_tuning
        row = {
            "model": model_name,
            "feature_set": fs_name,
            "split": "validation",
            "params": str(params),
            "MAE": mae,
            "RMSE": rmse,
            "training_time_seconds": train_sec,
        }
        tuning_rows.append(row)
        if (best_by_tuning is None) or (mae < float(best_by_tuning["MAE"])):
            best_by_tuning = row.copy()

    for fs_name, features in feature_sets.items():
        for alpha in [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]:
            params = {"alpha": alpha}
            _, _, mae, rmse, train_sec = evaluate_model("Ridge", params, train_df, val_df, features, target)
            record_tuning("Ridge", fs_name, params, mae, rmse, train_sec)

    for fs_name, features in feature_sets.items():
        for n_estimators in [200, 400]:
            for max_depth in [None, 10, 20]:
                for min_samples_leaf in [1, 2, 4]:
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "random_state": RANDOM_SEED,
                        "n_jobs": N_JOBS,
                    }
                    _, _, mae, rmse, train_sec = evaluate_model("RandomForestRegressor", params, train_df, val_df, features, target)
                    record_tuning("RandomForestRegressor", fs_name, params, mae, rmse, train_sec)

    gb_curve_rows: list[dict[str, object]] = []
    for fs_name, features in feature_sets.items():
        for n_estimators in [50, 100, 150, 200, 300, 400]:
            params = {
                "n_estimators": n_estimators,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": RANDOM_SEED,
            }
            _, _, mae, rmse, train_sec = evaluate_model("GradientBoostingRegressor", params, train_df, val_df, features, target)
            record_tuning("GradientBoostingRegressor", fs_name, params, mae, rmse, train_sec)
            gb_curve_rows.append({"feature_set": fs_name, "n_estimators": n_estimators, "MAE": mae})

        for learning_rate in [0.03, 0.05, 0.1]:
            for max_depth in [2, 3]:
                params = {
                    "n_estimators": 300,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "random_state": RANDOM_SEED,
                }
                _, _, mae, rmse, train_sec = evaluate_model("GradientBoostingRegressor", params, train_df, val_df, features, target)
                record_tuning("GradientBoostingRegressor", fs_name, params, mae, rmse, train_sec)

    for fs_name, features in feature_sets.items():
        for hidden in [(64,), (128,), (64, 32)]:
            for alpha in [1e-4, 1e-3]:
                for learning_rate_init in [0.001, 0.01]:
                    params = {
                        "hidden_layer_sizes": hidden,
                        "activation": "relu",
                        "alpha": alpha,
                        "learning_rate_init": learning_rate_init,
                        "max_iter": 500,
                        "early_stopping": True,
                        "random_state": RANDOM_SEED,
                    }
                    _, _, mae, rmse, train_sec = evaluate_model("MLPRegressor", params, train_df, val_df, features, target)
                    record_tuning("MLPRegressor", fs_name, params, mae, rmse, train_sec)

    tuning_df = pd.DataFrame(tuning_rows).sort_values("MAE", ascending=True).reset_index(drop=True)
    tuning_path = metrics_dir / "tuning_results.csv"
    tuning_df.to_csv(tuning_path, index=False)
    write_log(log_path, f"Saved tuning results to {tuning_path}.")

    gb_curve_df = pd.DataFrame(gb_curve_rows)
    plt.figure(figsize=(8, 6))
    for fs_name in ["F0", "F1"]:
        subset = gb_curve_df[gb_curve_df["feature_set"] == fs_name].sort_values("n_estimators")
        plt.plot(subset["n_estimators"], subset["MAE"], marker="o", linewidth=1.8, label=fs_name)
    plt.xlabel("n_estimators")
    plt.ylabel("Validation MAE")
    plt.title("Gradient Boosting Validation Curve")
    plt.legend()
    plt.tight_layout()
    gb_curve_path = figures_dir / "validation_curve_gb.png"
    plt.savefig(gb_curve_path, dpi=150)
    plt.close()
    write_log(log_path, f"Saved GB validation curve to {gb_curve_path}.")

    if best_by_tuning is None:
        raise RuntimeError("No tuning results were produced.")

    best_model_name = str(best_by_tuning["model"])
    best_feature_set_name = str(best_by_tuning["feature_set"])
    best_params = ast.literal_eval(str(best_by_tuning["params"]))
    best_features = feature_sets[best_feature_set_name]
    write_log(
        log_path,
        f"Best model by validation MAE: {best_model_name} on {best_feature_set_name} with params={best_params}.",
    )

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = make_model(best_model_name, best_params)
    x_train_val = train_val_df[best_features]
    y_train_val = train_val_df[target]
    final_model.fit(x_train_val, y_train_val)
    model_bundle = {
        "model_name": best_model_name,
        "feature_set": best_feature_set_name,
        "features": best_features,
        "params": best_params,
        "estimator": final_model,
    }
    final_model_path = models_dir / "final_model.pkl"
    with final_model_path.open("wb") as f:
        pickle.dump(model_bundle, f)
    write_log(log_path, f"Saved final model bundle to {final_model_path}.")

    x_test = test_df[best_features]
    y_test = test_df[target].to_numpy()
    y_pred_test = final_model.predict(x_test)
    residuals = y_test - y_pred_test
    abs_errors = np.abs(residuals)

    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    final_results_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "feature_set": best_feature_set_name,
                "split": "test",
                "MAE": test_mae,
                "RMSE": test_rmse,
            }
        ]
    )
    final_results_path = metrics_dir / "final_model_results.csv"
    final_results_df.to_csv(final_results_path, index=False)
    write_log(log_path, f"Saved final test metrics to {final_results_path}.")

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40, color="#1f77b4", alpha=0.85)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution (Test)")
    plt.tight_layout()
    plt.savefig(figures_dir / "residual_distribution.png", dpi=150)
    plt.close()

    if "hr" in test_df.columns:
        mae_by_hour = pd.DataFrame({"hr": test_df["hr"], "abs_error": abs_errors}).groupby("hr", as_index=False)["abs_error"].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(mae_by_hour["hr"], mae_by_hour["abs_error"], marker="o", linewidth=1.7, color="#ff7f0e")
        plt.xlabel("Hour")
        plt.ylabel("MAE")
        plt.title("MAE by Hour (Test)")
        plt.tight_layout()
        plt.savefig(figures_dir / "mae_by_hour.png", dpi=150)
        plt.close()

    if "weekday" in test_df.columns:
        mae_by_weekday = pd.DataFrame({"weekday": test_df["weekday"], "abs_error": abs_errors}).groupby("weekday", as_index=False)["abs_error"].mean()
        plt.figure(figsize=(8, 5))
        plt.plot(mae_by_weekday["weekday"], mae_by_weekday["abs_error"], marker="o", linewidth=1.7, color="#2ca02c")
        plt.xlabel("Weekday")
        plt.ylabel("MAE")
        plt.title("MAE by Weekday (Test)")
        plt.tight_layout()
        plt.savefig(figures_dir / "mae_by_weekday.png", dpi=150)
        plt.close()

    if "temp" in test_df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(test_df["temp"], residuals, s=10, alpha=0.35, c="#9467bd", edgecolors="none")
        plt.axhline(0.0, color="black", linewidth=1.0)
        plt.xlabel("Temperature (temp)")
        plt.ylabel("Residual")
        plt.title("Residual vs Temperature (Test)")
        plt.tight_layout()
        plt.savefig(figures_dir / "residual_vs_temperature.png", dpi=150)
        plt.close()

    time_series_df = test_df.copy()
    if "datetime" in time_series_df.columns:
        time_series_df["datetime"] = pd.to_datetime(time_series_df["datetime"], errors="coerce")
    elif {"dteday", "hr"}.issubset(time_series_df.columns):
        time_series_df["datetime"] = pd.to_datetime(time_series_df["dteday"], errors="coerce") + pd.to_timedelta(time_series_df["hr"], unit="h")
    else:
        time_series_df["datetime"] = np.arange(len(time_series_df))
    time_series_df["abs_error"] = abs_errors
    time_series_df = time_series_df.sort_values("datetime").reset_index(drop=True)
    rolling_window = min(168, max(24, len(time_series_df) // 10))
    time_series_df["rolling_mae"] = time_series_df["abs_error"].rolling(window=rolling_window, min_periods=1).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(time_series_df["datetime"], time_series_df["rolling_mae"], color="#d62728", linewidth=1.7)
    plt.xlabel("Time")
    plt.ylabel("Rolling MAE")
    plt.title(f"Rolling MAE Over Time (window={rolling_window})")
    plt.tight_layout()
    plt.savefig(figures_dir / "rolling_mae_over_time.png", dpi=150)
    plt.close()

    write_log(log_path, "Saved final evaluation figures for residuals and MAE diagnostics.")

    summary_path = docs_dir / "model_improvement_summary.txt"
    summary_lines = [
        "Task 4 Model Improvement Summary",
        f"Best validation model: {best_model_name} ({best_feature_set_name})",
        f"Best validation MAE: {float(best_by_tuning['MAE']):.4f}",
        f"Best validation RMSE: {float(best_by_tuning['RMSE']):.4f}",
        f"Test MAE: {test_mae:.4f}",
        f"Test RMSE: {test_rmse:.4f}",
        "Model selection used validation only; test was used only for final evaluation.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_log(log_path, f"Saved Task 4 summary to {summary_path}.")
    write_log(log_path, "Task 4 model improvement pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
