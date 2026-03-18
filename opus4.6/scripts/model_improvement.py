"""
Task 4: Model Comparison, Tuning & Final Evaluation
Ridge, Random Forest, Gradient Boosting, MLP — evaluated on F0 and F1.
Best model selected by validation MAE, then evaluated on held-out test set.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark")
EXPERIMENT_LOG = os.path.join(BENCHMARK_DIR, "experiment_log.txt")

for d in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, BENCHMARK_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(42)
RANDOM_STATE = 42
TARGET = "cnt"

F0_COLS = [
    "hr", "weekday", "workingday", "season", "mnth", "yr",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
F1_COLS = F0_COLS + ["sin_hour", "cos_hour", "sin_month", "cos_month"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(msg + "\n")


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    log("\n" + "=" * 60)
    log("TASK 4: Model Comparison, Tuning & Final Evaluation")
    log("=" * 60)

    # ==================================================================
    # 0. LOAD SPLITS
    # ==================================================================
    df_train = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(OUTPUT_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(OUTPUT_DIR, "test.csv"))
    log(f"\n[0] Loaded splits — train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ==================================================================
    # 1. MODEL COMPARISON  (Ridge, RF, GBR, MLP)  x  (F0, F1)
    # ==================================================================
    log("\n[1] Model comparison on validation set...")

    # Scalers for MLP — fit on train only
    scaler_f0 = StandardScaler().fit(df_train[F0_COLS])
    scaler_f1 = StandardScaler().fit(df_train[F1_COLS])

    model_defs = [
        ("Ridge", lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE), False),
        ("RandomForest", lambda: RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=1), False),
        ("GradientBoosting", lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            min_samples_leaf=10, random_state=RANDOM_STATE), False),
        ("MLP", lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=RANDOM_STATE), True),
    ]

    all_results = []
    trained_models = {}  # key: (model_name, feat_name) -> (model, scaler_or_None)

    for model_name, model_fn, needs_scaling in model_defs:
        for feat_name, feat_cols, scaler in [
            ("F0", F0_COLS, scaler_f0),
            ("F1", F1_COLS, scaler_f1),
        ]:
            X_train = df_train[feat_cols].values
            y_train = df_train[TARGET].values
            X_val = df_val[feat_cols].values
            y_val = df_val[TARGET].values

            sc = None
            if needs_scaling:
                sc = StandardScaler().fit(X_train)
                X_train = sc.transform(X_train)
                X_val = sc.transform(X_val)

            model = model_fn()
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred_tr = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            mae_tr, rmse_tr = compute_metrics(y_train, y_pred_tr)
            mae_val, rmse_val = compute_metrics(y_val, y_pred_val)

            for split, mae, rmse in [("train", mae_tr, rmse_tr), ("val", mae_val, rmse_val)]:
                all_results.append({
                    "model": model_name,
                    "feature_set": feat_name,
                    "split": split,
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "training_time_seconds": round(train_time, 4),
                })

            trained_models[(model_name, feat_name)] = (model, sc)
            log(f"  {model_name:20s} {feat_name}  "
                f"train MAE={mae_tr:7.2f}  val MAE={mae_val:7.2f}  "
                f"val RMSE={rmse_val:7.2f}  ({train_time:.2f}s)")

    # Save all_results.csv (append baseline results from Task 3)
    baseline_path = os.path.join(METRICS_DIR, "baseline_model_results.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        combined_df = pd.concat([baseline_df, pd.DataFrame(all_results)], ignore_index=True)
    else:
        combined_df = pd.DataFrame(all_results)

    combined_df.to_csv(os.path.join(METRICS_DIR, "all_results.csv"), index=False)
    log(f"\n  All results (incl. baseline) saved to outputs/metrics/all_results.csv")

    # ==================================================================
    # 2. TUNING
    # ==================================================================
    log("\n[2] Hyperparameter tuning...")

    tuning_results = []

    # --- 2a. Ridge: sweep alpha ---
    log("\n  [2a] Ridge alpha sweep...")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for feat_name, feat_cols in [("F0", F0_COLS), ("F1", F1_COLS)]:
            X_tr = df_train[feat_cols].values
            y_tr = df_train[TARGET].values
            X_v = df_val[feat_cols].values
            y_v = df_val[TARGET].values

            m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
            t0 = time.time()
            m.fit(X_tr, y_tr)
            tt = time.time() - t0
            mae_v, rmse_v = compute_metrics(y_v, m.predict(X_v))

            tuning_results.append({
                "model": "Ridge", "feature_set": feat_name,
                "param": f"alpha={alpha}", "val_MAE": round(mae_v, 4),
                "val_RMSE": round(rmse_v, 4), "training_time_seconds": round(tt, 4),
            })
            log(f"    Ridge {feat_name} alpha={alpha:6.2f}  "
                f"val MAE={mae_v:.2f}  RMSE={rmse_v:.2f}")

    # --- 2b. Random Forest: sweep n_estimators + max_depth ---
    log("\n  [2b] Random Forest sweep...")
    for n_est in [100, 200, 400]:
        for md in [10, 15, 20]:
            for feat_name, feat_cols in [("F0", F0_COLS), ("F1", F1_COLS)]:
                X_tr = df_train[feat_cols].values
                y_tr = df_train[TARGET].values
                X_v = df_val[feat_cols].values
                y_v = df_val[TARGET].values

                m = RandomForestRegressor(
                    n_estimators=n_est, max_depth=md, min_samples_leaf=5,
                    random_state=RANDOM_STATE, n_jobs=1)
                t0 = time.time()
                m.fit(X_tr, y_tr)
                tt = time.time() - t0
                mae_v, rmse_v = compute_metrics(y_v, m.predict(X_v))

                tuning_results.append({
                    "model": "RandomForest", "feature_set": feat_name,
                    "param": f"n_est={n_est},max_depth={md}",
                    "val_MAE": round(mae_v, 4), "val_RMSE": round(rmse_v, 4),
                    "training_time_seconds": round(tt, 4),
                })
                log(f"    RF {feat_name} n={n_est} d={md}  "
                    f"val MAE={mae_v:.2f}  RMSE={rmse_v:.2f}  ({tt:.1f}s)")

    # --- 2c. Gradient Boosting: sweep n_estimators (+ validation curve plot) ---
    log("\n  [2c] Gradient Boosting n_estimators sweep...")
    gb_n_est_values = [50, 100, 200, 300, 500]
    gb_curve_data = {"n_estimators": [], "feat_set": [], "val_MAE": []}

    for n_est in gb_n_est_values:
        for lr in [0.05, 0.1]:
            for feat_name, feat_cols in [("F0", F0_COLS), ("F1", F1_COLS)]:
                X_tr = df_train[feat_cols].values
                y_tr = df_train[TARGET].values
                X_v = df_val[feat_cols].values
                y_v = df_val[TARGET].values

                m = GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=6, learning_rate=lr,
                    min_samples_leaf=10, random_state=RANDOM_STATE)
                t0 = time.time()
                m.fit(X_tr, y_tr)
                tt = time.time() - t0
                mae_v, rmse_v = compute_metrics(y_v, m.predict(X_v))

                tuning_results.append({
                    "model": "GradientBoosting", "feature_set": feat_name,
                    "param": f"n_est={n_est},lr={lr}",
                    "val_MAE": round(mae_v, 4), "val_RMSE": round(rmse_v, 4),
                    "training_time_seconds": round(tt, 4),
                })
                log(f"    GB {feat_name} n={n_est} lr={lr}  "
                    f"val MAE={mae_v:.2f}  RMSE={rmse_v:.2f}  ({tt:.1f}s)")

                # Collect data for validation curve (lr=0.1 only)
                if lr == 0.1:
                    gb_curve_data["n_estimators"].append(n_est)
                    gb_curve_data["feat_set"].append(feat_name)
                    gb_curve_data["val_MAE"].append(mae_v)

    # --- Validation curve plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    curve_df = pd.DataFrame(gb_curve_data)
    for fs in ["F0", "F1"]:
        sub = curve_df[curve_df["feat_set"] == fs].sort_values("n_estimators")
        ax.plot(sub["n_estimators"], sub["val_MAE"], marker="o", label=fs)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Gradient Boosting: Validation MAE vs n_estimators (lr=0.1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "validation_curve_gb.png"), bbox_inches="tight")
    plt.close(fig)
    log("\n  Saved validation_curve_gb.png")

    # --- 2d. MLP: sweep architecture ---
    log("\n  [2d] MLP architecture sweep...")
    mlp_configs = [
        (64, 32),
        (128, 64),
        (256, 128),
        (128, 64, 32),
    ]
    for hidden in mlp_configs:
        for feat_name, feat_cols in [("F0", F0_COLS), ("F1", F1_COLS)]:
            X_tr = df_train[feat_cols].values
            y_tr = df_train[TARGET].values
            X_v = df_val[feat_cols].values
            y_v = df_val[TARGET].values

            sc = StandardScaler().fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_v_s = sc.transform(X_v)

            m = MLPRegressor(
                hidden_layer_sizes=hidden, max_iter=500, early_stopping=True,
                validation_fraction=0.1, random_state=RANDOM_STATE)
            t0 = time.time()
            m.fit(X_tr_s, y_tr)
            tt = time.time() - t0
            mae_v, rmse_v = compute_metrics(y_v, m.predict(X_v_s))

            tuning_results.append({
                "model": "MLP", "feature_set": feat_name,
                "param": f"hidden={hidden}",
                "val_MAE": round(mae_v, 4), "val_RMSE": round(rmse_v, 4),
                "training_time_seconds": round(tt, 4),
            })
            log(f"    MLP {feat_name} hidden={str(hidden):18s}  "
                f"val MAE={mae_v:.2f}  RMSE={rmse_v:.2f}  ({tt:.1f}s)")

    # Save tuning results
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(os.path.join(METRICS_DIR, "tuning_results.csv"), index=False)
    log(f"\n  Tuning results saved to outputs/metrics/tuning_results.csv")

    # ==================================================================
    # 3. SELECT BEST MODEL (lowest val MAE across tuning)
    # ==================================================================
    log("\n[3] Selecting best model...")

    best_row = tuning_df.loc[tuning_df["val_MAE"].idxmin()]
    best_model_name = best_row["model"]
    best_feat_name = best_row["feature_set"]
    best_param = best_row["param"]
    best_val_mae = best_row["val_MAE"]
    best_val_rmse = best_row["val_RMSE"]

    log(f"  Best: {best_model_name} | {best_feat_name} | {best_param}")
    log(f"        val MAE={best_val_mae:.4f}  RMSE={best_val_rmse:.4f}")

    # Retrain best model on training set with best params
    feat_cols = F0_COLS if best_feat_name == "F0" else F1_COLS
    X_train_final = df_train[feat_cols].values
    y_train_final = df_train[TARGET].values

    best_scaler = None

    if best_model_name == "Ridge":
        alpha_val = float(best_param.split("=")[1])
        best_model = Ridge(alpha=alpha_val, random_state=RANDOM_STATE)
    elif best_model_name == "RandomForest":
        parts = best_param.split(",")
        n_est = int(parts[0].split("=")[1])
        md = int(parts[1].split("=")[1])
        best_model = RandomForestRegressor(
            n_estimators=n_est, max_depth=md, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=1)
    elif best_model_name == "GradientBoosting":
        parts = best_param.split(",")
        n_est = int(parts[0].split("=")[1])
        lr = float(parts[1].split("=")[1])
        best_model = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=6, learning_rate=lr,
            min_samples_leaf=10, random_state=RANDOM_STATE)
    elif best_model_name == "MLP":
        hidden = eval(best_param.split("=", 1)[1])  # safe: controlled string
        best_scaler = StandardScaler().fit(X_train_final)
        X_train_final = best_scaler.transform(X_train_final)
        best_model = MLPRegressor(
            hidden_layer_sizes=hidden, max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=RANDOM_STATE)

    best_model.fit(X_train_final, y_train_final)

    # Save model
    model_bundle = {
        "model": best_model,
        "scaler": best_scaler,
        "feature_cols": feat_cols,
        "feature_set": best_feat_name,
        "model_name": best_model_name,
        "params": best_param,
    }
    model_path = os.path.join(MODELS_DIR, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    log(f"  Best model saved to {model_path}")

    # ==================================================================
    # 4. FINAL EVALUATION ON TEST SET
    # ==================================================================
    log("\n[4] Final evaluation on TEST set...")

    X_test = df_test[feat_cols].values
    y_test = df_test[TARGET].values
    if best_scaler is not None:
        X_test = best_scaler.transform(X_test)

    y_pred_test = best_model.predict(X_test)
    test_mae, test_rmse = compute_metrics(y_test, y_pred_test)

    log(f"  TEST MAE  = {test_mae:.4f}")
    log(f"  TEST RMSE = {test_rmse:.4f}")

    # Save final results
    final_results = pd.DataFrame([{
        "model": best_model_name,
        "feature_set": best_feat_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "RMSE": round(test_rmse, 4),
    }])
    final_results.to_csv(
        os.path.join(METRICS_DIR, "final_model_results.csv"), index=False)
    log(f"  Final results saved to outputs/metrics/final_model_results.csv")

    # ==================================================================
    # 5. DIAGNOSTIC PLOTS (test set)
    # ==================================================================
    log("\n[5] Generating diagnostic plots on test set...")

    residuals = y_test - y_pred_test

    # --- 5a. residual_distribution.png (overwrite with test-set version) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual (actual - predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution (Test Set)")
    axes[1].scatter(y_pred_test, residuals, alpha=0.2, s=10, color="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted cnt")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted (Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "residual_distribution.png"),
                bbox_inches="tight")
    plt.close(fig)
    log("  Saved residual_distribution.png")

    # --- 5b. mae_by_hour.png ---
    # Use original (unscaled) test data for grouping
    df_test_plot = df_test.copy()
    df_test_plot["pred"] = best_model.predict(
        best_scaler.transform(df_test[feat_cols].values) if best_scaler else df_test[feat_cols].values)
    df_test_plot["abs_error"] = np.abs(df_test_plot[TARGET] - df_test_plot["pred"])

    mae_hour = df_test_plot.groupby("hr")["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(mae_hour.index, mae_hour.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE by Hour (Test Set — {best_model_name})")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mae_by_hour.png"), bbox_inches="tight")
    plt.close(fig)
    log("  Saved mae_by_hour.png")

    # --- 5c. mae_by_weekday.png ---
    mae_wd = df_test_plot.groupby("weekday")["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mae_wd.index, mae_wd.values, color="teal", edgecolor="black")
    ax.set_xlabel("Weekday (0=Sun … 6=Sat)")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE by Weekday (Test Set — {best_model_name})")
    ax.set_xticks(range(7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mae_by_weekday.png"), bbox_inches="tight")
    plt.close(fig)
    log("  Saved mae_by_weekday.png")

    # --- 5d. residual_vs_temperature.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_test_plot["temp"], df_test_plot[TARGET] - df_test_plot["pred"],
               alpha=0.2, s=10, color="coral")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Normalized Temperature")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Temperature (Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "residual_vs_temperature.png"),
                bbox_inches="tight")
    plt.close(fig)
    log("  Saved residual_vs_temperature.png")

    # --- 5e. rolling_mae_over_time.png ---
    df_test_plot["dteday"] = pd.to_datetime(df_test_plot["dteday"])
    daily_mae = df_test_plot.groupby("dteday")["abs_error"].mean().reset_index()
    daily_mae.columns = ["date", "daily_mae"]
    daily_mae["rolling_mae_7d"] = daily_mae["daily_mae"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_mae["date"], daily_mae["daily_mae"],
            alpha=0.4, color="steelblue", label="Daily MAE")
    ax.plot(daily_mae["date"], daily_mae["rolling_mae_7d"],
            color="red", linewidth=2, label="7-day Rolling MAE")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAE")
    ax.set_title(f"Rolling MAE Over Time (Test Set — {best_model_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "rolling_mae_over_time.png"),
                bbox_inches="tight")
    plt.close(fig)
    log("  Saved rolling_mae_over_time.png")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    log("\n" + "=" * 60)
    log("TASK 4 SUMMARY")
    log("=" * 60)

    # Load baseline for comparison
    if os.path.exists(baseline_path):
        bl = pd.read_csv(baseline_path)
        bl_val = bl[(bl["split"] == "val") & (bl["feature_set"] == "F1")]
        if not bl_val.empty:
            bl_mae = bl_val["MAE"].values[0]
            improvement = (bl_mae - test_mae) / bl_mae * 100
            log(f"  Baseline (LR F1 val MAE):     {bl_mae:.2f}")
            log(f"  Best model (test MAE):         {test_mae:.2f}")
            log(f"  Improvement over baseline:     {improvement:+.1f}%")

    log(f"\n  Best model:     {best_model_name}")
    log(f"  Feature set:    {best_feat_name}")
    log(f"  Best params:    {best_param}")
    log(f"  Validation MAE: {best_val_mae:.4f}")
    log(f"  Test MAE:       {test_mae:.4f}")
    log(f"  Test RMSE:      {test_rmse:.4f}")
    log("=" * 60)


if __name__ == "__main__":
    run()
