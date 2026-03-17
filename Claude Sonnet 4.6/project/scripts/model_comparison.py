"""
Task 4: Model Comparison, Tuning, and Final Evaluation
Ridge, Random Forest, Gradient Boosting, MLP — F0 / F1 feature sets.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
MODELS_DIR  = os.path.join(OUT_DIR, "models")
BENCH_DIR   = os.path.join(OUT_DIR, "benchmark")
LOG_PATH    = os.path.join(BENCH_DIR, "experiment_log.txt")

for d in (FIG_DIR, METRICS_DIR, MODELS_DIR, BENCH_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ── Logger ────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def savefig(name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved figure: {name}")

# ── 1. Load splits ────────────────────────────────────────────────────────────
log("=" * 70)
log("TASK 4 — Model Comparison, Tuning & Final Evaluation")
log("=" * 70)

train = pd.read_csv(os.path.join(OUT_DIR, "train.csv"), parse_dates=["dteday"])
val   = pd.read_csv(os.path.join(OUT_DIR, "val.csv"),   parse_dates=["dteday"])
test  = pd.read_csv(os.path.join(OUT_DIR, "test.csv"),  parse_dates=["dteday"])

log(f"Loaded splits — train:{len(train)}  val:{len(val)}  test:{len(test)}")

# ── 2. Feature sets ───────────────────────────────────────────────────────────
F0 = ["hr", "weekday", "workingday", "season", "mnth", "yr",
      "weathersit", "temp", "atemp", "hum", "windspeed"]

def add_cyclical(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["sin_hour"]  = np.sin(2 * np.pi * d["hr"]   / 24)
    d["cos_hour"]  = np.cos(2 * np.pi * d["hr"]   / 24)
    d["sin_month"] = np.sin(2 * np.pi * d["mnth"]  / 12)
    d["cos_month"] = np.cos(2 * np.pi * d["mnth"]  / 12)
    return d

train = add_cyclical(train)
val   = add_cyclical(val)
test  = add_cyclical(test)

F1 = F0 + ["sin_hour", "cos_hour", "sin_month", "cos_month"]

y_train = train["cnt"].values
y_val   = val["cnt"].values
y_test  = test["cnt"].values

# Build scaled arrays — scaler always fit on train only
def make_arrays(fs_cols):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[fs_cols])
    X_vl = scaler.transform(val[fs_cols])
    X_te = scaler.transform(test[fs_cols])
    return X_tr, X_vl, X_te, scaler

arrays = {}
for fs_name, fs_cols in [("F0", F0), ("F1", F1)]:
    arrays[fs_name] = make_arrays(fs_cols)
    log(f"Scaler [{fs_name}] fit on train — {len(fs_cols)} features")

# ── 3. Model zoo — default / lightly-configured ──────────────────────────────
MODEL_CONFIGS = {
    "Ridge": Ridge(alpha=1.0, random_state=SEED),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=None,
        random_state=SEED, n_jobs=1),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        random_state=SEED),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu",
        max_iter=300, random_state=SEED),
}

# ── 4. Train & evaluate all models on F0 and F1 ──────────────────────────────
log("-" * 60)
log("Phase 1: Model comparison (default configs)")
log("-" * 60)

results = []

for model_name, model_proto in MODEL_CONFIGS.items():
    for fs_name, fs_cols in [("F0", F0), ("F1", F1)]:
        import copy
        model = copy.deepcopy(model_proto)
        X_tr, X_vl, X_te, _ = arrays[fs_name]

        t0 = time.perf_counter()
        model.fit(X_tr, y_train)
        train_time = time.perf_counter() - t0

        y_pred_tr = model.predict(X_tr)
        y_pred_vl = model.predict(X_vl)

        mae_tr  = mean_absolute_error(y_train, y_pred_tr)
        rmse_tr = rmse(y_train, y_pred_tr)
        mae_vl  = mean_absolute_error(y_val, y_pred_vl)
        rmse_vl = rmse(y_val, y_pred_vl)

        log(f"{model_name} [{fs_name}] — "
            f"train MAE={mae_tr:.2f} RMSE={rmse_tr:.2f} | "
            f"val MAE={mae_vl:.2f} RMSE={rmse_vl:.2f} | "
            f"time={train_time:.3f}s")

        for split, mae_v, rmse_v in [
                ("train", mae_tr, rmse_tr),
                ("val",   mae_vl, rmse_vl)]:
            results.append({
                "model": model_name, "feature_set": fs_name,
                "split": split,
                "MAE":  round(mae_v, 4),
                "RMSE": round(rmse_v, 4),
                "training_time_seconds": round(train_time, 6),
            })

# Append baseline results (load from Task 3)
baseline_path = os.path.join(METRICS_DIR, "baseline_model_results.csv")
if os.path.exists(baseline_path):
    baseline_df = pd.read_csv(baseline_path)
    results_df = pd.concat(
        [baseline_df, pd.DataFrame(results)], ignore_index=True)
    log("Appended baseline LinearRegression results from Task 3.")
else:
    results_df = pd.DataFrame(results)
    log("baseline_model_results.csv not found — omitting baseline rows.")

all_results_path = os.path.join(METRICS_DIR, "all_results.csv")
results_df.to_csv(all_results_path, index=False)
log(f"All results saved: {all_results_path}")

print("\nValidation MAE/RMSE summary (all models, F1):")
val_f1 = results_df[(results_df["split"] == "val") &
                    (results_df["feature_set"] == "F1")][
    ["model", "MAE", "RMSE"]].sort_values("MAE")
print(val_f1.to_string(index=False))

# ── 5. Tuning ─────────────────────────────────────────────────────────────────
log("-" * 60)
log("Phase 2: Hyperparameter tuning")
log("-" * 60)

tuning_records = []

# Use F1 for all tuning (better baseline)
X_tr_f1, X_vl_f1, X_te_f1, scaler_f1 = arrays["F1"]

# 5a. Ridge — sweep alpha ─────────────────────────────────────────────────────
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
for alpha in alphas:
    m = Ridge(alpha=alpha, random_state=SEED)
    t0 = time.perf_counter()
    m.fit(X_tr_f1, y_train)
    tt = time.perf_counter() - t0
    mae_v = mean_absolute_error(y_val, m.predict(X_vl_f1))
    rmse_v = rmse(y_val, m.predict(X_vl_f1))
    tuning_records.append({"model": "Ridge", "param": "alpha",
                           "value": alpha,
                           "val_MAE": round(mae_v, 4),
                           "val_RMSE": round(rmse_v, 4),
                           "training_time_seconds": round(tt, 6)})
log(f"Ridge tuning complete — best alpha: "
    f"{min(tuning_records, key=lambda r: r['val_MAE'])['value']}")

# 5b. Random Forest — sweep n_estimators + max_depth ─────────────────────────
rf_grid = [
    {"n_estimators": 50,  "max_depth": None},
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 100, "max_depth": 20},
    {"n_estimators": 200, "max_depth": 20},
]
for cfg in rf_grid:
    m = RandomForestRegressor(random_state=SEED, n_jobs=1, **cfg)
    t0 = time.perf_counter()
    m.fit(X_tr_f1, y_train)
    tt = time.perf_counter() - t0
    mae_v = mean_absolute_error(y_val, m.predict(X_vl_f1))
    rmse_v = rmse(y_val, m.predict(X_vl_f1))
    tuning_records.append({"model": "RandomForest",
                           "param": f"n_est={cfg['n_estimators']},depth={cfg['max_depth']}",
                           "value": cfg["n_estimators"],
                           "val_MAE": round(mae_v, 4),
                           "val_RMSE": round(rmse_v, 4),
                           "training_time_seconds": round(tt, 6)})
    log(f"  RF n_est={cfg['n_estimators']} depth={cfg['max_depth']} "
        f"val MAE={mae_v:.2f} RMSE={rmse_v:.2f}")

# 5c. Gradient Boosting — sweep n_estimators (required plot) ─────────────────
gb_n_estimators = [25, 50, 75, 100, 150, 200, 300, 400, 500]
gb_tune_maes  = []
gb_tune_rmses = []

for n_est in gb_n_estimators:
    m = GradientBoostingRegressor(
        n_estimators=n_est, learning_rate=0.1, max_depth=4,
        random_state=SEED)
    t0 = time.perf_counter()
    m.fit(X_tr_f1, y_train)
    tt = time.perf_counter() - t0
    mae_v  = mean_absolute_error(y_val, m.predict(X_vl_f1))
    rmse_v = rmse(y_val, m.predict(X_vl_f1))
    gb_tune_maes.append(mae_v)
    gb_tune_rmses.append(rmse_v)
    tuning_records.append({"model": "GradientBoosting",
                           "param": "n_estimators",
                           "value": n_est,
                           "val_MAE": round(mae_v, 4),
                           "val_RMSE": round(rmse_v, 4),
                           "training_time_seconds": round(tt, 6)})
    log(f"  GB n_est={n_est:4d}  val MAE={mae_v:.2f} RMSE={rmse_v:.2f}")

# Validation curve — GB n_estimators
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(gb_n_estimators, gb_tune_maes,  marker="o", color="#4C72B0",
        linewidth=2, label="Val MAE")
ax.plot(gb_n_estimators, gb_tune_rmses, marker="s", color="#DD8452",
        linestyle="--", linewidth=2, label="Val RMSE")
best_n_est_idx = int(np.argmin(gb_tune_maes))
best_n_est     = gb_n_estimators[best_n_est_idx]
ax.axvline(best_n_est, color="#55A868", linestyle=":", linewidth=1.5,
           label=f"Best n_estimators={best_n_est}")
ax.set_xlabel("n_estimators")
ax.set_ylabel("Metric value")
ax.set_title("GradientBoosting — Validation MAE / RMSE vs n_estimators (F1)")
ax.legend(fontsize=9)
fig.tight_layout()
savefig("validation_curve_gb.png")
log(f"Best GB n_estimators={best_n_est}  val MAE={gb_tune_maes[best_n_est_idx]:.2f}")

# 5d. MLP — sweep architectures ───────────────────────────────────────────────
mlp_configs = [
    {"hidden_layer_sizes": (64,),        "label": "(64,)"},
    {"hidden_layer_sizes": (128,),       "label": "(128,)"},
    {"hidden_layer_sizes": (128, 64),    "label": "(128,64)"},
    {"hidden_layer_sizes": (256, 128),   "label": "(256,128)"},
    {"hidden_layer_sizes": (128, 64, 32),"label": "(128,64,32)"},
]
for cfg in mlp_configs:
    m = MLPRegressor(hidden_layer_sizes=cfg["hidden_layer_sizes"],
                     activation="relu", max_iter=400,
                     random_state=SEED)
    t0 = time.perf_counter()
    m.fit(X_tr_f1, y_train)
    tt = time.perf_counter() - t0
    mae_v  = mean_absolute_error(y_val, m.predict(X_vl_f1))
    rmse_v = rmse(y_val, m.predict(X_vl_f1))
    tuning_records.append({"model": "MLP",
                           "param": f"layers={cfg['label']}",
                           "value": str(cfg["hidden_layer_sizes"]),
                           "val_MAE": round(mae_v, 4),
                           "val_RMSE": round(rmse_v, 4),
                           "training_time_seconds": round(tt, 6)})
    log(f"  MLP {cfg['label']}  val MAE={mae_v:.2f} RMSE={rmse_v:.2f}")

# Save tuning results
tuning_df = pd.DataFrame(tuning_records)
tuning_path = os.path.join(METRICS_DIR, "tuning_results.csv")
tuning_df.to_csv(tuning_path, index=False)
log(f"Tuning results saved: {tuning_path}")

# ── 6. Select best model (lowest val MAE on F1) ───────────────────────────────
log("-" * 60)
log("Phase 3: Best model selection")
log("-" * 60)

# Identify best config per model family from tuning results
best_per_model = {}

# Ridge best alpha
ridge_rows = tuning_df[tuning_df["model"] == "Ridge"].copy()
best_alpha = float(ridge_rows.loc[ridge_rows["val_MAE"].idxmin(), "value"])
best_per_model["Ridge"] = Ridge(alpha=best_alpha, random_state=SEED)
log(f"Best Ridge alpha={best_alpha}")

# RF best config
rf_rows = tuning_df[tuning_df["model"] == "RandomForest"].copy()
best_rf_row = rf_rows.loc[rf_rows["val_MAE"].idxmin()]
# Re-parse grid from param string
best_rf_cfg = rf_grid[rf_rows["val_MAE"].values.argmin()]
best_per_model["RandomForest"] = RandomForestRegressor(
    random_state=SEED, n_jobs=1, **best_rf_cfg)
log(f"Best RF config: {best_rf_cfg}")

# GB best n_estimators
best_per_model["GradientBoosting"] = GradientBoostingRegressor(
    n_estimators=best_n_est, learning_rate=0.1, max_depth=4,
    random_state=SEED)
log(f"Best GB n_estimators={best_n_est}")

# MLP best architecture
mlp_rows = tuning_df[tuning_df["model"] == "MLP"].copy()
best_mlp_idx = mlp_rows["val_MAE"].values.argmin()
best_mlp_cfg = mlp_configs[best_mlp_idx]
best_per_model["MLP"] = MLPRegressor(
    hidden_layer_sizes=best_mlp_cfg["hidden_layer_sizes"],
    activation="relu", max_iter=400,
    random_state=SEED)
log(f"Best MLP architecture: {best_mlp_cfg['label']}")

# Train each best config, evaluate on val, pick overall winner
candidate_results = []
trained_candidates = {}

for model_name, model in best_per_model.items():
    t0 = time.perf_counter()
    model.fit(X_tr_f1, y_train)
    tt = time.perf_counter() - t0
    mae_v  = mean_absolute_error(y_val, model.predict(X_vl_f1))
    rmse_v = rmse(y_val, model.predict(X_vl_f1))
    candidate_results.append({
        "model": model_name, "feature_set": "F1", "split": "val",
        "MAE": round(mae_v, 4), "RMSE": round(rmse_v, 4),
        "training_time_seconds": round(tt, 6),
    })
    trained_candidates[model_name] = model
    log(f"Tuned {model_name}  val MAE={mae_v:.2f}  RMSE={rmse_v:.2f}")

cand_df = pd.DataFrame(candidate_results)
best_row = cand_df.loc[cand_df["MAE"].idxmin()]
best_model_name = best_row["model"]
best_model      = trained_candidates[best_model_name]
best_val_mae    = best_row["MAE"]
best_val_rmse   = best_row["RMSE"]

log(f"BEST MODEL: {best_model_name}  val MAE={best_val_mae}  RMSE={best_val_rmse}")

# Save best model
model_path = os.path.join(MODELS_DIR, "final_model.pkl")
with open(model_path, "wb") as fh:
    pickle.dump({"model": best_model, "scaler": scaler_f1,
                 "feature_set": "F1", "features": F1,
                 "model_name": best_model_name}, fh)
log(f"Best model saved: {model_path}")

# ── 7. Final TEST evaluation ──────────────────────────────────────────────────
log("-" * 60)
log("Phase 4: Final TEST set evaluation")
log("-" * 60)

y_pred_test = best_model.predict(X_te_f1)
mae_test    = mean_absolute_error(y_test, y_pred_test)
rmse_test   = rmse(y_test, y_pred_test)
residuals   = y_test - y_pred_test

log(f"{best_model_name} on TEST — MAE={mae_test:.4f}  RMSE={rmse_test:.4f}")

final_results = pd.DataFrame([{
    "model": best_model_name, "feature_set": "F1",
    "split": "test",
    "MAE":  round(mae_test, 4),
    "RMSE": round(rmse_test, 4),
}])
final_path = os.path.join(METRICS_DIR, "final_model_results.csv")
final_results.to_csv(final_path, index=False)
log(f"Final test results saved: {final_path}")

# Add test row to all_results
test_row = final_results.copy()
test_row["training_time_seconds"] = float("nan")
results_df_upd = pd.concat([results_df, cand_df, test_row], ignore_index=True)
results_df_upd.to_csv(all_results_path, index=False)
log(f"Updated all_results.csv with tuned and test rows.")

# ── 8. Diagnostic plots (test set) ───────────────────────────────────────────
# Attach context columns to test for slice plots
test_ctx = test.copy()
test_ctx["y_pred"]   = y_pred_test
test_ctx["residual"] = residuals

# 8a. Residual distribution (overwrite Task 3 version with best model) ────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(y_pred_test, residuals, alpha=0.2, s=8,
           color="#4C72B0", rasterized=True)
ax.axhline(0, color="#C44E52", linewidth=1.5, linestyle="--")
ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
ax.set_title(f"Residuals vs Fitted\n{best_model_name} — Test set")

ax = axes[1]
ax.hist(residuals, bins=60, density=True, color="#4C72B0",
        edgecolor="white", linewidth=0.3, alpha=0.7)
bw = 1.06 * residuals.std() * len(residuals) ** (-0.2)
x_kde = np.linspace(residuals.min(), residuals.max(), 400)
diff  = (x_kde[np.newaxis, :] - residuals[:, np.newaxis]) / bw
kde   = np.mean(np.exp(-0.5 * diff ** 2) / (bw * np.sqrt(2 * np.pi)), axis=0)
ax.plot(x_kde, kde, color="#DD8452", linewidth=2, label="KDE")
ax.axvline(0, color="#C44E52", linewidth=1.5, linestyle="--", label="Zero")
ax.set_xlabel("Residual"); ax.set_ylabel("Density")
ax.set_title(f"Residual Distribution\n{best_model_name} — Test set")
ax.legend(fontsize=8)

fig.suptitle(f"Residual Diagnostics — {best_model_name} (Test set, F1)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
savefig("residual_distribution.png")

# 8b. MAE by hour ─────────────────────────────────────────────────────────────
mae_hr = (test_ctx.groupby("hr")
          .apply(lambda g: mean_absolute_error(g["cnt"], g["y_pred"]))
          .reset_index(name="MAE"))

fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(mae_hr["hr"], mae_hr["MAE"],
              color=sns.color_palette("YlOrRd", n_colors=24), edgecolor="white")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("MAE")
ax.set_title(f"MAE by Hour of Day — {best_model_name} (Test set)")
ax.set_xticks(range(24))
fig.tight_layout()
savefig("mae_by_hour.png")

# 8c. MAE by weekday ──────────────────────────────────────────────────────────
WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
mae_wd = (test_ctx.groupby("weekday")
          .apply(lambda g: mean_absolute_error(g["cnt"], g["y_pred"]))
          .reset_index(name="MAE"))
mae_wd["label"] = mae_wd["weekday"].map(dict(enumerate(WEEKDAY_LABELS)))

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(mae_wd["label"], mae_wd["MAE"],
       color=sns.color_palette("muted", n_colors=7), edgecolor="white")
ax.set_xlabel("Day of Week"); ax.set_ylabel("MAE")
ax.set_title(f"MAE by Weekday — {best_model_name} (Test set)")
for bar, val in zip(ax.patches, mae_wd["MAE"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1, f"{val:.1f}",
            ha="center", va="bottom", fontsize=8)
fig.tight_layout()
savefig("mae_by_weekday.png")

# 8d. Residual vs temperature ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(test_ctx["temp"], test_ctx["residual"],
           alpha=0.2, s=8, color="#55A868", rasterized=True)
# Bin-mean overlay
temp_bins = np.linspace(test_ctx["temp"].min(), test_ctx["temp"].max(), 20)
test_ctx["tb"] = pd.cut(test_ctx["temp"], bins=temp_bins)
bin_mean = (test_ctx.groupby("tb", observed=True)["residual"]
            .mean().reset_index())
bin_mid  = bin_mean["tb"].apply(lambda x: x.mid).astype(float)
ax.plot(bin_mid, bin_mean["residual"], color="#C44E52",
        linewidth=2.5, label="Bin mean")
ax.axhline(0, color="#4C72B0", linewidth=1.5, linestyle="--")
ax.set_xlabel("Normalised Temperature (temp)")
ax.set_ylabel("Residual (actual − predicted)")
ax.set_title(f"Residual vs Temperature — {best_model_name} (Test set)")
ax.legend(fontsize=9)
fig.tight_layout()
test_ctx = test_ctx.drop(columns=["tb"])
savefig("residual_vs_temperature.png")

# 8e. Rolling MAE over time ────────────────────────────────────────────────────
test_ctx_sorted = test_ctx.sort_values("dteday").copy()
test_ctx_sorted["abs_err"] = np.abs(test_ctx_sorted["residual"])
# 7-day rolling MAE (window in hours = 24*7 = 168)
test_ctx_sorted["rolling_mae"] = (
    test_ctx_sorted["abs_err"]
    .rolling(window=168, min_periods=24)
    .mean()
)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(test_ctx_sorted["dteday"], test_ctx_sorted["abs_err"],
        color="#4C72B0", alpha=0.25, linewidth=0.6, label="Hourly |error|")
ax.plot(test_ctx_sorted["dteday"], test_ctx_sorted["rolling_mae"],
        color="#DD8452", linewidth=2.2, label="7-day rolling MAE")
ax.axhline(mae_test, color="#C44E52", linewidth=1.5,
           linestyle="--", label=f"Overall MAE={mae_test:.1f}")
ax.set_xlabel("Date"); ax.set_ylabel("Absolute Error")
ax.set_title(f"Rolling MAE Over Time — {best_model_name} (Test set)")
ax.legend(fontsize=9)
fig.tight_layout()
savefig("rolling_mae_over_time.png")

# ── 9. Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL SUMMARY — MODEL COMPARISON & SELECTION")
print("=" * 65)

baseline_val = results_df[
    (results_df["model"] == "LinearRegression") &
    (results_df["feature_set"] == "F1") &
    (results_df["split"] == "val")]
if not baseline_val.empty:
    bl_mae = baseline_val["MAE"].values[0]
    bl_rmse = baseline_val["RMSE"].values[0]
    print(f"  Baseline LinearRegression [F1]  val MAE={bl_mae:.2f}  RMSE={bl_rmse:.2f}")

print(f"\n  Tuned model candidates (val, F1):")
for _, row in cand_df.sort_values("MAE").iterrows():
    print(f"    {row['model']:22s}  MAE={row['MAE']:.2f}  RMSE={row['RMSE']:.2f}")

print(f"\n  Selected: {best_model_name}")
print(f"  TEST  MAE={mae_test:.2f}  RMSE={rmse_test:.2f}")

if not baseline_val.empty:
    imp_mae  = (bl_mae - mae_test) / bl_mae * 100
    imp_rmse = (bl_rmse - rmse_test) / bl_rmse * 100
    print(f"\n  Improvement vs baseline — "
          f"MAE: {imp_mae:+.1f}%   RMSE: {imp_rmse:+.1f}%")

print("=" * 65)

log("Task 4 completed successfully.")
log("=" * 70)
