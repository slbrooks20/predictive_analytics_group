"""
Task 4: Model Comparison, Tuning, and Final Evaluation
Models: Ridge, RandomForest, GradientBoosting, MLPRegressor
Feature sets: F0 (original) and F1 (+ cyclical)
Metric: MAE and RMSE on validation; final eval on test.
"""

import os
import sys
import time
import pickle
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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
BENCH_DIR   = os.path.join(OUT_DIR, "benchmark")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
MODELS_DIR  = os.path.join(OUT_DIR, "models")
LOG_FILE    = os.path.join(BENCH_DIR, "experiment_log.txt")

TRAIN_CSV   = os.path.join(OUT_DIR, "train.csv")
VAL_CSV     = os.path.join(OUT_DIR, "val.csv")
TEST_CSV    = os.path.join(OUT_DIR, "test.csv")

ALL_RES_CSV    = os.path.join(METRICS_DIR, "all_results.csv")
TUNING_CSV     = os.path.join(METRICS_DIR, "tuning_results.csv")
FINAL_RES_CSV  = os.path.join(METRICS_DIR, "final_model_results.csv")
FINAL_PKL      = os.path.join(MODELS_DIR, "final_model.pkl")

for d in [BENCH_DIR, METRICS_DIR, FIG_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET = "cnt"

# ── Logger ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("TASK 4 — Model Comparison, Tuning & Final Evaluation")
log("=" * 60)

# ── 1. Load splits ─────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV, index_col=0)
val_df   = pd.read_csv(VAL_CSV,   index_col=0)
test_df  = pd.read_csv(TEST_CSV,  index_col=0)
log(f"Splits loaded — train:{train_df.shape}  val:{val_df.shape}  test:{test_df.shape}")

# ── 2. Feature sets ────────────────────────────────────────────────────────────
F0 = ["hr", "weekday", "workingday", "season", "mnth", "yr",
      "weathersit", "temp", "atemp", "hum", "windspeed"]

def add_cyclical(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sin_hour"]  = np.sin(2 * np.pi * out["hr"]  / 24)
    out["cos_hour"]  = np.cos(2 * np.pi * out["hr"]  / 24)
    out["sin_month"] = np.sin(2 * np.pi * out["mnth"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["mnth"] / 12)
    return out

train_cy = add_cyclical(train_df)
val_cy   = add_cyclical(val_df)
test_cy  = add_cyclical(test_df)

F1 = F0 + ["sin_hour", "cos_hour", "sin_month", "cos_month"]

FEAT_SETS = {"F0": (F0, train_df, val_df, test_df),
             "F1": (F1, train_cy, val_cy, test_cy)}

log(f"F0 ({len(F0)} features): {F0}")
log(f"F1 ({len(F1)} features): {F1}")

# ── 3. Scaler + fit/predict helper ────────────────────────────────────────────
def make_scaler_and_fit(X_tr):
    sc = StandardScaler()
    return sc, sc.fit_transform(X_tr)

def evaluate(model, X_tr_raw, y_tr, X_ev_raw, y_ev):
    sc, X_tr_s = make_scaler_and_fit(X_tr_raw)
    X_ev_s = sc.transform(X_ev_raw)
    t0 = time.perf_counter()
    model.fit(X_tr_s, y_tr)
    elapsed = time.perf_counter() - t0
    y_pred = np.clip(model.predict(X_ev_s), 0, None)
    mae  = mean_absolute_error(y_ev, y_pred)
    rmse = root_mean_squared_error(y_ev, y_pred)
    return mae, rmse, elapsed, sc, y_pred

# ── 4. Model grid (default hyper-params) ──────────────────────────────────────
def make_models():
    return {
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=None,
            random_state=SEED, n_jobs=1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            random_state=SEED),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=300,
            random_state=SEED, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20),
    }

# ── 5. Model comparison on validation ─────────────────────────────────────────
log("--- Model comparison (default hyper-params, validation set) ---")
sns.set_theme(style="whitegrid", font_scale=1.05)

comp_rows = []

for fname, (feats, tr, ev, _) in FEAT_SETS.items():
    X_tr  = tr[feats].values
    y_tr  = tr[TARGET].values
    X_ev  = ev[feats].values
    y_ev  = ev[TARGET].values

    for mname, model in make_models().items():
        mae, rmse, elapsed, _, _ = evaluate(model, X_tr, y_tr, X_ev, y_ev)
        row = dict(model=mname, feature_set=fname, split="validation",
                   MAE=round(mae, 4), RMSE=round(rmse, 4),
                   training_time_seconds=round(elapsed, 6))
        comp_rows.append(row)
        log(f"  {mname:20s} | {fname} | MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"time={elapsed:.3f}s")

# Append baseline for completeness
baseline_csv = os.path.join(METRICS_DIR, "baseline_model_results.csv")
if os.path.exists(baseline_csv):
    baseline_df = pd.read_csv(baseline_csv)
    # Rename model column value for consistency
    baseline_df["model"] = "LinearRegression"
    comp_rows_df = pd.concat(
        [pd.DataFrame(comp_rows), baseline_df], ignore_index=True)
else:
    comp_rows_df = pd.DataFrame(comp_rows)

comp_rows_df.to_csv(ALL_RES_CSV, index=False)
log(f"All results saved: {ALL_RES_CSV}")
print("\nModel comparison (validation):")
print(comp_rows_df[comp_rows_df["split"] == "validation"]
      .sort_values("MAE").to_string(index=False))

# ── 6. Tuning ─────────────────────────────────────────────────────────────────
log("--- Hyperparameter tuning ---")
tuning_rows = []

# Helper: tune and record
def tune_record(mname, fname, model, feats, tr, ev):
    feats_list, tr_frame, ev_frame, _ = FEAT_SETS[fname]
    X_tr = tr_frame[feats_list].values
    y_tr = tr_frame[TARGET].values
    X_ev = ev_frame[feats_list].values
    y_ev = ev_frame[TARGET].values
    mae, rmse, elapsed, _, _ = evaluate(model, X_tr, y_tr, X_ev, y_ev)
    row = dict(model=mname, feature_set=fname, config=str(feats),
               MAE=round(mae, 4), RMSE=round(rmse, 4),
               training_time_seconds=round(elapsed, 6))
    tuning_rows.append(row)
    log(f"  tune {mname:20s} | {fname} | {feats} → MAE={mae:.3f}  RMSE={rmse:.3f}")
    return mae, rmse

# ── 6a. Ridge — sweep alpha ───────────────────────────────────────────────────
log("  Ridge: sweeping alpha …")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    for fn in ["F0", "F1"]:
        tune_record("Ridge", fn,
                    Ridge(alpha=alpha, random_state=SEED),
                    f"alpha={alpha}", None, None)

# ── 6b. RandomForest — sweep n_estimators × max_depth ────────────────────────
log("  RandomForest: sweeping n_estimators × max_depth …")
for n_est in [50, 100, 200]:
    for depth in [8, 16, None]:
        cfg = f"n_est={n_est},depth={depth}"
        for fn in ["F0", "F1"]:
            tune_record("RandomForest", fn,
                        RandomForestRegressor(
                            n_estimators=n_est, max_depth=depth,
                            random_state=SEED, n_jobs=1),
                        cfg, None, None)

# ── 6c. GradientBoosting — sweep n_estimators (for plot) ─────────────────────
log("  GradientBoosting: sweeping n_estimators …")
gb_n_vals = [50, 100, 150, 200, 300, 400, 500]
gb_val_mae_f0, gb_val_mae_f1 = [], []

for n_est in gb_n_vals:
    # F0
    feats0, tr0, ev0, _ = FEAT_SETS["F0"]
    mae0, rmse0, el0, _, _ = evaluate(
        GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1,
                                  max_depth=4, random_state=SEED),
        tr0[feats0].values, tr0[TARGET].values,
        ev0[feats0].values, ev0[TARGET].values)
    gb_val_mae_f0.append(mae0)
    tuning_rows.append(dict(model="GradientBoosting", feature_set="F0",
                            config=f"n_est={n_est},lr=0.1,depth=4",
                            MAE=round(mae0, 4), RMSE=round(rmse0, 4),
                            training_time_seconds=round(el0, 6)))
    # F1
    feats1, tr1, ev1, _ = FEAT_SETS["F1"]
    mae1, rmse1, el1, _, _ = evaluate(
        GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1,
                                  max_depth=4, random_state=SEED),
        tr1[feats1].values, tr1[TARGET].values,
        ev1[feats1].values, ev1[TARGET].values)
    gb_val_mae_f1.append(mae1)
    tuning_rows.append(dict(model="GradientBoosting", feature_set="F1",
                            config=f"n_est={n_est},lr=0.1,depth=4",
                            MAE=round(mae1, 4), RMSE=round(rmse1, 4),
                            training_time_seconds=round(el1, 6)))
    log(f"    GB n_est={n_est}  F0 MAE={mae0:.3f}  F1 MAE={mae1:.3f}")

# Validation curve plot — GB
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(gb_n_vals, gb_val_mae_f0, marker="o", label="F0", color="#4878CF")
ax.plot(gb_n_vals, gb_val_mae_f1, marker="s", label="F1", color="#D65F5F")
ax.set_xlabel("n_estimators")
ax.set_ylabel("Validation MAE")
ax.set_title("Gradient Boosting — Validation MAE vs n_estimators")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "validation_curve_gb.png"), dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/validation_curve_gb.png")

# Best GB n_estimators per feature set
best_n_f0 = gb_n_vals[int(np.argmin(gb_val_mae_f0))]
best_n_f1 = gb_n_vals[int(np.argmin(gb_val_mae_f1))]
log(f"  Best GB n_estimators — F0: {best_n_f0}  F1: {best_n_f1}")

# Also sweep learning_rate for the best n_estimators
for lr in [0.05, 0.1, 0.2]:
    for fn, best_n in [("F0", best_n_f0), ("F1", best_n_f1)]:
        tune_record("GradientBoosting", fn,
                    GradientBoostingRegressor(
                        n_estimators=best_n, learning_rate=lr,
                        max_depth=4, random_state=SEED),
                    f"n_est={best_n},lr={lr},depth=4", None, None)

# ── 6d. MLP — sweep architecture + learning_rate_init ────────────────────────
log("  MLP: sweeping architecture and learning rate …")
for arch in [(128, 64), (256, 128, 64), (64, 32)]:
    for lr_init in [0.001, 0.01]:
        for fn in ["F0", "F1"]:
            tune_record("MLPRegressor", fn,
                        MLPRegressor(
                            hidden_layer_sizes=arch, max_iter=400,
                            learning_rate_init=lr_init,
                            random_state=SEED, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=20),
                        f"arch={arch},lr={lr_init}", None, None)

# Save tuning results
tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(TUNING_CSV, index=False)
log(f"Tuning results saved: {TUNING_CSV}  ({len(tuning_df)} runs)")

# ── 7. Select best model (lowest validation MAE) ───────────────────────────────
log("--- Selecting best model by validation MAE ---")

# Best per model from tuning rows
best_tune = (tuning_df.groupby(["model", "feature_set"])["MAE"]
             .min().reset_index())
best_tune = best_tune.sort_values("MAE")
log("Best tuned MAE per model:\n" + best_tune.to_string(index=False))

# Pick overall best
best_row   = best_tune.iloc[0]
best_model_name = best_row["model"]
best_feat_name  = best_row["feature_set"]
best_val_mae    = best_row["MAE"]
log(f"Best model: {best_model_name} ({best_feat_name})  val MAE={best_val_mae:.4f}")

# Reconstruct best model with its optimal hyper-params
def get_best_model_for(mname, fname):
    subset = tuning_df[(tuning_df["model"] == mname) &
                       (tuning_df["feature_set"] == fname)]
    best_cfg = subset.loc[subset["MAE"].idxmin(), "config"]
    log(f"  Best config for {mname}/{fname}: {best_cfg}")

    # Parse config string back to params
    cfg = dict(kv.split("=") for kv in best_cfg.split(","))

    if mname == "Ridge":
        return Ridge(alpha=float(cfg["alpha"]), random_state=SEED)

    if mname == "RandomForest":
        depth = None if cfg.get("depth") == "None" else int(cfg["depth"])
        return RandomForestRegressor(
            n_estimators=int(cfg["n_est"]), max_depth=depth,
            random_state=SEED, n_jobs=1)

    if mname == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=int(cfg["n_est"]),
            learning_rate=float(cfg["lr"]),
            max_depth=int(cfg["depth"]),
            random_state=SEED)

    if mname == "MLPRegressor":
        arch_str = best_cfg.split("arch=")[1].split(",lr=")[0]
        arch = tuple(int(x) for x in arch_str.strip("()").split(", "))
        lr_init = float(best_cfg.split("lr=")[1])
        return MLPRegressor(
            hidden_layer_sizes=arch, max_iter=400,
            learning_rate_init=lr_init,
            random_state=SEED, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20)

    raise ValueError(f"Unknown model: {mname}")

best_model_obj = get_best_model_for(best_model_name, best_feat_name)

# Refit on full training set with best config
feats_list, tr_frame, ev_frame, te_frame = FEAT_SETS[best_feat_name]
X_tr_raw = tr_frame[feats_list].values
y_tr     = tr_frame[TARGET].values

sc_final = StandardScaler()
X_tr_s   = sc_final.fit_transform(X_tr_raw)
best_model_obj.fit(X_tr_s, y_tr)
log(f"Best model refit on training set.")

# Save model + scaler as a dict
with open(FINAL_PKL, "wb") as f:
    pickle.dump({"model": best_model_obj, "scaler": sc_final,
                 "feature_set": best_feat_name, "features": feats_list}, f)
log(f"Final model saved: {FINAL_PKL}")

# ── 8. Final evaluation on TEST set ───────────────────────────────────────────
log("--- Final evaluation on TEST set ---")

X_te_raw = te_frame[feats_list].values
y_te     = te_frame[TARGET].values
X_te_s   = sc_final.transform(X_te_raw)
y_pred_te = np.clip(best_model_obj.predict(X_te_s), 0, None)

test_mae  = mean_absolute_error(y_te, y_pred_te)
test_rmse = root_mean_squared_error(y_te, y_pred_te)
log(f"TEST  MAE={test_mae:.4f}  RMSE={test_rmse:.4f}")
print(f"\nFinal test  MAE={test_mae:.4f}  RMSE={test_rmse:.4f}")

final_df = pd.DataFrame([{
    "model": best_model_name,
    "feature_set": best_feat_name,
    "split": "test",
    "MAE":  round(test_mae,  4),
    "RMSE": round(test_rmse, 4),
}])
final_df.to_csv(FINAL_RES_CSV, index=False)
log(f"Final results saved: {FINAL_RES_CSV}")

residuals = y_te - y_pred_te

# ── 8a. residual_distribution.png ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(residuals, bins=60, color="#4878CF", edgecolor="white",
             linewidth=0.4, density=True, alpha=0.85)
pd.Series(residuals).plot.kde(ax=axes[0], color="#D65F5F", linewidth=2)
axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_xlabel("Residual (actual − predicted)")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Residual Distribution — {best_model_name} (test)")

axes[1].scatter(y_pred_te, residuals, alpha=0.25, s=8,
                color="#6ACC65", rasterized=True)
axes[1].axhline(0, color="#D65F5F", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Predicted cnt")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Fitted (test)")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "residual_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/residual_distribution.png")

# ── 8b. mae_by_hour.png ───────────────────────────────────────────────────────
te_diag = te_frame.copy()
te_diag["_pred"]  = y_pred_te
te_diag["_abserr"] = np.abs(residuals)

hour_mae = te_diag.groupby("hr")["_abserr"].mean()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(hour_mae.index, hour_mae.values, color="#4878CF", edgecolor="white",
       linewidth=0.4)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean Absolute Error")
ax.set_title(f"MAE by Hour of Day — {best_model_name} (test)")
ax.set_xticks(range(0, 24))
ax.axhline(test_mae, color="#D65F5F", linewidth=1.5, linestyle="--",
           label=f"Overall MAE = {test_mae:.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "mae_by_hour.png"), dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/mae_by_hour.png")

# ── 8c. mae_by_weekday.png ────────────────────────────────────────────────────
day_labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
              4: "Fri", 5: "Sat", 6: "Sun"}
weekday_mae = te_diag.groupby("weekday")["_abserr"].mean()
weekday_mae.index = weekday_mae.index.map(day_labels)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(weekday_mae.index, weekday_mae.values, color="#6ACC65", edgecolor="white",
       linewidth=0.4)
ax.set_xlabel("Day of week")
ax.set_ylabel("Mean Absolute Error")
ax.set_title(f"MAE by Day of Week — {best_model_name} (test)")
ax.axhline(test_mae, color="#D65F5F", linewidth=1.5, linestyle="--",
           label=f"Overall MAE = {test_mae:.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "mae_by_weekday.png"), dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/mae_by_weekday.png")

# ── 8d. residual_vs_temperature.png ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(te_diag["temp"], residuals, alpha=0.3, s=8,
           color="#9467BD", rasterized=True)
# Overlay binned mean residual
te_diag["_temp_bin"] = pd.cut(te_diag["temp"], bins=20)
bin_res = te_diag.groupby("_temp_bin", observed=True)["_abserr"].mean()
bin_mid = bin_res.index.map(lambda x: x.mid).astype(float)
ax.plot(bin_mid, bin_res.values, color="#D65F5F", linewidth=2,
        marker="o", markersize=4, label="Binned mean |error|")
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("temp (normalised)")
ax.set_ylabel("Residual (actual − predicted)")
ax.set_title(f"Residuals vs Temperature — {best_model_name} (test)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "residual_vs_temperature.png"),
            dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/residual_vs_temperature.png")

# ── 8e. rolling_mae_over_time.png ────────────────────────────────────────────
rolling_abserr = pd.Series(np.abs(residuals)).rolling(window=100, min_periods=10).mean()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(range(len(rolling_abserr)), rolling_abserr, alpha=0.25, color="#4878CF")
ax.plot(rolling_abserr, color="#4878CF", linewidth=1.5,
        label="100-sample rolling MAE")
ax.axhline(test_mae, color="#D65F5F", linewidth=1.5, linestyle="--",
           label=f"Overall MAE = {test_mae:.1f}")
ax.set_xlabel("Test sample index (chronological)")
ax.set_ylabel("Rolling MAE (window=100)")
ax.set_title(f"Rolling MAE Over Time — {best_model_name} (test)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rolling_mae_over_time.png"),
            dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/rolling_mae_over_time.png")

# ── 9. Summary comparison table ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 4 SUMMARY")
print("=" * 70)

all_df = pd.read_csv(ALL_RES_CSV)
val_summary = (all_df[all_df["split"] == "validation"]
               .sort_values("MAE")[["model", "feature_set", "MAE", "RMSE"]])
print("\nModel comparison — validation MAE (default hyper-params):")
print(val_summary.to_string(index=False))

print(f"\nBest model selected : {best_model_name} ({best_feat_name})")
print(f"  Validation MAE    : {best_val_mae:.4f}")
print(f"  TEST MAE          : {test_mae:.4f}")
print(f"  TEST RMSE         : {test_rmse:.4f}")
print("=" * 70)

log("Task 4 complete.")
log("=" * 60)
