"""
Task 3: Baseline Model — Linear Regression with F0 and F1 feature sets.
Chronological 70/15/15 split. MAE and RMSE on validation set.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV    = os.path.join(BASE_DIR, "outputs", "cleaned_data.csv")
RAW_CSV      = os.path.join(BASE_DIR, "dataset", "hour.csv")
OUT_DIR      = os.path.join(BASE_DIR, "outputs")
BENCH_DIR    = os.path.join(OUT_DIR,  "benchmark")
METRICS_DIR  = os.path.join(OUT_DIR,  "metrics")
FIG_DIR      = os.path.join(OUT_DIR,  "figures")
LOG_FILE     = os.path.join(BENCH_DIR, "experiment_log.txt")

TRAIN_CSV    = os.path.join(OUT_DIR, "train.csv")
VAL_CSV      = os.path.join(OUT_DIR, "val.csv")
TEST_CSV     = os.path.join(OUT_DIR, "test.csv")
PREP_REPORT  = os.path.join(BENCH_DIR, "preprocessing_report.csv")
RESULTS_CSV  = os.path.join(METRICS_DIR, "baseline_model_results.csv")

for d in [BENCH_DIR, METRICS_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("TASK 3 — Baseline Model (Linear Regression)")
log("=" * 60)

# ── 1. Load data ───────────────────────────────────────────────────────────────
if os.path.exists(CLEAN_CSV):
    df = pd.read_csv(CLEAN_CSV, parse_dates=["dteday"])
    log(f"Loaded cleaned data: {CLEAN_CSV}  shape={df.shape}")
else:
    df = pd.read_csv(RAW_CSV, parse_dates=["dteday"])
    log(f"Cleaned data not found — loaded raw: {RAW_CSV}  shape={df.shape}")
    for col in ["casual", "registered"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    log("Removed leakage columns: casual, registered")

TARGET = "cnt"

# Drop non-predictive columns
DROP_COLS = ["instant", "dteday"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
log(f"Dropped columns: {DROP_COLS}")

# ── 2. Chronological split (70 / 15 / 15) ────────────────────────────────────
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

log(f"Split sizes — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# ── 2a. Assert strict chronological ordering ──────────────────────────────────
# Use the row index as a proxy for time (data is already time-sorted by hour)
train_max_idx = train_df.index.max()
val_min_idx   = val_df.index.min()
val_max_idx   = val_df.index.max()
test_min_idx  = test_df.index.min()

if not (train_max_idx < val_min_idx):
    raise AssertionError(
        f"Chronological split FAILED: max train index ({train_max_idx}) "
        f">= min val index ({val_min_idx})"
    )
if not (val_max_idx < test_min_idx):
    raise AssertionError(
        f"Chronological split FAILED: max val index ({val_max_idx}) "
        f">= min test index ({test_min_idx})"
    )

chron_pass = True
log("Chronological split assertions PASSED.")
log(f"  train rows [{train_df.index.min()}–{train_max_idx}]  "
    f"val [{val_min_idx}–{val_max_idx}]  test [{test_min_idx}–{test_df.index.max()}]")

# ── 2b. Save splits ───────────────────────────────────────────────────────────
train_df.to_csv(TRAIN_CSV, index=True)
val_df.to_csv(VAL_CSV,     index=True)
test_df.to_csv(TEST_CSV,   index=True)
log(f"Splits saved: train.csv  val.csv  test.csv")

# ── 3. Feature sets ───────────────────────────────────────────────────────────
F0 = ["hr", "weekday", "workingday", "season", "mnth", "yr",
      "weathersit", "temp", "atemp", "hum", "windspeed"]

# Verify all F0 features exist
missing_f0 = [c for c in F0 if c not in df.columns]
if missing_f0:
    raise ValueError(f"F0 features missing from data: {missing_f0}")

# Build cyclical features on each split independently (no leakage)
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

missing_f1 = [c for c in F1 if c not in train_cy.columns]
if missing_f1:
    raise ValueError(f"F1 features missing: {missing_f1}")

log(f"F0 features ({len(F0)}): {F0}")
log(f"F1 features ({len(F1)}): {F1}")

# ── 4. Preprocessing report ───────────────────────────────────────────────────
prep_rows = [
    {"check": "train_rows",              "value": len(train_df)},
    {"check": "val_rows",               "value": len(val_df)},
    {"check": "test_rows",              "value": len(test_df)},
    {"check": "chronological_split_pass","value": str(chron_pass)},
    {"check": "feature_set_F0_defined", "value": str(len(missing_f0) == 0)},
    {"check": "feature_set_F1_defined", "value": str(len(missing_f1) == 0)},
]
pd.DataFrame(prep_rows).to_csv(PREP_REPORT, index=False)
log(f"Preprocessing report saved: {PREP_REPORT}")

# ── 5. Train & evaluate helper ────────────────────────────────────────────────
def train_eval(feat_cols, feat_name, train_frame, val_frame):
    X_tr = train_frame[feat_cols].values
    y_tr = train_frame[TARGET].values
    X_va = val_frame[feat_cols].values
    y_va = val_frame[TARGET].values

    # Fit scaler on train only
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # Train
    model = LinearRegression(n_jobs=1)
    t0 = time.perf_counter()
    model.fit(X_tr_s, y_tr)
    train_time = time.perf_counter() - t0

    # Predict (clip to non-negative — counts cannot be negative)
    y_pred_va = np.clip(model.predict(X_va_s), 0, None)

    mae  = mean_absolute_error(y_va, y_pred_va)
    rmse = root_mean_squared_error(y_va, y_pred_va)

    log(f"Linear Regression | {feat_name} | val MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"time={train_time:.4f}s")

    return {
        "model":   "LinearRegression",
        "feature_set": feat_name,
        "split":   "validation",
        "MAE":     round(mae, 4),
        "RMSE":    round(rmse, 4),
        "training_time_seconds": round(train_time, 6),
    }, model, scaler, y_pred_va, y_va

# ── 6. Run on F0 and F1 ───────────────────────────────────────────────────────
res_f0, model_f0, scaler_f0, pred_f0, true_va = train_eval(
    F0, "F0", train_df, val_df)

res_f1, model_f1, scaler_f1, pred_f1, _ = train_eval(
    F1, "F1", train_cy, val_cy)

results = pd.DataFrame([res_f0, res_f1])
results.to_csv(RESULTS_CSV, index=False)
log(f"Baseline results saved: {RESULTS_CSV}")

print("\nBaseline model results:")
print(results.to_string(index=False))

# ── 7. Diagnostic plots (using best feature set = F1 for plots) ───────────────
sns.set_theme(style="whitegrid", font_scale=1.05)

# Choose the feature set with lower RMSE for diagnostic plots
if res_f1["RMSE"] <= res_f0["RMSE"]:
    plot_pred  = pred_f1
    plot_label = "F1"
else:
    plot_pred  = pred_f0
    plot_label = "F0"

residuals = true_va - plot_pred

# 7a. Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(true_va, plot_pred, alpha=0.25, s=8, color="#4878CF", rasterized=True)
lim = max(true_va.max(), plot_pred.max()) * 1.02
ax.plot([0, lim], [0, lim], color="#D65F5F", linewidth=1.5, linestyle="--",
        label="Perfect prediction")
ax.set_xlabel("Actual cnt")
ax.set_ylabel("Predicted cnt")
ax.set_title(f"Actual vs Predicted — Linear Regression ({plot_label}, validation set)")
ax.legend()
fig.tight_layout()
path = os.path.join(FIG_DIR, "actual_vs_predicted.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/actual_vs_predicted.png")

# 7b. Residual distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(residuals, bins=60, color="#4878CF", edgecolor="white",
             linewidth=0.4, density=True, alpha=0.85)
pd.Series(residuals).plot.kde(ax=axes[0], color="#D65F5F", linewidth=2)
axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_xlabel("Residual (actual − predicted)")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Residual Distribution ({plot_label})")

# Residuals vs fitted
axes[1].scatter(plot_pred, residuals, alpha=0.25, s=8,
                color="#6ACC65", rasterized=True)
axes[1].axhline(0, color="#D65F5F", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Predicted cnt")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Fitted")

fig.tight_layout()
path = os.path.join(FIG_DIR, "residual_distribution.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
log("Figure saved: outputs/figures/residual_distribution.png")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3 SUMMARY")
print("=" * 60)
print(f"  Train rows : {len(train_df)}")
print(f"  Val rows   : {len(val_df)}")
print(f"  Test rows  : {len(test_df)}  (held out)")
print(f"  Chron split: PASS")
print()
for _, row in results.iterrows():
    print(f"  LinReg {row['feature_set']}  — MAE={row['MAE']:.2f}  "
          f"RMSE={row['RMSE']:.2f}  time={row['training_time_seconds']:.4f}s")
print("=" * 60)

log("Task 3 complete.")
log("=" * 60)
