"""
Task 3: Baseline Model & Evaluation Harness
Linear Regression on F0 and F1 feature sets — MAE / RMSE on validation split.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH   = os.path.join(BASE_DIR, "outputs", "cleaned_data.csv")
RAW_PATH     = os.path.join(BASE_DIR, "dataset", "hour.csv")
OUT_DIR      = os.path.join(BASE_DIR, "outputs")
FIG_DIR      = os.path.join(OUT_DIR, "figures")
METRICS_DIR  = os.path.join(OUT_DIR, "metrics")
MODELS_DIR   = os.path.join(OUT_DIR, "models")
BENCH_DIR    = os.path.join(OUT_DIR, "benchmark")
LOG_PATH     = os.path.join(BENCH_DIR, "experiment_log.txt")

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

# ── 1. Load data ──────────────────────────────────────────────────────────────
log("=" * 70)
log("TASK 3 — Baseline Model & Evaluation Harness")
log("=" * 70)

if os.path.exists(CLEAN_PATH):
    df = pd.read_csv(CLEAN_PATH)
    log(f"Loaded cleaned data: {CLEAN_PATH}  shape={df.shape}")
else:
    df = pd.read_csv(RAW_PATH)
    for col in ["casual", "registered"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    log(f"Cleaned data not found; loaded raw: {RAW_PATH}  shape={df.shape}")

df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values("dteday").reset_index(drop=True)
log(f"Date range: {df['dteday'].min().date()} → {df['dteday'].max().date()}")

# ── 2. Chronological 70 / 15 / 15 split ──────────────────────────────────────
# Split on unique calendar dates so no day straddles two splits.
unique_dates = np.sort(df["dteday"].dt.normalize().unique())
n_dates  = len(unique_dates)
n_d_tr   = int(n_dates * 0.70)
n_d_val  = int(n_dates * 0.15)
# test gets remaining dates
cutoff_tv = unique_dates[n_d_tr]       # first date in val
cutoff_vt = unique_dates[n_d_tr + n_d_val]  # first date in test

train = df[df["dteday"] <  cutoff_tv].copy()
val   = df[(df["dteday"] >= cutoff_tv) & (df["dteday"] < cutoff_vt)].copy()
test  = df[df["dteday"] >= cutoff_vt].copy()

log(f"Split sizes — train: {len(train)}  val: {len(val)}  test: {len(test)}")
log(f"Train dates: {train['dteday'].min().date()} → {train['dteday'].max().date()}")
log(f"Val   dates: {val['dteday'].min().date()}   → {val['dteday'].max().date()}")
log(f"Test  dates: {test['dteday'].min().date()}  → {test['dteday'].max().date()}")

# ── Chronological assertions ──────────────────────────────────────────────────
chron_pass = True
try:
    assert train["dteday"].max() < val["dteday"].min(), \
        f"Chronological violation: max(train)={train['dteday'].max().date()} " \
        f">= min(val)={val['dteday'].min().date()}"
    assert val["dteday"].max() < test["dteday"].min(), \
        f"Chronological violation: max(val)={val['dteday'].max().date()} " \
        f">= min(test)={test['dteday'].min().date()}"
    log("Chronological split assertions PASSED.")
except AssertionError as e:
    chron_pass = False
    log(f"FATAL — {e}")
    raise

# ── Save splits ───────────────────────────────────────────────────────────────
train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(OUT_DIR, "val.csv"),   index=False)
test.to_csv(os.path.join(OUT_DIR, "test.csv"),  index=False)
log("Splits saved: outputs/train.csv, val.csv, test.csv")

# ── 3. Feature sets ───────────────────────────────────────────────────────────
F0 = ["hr", "weekday", "workingday", "season", "mnth", "yr",
      "weathersit", "temp", "atemp", "hum", "windspeed"]

# Cyclical encodings — fit formula uses train statistics (min/max)
# For period-based cyclicals the formula is deterministic; no fitting needed.
def add_cyclical(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["sin_hour"]  = np.sin(2 * np.pi * d["hr"]  / 24)
    d["cos_hour"]  = np.cos(2 * np.pi * d["hr"]  / 24)
    d["sin_month"] = np.sin(2 * np.pi * d["mnth"] / 12)
    d["cos_month"] = np.cos(2 * np.pi * d["mnth"] / 12)
    return d

train = add_cyclical(train)
val   = add_cyclical(val)
test  = add_cyclical(test)

F1 = F0 + ["sin_hour", "cos_hour", "sin_month", "cos_month"]

log(f"F0 features ({len(F0)}): {F0}")
log(f"F1 features ({len(F1)}): {F1}")

# ── 4. Preprocessing — StandardScaler fit on train only ──────────────────────
# One scaler per feature set (fitted independently)
scalers = {}
X_sets  = {}

for fs_name, fs_cols in [("F0", F0), ("F1", F1)]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[fs_cols])
    X_vl = scaler.transform(val[fs_cols])
    X_te = scaler.transform(test[fs_cols])
    scalers[fs_name] = scaler
    X_sets[fs_name]  = {"train": X_tr, "val": X_vl, "test": X_te}
    log(f"Scaler for {fs_name} fit on train only — "
        f"mean range [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")

y_train = train["cnt"].values
y_val   = val["cnt"].values
y_test  = test["cnt"].values

# ── 5. Train Linear Regression on both feature sets ──────────────────────────
results = []
models  = {}
preds   = {}  # store val predictions for plots

for fs_name in ("F0", "F1"):
    X_tr = X_sets[fs_name]["train"]
    X_vl = X_sets[fs_name]["val"]

    lr = LinearRegression(n_jobs=1)
    t0 = time.perf_counter()
    lr.fit(X_tr, y_train)
    train_time = time.perf_counter() - t0

    y_pred_train = lr.predict(X_tr)
    y_pred_val   = lr.predict(X_vl)

    mae_train  = mean_absolute_error(y_train, y_pred_train)
    rmse_train = rmse(y_train, y_pred_train)
    mae_val    = mean_absolute_error(y_val, y_pred_val)
    rmse_val   = rmse(y_val, y_pred_val)

    models[fs_name] = lr
    preds[fs_name]  = {"y_true": y_val, "y_pred": y_pred_val}

    log(f"LinearRegression [{fs_name}] — "
        f"train MAE={mae_train:.2f} RMSE={rmse_train:.2f} | "
        f"val MAE={mae_val:.2f} RMSE={rmse_val:.2f} | "
        f"time={train_time:.4f}s")

    results.append({"model": "LinearRegression", "feature_set": fs_name,
                    "split": "train",
                    "MAE": round(mae_train, 4), "RMSE": round(rmse_train, 4),
                    "training_time_seconds": round(train_time, 6)})
    results.append({"model": "LinearRegression", "feature_set": fs_name,
                    "split": "val",
                    "MAE": round(mae_val, 4), "RMSE": round(rmse_val, 4),
                    "training_time_seconds": round(train_time, 6)})

# ── 6. Save metrics ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_path = os.path.join(METRICS_DIR, "baseline_model_results.csv")
results_df.to_csv(results_path, index=False)
log(f"Baseline results saved to: {results_path}")
print("\nBaseline results:\n" + results_df.to_string(index=False))

# ── 7. Diagnostic plots ───────────────────────────────────────────────────────
# Use F1 (richer feature set) for diagnostic plots; label accordingly.
best_fs = "F1"
y_true_plot = preds[best_fs]["y_true"]
y_pred_plot = preds[best_fs]["y_pred"]
residuals   = y_true_plot - y_pred_plot

# 7a. Actual vs Predicted ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, fs in zip(axes, ("F0", "F1")):
    yt = preds[fs]["y_true"]
    yp = preds[fs]["y_pred"]
    res = yt - yp

    lim = max(yt.max(), yp.max()) * 1.02
    ax.scatter(yt, yp, alpha=0.18, s=10, color="#4C72B0", rasterized=True)
    ax.plot([0, lim], [0, lim], color="#DD8452", linewidth=1.5,
            linestyle="--", label="Perfect prediction")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual cnt"); ax.set_ylabel("Predicted cnt")
    mae_v = mean_absolute_error(yt, yp)
    rmse_v = rmse(yt, yp)
    ax.set_title(f"LinearRegression [{fs}]\nMAE={mae_v:.1f}  RMSE={rmse_v:.1f}")
    ax.legend(fontsize=8)

fig.suptitle("Actual vs Predicted — Validation Set", fontsize=13, fontweight="bold")
fig.tight_layout()
avp_path = os.path.join(FIG_DIR, "actual_vs_predicted.png")
fig.savefig(avp_path, dpi=150, bbox_inches="tight")
plt.close()
log("Saved figure: actual_vs_predicted.png")

# 7b. Residual distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col_idx, fs in enumerate(("F0", "F1")):
    yt = preds[fs]["y_true"]
    yp = preds[fs]["y_pred"]
    res = yt - yp

    # Residuals vs Fitted
    ax = axes[0, col_idx]
    ax.scatter(yp, res, alpha=0.18, s=8, color="#55A868", rasterized=True)
    ax.axhline(0, color="#C44E52", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs Fitted [{fs}]")

    # Residual histogram + KDE
    ax = axes[1, col_idx]
    ax.hist(res, bins=60, density=True, color="#4C72B0",
            edgecolor="white", linewidth=0.3, alpha=0.7, label="Residuals")
    # Manual KDE using Gaussian kernel
    bw = 1.06 * res.std() * len(res) ** (-0.2)
    x_kde = np.linspace(res.min(), res.max(), 400)
    kde_vals = np.array([
        np.mean(np.exp(-0.5 * ((x_kde - xi) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for xi in res
    ])  # exact KDE — O(n²) but n_val ~2600 so fast enough
    # Use vectorised form for speed
    diff = (x_kde[np.newaxis, :] - res[:, np.newaxis]) / bw  # (n_val, 400)
    kde_vals = np.mean(np.exp(-0.5 * diff ** 2) / (bw * np.sqrt(2 * np.pi)), axis=0)
    ax.plot(x_kde, kde_vals, color="#DD8452", linewidth=2, label="KDE")
    ax.axvline(0, color="#C44E52", linewidth=1.5, linestyle="--", label="Zero")
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution [{fs}]")
    ax.legend(fontsize=8)

fig.suptitle("Residual Diagnostics — Validation Set (LinearRegression)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
rd_path = os.path.join(FIG_DIR, "residual_distribution.png")
fig.savefig(rd_path, dpi=150, bbox_inches="tight")
plt.close()
log("Saved figure: residual_distribution.png")

# ── 8. Preprocessing report ───────────────────────────────────────────────────
prep_records = [
    {"metric": "train_rows",              "value": len(train)},
    {"metric": "val_rows",                "value": len(val)},
    {"metric": "test_rows",               "value": len(test)},
    {"metric": "chronological_split_pass","value": chron_pass},
    {"metric": "feature_set_F0_defined",  "value": True},
    {"metric": "feature_set_F1_defined",  "value": True},
    {"metric": "F0_n_features",           "value": len(F0)},
    {"metric": "F1_n_features",           "value": len(F1)},
    {"metric": "train_date_min",          "value": str(train["dteday"].min().date())},
    {"metric": "train_date_max",          "value": str(train["dteday"].max().date())},
    {"metric": "val_date_min",            "value": str(val["dteday"].min().date())},
    {"metric": "val_date_max",            "value": str(val["dteday"].max().date())},
    {"metric": "test_date_min",           "value": str(test["dteday"].min().date())},
    {"metric": "test_date_max",           "value": str(test["dteday"].max().date())},
]
prep_df = pd.DataFrame(prep_records)
prep_path = os.path.join(BENCH_DIR, "preprocessing_report.csv")
prep_df.to_csv(prep_path, index=False)
log(f"Preprocessing report saved to: {prep_path}")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)
val_rows = results_df[results_df["split"] == "val"]
for _, row in val_rows.iterrows():
    print(f"  LinearRegression [{row['feature_set']}]  "
          f"val MAE={row['MAE']:.2f}  RMSE={row['RMSE']:.2f}  "
          f"train_time={row['training_time_seconds']:.4f}s")
print("=" * 60)

log("Task 3 — Baseline model completed successfully.")
log("=" * 70)
