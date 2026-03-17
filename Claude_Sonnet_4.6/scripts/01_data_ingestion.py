"""
Task 1: Dataset Ingestion, Schema Checks, and Missingness Handling
Bike-sharing regression dataset — target variable: cnt
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE_DIR, "dataset", "hour.csv")
OUT_CLEAN     = os.path.join(BASE_DIR, "outputs", "cleaned_data.csv")
OUT_REPORT    = os.path.join(BASE_DIR, "outputs", "benchmark", "data_validation_report.csv")
LOG_FILE      = os.path.join(BASE_DIR, "outputs", "benchmark", "experiment_log.txt")

os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CLEAN),  exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── 1. Load dataset ────────────────────────────────────────────────────────────
log("=" * 60)
log("TASK 1 — Data Ingestion, Schema Checks & Missingness Handling")
log("=" * 60)

if not os.path.exists(DATA_PATH):
    log(f"ERROR: Dataset not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
log(f"Dataset loaded from: {DATA_PATH}")

# ── 2. Shape ───────────────────────────────────────────────────────────────────
rows, cols = df.shape
log(f"Dataset shape: {rows} rows × {cols} columns")
print(f"\nShape: {df.shape}")

# ── 3. Column names and data types ────────────────────────────────────────────
print("\nColumn names and data types:")
print(df.dtypes.to_string())
log("Column dtypes logged.")

# ── 4. Numeric vs categorical ─────────────────────────────────────────────────
# Semantic categorical columns in this dataset (stored as int codes)
SEMANTIC_CAT = {"season", "yr", "mnth", "hr", "holiday", "weekday",
                "workingday", "weathersit"}

numeric_cols     = [c for c in df.columns if df[c].dtype in [np.float64, np.float32,
                    np.int64, np.int32] and c not in SEMANTIC_CAT]
categorical_cols = [c for c in df.columns if c in SEMANTIC_CAT]

print(f"\nNumeric columns    ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
log(f"Numeric cols: {numeric_cols}")
log(f"Categorical cols: {categorical_cols}")

# ── 5. Missing values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_total = int(missing.sum())
print(f"\nMissing values per column:\n{missing[missing > 0] if missing_total > 0 else 'None'}")
log(f"Total missing values: {missing_total}")

# ── 6. Duplicate rows ─────────────────────────────────────────────────────────
dup_count = int(df.duplicated().sum())
print(f"\nDuplicate rows: {dup_count}")
log(f"Duplicate rows found: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    log(f"Dropped {dup_count} duplicate rows.")

# ── 7. Leakage validation ─────────────────────────────────────────────────────
leakage_detected  = False
leakage_cols_removed = []

if "casual" in df.columns and "registered" in df.columns and "cnt" in df.columns:
    identity_check = (df["casual"] + df["registered"] == df["cnt"]).all()
    leakage_detected = bool(identity_check)
    if leakage_detected:
        df = df.drop(columns=["casual", "registered"])
        leakage_cols_removed = ["casual", "registered"]
        log("Leakage detected: casual + registered == cnt. Columns removed.")
    else:
        log("No perfect additive leakage found (casual + registered != cnt).")
else:
    log("Columns 'casual' or 'registered' not found — leakage check skipped.")

print(f"\nLeakage identity detected: {leakage_detected}")
print(f"Leakage columns removed:   {leakage_cols_removed}")

# ── 8. Correlation with target ────────────────────────────────────────────────
TARGET = "cnt"
feature_cols = [c for c in df.columns if c != TARGET and df[c].dtype in
                [np.float64, np.float32, np.int64, np.int32]]

corr_series = df[feature_cols + [TARGET]].corr()[TARGET].drop(TARGET).abs()
HIGH_CORR_THRESH = 0.95
high_corr_features = corr_series[corr_series > HIGH_CORR_THRESH].index.tolist()
max_corr = float(corr_series.max())

print(f"\nFeature–target correlations (|r|):\n{corr_series.sort_values(ascending=False).to_string()}")
print(f"\nMax |correlation| with cnt: {max_corr:.4f}")
print(f"Features with |corr| > {HIGH_CORR_THRESH}: {high_corr_features}")
log(f"Max |corr| with cnt: {max_corr:.4f}")
log(f"High-corr features (>{HIGH_CORR_THRESH}): {high_corr_features}")

# ── 9. Validate cnt >= 0 ──────────────────────────────────────────────────────
cnt_nonneg = bool((df[TARGET] >= 0).all())
print(f"\ncnt non-negative check: {cnt_nonneg}")
log(f"cnt >= 0 for all rows: {cnt_nonneg}")

# ── 10. Impossible values in count-like fields ────────────────────────────────
count_like = ["cnt"] + [c for c in ["casual", "registered"] if c in df.columns]
impossible_flags = {}
for col in count_like:
    n_neg = int((df[col] < 0).sum())
    impossible_flags[col] = n_neg
    if n_neg > 0:
        log(f"WARNING: {n_neg} negative values found in '{col}'")

print(f"\nNegative value counts in count fields: {impossible_flags}")

# ── 11. Save cleaned dataset ──────────────────────────────────────────────────
df.to_csv(OUT_CLEAN, index=False)
log(f"Cleaned dataset saved to: {OUT_CLEAN}  ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"\nCleaned dataset saved → {OUT_CLEAN}")

# ── 12. Validation report ─────────────────────────────────────────────────────
report_rows = [
    {"check": "missing_values_total",       "value": missing_total,            "status": "PASS" if missing_total == 0 else "WARN"},
    {"check": "duplicate_rows",             "value": dup_count,                "status": "PASS" if dup_count == 0     else "WARN"},
    {"check": "leakage_identity_detected",  "value": str(leakage_detected),    "status": "WARN" if leakage_detected   else "PASS"},
    {"check": "leakage_columns_removed",    "value": str(leakage_cols_removed),"status": "INFO"},
    {"check": "max_feature_target_corr",    "value": round(max_corr, 4),       "status": "WARN" if max_corr > HIGH_CORR_THRESH else "PASS"},
    {"check": "target_nonnegative_check",   "value": str(cnt_nonneg),          "status": "PASS" if cnt_nonneg else "FAIL"},
    {"check": "high_corr_features",         "value": str(high_corr_features),  "status": "INFO"},
    {"check": "final_shape_rows",           "value": df.shape[0],              "status": "INFO"},
    {"check": "final_shape_cols",           "value": df.shape[1],              "status": "INFO"},
]

report_df = pd.DataFrame(report_rows)
report_df.to_csv(OUT_REPORT, index=False)
log(f"Validation report saved to: {OUT_REPORT}")
print(f"Validation report saved → {OUT_REPORT}")

# ── 13. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)
print(f"  Rows × Cols (original)  : {rows} × {cols}")
print(f"  Missing values          : {missing_total}")
print(f"  Duplicate rows          : {dup_count}")
print(f"  Leakage detected        : {leakage_detected}  → removed {leakage_cols_removed}")
print(f"  Max |feature–cnt corr|  : {max_corr:.4f}")
print(f"  High-corr features      : {high_corr_features}")
print(f"  cnt non-negative        : {cnt_nonneg}")
print(f"  Final shape             : {df.shape[0]} × {df.shape[1]}")
print("=" * 60)

log("Task 1 complete.")
log("=" * 60)
