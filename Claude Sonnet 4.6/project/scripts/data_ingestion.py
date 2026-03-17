"""
Task 1: Dataset Ingestion, Schema Checks, and Missingness Handling
Bike-Sharing Regression Dataset — target variable: cnt
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "dataset", "hour.csv")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
BENCH_DIR   = os.path.join(OUT_DIR, "benchmark")
LOG_PATH    = os.path.join(BENCH_DIR, "experiment_log.txt")
CLEAN_PATH  = os.path.join(OUT_DIR, "cleaned_data.csv")
REPORT_PATH = os.path.join(BENCH_DIR, "data_validation_report.csv")

os.makedirs(BENCH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Logger ────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ── 1. Load dataset ───────────────────────────────────────────────────────────
log("=" * 70)
log("TASK 1 — Data Ingestion, Schema Checks, Missingness Handling")
log("=" * 70)

log(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
log("Dataset loaded successfully.")

# ── 2. Shape ──────────────────────────────────────────────────────────────────
shape_msg = f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns"
print(shape_msg)
log(shape_msg)

# ── 3. Column names and data types ────────────────────────────────────────────
log("Column names and data types:")
dtype_info = df.dtypes.to_string()
print("\nColumn names and dtypes:\n" + dtype_info)
log(dtype_info)

# ── 4. Identify numeric vs categorical ────────────────────────────────────────
# Semantic categoricals based on dataset documentation
SEMANTIC_CATS = ["season", "yr", "mnth", "hr", "holiday", "weekday",
                 "workingday", "weathersit"]

numeric_cols     = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in SEMANTIC_CATS]
categorical_cols = [c for c in df.columns
                    if c in SEMANTIC_CATS or df[c].dtype == object]

log(f"Numeric columns     ({len(numeric_cols)}): {numeric_cols}")
log(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"\nNumeric columns     : {numeric_cols}")
print(f"Categorical columns : {categorical_cols}")

# ── 5. Missing values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_total = int(missing.sum())
log(f"Missing values per column:\n{missing.to_string()}")
log(f"Total missing values: {missing_total}")
print(f"\nMissing values per column:\n{missing.to_string()}")
print(f"Total missing values: {missing_total}")

# ── 6. Duplicate rows ─────────────────────────────────────────────────────────
dup_count = int(df.duplicated().sum())
log(f"Duplicate rows: {dup_count}")
print(f"\nDuplicate rows: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    log(f"Dropped {dup_count} duplicate rows. New shape: {df.shape}")

# ── 7. Leakage validation: casual + registered == cnt? ────────────────────────
leakage_detected = False
leakage_cols_removed = []

if {"casual", "registered"}.issubset(df.columns):
    identity_check = (df["casual"] + df["registered"] == df["cnt"]).all()
    leakage_detected = bool(identity_check)
    log(f"Leakage identity check (casual + registered == cnt): {leakage_detected}")
    print(f"\nLeakage identity check (casual + registered == cnt): {leakage_detected}")
    if leakage_detected:
        leakage_cols_removed = ["casual", "registered"]
        df = df.drop(columns=leakage_cols_removed)
        log(f"Removed leakage columns: {leakage_cols_removed}")
        print(f"Removed leakage columns: {leakage_cols_removed}")
else:
    log("Columns 'casual' and/or 'registered' not found — skipping leakage check.")

# ── 8. Correlation of each feature with cnt ───────────────────────────────────
# Use select_dtypes to safely exclude string/object columns (handles pandas 3.x StringDtype)
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "cnt"]
corr_with_target = df[feature_cols].corrwith(df["cnt"]).abs().sort_values(ascending=False)

log("Absolute correlation with cnt (top features):")
log(corr_with_target.to_string())
print("\nAbsolute correlation with cnt:")
print(corr_with_target.to_string())

HIGH_CORR_THRESH = 0.95
high_corr_features = corr_with_target[corr_with_target > HIGH_CORR_THRESH].index.tolist()
max_corr_val = float(corr_with_target.max())

if high_corr_features:
    log(f"WARNING: Features with |corr| > {HIGH_CORR_THRESH}: {high_corr_features}")
    print(f"\nWARNING: Features with |corr| > {HIGH_CORR_THRESH}: {high_corr_features}")
else:
    log(f"No features exceed the |corr| > {HIGH_CORR_THRESH} threshold (max={max_corr_val:.4f}).")
    print(f"\nNo features exceed |corr| > {HIGH_CORR_THRESH} (max={max_corr_val:.4f}).")

# ── 9. Validate cnt >= 0 ──────────────────────────────────────────────────────
cnt_nonneg = bool((df["cnt"] >= 0).all())
cnt_neg_count = int((df["cnt"] < 0).sum())
log(f"cnt >= 0 for all rows: {cnt_nonneg} (negative count: {cnt_neg_count})")
print(f"\ncnt >= 0 for all rows: {cnt_nonneg}")

# ── 10. Check for impossible values in count-like fields ─────────────────────
count_fields = [c for c in ["cnt", "casual", "registered"]
                if c in df.columns]
impossible_issues = {}
for col in count_fields:
    neg_rows = int((df[col] < 0).sum())
    if neg_rows:
        impossible_issues[col] = neg_rows
        log(f"ISSUE: Column '{col}' has {neg_rows} negative values.")

if not impossible_issues:
    log("No impossible (negative) values found in count-like fields.")
    print("No impossible values found in count-like fields.")
else:
    print(f"Impossible value issues: {impossible_issues}")

# Also check temp, atemp, hum, windspeed are within plausible normalised ranges [0,1]
range_cols = {"temp": (0, 1), "atemp": (0, 1), "hum": (0, 1), "windspeed": (0, 1)}
for col, (lo, hi) in range_cols.items():
    if col in df.columns:
        out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
        if out_of_range:
            log(f"ISSUE: Column '{col}' has {out_of_range} values outside [{lo},{hi}].")
        else:
            log(f"Column '{col}': all values within [{lo},{hi}].")

# ── 11. Save cleaned dataset ─────────────────────────────────────────────────
df.to_csv(CLEAN_PATH, index=False)
log(f"Cleaned dataset saved to: {CLEAN_PATH}  (shape: {df.shape})")
print(f"\nCleaned dataset saved to: {CLEAN_PATH}")

# ── 12. Save validation report ───────────────────────────────────────────────
report_records = [
    {"check": "missing_values_total",        "value": missing_total,
     "status": "PASS" if missing_total == 0 else "WARN"},
    {"check": "duplicate_rows",              "value": dup_count,
     "status": "PASS" if dup_count == 0 else "WARN"},
    {"check": "leakage_identity_detected",   "value": int(leakage_detected),
     "status": "REMOVED" if leakage_detected else "PASS"},
    {"check": "leakage_columns_removed",     "value": str(leakage_cols_removed),
     "status": "REMOVED" if leakage_cols_removed else "NONE"},
    {"check": "max_feature_target_corr",     "value": round(max_corr_val, 6),
     "status": "WARN" if max_corr_val > HIGH_CORR_THRESH else "PASS"},
    {"check": "target_nonnegative_check",    "value": int(cnt_nonneg),
     "status": "PASS" if cnt_nonneg else "FAIL"},
    {"check": "impossible_values_in_counts", "value": str(impossible_issues),
     "status": "PASS" if not impossible_issues else "FAIL"},
    {"check": "final_dataset_shape_rows",    "value": df.shape[0],
     "status": "INFO"},
    {"check": "final_dataset_shape_cols",    "value": df.shape[1],
     "status": "INFO"},
]

report_df = pd.DataFrame(report_records)
report_df.to_csv(REPORT_PATH, index=False)
log(f"Validation report saved to: {REPORT_PATH}")
print(f"Validation report saved to: {REPORT_PATH}")

# ── 13. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS AND CHANGES")
print("=" * 60)
summary_lines = [
    f"  Original shape            : {pd.read_csv(DATA_PATH).shape}",
    f"  Final shape               : {df.shape}",
    f"  Total missing values      : {missing_total}",
    f"  Duplicate rows dropped    : {dup_count}",
    f"  Leakage detected          : {leakage_detected}",
    f"  Leakage columns removed   : {leakage_cols_removed}",
    f"  Max |corr| with cnt       : {max_corr_val:.4f}",
    f"  High-corr features (>0.95): {high_corr_features}",
    f"  cnt >= 0 check            : {'PASSED' if cnt_nonneg else 'FAILED'}",
    f"  Impossible value issues   : {impossible_issues if impossible_issues else 'None'}",
]
for line in summary_lines:
    print(line)
    log(line)

print("=" * 60)
log("Task 1 completed successfully.")
log("=" * 70)
