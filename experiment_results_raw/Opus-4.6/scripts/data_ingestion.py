"""
Task 1: Dataset Ingestion, Schema Checks, and Missingness Handling
Bike-sharing regression dataset with target variable cnt.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "hour.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
VALIDATION_REPORT_PATH = os.path.join(BENCHMARK_DIR, "data_validation_report.csv")
EXPERIMENT_LOG_PATH = os.path.join(BENCHMARK_DIR, "experiment_log.txt")

for d in [FIGURES_DIR, METRICS_DIR, BENCHMARK_DIR,
          os.path.join(OUTPUT_DIR, "models"), os.path.join(OUTPUT_DIR, "docs")]:
    os.makedirs(d, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Print and append to experiment log."""
    print(msg)
    with open(EXPERIMENT_LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run() -> None:
    # Clear log for fresh run
    open(EXPERIMENT_LOG_PATH, "w").close()

    log("=" * 60)
    log("TASK 1: Data Ingestion, Schema Checks & Missingness Handling")
    log("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    log(f"\n[1] Dataset loaded from {DATA_PATH}")

    # ------------------------------------------------------------------
    # 2. Dataset shape
    # ------------------------------------------------------------------
    log(f"[2] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 3. Column names and data types
    # ------------------------------------------------------------------
    log("\n[3] Column names and data types:")
    for col in df.columns:
        log(f"    {col:20s} -> {df[col].dtype}")

    # ------------------------------------------------------------------
    # 4. Identify numeric vs categorical variables
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    log(f"\n[4] Numeric columns ({len(numeric_cols)}):     {numeric_cols}")
    log(f"    Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # ------------------------------------------------------------------
    # 5. Missing values per column
    # ------------------------------------------------------------------
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    log(f"\n[5] Missing values per column:")
    for col in df.columns:
        log(f"    {col:20s} -> {missing[col]}")
    log(f"    TOTAL missing values: {total_missing}")

    # ------------------------------------------------------------------
    # 6. Duplicate rows
    # ------------------------------------------------------------------
    duplicate_count = int(df.duplicated().sum())
    log(f"\n[6] Duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log(f"    Removed {duplicate_count} duplicate rows. New shape: {df.shape}")

    # ------------------------------------------------------------------
    # 7. Leakage validation: casual + registered == cnt
    # ------------------------------------------------------------------
    leakage_check = (df["casual"] + df["registered"] == df["cnt"]).all()
    log(f"\n[7] Leakage check (casual + registered == cnt): {leakage_check}")
    if leakage_check:
        log("    -> Confirmed: casual + registered = cnt (identity leak).")
        log("    -> Dropping 'casual' and 'registered' to prevent leakage.")
        df = df.drop(columns=["casual", "registered"])
    else:
        log("    -> No perfect identity leakage detected; keeping columns.")

    # ------------------------------------------------------------------
    # 8. Correlation with cnt & flag |corr| > 0.95
    # ------------------------------------------------------------------
    numeric_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()
    if "cnt" in numeric_for_corr:
        numeric_for_corr.remove("cnt")

    correlations = df[numeric_for_corr].corrwith(df["cnt"])
    log("\n[8] Correlation of each feature with cnt:")
    flagged_features = []
    for feat, corr_val in correlations.items():
        flag = " *** FLAGGED (|corr| > 0.95)" if abs(corr_val) > 0.95 else ""
        log(f"    {feat:20s} -> {corr_val:+.4f}{flag}")
        if abs(corr_val) > 0.95:
            flagged_features.append(feat)

    if flagged_features:
        log(f"    Flagged features (|corr| > 0.95): {flagged_features}")
    else:
        log("    No features flagged with |corr| > 0.95.")

    # ------------------------------------------------------------------
    # 9. Validate cnt >= 0
    # ------------------------------------------------------------------
    negative_cnt = int((df["cnt"] < 0).sum())
    target_nonneg = negative_cnt == 0
    log(f"\n[9] cnt >= 0 for all rows: {target_nonneg}  (negative rows: {negative_cnt})")

    # ------------------------------------------------------------------
    # 10. Check impossible values in count-like fields
    # ------------------------------------------------------------------
    count_like = [c for c in ["cnt", "casual", "registered"] if c in df.columns]
    log(f"\n[10] Impossible-value check on count-like fields {count_like}:")
    impossible_found = False
    for col in count_like:
        neg = int((df[col] < 0).sum())
        if neg > 0:
            log(f"     {col}: {neg} negative values found!")
            impossible_found = True
        else:
            log(f"     {col}: OK (no negative values)")
    if not impossible_found:
        log("     No impossible values detected.")

    # ------------------------------------------------------------------
    # 11. Save cleaned dataset
    # ------------------------------------------------------------------
    df.to_csv(CLEANED_DATA_PATH, index=False)
    log(f"\n[11] Cleaned dataset saved to {CLEANED_DATA_PATH}")
    log(f"     Final shape: {df.shape}")

    # ------------------------------------------------------------------
    # 12. Save validation report
    # ------------------------------------------------------------------
    report_rows = [
        {"check": "missing_values_total", "result": total_missing},
        {"check": "duplicate_rows", "result": duplicate_count},
        {"check": "leakage_identity_detected", "result": leakage_check},
        {"check": "target_nonnegative_check", "result": target_nonneg},
        {"check": "flagged_high_corr_features", "result": str(flagged_features) if flagged_features else "none"},
        {"check": "impossible_count_values", "result": impossible_found},
        {"check": "rows_after_cleaning", "result": df.shape[0]},
        {"check": "columns_after_cleaning", "result": df.shape[1]},
    ]
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(VALIDATION_REPORT_PATH, index=False)
    log(f"\n[12] Validation report saved to {VALIDATION_REPORT_PATH}")

    # ------------------------------------------------------------------
    # 13. Summary
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("SUMMARY OF FINDINGS AND CHANGES")
    log("=" * 60)
    log(f"  - Original shape:          17379 rows x 17 columns")
    log(f"  - Total missing values:    {total_missing}")
    log(f"  - Duplicate rows removed:  {duplicate_count}")
    log(f"  - Leakage columns dropped: {'casual, registered' if leakage_check else 'none'}")
    log(f"  - High-corr features:      {flagged_features if flagged_features else 'none'}")
    log(f"  - Negative cnt values:     {negative_cnt}")
    log(f"  - Final cleaned shape:     {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  - Cleaned data saved to:   {CLEANED_DATA_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    run()
