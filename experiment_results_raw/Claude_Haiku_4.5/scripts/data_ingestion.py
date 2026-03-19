"""
Bike-sharing dataset ingestion, validation, and cleaning.
Target variable: cnt (bike rental count)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Suppress pandas deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning)

# Set random seed
np.random.seed(42)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "hour.csv"
OUTPUT_CLEANED = PROJECT_ROOT / "outputs" / "cleaned_data.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "outputs" / "benchmark" / "data_validation_report.csv"
LOG_PATH = PROJECT_ROOT / "outputs" / "benchmark" / "experiment_log.txt"

# Ensure output directories exist
OUTPUT_CLEANED.parent.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_message(msg):
    """Log message to both console and log file."""
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def main():
    # Clear log file
    with open(LOG_PATH, "w") as f:
        f.write("=== Bike-Sharing Data Ingestion & Validation Log ===\n\n")

    log_message("Starting data ingestion...")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    log_message(f"\n1. Dataset Loaded")
    log_message(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Print column names and data types
    log_message(f"\n2. Column Names and Data Types:")
    for col in df.columns:
        log_message(f"   {col:20s} {df[col].dtype}")

    # Identify numeric vs categorical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    log_message(f"\n3. Variable Classification:")
    log_message(f"   Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    log_message(f"   Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # Check missing values
    log_message(f"\n4. Missing Values Per Column:")
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    log_message(f"   Total missing values: {total_missing}")
    if total_missing > 0:
        for col in missing_summary[missing_summary > 0].index:
            log_message(f"   {col}: {missing_summary[col]}")
    else:
        log_message("   No missing values found")

    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    log_message(f"\n5. Duplicate Rows: {duplicate_rows}")

    # Explicit leakage validation: casual + registered = cnt
    log_message(f"\n6. Leakage Validation:")
    if 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns:
        leakage_check = np.isclose(df['casual'] + df['registered'], df['cnt']).all()
        log_message(f"   casual + registered = cnt? {leakage_check}")

        if leakage_check:
            log_message(f"   [WARNING] LEAKAGE DETECTED: casual and registered are components of cnt")
            log_message(f"   Removing 'casual' and 'registered' from dataset")
            df_cleaned = df.drop(columns=['casual', 'registered'])
            leakage_detected = True
            columns_removed = 2
        else:
            log_message(f"   No leakage identity detected")
            df_cleaned = df.copy()
            leakage_detected = False
            columns_removed = 0
    else:
        log_message("   Leakage check: required columns not found")
        df_cleaned = df.copy()
        leakage_detected = False
        columns_removed = 0

    # Correlation analysis
    log_message(f"\n7. Feature-Target Correlation:")
    numeric_cols_for_corr = [c for c in df_cleaned.select_dtypes(include=[np.number]).columns if c != 'cnt']
    correlations = df_cleaned[numeric_cols_for_corr + ['cnt']].corr()['cnt'].drop('cnt').abs().sort_values(ascending=False)

    max_corr = correlations.max()
    max_corr_feature = correlations.idxmax()
    log_message(f"   Max feature-target correlation: {max_corr:.4f} ({max_corr_feature})")

    high_corr_features = correlations[correlations > 0.95]
    if len(high_corr_features) > 0:
        log_message(f"   [WARNING] Features with |corr| > 0.95 with cnt:")
        for feat, corr_val in high_corr_features.items():
            log_message(f"     {feat}: {corr_val:.4f}")
    else:
        log_message(f"   No features with |correlation| > 0.95")

    # Validate cnt >= 0
    log_message(f"\n8. Target Variable Validation:")
    target_nonnegative = (df_cleaned['cnt'] >= 0).all()
    log_message(f"   All cnt values >= 0? {target_nonnegative}")

    if not target_nonnegative:
        negative_count = (df_cleaned['cnt'] < 0).sum()
        log_message(f"   [WARNING] Found {negative_count} negative cnt values")

    # Check for impossible values in count-like fields
    log_message(f"\n9. Impossible Values Check:")
    count_like_cols = [c for c in df_cleaned.columns if c in ['cnt', 'casual', 'registered'] and c in df_cleaned.columns]

    impossible_found = False
    for col in count_like_cols:
        negative = (df_cleaned[col] < 0).sum()
        non_integer = ((df_cleaned[col] % 1) != 0).sum()

        if negative > 0 or non_integer > 0:
            log_message(f"   {col}: {negative} negative, {non_integer} non-integer values")
            impossible_found = True

    if not impossible_found:
        log_message(f"   No impossible values found in count-like fields")

    # Save cleaned dataset
    log_message(f"\n10. Saving Cleaned Dataset:")
    df_cleaned.to_csv(OUTPUT_CLEANED, index=False)
    log_message(f"    Saved to: {OUTPUT_CLEANED}")
    log_message(f"    Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")

    # Create validation report
    log_message(f"\n11. Creating Validation Report:")
    validation_report = pd.DataFrame({
        'metric': [
            'missing_values_total',
            'duplicate_rows',
            'leakage_identity_detected',
            'leakage_columns_removed',
            'max_feature_target_corr',
            'target_nonnegative_check'
        ],
        'value': [
            total_missing,
            duplicate_rows,
            leakage_detected,
            columns_removed,
            f"{max_corr:.6f}",
            target_nonnegative
        ]
    })

    validation_report.to_csv(VALIDATION_REPORT_PATH, index=False)
    log_message(f"    Saved to: {VALIDATION_REPORT_PATH}")

    # Print summary
    log_message(f"\n{'='*60}")
    log_message(f"SUMMARY OF FINDINGS AND CHANGES")
    log_message(f"{'='*60}")
    log_message(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns → {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
    log_message(f"Missing values: {total_missing}")
    log_message(f"Duplicate rows: {duplicate_rows}")
    log_message(f"Leakage detected: {leakage_detected} (removed {columns_removed} columns)")
    log_message(f"Max feature-target correlation: {max_corr:.6f}")
    log_message(f"Target non-negative validation: {target_nonnegative}")
    log_message(f"Output: {OUTPUT_CLEANED}")
    log_message(f"{'='*60}\n")

    return df_cleaned


if __name__ == "__main__":
    df_cleaned = main()
