"""
Data Ingestion and Validation Script
Loads, validates, and cleans the bike-sharing dataset
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Set random seed
np.random.seed(42)

def load_dataset(csv_path):
    """Load the dataset from CSV"""
    print(f"\n{'='*70}")
    print("DATA INGESTION AND VALIDATION REPORT")
    print(f"{'='*70}\n")

    try:
        df = pd.read_csv(csv_path)
        print(f"[OK] Dataset loaded successfully from: {csv_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found at {csv_path}")
        sys.exit(1)

def log_to_file(message, log_path):
    """Append message to log file"""
    with open(log_path, 'a') as f:
        f.write(message + '\n')

def check_dataset_shape(df):
    """Check and log dataset shape"""
    shape = df.shape
    print(f"\n--- DATASET SHAPE ---")
    print(f"Rows: {shape[0]}, Columns: {shape[1]}")
    return shape

def check_column_info(df):
    """Print column names and data types"""
    print(f"\n--- COLUMN INFORMATION ---")
    print(f"{'Column':<20} {'Data Type':<15} {'Non-Null Count':<15}")
    print("-" * 50)
    for col in df.columns:
        non_null = df[col].notna().sum()
        dtype = str(df[col].dtype)
        print(f"{col:<20} {dtype:<15} {non_null:<15}")

def identify_variable_types(df):
    """Identify numeric and categorical variables"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"\n--- VARIABLE TYPES ---")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    return numeric_cols, categorical_cols

def check_missing_values(df):
    """Check for missing values"""
    print(f"\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    if missing.sum() == 0:
        print("[OK] No missing values detected")
        return 0
    else:
        print(f"Columns with missing values:")
        for col in df.columns:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        return missing.sum()

def check_duplicate_rows(df):
    """Check for duplicate rows"""
    print(f"\n--- DUPLICATE ROWS ---")
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print("[OK] No duplicate rows detected")
    else:
        print(f"[WARNING] Found {duplicates} duplicate rows")
    return duplicates

def check_leakage(df):
    """Check for data leakage: casual + registered = cnt"""
    print(f"\n--- DATA LEAKAGE CHECK ---")

    if 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns:
        leakage_check = (df['casual'] + df['registered'] == df['cnt']).all()

        if leakage_check:
            print("[ERROR] LEAKAGE DETECTED: casual + registered = cnt")
            print("  Action: Removing 'casual' and 'registered' columns")
            df = df.drop(columns=['casual', 'registered'])
            return True, df
        else:
            # Check percentage of rows where leakage exists
            leakage_rows = (df['casual'] + df['registered'] == df['cnt']).sum()
            leakage_pct = (leakage_rows / len(df)) * 100
            print(f"Partial leakage: {leakage_pct:.2f}% of rows have casual + registered = cnt")
            print("  Action: Keeping columns for investigation")
            return False, df
    else:
        print("[OK] Required leakage columns not all present, skipping check")
        return False, df

def compute_correlations(df):
    """Compute correlation between features and target"""
    print(f"\n--- FEATURE-TARGET CORRELATION ---")

    if 'cnt' not in df.columns:
        print("[ERROR] Target column 'cnt' not found")
        return None

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    correlations = {}

    for col in numeric_cols:
        if col != 'cnt':
            corr = df[col].corr(df['cnt'])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 10 correlations with cnt:")
    high_corr_features = []
    for i, (col, corr) in enumerate(sorted_corr[:10]):
        print(f"  {i+1}. {col:<20} {corr:>8.4f}")
        if abs(corr) > 0.95:
            high_corr_features.append((col, corr))

    if high_corr_features:
        print(f"\n[WARNING] Features with |correlation| > 0.95:")
        for col, corr in high_corr_features:
            print(f"  {col}: {corr:.4f}")
    else:
        print("\n[OK] No features with |correlation| > 0.95")

    return correlations

def validate_target_values(df):
    """Validate that cnt >= 0 for all rows"""
    print(f"\n--- TARGET VALUE VALIDATION ---")

    if 'cnt' not in df.columns:
        print("[ERROR] Target column 'cnt' not found")
        return False

    invalid_rows = (df['cnt'] < 0).sum()

    if invalid_rows == 0:
        print("[OK] All values in 'cnt' are non-negative")
        print(f"  Min: {df['cnt'].min()}, Max: {df['cnt'].max()}, Mean: {df['cnt'].mean():.2f}")
        return True
    else:
        print(f"[ERROR] Found {invalid_rows} rows with cnt < 0")
        return False

def check_impossible_values(df):
    """Check for impossible values in count-like fields"""
    print(f"\n--- IMPOSSIBLE VALUES CHECK ---")

    count_cols = [col for col in df.columns if 'cnt' in col or col in ['casual', 'registered']]
    impossible_found = False

    for col in count_cols:
        if col not in df.columns:
            continue

        # Check for negative values
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  [ERROR] {col}: {neg_count} negative values")
            impossible_found = True

    if not impossible_found:
        print("[OK] No impossible values in count-like fields")

    return impossible_found

def save_cleaned_data(df, output_path):
    """Save cleaned dataset"""
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Cleaned dataset saved to: {output_path}")
    print(f"  Shape: {df.shape}")

def create_validation_report(df, original_shape, missing_count, duplicate_count,
                            leakage_detected, correlations, output_path):
    """Create validation report"""

    report_data = {
        'Metric': [],
        'Value': []
    }

    # Extract high correlation features
    high_corr_features = []
    if correlations:
        for col, corr in correlations.items():
            if abs(corr) > 0.95:
                high_corr_features.append(col)

    # Build report
    report_data['Metric'].extend([
        'original_rows',
        'original_columns',
        'final_rows',
        'final_columns',
        'missing_values_total',
        'duplicate_rows',
        'leakage_identity_detected',
        'leakage_columns_removed',
        'target_nonnegative_check',
        'max_feature_target_corr',
        'high_corr_features_count'
    ])

    max_corr = max([abs(c) for c in correlations.values()]) if correlations else 0

    report_data['Value'].extend([
        str(original_shape[0]),
        str(original_shape[1]),
        str(df.shape[0]),
        str(df.shape[1]),
        str(missing_count),
        str(duplicate_count),
        'Yes' if leakage_detected else 'No',
        str(2 if leakage_detected else 0),  # columns removed
        'Pass' if (df['cnt'] >= 0).all() else 'Fail',
        f"{max_corr:.4f}",
        str(len(high_corr_features))
    ])

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_path, index=False)
    print(f"\n[OK] Validation report saved to: {output_path}")

    return report_df

def main():
    """Main execution"""
    # Paths
    dataset_path = Path('dataset/hour.csv')
    output_dir = Path('outputs')
    cleaned_data_path = output_dir / 'cleaned_data.csv'
    validation_report_path = output_dir / 'benchmark/data_validation_report.csv'
    log_path = output_dir / 'benchmark/experiment_log.txt'

    # Initialize log file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("DATA INGESTION AND VALIDATION LOG\n")
        f.write(f"{'='*70}\n\n")

    # Load dataset
    df = load_dataset(dataset_path)
    original_shape = df.shape

    # Dataset checks
    check_dataset_shape(df)
    check_column_info(df)
    numeric_cols, categorical_cols = identify_variable_types(df)
    missing_count = check_missing_values(df)
    duplicate_count = check_duplicate_rows(df)

    # Data leakage check
    leakage_detected, df = check_leakage(df)

    # Correlation analysis
    correlations = compute_correlations(df)

    # Target validation
    validate_target_values(df)
    check_impossible_values(df)

    # Save cleaned data
    save_cleaned_data(df, cleaned_data_path)

    # Create validation report
    report_df = create_validation_report(df, original_shape, missing_count, duplicate_count,
                                        leakage_detected, correlations, validation_report_path)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF FINDINGS AND CHANGES")
    print(f"{'='*70}")
    print(f"\nOriginal dataset: {original_shape[0]} rows x {original_shape[1]} columns")
    print(f"Cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nMissing values: {missing_count}")
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Data leakage detected: {'Yes (casual + registered = cnt removed)' if leakage_detected else 'No'}")

    if correlations:
        max_corr = max([abs(c) for c in correlations.values()])
        print(f"Max |feature-target correlation|: {max_corr:.4f}")

    print(f"\nTarget validation (cnt >= 0): {'[OK] PASS' if (df['cnt'] >= 0).all() else '[ERROR] FAIL'}")
    print(f"\nFiles saved:")
    print(f"  - Cleaned data: {cleaned_data_path}")
    print(f"  - Validation report: {validation_report_path}")
    print(f"  - Log file: {log_path}")
    print(f"\n{'='*70}\n")

    # Log summary to file
    with open(log_path, 'a') as f:
        f.write(f"\nOriginal shape: {original_shape}\n")
        f.write(f"Final shape: {df.shape}\n")
        f.write(f"Missing values: {missing_count}\n")
        f.write(f"Duplicate rows: {duplicate_count}\n")
        f.write(f"Leakage detected: {leakage_detected}\n")
        f.write(f"\nValidation Report:\n")
        f.write(report_df.to_string())

if __name__ == '__main__':
    main()
