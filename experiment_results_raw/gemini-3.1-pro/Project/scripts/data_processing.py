import pandas as pd
import numpy as np
import os

# Set random seed
np.random.seed(42)

def run_data_processing():
    # Define paths
    input_path = 'dataset/hour.csv'
    output_cleaned_path = 'outputs/cleaned_data.csv'
    output_validation_path = 'outputs/benchmark/data_validation_report.csv'
    log_path = 'outputs/benchmark/experiment_log.txt'

    # Ensure directories exist
    os.makedirs('outputs/benchmark', exist_ok=True)

    # Logging helper
    def log(message):
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + '\n')

    log("--- DATA PROCESSING STARTED ---")

    # Load dataset
    df = pd.read_csv(input_path)
    log(f"Dataset shape: {df.shape}")

    # Column names and data types
    log("\nColumn names and data types:")
    log(str(df.dtypes))

    # Numeric vs Categorical
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_vars = df.select_dtypes(exclude=[np.number]).columns.tolist()
    log(f"\nNumeric variables: {numeric_vars}")
    log(f"Categorical variables: {categorical_vars}")

    # Missing values
    missing_values = df.isnull().sum()
    log("\nMissing values per column:")
    log(str(missing_values))

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    log(f"\nDuplicate rows: {duplicate_rows}")

    # Leakage validation: casual + registered = cnt
    leakage_identity_detected = False
    leakage_columns_removed = False
    if 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns:
        identity_check = (df['casual'] + df['registered'] == df['cnt']).all()
        if identity_check:
            leakage_identity_detected = True
            log("\nLeakage detected: casual + registered = cnt. Removing casual and registered columns.")
            df = df.drop(columns=['casual', 'registered'])
            leakage_columns_removed = True
        else:
            log("\nNo exact leakage identity (casual + registered = cnt) detected.")

    # Correlation with cnt
    target_col = 'cnt'
    correlations = df.select_dtypes(include=[np.number]).corr()[target_col].abs()
    max_feature_target_corr = correlations.drop(target_col, errors='ignore').max()
    log(f"\nMax feature correlation with target: {max_feature_target_corr}")

    # Flag features with absolute correlation > 0.95
    high_corr_features = correlations[correlations > 0.95].index.tolist()
    high_corr_features = [f for f in high_corr_features if f != target_col]
    if high_corr_features:
        log(f"High correlation features (> 0.95) flagged: {high_corr_features}")

    # Non-negative check for cnt
    target_nonnegative_check = (df[target_col] >= 0).all()
    log(f"\nTarget non-negative check: {target_nonnegative_check}")

    # Impossible values in count-like fields (e.g., negative counts or non-integer counts if applicable)
    # Since we checked cnt >= 0, we can also check for integer values if necessary.
    # For now, non-negative is the primary 'impossible' check requested.

    # Save cleaned dataset
    df.to_csv(output_cleaned_path, index=False)
    log(f"\nCleaned dataset saved to: {output_cleaned_path}")

    # Validation report
    validation_report = {
        'missing_values_total': missing_values.sum(),
        'duplicate_rows': duplicate_rows,
        'leakage_identity_detected': leakage_identity_detected,
        'leakage_columns_removed': leakage_columns_removed,
        'max_feature_target_corr': max_feature_target_corr,
        'target_nonnegative_check': target_nonnegative_check
    }
    report_df = pd.DataFrame(list(validation_report.items()), columns=['Metric', 'Value'])
    report_df.to_csv(output_validation_path, index=False)
    log(f"Validation report saved to: {output_validation_path}")

    log("\n--- DATA PROCESSING COMPLETED ---")

if __name__ == '__main__':
    run_data_processing()
