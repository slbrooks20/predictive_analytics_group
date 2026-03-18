import pandas as pd
import numpy as np
import os
import logging

# Set up logging
log_dir = "outputs/benchmark"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "experiment_log.txt"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    np.random.seed(42)
    logger.info("Started data preparation pipeline.")

    # Load dataset
    dataset_path = "dataset/hour.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Print and log dataset shape
    shape = df.shape
    print(f"Dataset shape: {shape}")
    logger.info(f"Dataset shape: {shape}")

    # Print column names and data types
    print("\nColumn names and data types:")
    print(df.dtypes)
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes.to_string()}")

    # Identify numeric vs categorical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")

    # Check missing values per column
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    logger.info(f"Total missing values: {total_missing}")

    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_rows}")
    logger.info(f"Duplicate rows: {duplicate_rows}")

    # Perform explicit leakage validation
    leakage_identity_detected = False
    leakage_columns_removed = []

    if 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns:
        # Verify whether casual + registered = cnt
        if (df['casual'] + df['registered'] == df['cnt']).all():
            leakage_identity_detected = True
            logger.info("Leakage detected: casual + registered == cnt")
            # Remove casual and registered
            df = df.drop(columns=['casual', 'registered'])
            leakage_columns_removed = ['casual', 'registered']
            logger.info("Removed 'casual' and 'registered' columns.")
            print("\nLeakage detected: 'casual' + 'registered' == 'cnt'. Columns removed.")

    # Compute correlation between each feature and cnt
    max_feature_target_corr = 0.0
    if 'cnt' in df.columns:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'cnt' in numeric_features:
            numeric_features.remove('cnt')
        
        correlations = df[numeric_features + ['cnt']].corr()['cnt'].drop('cnt')
        max_feature_target_corr = correlations.abs().max()
        
        # Flag any feature with absolute correlation > 0.95
        high_corr_features = correlations[correlations.abs() > 0.95]
        if not high_corr_features.empty:
            logger.warning(f"Features with > 0.95 absolute correlation with cnt: {high_corr_features.to_dict()}")
            print(f"\nWarning: High correlation (> 0.95) features detected:\n{high_corr_features}")

    # Validate that cnt >= 0 for all rows
    target_nonnegative_check = True
    if 'cnt' in df.columns:
        if not (df['cnt'] >= 0).all():
            target_nonnegative_check = False
            logger.error("Validation failed: 'cnt' contains negative values.")
            print("\nError: 'cnt' contains negative values.")
        else:
            logger.info("Validation passed: 'cnt' is non-negative for all rows.")

    # Check for impossible values in count-like fields (already covered by cnt check, but can generalize if needed)

    # Save cleaned dataset
    os.makedirs("outputs", exist_ok=True)
    cleaned_data_path = "outputs/cleaned_data.csv"
    df.to_csv(cleaned_data_path, index=False)
    logger.info(f"Saved cleaned dataset to {cleaned_data_path}")

    # Save a validation report
    report_data = {
        "metric": [
            "missing_values_total",
            "duplicate_rows",
            "leakage_identity_detected",
            "leakage_columns_removed",
            "max_feature_target_corr",
            "target_nonnegative_check"
        ],
        "value": [
            total_missing,
            duplicate_rows,
            leakage_identity_detected,
            "|".join(leakage_columns_removed) if leakage_columns_removed else "None",
            max_feature_target_corr,
            target_nonnegative_check
        ]
    }
    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(log_dir, "data_validation_report.csv")
    report_df.to_csv(report_path, index=False)
    logger.info(f"Saved validation report to {report_path}")

    print("\n--- Summary of Findings and Changes ---")
    print(f"Total missing values: {total_missing}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Leakage identity detected (casual+registered=cnt): {leakage_identity_detected}")
    if leakage_identity_detected:
         print(f"Columns removed due to leakage: {leakage_columns_removed}")
    print(f"Max absolute feature-target correlation: {max_feature_target_corr:.4f}")
    print(f"Target is non-negative: {target_nonnegative_check}")
    print("---------------------------------------")

if __name__ == "__main__":
    main()