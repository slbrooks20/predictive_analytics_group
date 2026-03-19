import pandas as pd
import numpy as np
import os
from datetime import datetime

# Set random seed
np.random.seed(42)

def log_message(message):
    """Log message to experiment log file"""
    with open('outputs/benchmark/experiment_log.txt', 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

def main():
    # Create output directories if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/docs', exist_ok=True)
    os.makedirs('outputs/benchmark', exist_ok=True)
    
    # Clear or create experiment log
    with open('outputs/benchmark/experiment_log.txt', 'w') as f:
        f.write("=== DATA VALIDATION EXPERIMENT LOG ===\n")
    
    log_message("Starting data validation pipeline")
    
    # 1. Load dataset
    log_message("Loading dataset from dataset/hour.csv")
    df = pd.read_csv('dataset/hour.csv')
    
    # 2. Print and log dataset shape
    log_message(f"Dataset shape: {df.shape}")
    print(f"Dataset shape: {df.shape}")
    
    # 3. Print column names and data types
    log_message("Column names and data types:")
    print("\nColumn names and data types:")
    print(df.dtypes)
    
    # 4. Identify numeric vs categorical variables
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    log_message(f"Numeric columns: {numeric_cols}")
    log_message(f"Categorical columns: {categorical_cols}")
    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # 5. Check missing values per column
    missing_values = df.isnull().sum()
    missing_values_total = missing_values.sum()
    
    log_message(f"Missing values per column:\n{missing_values[missing_values > 0]}")
    log_message(f"Total missing values: {missing_values_total}")
    print(f"\nMissing values per column:")
    print(missing_values[missing_values > 0])
    print(f"Total missing values: {missing_values_total}")
    
    # 6. Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    log_message(f"Duplicate rows: {duplicate_rows}")
    print(f"\nDuplicate rows: {duplicate_rows}")
    
    # 7. Perform explicit leakage validation
    leakage_identity_detected = False
    leakage_columns_removed = []
    
    if 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns:
        # Check if casual + registered = cnt
        leakage_check = (df['casual'] + df['registered'] == df['cnt']).all()
        
        if leakage_check:
            leakage_identity_detected = True
            log_message("Leakage detected: casual + registered = cnt")
            print("\nLeakage detected: casual + registered = cnt")
            
            # Remove casual and registered columns
            df = df.drop(['casual', 'registered'], axis=1)
            leakage_columns_removed = ['casual', 'registered']
            log_message(f"Removed columns due to leakage: {leakage_columns_removed}")
            print(f"Removed columns due to leakage: {leakage_columns_removed}")
    
    # 8. Compute correlation between each feature and cnt
    if 'cnt' in df.columns:
        # Drop non-numeric columns before correlation calculation
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        correlations = numeric_df.corr()['cnt'].abs().sort_values(ascending=False)
        max_feature_target_corr = correlations.max()
        
        log_message(f"Max correlation with target: {max_feature_target_corr}")
        print(f"\nMax correlation with target (cnt): {max_feature_target_corr}")
        
        # Flag features with absolute correlation > 0.95
        high_corr_features = correlations[correlations > 0.95]
        if len(high_corr_features) > 1:  # Exclude cnt itself
            log_message(f"Features with correlation > 0.95: {high_corr_features.index.tolist()}")
            print(f"Features with correlation > 0.95: {high_corr_features.index.tolist()}")
    
    # 9. Validate that cnt >= 0 for all rows
    target_nonnegative_check = True
    if 'cnt' in df.columns:
        negative_count = (df['cnt'] < 0).sum()
        if negative_count > 0:
            target_nonnegative_check = False
            log_message(f"WARNING: {negative_count} rows have negative cnt values")
            print(f"\nWARNING: {negative_count} rows have negative cnt values")
        else:
            log_message("All cnt values are non-negative")
            print("\nAll cnt values are non-negative")
    
    # 10. Check for impossible values in count-like fields
    count_columns = ['cnt']
    impossible_values_found = False
    
    for col in count_columns:
        if col in df.columns:
            # Check for negative values
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                impossible_values_found = True
                log_message(f"WARNING: {negative_count} negative values in {col}")
                print(f"\nWARNING: {negative_count} negative values in {col}")
            
            # Check for unrealistically high values (more than 2000 bikes)
            unrealistic_count = (df[col] > 2000).sum()
            if unrealistic_count > 0:
                impossible_values_found = True
                log_message(f"WARNING: {unrealistic_count} unrealistic values (>2000) in {col}")
                print(f"\nWARNING: {unrealistic_count} unrealistic values (>2000) in {col}")
    
    # 11. Save cleaned dataset
    log_message("Saving cleaned dataset to outputs/cleaned_data.csv")
    df.to_csv('outputs/cleaned_data.csv', index=False)
    print("\nCleaned dataset saved to outputs/cleaned_data.csv")
    
    # 12. Save validation report
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
            missing_values_total,
            duplicate_rows,
            leakage_identity_detected,
            ','.join(leakage_columns_removed) if leakage_columns_removed else 'None',
            max_feature_target_corr if 'cnt' in df.columns else 0,
            target_nonnegative_check
        ]
    })
    
    validation_report.to_csv('outputs/benchmark/data_validation_report.csv', index=False)
    log_message("Validation report saved to outputs/benchmark/data_validation_report.csv")
    print("\nValidation report saved to outputs/benchmark/data_validation_report.csv")
    
    # 13. Print summary of findings and changes
    log_message("\n=== SUMMARY OF FINDINGS AND CHANGES ===")
    print("\n=== SUMMARY OF FINDINGS AND CHANGES ===")
    
    summary_points = []
    summary_points.append(f"Initial dataset shape: {df.shape}")
    summary_points.append(f"Missing values: {missing_values_total}")
    summary_points.append(f"Duplicate rows: {duplicate_rows}")
    
    if leakage_identity_detected:
        summary_points.append(f"Leakage detected and removed: {leakage_columns_removed}")
    else:
        summary_points.append("No leakage detected")
    
    if 'cnt' in df.columns:
        summary_points.append(f"Max feature-target correlation: {max_feature_target_corr:.4f}")
        summary_points.append(f"Target non-negative check: {'PASSED' if target_nonnegative_check else 'FAILED'}")
    
    if impossible_values_found:
        summary_points.append("WARNING: Impossible values detected in count fields")
    
    for point in summary_points:
        log_message(point)
        print(point)
    
    log_message("Data validation pipeline completed successfully")
    print("\nData validation pipeline completed successfully")

if __name__ == "__main__":
    main()
