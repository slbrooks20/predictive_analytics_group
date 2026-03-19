import pandas as pd
import numpy as np
import os
from datetime import datetime

# Set random seed
np.random.seed(42)

# Ensure output directories exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/metrics', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/docs', exist_ok=True)
os.makedirs('outputs/benchmark', exist_ok=True)

# Initialize logging
log_file = 'outputs/benchmark/experiment_log.txt'
with open(log_file, 'w') as f:
    f.write(f"Experiment started at {datetime.now()}\n")

def log_step(message):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")
    print(message)

# 1. Load dataset
log_step("Loading dataset...")
df = pd.read_csv('dataset/hour.csv')
log_step(f"Dataset loaded successfully.")

# 2. Print and log dataset shape
log_step(f"Dataset shape: {df.shape}")

# 3. Print column names and data types
log_step("Column names and data types:")
for col, dtype in df.dtypes.items():
    log_step(f"  {col}: {dtype}")

# 4. Identify numeric vs categorical variables
numeric_vars = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
categorical_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
log_step(f"Numeric variables: {numeric_vars}")
log_step(f"Categorical variables: {categorical_vars}")

# 5. Check missing values per column
missing_values = df.isnull().sum()
log_step("Missing values per column:")
for col, missing in missing_values.items():
    log_step(f"  {col}: {missing}")

# 6. Check for duplicate rows
duplicate_rows = df.duplicated().sum()
log_step(f"Duplicate rows: {duplicate_rows}")

# 7. Perform explicit leakage validation
leakage_identity_detected = False
leakage_columns_removed = []
if 'casual' in df.columns and 'registered' in df.columns and 'cnt' in df.columns:
    leakage_check = (df['casual'] + df['registered'] == df['cnt']).all()
    if leakage_check:
        leakage_identity_detected = True
        leakage_columns_removed = ['casual', 'registered']
        df = df.drop(columns=['casual', 'registered'])
        log_step("Leakage detected: casual + registered = cnt. Removed casual and registered columns.")
    else:
        log_step("No leakage detected: casual + registered != cnt.")
else:
    log_step("Leakage check skipped: required columns not found.")

# 8. Compute correlation between each feature and cnt
if 'cnt' in df.columns:
    correlations = df.corr(numeric_only=True)['cnt'].drop('cnt')
    max_corr = correlations.abs().max()
    log_step(f"Max feature-target correlation: {max_corr:.4f}")
    
    high_corr_features = correlations[correlations.abs() > 0.95].index.tolist()
    if high_corr_features:
        log_step(f"High correlation features (|corr| > 0.95): {high_corr_features}")
else:
    log_step("Correlation check skipped: 'cnt' column not found.")
    max_corr = np.nan

# 9. Validate that cnt >= 0 for all rows
target_nonnegative_check = False
if 'cnt' in df.columns:
    target_nonnegative_check = (df['cnt'] >= 0).all()
    log_step(f"Target nonnegative check: {target_nonnegative_check}")
else:
    log_step("Target nonnegative check skipped: 'cnt' column not found.")

# 10. Check for impossible values in count-like fields
count_like_fields = [col for col in df.columns if 'cnt' in col.lower() or 'count' in col.lower()]
if count_like_fields:
    log_step("Checking for impossible values in count-like fields:")
    for col in count_like_fields:
        invalid_values = df[col] < 0
        if invalid_values.any():
            log_step(f"  {col}: {invalid_values.sum()} invalid values (negative)")
        else:
            log_step(f"  {col}: No invalid values")

# 11. Save cleaned dataset
cleaned_data_path = 'outputs/cleaned_data.csv'
df.to_csv(cleaned_data_path, index=False)
log_step(f"Cleaned dataset saved to {cleaned_data_path}")

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
        missing_values.sum(),
        duplicate_rows,
        leakage_identity_detected,
        ', '.join(leakage_columns_removed) if leakage_columns_removed else 'None',
        max_corr,
        target_nonnegative_check
    ]
})
validation_report_path = 'outputs/benchmark/data_validation_report.csv'
validation_report.to_csv(validation_report_path, index=False)
log_step(f"Validation report saved to {validation_report_path}")

# 13. Print summary of findings and changes
log_step("\n=== Summary of Findings and Changes ===")
log_step(f"- Dataset shape: {df.shape}")
log_step(f"- Missing values: {missing_values.sum()}")
log_step(f"- Duplicate rows: {duplicate_rows}")
log_step(f"- Leakage detected: {leakage_identity_detected}")
log_step(f"- Columns removed due to leakage: {leakage_columns_removed}")
log_step(f"- Max feature-target correlation: {max_corr:.4f}")
log_step(f"- Target nonnegative check: {target_nonnegative_check}")

log_step("Data ingestion and validation completed successfully.")