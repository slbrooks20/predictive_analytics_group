import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
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
os.makedirs('outputs', exist_ok=True)

# Initialize logging
log_file = 'outputs/benchmark/experiment_log.txt'
with open(log_file, 'a') as f:
    f.write(f"\n{datetime.now()} - Baseline Model Training started\n")

def log_step(message):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")
    print(message)

# 1. Data preparation
log_step("Loading data...")
if os.path.exists('outputs/cleaned_data.csv'):
    df = pd.read_csv('outputs/cleaned_data.csv')
    log_step("Loaded cleaned data from outputs/cleaned_data.csv")
else:
    df = pd.read_csv('dataset/hour.csv')
    log_step("Loaded raw data from dataset/hour.csv")

# Check if target variable exists
target_var = 'cnt'
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in dataset")

# Create chronological split
log_step("Creating chronological split...")
if 'dteday' in df.columns and 'hr' in df.columns:
    df['dteday'] = pd.to_datetime(df['dteday'])
    # Sort by both date and hour to ensure proper chronological order
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)
    
    # Calculate split points
    total_rows = len(df)
    train_end = int(0.70 * total_rows)
    val_end = int(0.85 * total_rows)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Verify chronological split using datetime + hour
    train_max = pd.to_datetime(train_df['dteday'].max()) + pd.Timedelta(hours=train_df.loc[train_df['dteday'] == train_df['dteday'].max(), 'hr'].max())
    val_min = pd.to_datetime(val_df['dteday'].min()) + pd.Timedelta(hours=val_df.loc[val_df['dteday'] == val_df['dteday'].min(), 'hr'].min())
    val_max = pd.to_datetime(val_df['dteday'].max()) + pd.Timedelta(hours=val_df.loc[val_df['dteday'] == val_df['dteday'].max(), 'hr'].max())
    test_min = pd.to_datetime(test_df['dteday'].min()) + pd.Timedelta(hours=test_df.loc[test_df['dteday'] == test_df['dteday'].min(), 'hr'].min())
    
    chronological_split_pass = True
    if train_max >= val_min:
        chronological_split_pass = False
        raise ValueError(f"Chronological split validation failed: max(train_time) {train_max} >= min(validation_time) {val_min}")
    
    if val_max >= test_min:
        chronological_split_pass = False
        raise ValueError(f"Chronological split validation failed: max(validation_time) {val_max} >= min(test_time) {test_min}")
    
    log_step(f"Chronological split validation passed")
    log_step(f"Train: {len(train_df)} rows ({train_df['dteday'].min()} {train_df.loc[train_df['dteday'] == train_df['dteday'].min(), 'hr'].min():02d}:00 to {train_df['dteday'].max()} {train_df.loc[train_df['dteday'] == train_df['dteday'].max(), 'hr'].max():02d}:00)")
    log_step(f"Val: {len(val_df)} rows ({val_df['dteday'].min()} {val_df.loc[val_df['dteday'] == val_df['dteday'].min(), 'hr'].min():02d}:00 to {val_df['dteday'].max()} {val_df.loc[val_df['dteday'] == val_df['dteday'].max(), 'hr'].max():02d}:00)")
    log_step(f"Test: {len(test_df)} rows ({test_df['dteday'].min()} {test_df.loc[test_df['dteday'] == test_df['dteday'].min(), 'hr'].min():02d}:00 to {test_df['dteday'].max()} {test_df.loc[test_df['dteday'] == test_df['dteday'].max(), 'hr'].max():02d}:00)")
elif 'instant' in df.columns:
    # Use instant column if available (it's already a unique chronological index)
    df['instant'] = pd.to_datetime(df['instant'])
    df = df.sort_values('instant').reset_index(drop=True)
    
    # Calculate split points
    total_rows = len(df)
    train_end = int(0.70 * total_rows)
    val_end = int(0.85 * total_rows)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Verify chronological split
    chronological_split_pass = True
    if max(train_df['instant']) >= min(val_df['instant']):
        chronological_split_pass = False
        raise ValueError(f"Chronological split validation failed: max(train_time) {max(train_df['instant'])} >= min(validation_time) {min(val_df['instant'])}")
    
    if max(val_df['instant']) >= min(test_df['instant']):
        chronological_split_pass = False
        raise ValueError(f"Chronological split validation failed: max(validation_time) {max(val_df['instant'])} >= min(test_time) {min(test_df['instant'])}")
    
    log_step(f"Chronological split validation passed using 'instant' column")
    log_step(f"Train: {len(train_df)} rows ({train_df['instant'].min()} to {train_df['instant'].max()})")
    log_step(f"Val: {len(val_df)} rows ({val_df['instant'].min()} to {val_df['instant'].max()})")
    log_step(f"Test: {len(test_df)} rows ({test_df['instant'].min()} to {test_df['instant'].max()})")
else:
    # Fallback to random split if no datetime column
    log_step("Warning: Neither 'dteday' nor 'instant' column found, using random split instead")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(0.70 * len(df))
    val_end = int(0.85 * len(df))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    chronological_split_pass = False

# Save splits
log_step("Saving data splits...")
train_df.to_csv('outputs/train.csv', index=False)
val_df.to_csv('outputs/val.csv', index=False)
test_df.to_csv('outputs/test.csv', index=False)
log_step("Data splits saved to outputs/train.csv, outputs/val.csv, outputs/test.csv")

# Define feature sets
log_step("Defining feature sets...")

# F0: Original predictors
F0_features = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
F0_features = [f for f in F0_features if f in df.columns]
feature_set_F0_defined = len(F0_features) > 0

# F1: F0 + cyclical features
F1_features = F0_features.copy()
if 'hr' in df.columns:
    F1_features.extend(['sin_hour', 'cos_hour'])
    train_df['sin_hour'] = np.sin(2 * np.pi * train_df['hr'] / 24)
    train_df['cos_hour'] = np.cos(2 * np.pi * train_df['hr'] / 24)
    val_df['sin_hour'] = np.sin(2 * np.pi * val_df['hr'] / 24)
    val_df['cos_hour'] = np.cos(2 * np.pi * val_df['hr'] / 24)
    test_df['sin_hour'] = np.sin(2 * np.pi * test_df['hr'] / 24)
    test_df['cos_hour'] = np.cos(2 * np.pi * test_df['hr'] / 24)

if 'mnth' in df.columns:
    F1_features.extend(['sin_month', 'cos_month'])
    train_df['sin_month'] = np.sin(2 * np.pi * train_df['mnth'] / 12)
    train_df['cos_month'] = np.cos(2 * np.pi * train_df['mnth'] / 12)
    val_df['sin_month'] = np.sin(2 * np.pi * val_df['mnth'] / 12)
    val_df['cos_month'] = np.cos(2 * np.pi * val_df['mnth'] / 12)
    test_df['sin_month'] = np.sin(2 * np.pi * test_df['mnth'] / 12)
    test_df['cos_month'] = np.cos(2 * np.pi * test_df['mnth'] / 12)

# Remove duplicates from F1_features
F1_features = list(set(F1_features))
feature_set_F1_defined = len(F1_features) > len(F0_features)

log_step(f"F0 features: {F0_features}")
log_step(f"F1 features: {F1_features}")

# 2. Save preprocessing report
preprocessing_report = pd.DataFrame({
    'metric': [
        'train_rows',
        'val_rows',
        'test_rows',
        'chronological_split_pass',
        'feature_set_F0_defined',
        'feature_set_F1_defined'
    ],
    'value': [
        len(train_df),
        len(val_df),
        len(test_df),
        chronological_split_pass,
        feature_set_F0_defined,
        feature_set_F1_defined
    ]
})
preprocessing_report.to_csv('outputs/benchmark/preprocessing_report.csv', index=False)
log_step("Preprocessing report saved to outputs/benchmark/preprocessing_report.csv")

# 3. Train baseline model
log_step("Training baseline models...")

results = []

for feature_set_name, features in [('F0', F0_features), ('F1', F1_features)]:
    if not features:
        log_step(f"Skipping feature set {feature_set_name} - no features defined")
        continue
    
    log_step(f"Training Linear Regression with feature set {feature_set_name}")
    
    # Prepare data
    X_train = train_df[features]
    y_train = train_df[target_var]
    X_val = val_df[features]
    y_val = val_df[target_var]
    
    # Train model
    start_time = time.time()
    model = LinearRegression(n_jobs=1)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    log_step(f"Feature set {feature_set_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, Training time: {training_time:.4f}s")
    
    # Save model
    model_path = f'outputs/models/linear_regression_{feature_set_name}.joblib'
    import joblib
    joblib.dump(model, model_path)
    log_step(f"Model saved to {model_path}")
    
    # Store results
    results.append({
        'model': 'LinearRegression',
        'feature_set': feature_set_name,
        'split': 'validation',
        'MAE': mae,
        'RMSE': rmse,
        'training_time_seconds': training_time
    })
    
    # Generate diagnostic plots for this feature set
    log_step(f"Generating diagnostic plots for feature set {feature_set_name}")
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted (Feature Set {feature_set_name})')
    plt.savefig(f'outputs/figures/actual_vs_predicted_{feature_set_name}.png')
    plt.close()
    
    # Residual distribution plot
    residuals = y_val - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution (Feature Set {feature_set_name})')
    plt.savefig(f'outputs/figures/residual_distribution_{feature_set_name}.png')
    plt.close()

# 4. Save baseline results
if results:
    baseline_results = pd.DataFrame(results)
    baseline_results.to_csv('outputs/metrics/baseline_model_results.csv', index=False)
    log_step("Baseline model results saved to outputs/metrics/baseline_model_results.csv")
else:
    log_step("Warning: No baseline results to save")

log_step("Baseline model training completed successfully.")