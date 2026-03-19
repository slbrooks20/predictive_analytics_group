import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(42)

def log_message(message):
    """Log message to experiment log file"""
    with open('outputs/benchmark/experiment_log.txt', 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

def create_cyclical_features(df):
    """Create cyclical features for hour and month"""
    df = df.copy()
    df['sin_hour'] = np.sin(2 * np.pi * df['hr'] / 23)
    df['cos_hour'] = np.cos(2 * np.pi * df['hr'] / 23)
    df['sin_month'] = np.sin(2 * np.pi * df['mnth'] / 11)
    df['cos_month'] = np.cos(2 * np.pi * df['mnth'] / 11)
    return df

def main():
    # Create output directories if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/docs', exist_ok=True)
    os.makedirs('outputs/benchmark', exist_ok=True)
    
    log_message("Starting baseline modeling pipeline")
    
    # 1. Data preparation
    log_message("Loading data for chronological splitting")
    # Load original data to preserve datetime for proper splitting
    df = pd.read_csv('dataset/hour.csv')
    
    # Parse datetime for chronological splitting
    df['datetime'] = pd.to_datetime(df['dteday'])
    df = df.sort_values('datetime')
    
    # Apply the same cleaning as in data_validation.py
    # Remove leakage columns
    if 'casual' in df.columns and 'registered' in df.columns:
        df = df.drop(['casual', 'registered'], axis=1)
    
    # Chronological split: 70% train / 15% validation / 15% test
    # Use time-based splitting to avoid boundary issues
    unique_dates = df['datetime'].dt.date.unique()
    n_dates = len(unique_dates)
    
    train_end_idx = int(n_dates * 0.7)
    val_end_idx = int(n_dates * 0.85)
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    train_df = df[df['datetime'].dt.date.isin(train_dates)].copy()
    val_df = df[df['datetime'].dt.date.isin(val_dates)].copy()
    test_df = df[df['datetime'].dt.date.isin(test_dates)].copy()
    
    # Verify chronological split
    chronological_split_pass = True
    if max(train_df['datetime']) >= min(val_df['datetime']):
        chronological_split_pass = False
        raise ValueError("Train and validation sets overlap in time!")
    
    if max(val_df['datetime']) >= min(test_df['datetime']):
        chronological_split_pass = False
        raise ValueError("Validation and test sets overlap in time!")
    
    log_message(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")
    
    # Define feature sets
    F0_features = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    
    # Create F1 features (F0 + cyclical)
    train_df = create_cyclical_features(train_df)
    val_df = create_cyclical_features(val_df)
    test_df = create_cyclical_features(test_df)
    
    F1_features = F0_features + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    
    # Save splits
    train_df.to_csv('outputs/train.csv', index=False)
    val_df.to_csv('outputs/val.csv', index=False)
    test_df.to_csv('outputs/test.csv', index=False)
    
    log_message("Saved data splits to outputs/")
    print("Saved data splits to outputs/")
    
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
            len(F0_features),
            len(F1_features)
        ]
    })
    
    preprocessing_report.to_csv('outputs/benchmark/preprocessing_report.csv', index=False)
    log_message("Saved preprocessing report to outputs/benchmark/preprocessing_report.csv")
    print("Saved preprocessing report to outputs/benchmark/preprocessing_report.csv")
    
    # 3. Train baseline model: Linear Regression
    log_message("Training baseline models")
    
    results = []
    
    # Train and evaluate with F0 features
    log_message("Training with F0 feature set")
    start_time = datetime.now()
    
    # Fit scaler on train only
    scaler_F0 = StandardScaler()
    X_train_F0 = train_df[F0_features]
    X_train_F0_scaled = scaler_F0.fit_transform(X_train_F0)
    
    # Train model
    model_F0 = LinearRegression(n_jobs=1)
    model_F0.fit(X_train_F0_scaled, train_df['cnt'])
    
    # Evaluate on validation
    X_val_F0 = val_df[F0_features]
    X_val_F0_scaled = scaler_F0.transform(X_val_F0)
    val_pred_F0 = model_F0.predict(X_val_F0_scaled)
    
    mae_F0 = mean_absolute_error(val_df['cnt'], val_pred_F0)
    rmse_F0 = np.sqrt(mean_squared_error(val_df['cnt'], val_pred_F0))
    training_time_F0 = (datetime.now() - start_time).total_seconds()
    
    results.append({
        'model': 'LinearRegression',
        'feature_set': 'F0',
        'split': 'validation',
        'MAE': mae_F0,
        'RMSE': rmse_F0,
        'training_time_seconds': training_time_F0
    })
    
    log_message(f"F0 - MAE: {mae_F0:.2f}, RMSE: {rmse_F0:.2f}, Time: {training_time_F0:.2f}s")
    print(f"F0 - MAE: {mae_F0:.2f}, RMSE: {rmse_F0:.2f}, Time: {training_time_F0:.2f}s")
    
    # Train and evaluate with F1 features
    log_message("Training with F1 feature set")
    start_time = datetime.now()
    
    # Fit scaler on train only
    scaler_F1 = StandardScaler()
    X_train_F1 = train_df[F1_features]
    X_train_F1_scaled = scaler_F1.fit_transform(X_train_F1)
    
    # Train model
    model_F1 = LinearRegression(n_jobs=1)
    model_F1.fit(X_train_F1_scaled, train_df['cnt'])
    
    # Evaluate on validation
    X_val_F1 = val_df[F1_features]
    X_val_F1_scaled = scaler_F1.transform(X_val_F1)
    val_pred_F1 = model_F1.predict(X_val_F1_scaled)
    
    mae_F1 = mean_absolute_error(val_df['cnt'], val_pred_F1)
    rmse_F1 = np.sqrt(mean_squared_error(val_df['cnt'], val_pred_F1))
    training_time_F1 = (datetime.now() - start_time).total_seconds()
    
    results.append({
        'model': 'LinearRegression',
        'feature_set': 'F1',
        'split': 'validation',
        'MAE': mae_F1,
        'RMSE': rmse_F1,
        'training_time_seconds': training_time_F1
    })
    
    log_message(f"F1 - MAE: {mae_F1:.2f}, RMSE: {rmse_F1:.2f}, Time: {training_time_F1:.2f}s")
    print(f"F1 - MAE: {mae_F1:.2f}, RMSE: {rmse_F1:.2f}, Time: {training_time_F1:.2f}s")
    
    # Save baseline results
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/metrics/baseline_model_results.csv', index=False)
    log_message("Saved baseline results to outputs/metrics/baseline_model_results.csv")
    print("Saved baseline results to outputs/metrics/baseline_model_results.csv")
    
    # Save models
    import joblib
    joblib.dump({'model': model_F0, 'scaler': scaler_F0, 'features': F0_features}, 
                'outputs/models/linear_regression_F0.joblib')
    joblib.dump({'model': model_F1, 'scaler': scaler_F1, 'features': F1_features}, 
                'outputs/models/linear_regression_F1.joblib')
    log_message("Saved models to outputs/models/")
    print("Saved models to outputs/models/")
    
    # 4. Generate diagnostic plots (using F1 model as it should perform better)
    log_message("Generating diagnostic plots")
    
    # Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(val_df['cnt'], val_pred_F1, alpha=0.6)
    plt.plot([val_df['cnt'].min(), val_df['cnt'].max()], 
             [val_df['cnt'].min(), val_df['cnt'].max()], 'r--', lw=2)
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title('Actual vs Predicted (F1 Feature Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved actual_vs_predicted.png")
    print("Saved actual_vs_predicted.png")
    
    # Residual distribution
    residuals = val_df['cnt'] - val_pred_F1
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Distribution (F1 Feature Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved residual_distribution.png")
    print("Saved residual_distribution.png")
    
    log_message("Baseline modeling pipeline completed successfully")
    print("\nBaseline modeling pipeline completed successfully")

if __name__ == "__main__":
    main()
