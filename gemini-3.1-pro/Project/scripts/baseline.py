import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging
import time

np.random.seed(42)

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
    logger.info("Started baseline modeling pipeline.")
    
    # Load dataset
    data_path = "outputs/cleaned_data.csv"
    if not os.path.exists(data_path):
        data_path = "dataset/hour.csv"
        logger.info(f"Cleaned data not found, falling back to {data_path}")
    else:
        logger.info(f"Loading cleaned data from {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Ensure data is sorted chronologically
    # "instant" is the chronological index in the bike sharing dataset
    if 'instant' in df.columns:
        df = df.sort_values('instant').reset_index(drop=True)
        time_col = 'instant'
    else:
        # Fallback if instant is not present
        time_col = None
    
    # Split data chronologically: 70% train / 15% validation / 15% test
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Explicitly verify and assert chronological split
    if time_col == 'instant':
        train_max_time = train_df[time_col].max()
        val_min_time = val_df[time_col].min()
        val_max_time = val_df[time_col].max()
        test_min_time = test_df[time_col].min()
        
        chronological_split_pass = (train_max_time < val_min_time) and (val_max_time < test_min_time)
        if not chronological_split_pass:
            raise ValueError("Chronological split failed! Overlapping time indices detected.")
        logger.info("Chronological split verified successfully.")
    else:
        chronological_split_pass = True
        
    # Define feature sets
    F0 = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    F0 = [f for f in F0 if f in df.columns]
    
    # Create cyclical features (F1)
    def add_cyclical_features(data):
        d = data.copy()
        if 'hr' in d.columns:
            d['sin_hour'] = np.sin(2 * np.pi * d['hr'] / 24.0)
            d['cos_hour'] = np.cos(2 * np.pi * d['hr'] / 24.0)
        if 'mnth' in d.columns:
            d['sin_month'] = np.sin(2 * np.pi * d['mnth'] / 12.0)
            d['cos_month'] = np.cos(2 * np.pi * d['mnth'] / 12.0)
        return d
    
    train_df = add_cyclical_features(train_df)
    val_df = add_cyclical_features(val_df)
    test_df = add_cyclical_features(test_df)
    
    F1 = F0.copy()
    for col in ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']:
        if col in train_df.columns:
            F1.append(col)
            
    # Save splits
    os.makedirs("outputs", exist_ok=True)
    train_df.to_csv("outputs/train.csv", index=False)
    val_df.to_csv("outputs/val.csv", index=False)
    test_df.to_csv("outputs/test.csv", index=False)
    logger.info("Saved train, val, test splits.")
    
    # Save preprocessing report
    report_data = {
        "train_rows": [len(train_df)],
        "val_rows": [len(val_df)],
        "test_rows": [len(test_df)],
        "chronological_split_pass": [chronological_split_pass],
        "feature_set_F0_defined": [True],
        "feature_set_F1_defined": [True]
    }
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(log_dir, "preprocessing_report.csv"), index=False)
    logger.info("Saved preprocessing report.")
    
    # Target
    target = 'cnt'
    
    results = []
    
    # Prepare data for F0
    X_train_F0 = train_df[F0]
    y_train = train_df[target]
    X_val_F0 = val_df[F0]
    y_val = val_df[target]
    
    # Model F0
    model_F0 = LinearRegression(n_jobs=1)
    start_time = time.time()
    model_F0.fit(X_train_F0, y_train)
    train_time_F0 = time.time() - start_time
    
    y_val_pred_F0 = model_F0.predict(X_val_F0)
    mae_F0 = mean_absolute_error(y_val, y_val_pred_F0)
    rmse_F0 = np.sqrt(mean_squared_error(y_val, y_val_pred_F0))
    
    results.append({
        'model': 'Linear Regression',
        'feature_set': 'F0',
        'split': 'validation',
        'MAE': mae_F0,
        'RMSE': rmse_F0,
        'training_time_seconds': train_time_F0
    })
    
    # Prepare data for F1
    X_train_F1 = train_df[F1]
    X_val_F1 = val_df[F1]
    
    # Model F1
    model_F1 = LinearRegression(n_jobs=1)
    start_time = time.time()
    model_F1.fit(X_train_F1, y_train)
    train_time_F1 = time.time() - start_time
    
    y_val_pred_F1 = model_F1.predict(X_val_F1)
    mae_F1 = mean_absolute_error(y_val, y_val_pred_F1)
    rmse_F1 = np.sqrt(mean_squared_error(y_val, y_val_pred_F1))
    
    results.append({
        'model': 'Linear Regression',
        'feature_set': 'F1',
        'split': 'validation',
        'MAE': mae_F1,
        'RMSE': rmse_F1,
        'training_time_seconds': train_time_F1
    })
    
    # Save results
    metrics_dir = "outputs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(metrics_dir, "baseline_model_results.csv"), index=False)
    logger.info("Saved baseline model results.")
    
    # Generate diagnostic plots (using F1 predictions)
    fig_dir = "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # actual_vs_predicted.png
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, y_val_pred_F1, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title('Actual vs Predicted (Validation Set, F1)')
    plt.xlabel('Actual cnt')
    plt.ylabel('Predicted cnt')
    plt.savefig(os.path.join(fig_dir, 'actual_vs_predicted.png'), bbox_inches='tight')
    plt.close()
    logger.info("Saved actual_vs_predicted.png")
    
    # residual_distribution.png
    residuals = y_val - y_val_pred_F1
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title('Residual Distribution (Validation Set, F1)')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(fig_dir, 'residual_distribution.png'), bbox_inches='tight')
    plt.close()
    logger.info("Saved residual_distribution.png")

    logger.info("Baseline modeling pipeline completed successfully.")

if __name__ == "__main__":
    main()