import pandas as pd
import numpy as np
import os
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set random seed
np.random.seed(42)

def run_baseline_model():
    # Define paths
    input_path = 'outputs/cleaned_data.csv'
    if not os.path.exists(input_path):
        input_path = 'dataset/hour.csv'
    
    outputs_dir = 'outputs/'
    figures_dir = 'outputs/figures/'
    metrics_dir = 'outputs/metrics/'
    models_dir = 'outputs/models/'
    benchmark_dir = 'outputs/benchmark/'
    log_path = 'outputs/benchmark/experiment_log.txt'

    for d in [outputs_dir, figures_dir, metrics_dir, models_dir, benchmark_dir]:
        os.makedirs(d, exist_ok=True)

    # Logging helper
    def log(message):
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + '\n')

    log("\n--- BASELINE MODEL TRAINING STARTED ---")

    # 1. Load data
    df = pd.read_csv(input_path)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])

    # 2. Chronological Split (70/15/15)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Verify chronological split
    # Use 'instant' or ('dteday' + 'hr') for verification
    # 'instant' is a monotonic index in the original dataset
    train_max_time = train_df['instant'].max()
    val_min_time = val_df['instant'].min()
    val_max_time = val_df['instant'].max()
    test_min_time = test_df['instant'].min()

    chronological_split_pass = (train_max_time < val_min_time) and (val_max_time < test_min_time)
    
    if not chronological_split_pass:
        log("ERROR: Chronological split verification failed.")
        raise ValueError("Chronological split failed verification.")
    
    log(f"Chronological split verified. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Feature Engineering (Cyclical Features)
    def add_cyclical_features(data):
        d = data.copy()
        d['sin_hour'] = np.sin(2 * np.pi * d['hr'] / 24)
        d['cos_hour'] = np.cos(2 * np.pi * d['hr'] / 24)
        d['sin_month'] = np.sin(2 * np.pi * (d['mnth'] - 1) / 12)
        d['cos_month'] = np.cos(2 * np.pi * (d['mnth'] - 1) / 12)
        return d

    train_df = add_cyclical_features(train_df)
    val_df = add_cyclical_features(val_df)
    test_df = add_cyclical_features(test_df)

    # Define feature sets
    F0 = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    F1 = F0 + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    target = 'cnt'

    # Save splits
    train_df.to_csv(os.path.join(outputs_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(outputs_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(outputs_dir, 'test.csv'), index=False)

    # Preprocessing Report
    preprocessing_report = pd.DataFrame([{
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'chronological_split_pass': chronological_split_pass,
        'feature_set_F0_defined': True,
        'feature_set_F1_defined': True
    }])
    preprocessing_report.to_csv(os.path.join(benchmark_dir, 'preprocessing_report.csv'), index=False)

    # 4. Model Training and Evaluation
    results = []
    best_val_mae = float('inf')
    best_model = None
    best_preds = None
    best_y_val = None

    for f_name, features in [('F0', F0), ('F1', F1)]:
        X_train = train_df[features]
        y_train = train_df[target]
        X_val = val_df[features]
        y_val = val_df[target]

        start_time = time.time()
        model = LinearRegression(n_jobs=1)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Validation set evaluation
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        results.append({
            'model': 'LinearRegression',
            'feature_set': f_name,
            'split': 'validation',
            'MAE': mae,
            'RMSE': rmse,
            'training_time_seconds': training_time
        })

        log(f"Model: LinearRegression, Features: {f_name}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Track best for plots
        if mae < best_val_mae:
            best_val_mae = mae
            best_model = model
            best_preds = y_pred
            best_y_val = y_val
            joblib.dump(model, os.path.join(models_dir, 'baseline_linear_regression.joblib'))

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(metrics_dir, 'baseline_model_results.csv'), index=False)

    # 5. Diagnostic Plots (using best validation results)
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(best_y_val, best_preds, alpha=0.3, color='blue')
    plt.plot([best_y_val.min(), best_y_val.max()], [best_y_val.min(), best_y_val.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Demand (Baseline)')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.savefig(os.path.join(figures_dir, 'actual_vs_predicted.png'))
    plt.close()

    # Residual Distribution
    residuals = best_y_val - best_preds
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residual Distribution (Baseline)')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_dir, 'residual_distribution.png'))
    plt.close()

    log("--- BASELINE MODEL TRAINING COMPLETED ---")

if __name__ == '__main__':
    run_baseline_model()
