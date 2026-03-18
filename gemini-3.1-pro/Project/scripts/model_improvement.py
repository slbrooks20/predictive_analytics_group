import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seed and state
np.random.seed(42)
RANDOM_STATE = 42

def run_model_improvement():
    # Define paths
    train_path = 'outputs/train.csv'
    val_path = 'outputs/val.csv'
    test_path = 'outputs/test.csv'
    
    figures_dir = 'outputs/figures/'
    metrics_dir = 'outputs/metrics/'
    models_dir = 'outputs/models/'
    benchmark_dir = 'outputs/benchmark/'
    log_path = 'outputs/benchmark/experiment_log.txt'

    # Ensure directories exist
    for d in [figures_dir, metrics_dir, models_dir, benchmark_dir]:
        os.makedirs(d, exist_ok=True)

    # Logging helper
    def log(message):
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + '\n')

    log("\n--- MODEL IMPROVEMENT STARTED ---")

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Feature sets
    F0 = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    F1 = F0 + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    target = 'cnt'

    models_to_compare = {
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'MLP': MLPRegressor(random_state=RANDOM_STATE, max_iter=500)
    }

    comparison_results = []

    # 1. Model Comparison on F0 and F1 (Validation Set)
    for model_name, model_obj in models_to_compare.items():
        for f_name, features in [('F0', F0), ('F1', F1)]:
            X_train, y_train = train_df[features], train_df[target]
            X_val, y_val = val_df[features], val_df[target]

            start_time = time.time()
            model_obj.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = model_obj.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            comparison_results.append({
                'model': model_name,
                'feature_set': f_name,
                'split': 'validation',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': train_time
            })
            log(f"Model: {model_name}, Features: {f_name}, MAE: {mae:.2f}")

    pd.DataFrame(comparison_results).to_csv(os.path.join(metrics_dir, 'all_results.csv'), index=False)

    # 2. Tuning (Simplified for runtime efficiency while demonstrating the process)
    log("\n--- Tuning Models ---")
    tuning_results = []

    # Gradient Boosting Sweep for n_estimators
    n_estimators_list = [50, 100, 200, 300, 500]
    gb_mae_list = []
    best_gb_mae = float('inf')
    best_n_estimators = 100

    X_train_f1, y_train_f1 = train_df[F1], train_df[target]
    X_val_f1, y_val_f1 = val_df[F1], val_df[target]

    for n in n_estimators_list:
        model = GradientBoostingRegressor(n_estimators=n, random_state=RANDOM_STATE)
        model.fit(X_train_f1, y_train_f1)
        y_pred = model.predict(X_val_f1)
        mae = mean_absolute_error(y_val_f1, y_pred)
        gb_mae_list.append(mae)
        tuning_results.append({'model': 'GradientBoosting', 'param': 'n_estimators', 'value': n, 'MAE': mae})
        if mae < best_gb_mae:
            best_gb_mae = mae
            best_n_estimators = n

    # Plot GB validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, gb_mae_list, marker='o')
    plt.title('Gradient Boosting: Validation MAE vs n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Validation MAE')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'validation_curve_gb.png'))
    plt.close()

    # Quick tuning for others (Ridge, RF, MLP)
    # Ridge alpha
    best_ridge_mae = float('inf')
    best_alpha = 1.0
    for alpha in [0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_train_f1, y_train_f1)
        mae = mean_absolute_error(y_val_f1, model.predict(X_val_f1))
        tuning_results.append({'model': 'Ridge', 'param': 'alpha', 'value': alpha, 'MAE': mae})
        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_alpha = alpha

    # Random Forest depth
    best_rf_mae = float('inf')
    best_depth = 20
    for depth in [10, 20, 30]:
        model = RandomForestRegressor(max_depth=depth, n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
        model.fit(X_train_f1, y_train_f1)
        mae = mean_absolute_error(y_val_f1, model.predict(X_val_f1))
        tuning_results.append({'model': 'RandomForest', 'param': 'max_depth', 'value': depth, 'MAE': mae})
        if mae < best_rf_mae:
            best_rf_mae = mae
            best_depth = depth

    # MLP architecture
    best_mlp_mae = float('inf')
    best_layer = (100,)
    for layers in [(100,), (100, 50)]:
        model = MLPRegressor(hidden_layer_sizes=layers, max_iter=500, random_state=RANDOM_STATE)
        model.fit(X_train_f1, y_train_f1)
        mae = mean_absolute_error(y_val_f1, model.predict(X_val_f1))
        tuning_results.append({'model': 'MLP', 'param': 'hidden_layer_sizes', 'value': str(layers), 'MAE': mae})
        if mae < best_mlp_mae:
            best_mlp_mae = mae
            best_layer = layers

    pd.DataFrame(tuning_results).to_csv(os.path.join(metrics_dir, 'tuning_results.csv'), index=False)

    # 3. Select Best Model
    # Compare the best MAE from each tuned category
    best_candidates = [
        ('Ridge', best_ridge_mae, Ridge(alpha=best_alpha, random_state=RANDOM_STATE)),
        ('RandomForest', best_rf_mae, RandomForestRegressor(max_depth=best_depth, n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)),
        ('GradientBoosting', best_gb_mae, GradientBoostingRegressor(n_estimators=best_n_estimators, random_state=RANDOM_STATE)),
        ('MLP', best_mlp_mae, MLPRegressor(hidden_layer_sizes=best_layer, max_iter=1000, random_state=RANDOM_STATE))
    ]
    
    # Re-evaluating the absolute best with slightly more estimators for GB/RF if selected
    best_model_name, _, best_model_obj = min(best_candidates, key=lambda x: x[1])
    log(f"\nBest Model Selected: {best_model_name}")

    # Final fit of the best model on Train + Val if desired, but here just Train to keep it simple and strictly reproducible
    best_model_obj.fit(X_train_f1, y_train_f1)
    with open(os.path.join(models_dir, 'final_model.pkl'), 'wb') as f:
        pickle.dump(best_model_obj, f)

    # 4. Final Evaluation on TEST SET
    X_test_f1, y_test_f1 = test_df[F1], test_df[target]
    y_test_pred = best_model_obj.predict(X_test_f1)
    test_mae = mean_absolute_error(y_test_f1, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_f1, y_test_pred))

    final_results = [{
        'model': best_model_name,
        'feature_set': 'F1',
        'split': 'test',
        'MAE': test_mae,
        'RMSE': test_rmse
    }]
    pd.DataFrame(final_results).to_csv(os.path.join(metrics_dir, 'final_model_results.csv'), index=False)
    log(f"Test Set Evaluation: MAE={test_mae:.2f}, RMSE={test_rmse:.2f}")

    # 5. Diagnostic Plots on Test Set
    # Residual distribution
    residuals = y_test_f1 - y_test_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='red')
    plt.title(f'Residual Distribution (Best Model: {best_model_name})')
    plt.savefig(os.path.join(figures_dir, 'residual_distribution_final.png'))
    plt.close()

    # MAE by hour
    test_df['abs_error'] = np.abs(residuals)
    mae_by_hr = test_df.groupby('hr')['abs_error'].mean()
    plt.figure(figsize=(10, 6))
    mae_by_hr.plot(kind='bar', color='skyblue')
    plt.title('MAE by Hour of Day')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(os.path.join(figures_dir, 'mae_by_hour.png'))
    plt.close()

    # MAE by weekday
    mae_by_wd = test_df.groupby('weekday')['abs_error'].mean()
    plt.figure(figsize=(10, 6))
    mae_by_wd.plot(kind='bar', color='salmon')
    plt.title('MAE by Weekday')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(os.path.join(figures_dir, 'mae_by_weekday.png'))
    plt.close()

    # Residual vs temperature
    plt.figure(figsize=(10, 6))
    plt.scatter(test_df['temp'], residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Temperature')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Residual')
    plt.savefig(os.path.join(figures_dir, 'residual_vs_temperature.png'))
    plt.close()

    # Rolling MAE over time
    test_df_sorted = test_df.sort_values('instant')
    test_df_sorted['rolling_mae'] = test_df_sorted['abs_error'].rolling(window=100).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(test_df_sorted['instant'], test_df_sorted['rolling_mae'], color='green')
    plt.title('Rolling MAE over Time (Window=100)')
    plt.xlabel('Time (Instant)')
    plt.ylabel('MAE')
    plt.savefig(os.path.join(figures_dir, 'rolling_mae_over_time.png'))
    plt.close()

    log("--- MODEL IMPROVEMENT COMPLETED ---")

if __name__ == '__main__':
    run_model_improvement()
