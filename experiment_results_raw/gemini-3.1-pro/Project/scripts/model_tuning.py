import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging
import time
import pickle

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
    logger.info("Started model tuning pipeline.")
    
    # Load splits
    try:
        train_df = pd.read_csv("outputs/train.csv")
        val_df = pd.read_csv("outputs/val.csv")
        test_df = pd.read_csv("outputs/test.csv")
        logger.info("Loaded train, val, test splits.")
    except Exception as e:
        logger.error(f"Failed to load data splits: {e}")
        return

    # Define feature sets
    target = 'cnt'
    F0_base = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    F0 = [f for f in F0_base if f in train_df.columns]
    
    F1_base = F0 + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    F1 = [f for f in F1_base if f in train_df.columns]
    
    X_train_F0 = train_df[F0]
    y_train = train_df[target]
    X_val_F0 = val_df[F0]
    y_val = val_df[target]
    
    X_train_F1 = train_df[F1]
    X_val_F1 = val_df[F1]
    
    X_test_F1 = test_df[F1]
    y_test = test_df[target]

    # Directories
    metrics_dir = "outputs/metrics"
    models_dir = "outputs/models"
    fig_dir = "outputs/figures"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Model Comparison
    models_to_compare = {
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'MLP': MLPRegressor(random_state=42, max_iter=500)
    }
    
    feature_sets = {'F0': (X_train_F0, X_val_F0), 'F1': (X_train_F1, X_val_F1)}
    
    all_results = []
    
    for model_name, model in models_to_compare.items():
        for fs_name, (X_tr, X_v) in feature_sets.items():
            logger.info(f"Training {model_name} on {fs_name} for comparison...")
            start_time = time.time()
            model.fit(X_tr, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_v)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            all_results.append({
                'model': model_name,
                'feature_set': fs_name,
                'split': 'validation',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': train_time
            })
            
    all_results_df = pd.DataFrame(all_results)
    all_results_path = os.path.join(metrics_dir, 'all_results.csv')
    all_results_df.to_csv(all_results_path, index=False)
    logger.info(f"Saved all_results.csv to {all_results_path}")

    # 2. Tuning
    # To keep it standard, we will tune on F1 feature set as it typically contains richer representation
    tuning_results = []
    best_mae = float('inf')
    best_model = None
    best_model_name = ""
    
    # Tune Ridge
    for alpha in [0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha, random_state=42)
        start_time = time.time()
        model.fit(X_train_F1, y_train)
        tt = time.time() - start_time
        y_pred = model.predict(X_val_F1)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        tuning_results.append({'model': 'Ridge', 'params': f'alpha={alpha}', 'MAE': mae, 'RMSE': rmse, 'time': tt})
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = "Ridge"

    # Tune RandomForest
    for max_depth in [10, 20, None]:
        model = RandomForestRegressor(max_depth=max_depth, random_state=42, n_jobs=1)
        start_time = time.time()
        model.fit(X_train_F1, y_train)
        tt = time.time() - start_time
        y_pred = model.predict(X_val_F1)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        tuning_results.append({'model': 'RandomForest', 'params': f'max_depth={max_depth}', 'MAE': mae, 'RMSE': rmse, 'time': tt})
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = "RandomForest"

    # Tune GradientBoosting
    gb_estimators = [50, 100, 200, 300]
    gb_maes = []
    for n_est in gb_estimators:
        model = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
        start_time = time.time()
        model.fit(X_train_F1, y_train)
        tt = time.time() - start_time
        y_pred = model.predict(X_val_F1)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        gb_maes.append(mae)
        tuning_results.append({'model': 'GradientBoosting', 'params': f'n_estimators={n_est}', 'MAE': mae, 'RMSE': rmse, 'time': tt})
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = "GradientBoosting"

    # Plot GB validation curve
    plt.figure(figsize=(8, 6))
    plt.plot(gb_estimators, gb_maes, marker='o', linestyle='-')
    plt.title('Gradient Boosting: Validation MAE vs n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Validation MAE')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'validation_curve_gb.png'), bbox_inches='tight')
    plt.close()
    logger.info("Saved validation_curve_gb.png")

    # Tune MLP
    for hidden_layer_sizes in [(50,), (100,), (50, 50)]:
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=42)
        start_time = time.time()
        model.fit(X_train_F1, y_train)
        tt = time.time() - start_time
        y_pred = model.predict(X_val_F1)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        tuning_results.append({'model': 'MLP', 'params': f'hidden_layer_sizes={hidden_layer_sizes}', 'MAE': mae, 'RMSE': rmse, 'time': tt})
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = "MLP"
            
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(os.path.join(metrics_dir, 'tuning_results.csv'), index=False)
    logger.info("Saved tuning_results.csv")
    
    logger.info(f"Best model selected: {best_model_name} with Validation MAE: {best_mae:.4f}")

    # Save best model
    final_model_path = os.path.join(models_dir, 'final_model.pkl')
    with open(final_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Saved best model to {final_model_path}")

    # 3. Final evaluation on TEST set
    y_test_pred = best_model.predict(X_test_F1)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    final_results = [{
        'model': best_model_name,
        'feature_set': 'F1',
        'split': 'test',
        'MAE': test_mae,
        'RMSE': test_rmse
    }]
    pd.DataFrame(final_results).to_csv(os.path.join(metrics_dir, 'final_model_results.csv'), index=False)
    logger.info("Saved final_model_results.csv")
    
    # Generate diagnostic plots for TEST set
    test_residuals = y_test - y_test_pred
    test_df['predicted_cnt'] = y_test_pred
    test_df['abs_error'] = np.abs(test_residuals)
    
    # residual_distribution.png
    plt.figure(figsize=(10, 6))
    sns.histplot(test_residuals, kde=True, bins=50)
    plt.title('Residual Distribution (Test Set)')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(fig_dir, 'residual_distribution.png'), bbox_inches='tight')
    plt.close()
    
    # mae_by_hour.png
    if 'hr' in test_df.columns:
        plt.figure(figsize=(10, 6))
        mae_hr = test_df.groupby('hr')['abs_error'].mean()
        mae_hr.plot(kind='bar', color='skyblue')
        plt.title('Test MAE by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(os.path.join(fig_dir, 'mae_by_hour.png'), bbox_inches='tight')
        plt.close()
        
    # mae_by_weekday.png
    if 'weekday' in test_df.columns:
        plt.figure(figsize=(10, 6))
        mae_wd = test_df.groupby('weekday')['abs_error'].mean()
        mae_wd.plot(kind='bar', color='lightgreen')
        plt.title('Test MAE by Weekday')
        plt.xlabel('Weekday')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(os.path.join(fig_dir, 'mae_by_weekday.png'), bbox_inches='tight')
        plt.close()
        
    # residual_vs_temperature.png
    if 'temp' in test_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=test_df['temp'], y=test_residuals, alpha=0.4)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Residuals vs Temperature (Test Set)')
        plt.xlabel('Temperature (normalized)')
        plt.ylabel('Residuals')
        plt.savefig(os.path.join(fig_dir, 'residual_vs_temperature.png'), bbox_inches='tight')
        plt.close()
        
    # rolling_mae_over_time.png
    plt.figure(figsize=(12, 6))
    rolling_mae = test_df['abs_error'].rolling(window=24, min_periods=1).mean()
    plt.plot(range(len(test_df)), rolling_mae, color='purple')
    plt.title('Rolling MAE Over Time (Test Set, 24-hour window)')
    plt.xlabel('Test Set Timeline (chronological steps)')
    plt.ylabel('Rolling MAE')
    plt.savefig(os.path.join(fig_dir, 'rolling_mae_over_time.png'), bbox_inches='tight')
    plt.close()
    
    logger.info("Saved all test diagnostic plots.")
    logger.info("Model tuning pipeline completed successfully.")

if __name__ == "__main__":
    main()