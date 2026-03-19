import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
import time
import joblib
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
with open(log_file, 'a') as f:
    f.write(f"\n{datetime.now()} - Advanced Modeling started\n")

def log_step(message):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")
    print(message)

# Load data splits
log_step("Loading data splits...")
train_df = pd.read_csv('outputs/train.csv')
val_df = pd.read_csv('outputs/val.csv')
test_df = pd.read_csv('outputs/test.csv')
log_step(f"Loaded train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} rows")

# Recreate cyclical features (they may not be saved in the CSV files)
log_step("Recreating cyclical features...")
if 'hr' in train_df.columns:
    train_df['sin_hour'] = np.sin(2 * np.pi * train_df['hr'] / 24)
    train_df['cos_hour'] = np.cos(2 * np.pi * train_df['hr'] / 24)
    val_df['sin_hour'] = np.sin(2 * np.pi * val_df['hr'] / 24)
    val_df['cos_hour'] = np.cos(2 * np.pi * val_df['hr'] / 24)
    test_df['sin_hour'] = np.sin(2 * np.pi * test_df['hr'] / 24)
    test_df['cos_hour'] = np.cos(2 * np.pi * test_df['hr'] / 24)

if 'mnth' in train_df.columns:
    train_df['sin_month'] = np.sin(2 * np.pi * train_df['mnth'] / 12)
    train_df['cos_month'] = np.cos(2 * np.pi * train_df['mnth'] / 12)
    val_df['sin_month'] = np.sin(2 * np.pi * val_df['mnth'] / 12)
    val_df['cos_month'] = np.cos(2 * np.pi * val_df['mnth'] / 12)
    test_df['sin_month'] = np.sin(2 * np.pi * test_df['mnth'] / 12)
    test_df['cos_month'] = np.cos(2 * np.pi * test_df['mnth'] / 12)

# Define feature sets (same as Task 3)
F0_features = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
F0_features = [f for f in F0_features if f in train_df.columns]

F1_features = F0_features.copy()
if 'hr' in train_df.columns:
    F1_features.extend(['sin_hour', 'cos_hour'])
if 'mnth' in train_df.columns:
    F1_features.extend(['sin_month', 'cos_month'])
F1_features = list(set(F1_features))

log_step(f"F0 features: {F0_features}")
log_step(f"F1 features: {F1_features}")

target_var = 'cnt'

# 1. Model comparison
log_step("Starting model comparison...")

all_results = []

# Define models and their parameter grids for tuning
models_to_tune = {
    'Ridge': {
        'model': Ridge(random_state=42),
        'param_grid': {'alpha': [0.1, 1.0, 10.0, 100.0]}
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=1),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'MLP': {
        'model': MLPRegressor(random_state=42, early_stopping=True),
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [200, 500]
        }
    }
}

# Train and evaluate all models
for model_name, model_info in models_to_tune.items():
    for feature_set_name, features in [('F0', F0_features), ('F1', F1_features)]:
        if not features:
            continue
            
        log_step(f"Training {model_name} with feature set {feature_set_name}")
        
        # Prepare data
        X_train = train_df[features]
        y_train = train_df[target_var]
        X_val = val_df[features]
        y_val = val_df[target_var]
        
        # Find best parameters through grid search
        best_score = float('inf')
        best_params = None
        best_model = None
        tuning_results = []
        
        param_grid = model_info['param_grid']
        
        for params in ParameterGrid(param_grid):
            try:
                # Create and train model
                model = model_info['model'].set_params(**params)
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                tuning_results.append({
                    'model': model_name,
                    'feature_set': feature_set_name,
                    'params': str(params),
                    'MAE': mae,
                    'RMSE': rmse,
                    'training_time_seconds': training_time
                })
                
                if mae < best_score:
                    best_score = mae
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                log_step(f"Error with {model_name} params {params}: {e}")
                continue
        
        if best_model is not None:
            # Evaluate best model
            y_pred = best_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            # Save model
            model_path = f'outputs/models/{model_name}_{feature_set_name}.pkl'
            joblib.dump(best_model, model_path)
            
            all_results.append({
                'model': model_name,
                'feature_set': feature_set_name,
                'split': 'validation',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': best_model.training_time if hasattr(best_model, 'training_time') else training_time
            })
            
            log_step(f"{model_name} {feature_set_name} - Best MAE: {mae:.4f}, RMSE: {rmse:.4f}, Params: {best_params}")
        else:
            log_step(f"Warning: No valid model found for {model_name} {feature_set_name}")

# Save all results
if all_results:
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv('outputs/metrics/all_results.csv', index=False)
    log_step("All model results saved to outputs/metrics/all_results.csv")

# Save tuning results
tuning_results_df = pd.DataFrame(tuning_results)
if not tuning_results_df.empty:
    tuning_results_df.to_csv('outputs/metrics/tuning_results.csv', index=False)
    log_step("Tuning results saved to outputs/metrics/tuning_results.csv")

# 4. Gradient Boosting validation curve
log_step("Generating Gradient Boosting validation curve...")

# Extract GB tuning results
gb_tuning = tuning_results_df[(tuning_results_df['model'] == 'GradientBoosting') & (tuning_results_df['feature_set'] == 'F1')]

if not gb_tuning.empty:
    plt.figure(figsize=(12, 6))
    
    # Group by n_estimators and take mean MAE
    gb_grouped = gb_tuning.copy()
    gb_grouped['n_estimators'] = gb_grouped['params'].apply(lambda x: eval(x)['n_estimators'] if 'n_estimators' in x else None)
    gb_grouped = gb_grouped.dropna(subset=['n_estimators'])
    gb_mean = gb_grouped.groupby('n_estimators')['MAE'].mean().reset_index()
    
    sns.lineplot(data=gb_mean, x='n_estimators', y='MAE', marker='o')
    plt.title('Gradient Boosting Validation MAE vs n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Validation MAE')
    plt.grid(True)
    plt.savefig('outputs/figures/validation_curve_gb.png')
    plt.close()
    log_step("Saved validation_curve_gb.png")

# 5. Select best model based on lowest validation MAE
if all_results:
    best_model_row = min(all_results, key=lambda x: x['MAE'])
    log_step(f"Best model: {best_model_row['model']} with {best_model_row['feature_set']}")
    log_step(f"Best validation MAE: {best_model_row['MAE']:.4f}, RMSE: {best_model_row['RMSE']:.4f}")
    
    # Load best model
    best_model_path = f"outputs/models/{best_model_row['model']}_{best_model_row['feature_set']}.pkl"
    best_model = joblib.load(best_model_path)
    best_features = F1_features if best_model_row['feature_set'] == 'F1' else F0_features
    
    # Save best model
    joblib.dump(best_model, 'outputs/models/final_model.pkl')
    log_step("Best model saved to outputs/models/final_model.pkl")
    
    # 6. Final evaluation on TEST set
    log_step("Evaluating best model on test set...")
    
    X_test = test_df[best_features]
    y_test = test_df[target_var]
    
    y_pred_test = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    final_results = pd.DataFrame([{
        'model': best_model_row['model'],
        'feature_set': best_model_row['feature_set'],
        'split': 'test',
        'MAE': test_mae,
        'RMSE': test_rmse
    }])
    final_results.to_csv('outputs/metrics/final_model_results.csv', index=False)
    log_step(f"Test set results - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # Generate diagnostic plots
    log_step("Generating diagnostic plots...")
    
    # Residual distribution
    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('Residual Distribution (Test Set)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('outputs/figures/residual_distribution.png')
    plt.close()
    
    # MAE by hour
    if 'hr' in test_df.columns:
        test_df['residual'] = residuals
        mae_by_hour = test_df.groupby('hr')['residual'].apply(lambda x: mean_absolute_error(y_test.iloc[x.index], y_pred_test[x.index])).reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=mae_by_hour, x='hr', y='residual')
        plt.title('MAE by Hour of Day (Test Set)')
        plt.xlabel('Hour of Day')
        plt.ylabel('MAE')
        plt.savefig('outputs/figures/mae_by_hour.png')
        plt.close()
    
    # MAE by weekday
    if 'weekday' in test_df.columns:
        mae_by_weekday = test_df.groupby('weekday')['residual'].apply(lambda x: mean_absolute_error(y_test.iloc[x.index], y_pred_test[x.index])).reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=mae_by_weekday, x='weekday', y='residual')
        plt.title('MAE by Weekday (Test Set)')
        plt.xlabel('Weekday')
        plt.ylabel('MAE')
        plt.savefig('outputs/figures/mae_by_weekday.png')
        plt.close()
    
    # Residual vs temperature
    if 'temp' in test_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=test_df['temp'], y=residuals, alpha=0.5)
        plt.title('Residuals vs Temperature (Test Set)')
        plt.xlabel('Normalized Temperature')
        plt.ylabel('Residuals')
        plt.savefig('outputs/figures/residual_vs_temperature.png')
        plt.close()
    
    # Rolling MAE over time
    if 'dteday' in test_df.columns:
        test_df['dteday'] = pd.to_datetime(test_df['dteday'])
        test_df = test_df.sort_values('dteday')
        test_df['absolute_error'] = np.abs(residuals)
        
        # Calculate rolling MAE with window of 24 hours
        rolling_mae = test_df.set_index('dteday')['absolute_error'].rolling(window=24, min_periods=1).mean()
        
        plt.figure(figsize=(14, 6))
        rolling_mae.plot()
        plt.title('Rolling MAE over Time (24-hour window)')
        plt.xlabel('Date')
        plt.ylabel('Rolling MAE')
        plt.grid(True)
        plt.savefig('outputs/figures/rolling_mae_over_time.png')
        plt.close()

log_step("Advanced modeling completed successfully.")