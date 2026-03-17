"""
Improved Model Building and Comparison
Tests Ridge, Random Forest, Gradient Boosting, and MLP models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seeds
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_splits():
    """Load train/val/test splits"""
    print(f"\n--- LOADING DATA SPLITS ---")

    train_df = pd.read_csv(Path('outputs/train.csv'))
    val_df = pd.read_csv(Path('outputs/val.csv'))
    test_df = pd.read_csv(Path('outputs/test.csv'))

    # Convert dteday to datetime
    train_df['dteday'] = pd.to_datetime(train_df['dteday'])
    val_df['dteday'] = pd.to_datetime(val_df['dteday'])
    test_df['dteday'] = pd.to_datetime(test_df['dteday'])

    print(f"[OK] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

def get_feature_sets():
    """Define F0 and F1 feature sets"""
    F0_features = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr',
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

    F1_features = F0_features + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']

    return F0_features, F1_features

def prepare_data(train_df, val_df, test_df, features):
    """Prepare and scale data"""
    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    X_test = test_df[features].copy()

    y_train = train_df['cnt'].copy()
    y_val = val_df['cnt'].copy()
    y_test = test_df['cnt'].copy()

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_model(model, X_train, y_train):
    """Train a model and return training time"""
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def evaluate_model(model, X_val, y_val):
    """Evaluate model and return predictions and metrics"""
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return y_pred, mae, rmse

def train_and_evaluate_models(train_df, val_df, test_df):
    """Train all baseline models on both feature sets"""
    print(f"\n{'='*70}")
    print("TRAINING BASELINE MODELS")
    print(f"{'='*70}\n")

    F0_features, F1_features = get_feature_sets()

    results = []

    # Ridge Regression
    print(f"--- RIDGE REGRESSION ---")
    for features, feature_name in [(F0_features, 'F0'), (F1_features, 'F1')]:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
            train_df, val_df, test_df, features)

        model = Ridge(alpha=1.0, random_state=42)
        model, train_time = train_model(model, X_train, y_train)
        y_pred, mae, rmse = evaluate_model(model, X_val, y_val)

        print(f"  {feature_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, Time={train_time:.4f}s")

        results.append({
            'model': 'Ridge',
            'feature_set': feature_name,
            'split': 'validation',
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'training_time_seconds': round(train_time, 4)
        })

    # Random Forest
    print(f"\n--- RANDOM FOREST ---")
    for features, feature_name in [(F0_features, 'F0'), (F1_features, 'F1')]:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
            train_df, val_df, test_df, features)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1, max_depth=20)
        model, train_time = train_model(model, X_train, y_train)
        y_pred, mae, rmse = evaluate_model(model, X_val, y_val)

        print(f"  {feature_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, Time={train_time:.4f}s")

        results.append({
            'model': 'Random Forest',
            'feature_set': feature_name,
            'split': 'validation',
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'training_time_seconds': round(train_time, 4)
        })

    # Gradient Boosting
    print(f"\n--- GRADIENT BOOSTING ---")
    for features, feature_name in [(F0_features, 'F0'), (F1_features, 'F1')]:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
            train_df, val_df, test_df, features)

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                         random_state=42)
        model, train_time = train_model(model, X_train, y_train)
        y_pred, mae, rmse = evaluate_model(model, X_val, y_val)

        print(f"  {feature_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, Time={train_time:.4f}s")

        results.append({
            'model': 'Gradient Boosting',
            'feature_set': feature_name,
            'split': 'validation',
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'training_time_seconds': round(train_time, 4)
        })

    # MLP
    print(f"\n--- MLP REGRESSOR ---")
    for features, feature_name in [(F0_features, 'F0'), (F1_features, 'F1')]:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
            train_df, val_df, test_df, features)

        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        model, train_time = train_model(model, X_train, y_train)
        y_pred, mae, rmse = evaluate_model(model, X_val, y_val)

        print(f"  {feature_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, Time={train_time:.4f}s")

        results.append({
            'model': 'MLP',
            'feature_set': feature_name,
            'split': 'validation',
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'training_time_seconds': round(train_time, 4)
        })

    return pd.DataFrame(results)

def tune_gradient_boosting(train_df, val_df):
    """Tune Gradient Boosting and create validation curve"""
    print(f"\n{'='*70}")
    print("TUNING GRADIENT BOOSTING - VALIDATION CURVE")
    print(f"{'='*70}\n")

    F0_features, F1_features = get_feature_sets()

    # Use F1 features (better performance)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        train_df, val_df, val_df, F1_features)

    n_estimators_range = [10, 25, 50, 75, 100, 150, 200]
    tuning_results = []

    for n_est in n_estimators_range:
        model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1,
                                         max_depth=5, random_state=42)
        model, _ = train_model(model, X_train, y_train)
        y_pred, mae, rmse = evaluate_model(model, X_val, y_val)

        tuning_results.append({
            'n_estimators': n_est,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        })

        print(f"  n_estimators={n_est}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    tuning_df = pd.DataFrame(tuning_results)

    # Plot validation curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tuning_df['n_estimators'], tuning_df['MAE'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Estimators', fontsize=11)
    ax.set_ylabel('Validation MAE', fontsize=11)
    ax.set_title('Gradient Boosting Tuning - Validation MAE vs n_estimators', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path('outputs/figures/validation_curve_gb.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: outputs/figures/validation_curve_gb.png")

    return tuning_df

def select_best_model(results_df, train_df, val_df):
    """Select and train the best model based on validation MAE"""
    print(f"\n{'='*70}")
    print("SELECTING BEST MODEL")
    print(f"{'='*70}\n")

    # Find best result by lowest MAE
    best_result = results_df.loc[results_df['MAE'].idxmin()]

    best_model_name = best_result['model']
    best_feature_set = best_result['feature_set']
    best_mae = best_result['MAE']

    print(f"Best model: {best_model_name} with {best_feature_set}")
    print(f"Validation MAE: {best_mae:.2f}")

    # Get features
    F0_features, F1_features = get_feature_sets()
    features = F1_features if best_feature_set == 'F1' else F0_features

    # Prepare data
    X_train, X_val, _, y_train, y_val, _, scaler = prepare_data(
        train_df, val_df, val_df, features)

    # Train best model
    if best_model_name == 'Ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif best_model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1, max_depth=20)
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                         random_state=42)
    elif best_model_name == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)

    model, _ = train_model(model, X_train, y_train)

    # Save best model
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'feature_set': best_feature_set,
        'model_name': best_model_name
    }

    with open(Path('outputs/models/final_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    print(f"[OK] Saved best model to: outputs/models/final_model.pkl")

    return model_data

def test_evaluation(model_data, test_df):
    """Evaluate best model on test set and generate diagnostic plots"""
    print(f"\n{'='*70}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*70}\n")

    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    model_name = model_data['model_name']
    feature_set = model_data['feature_set']

    # Prepare test data
    X_test = test_df[features].copy()
    y_test = test_df['cnt'].copy()
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    # Save results
    test_results = pd.DataFrame([{
        'model': model_name,
        'feature_set': feature_set,
        'split': 'test',
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }])

    test_results.to_csv(Path('outputs/metrics/final_model_results.csv'), index=False)
    print(f"[OK] Saved: outputs/metrics/final_model_results.csv")

    # Generate diagnostic plots
    print(f"\n--- GENERATING DIAGNOSTIC PLOTS ---")

    residuals = y_test - y_pred

    # 1. Residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Test Set Residual Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.4, s=20, color='steelblue')
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Demand', fontsize=11)
    axes[1].set_ylabel('Residual', fontsize=11)
    axes[1].set_title('Test Set - Residuals vs Predictions', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path('outputs/figures/residual_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: outputs/figures/residual_distribution.png")

    # 2. MAE by hour
    test_df_results = test_df.copy()
    test_df_results['prediction'] = y_pred
    test_df_results['residual'] = residuals
    test_df_results['abs_error'] = np.abs(residuals)

    hourly_mae = test_df_results.groupby('hr')['abs_error'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly_mae.index, hourly_mae.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Test Set - MAE by Hour of Day', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(Path('outputs/figures/mae_by_hour.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: outputs/figures/mae_by_hour.png")

    # 3. MAE by weekday
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_mae = test_df_results.groupby('weekday')['abs_error'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([day_names[i] for i in weekday_mae.index], weekday_mae.values, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Test Set - MAE by Day of Week', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path('outputs/figures/mae_by_weekday.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: outputs/figures/mae_by_weekday.png")

    # 4. Residuals vs temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(test_df_results['temp'], residuals, alpha=0.4, s=20, c=residuals,
                        cmap='coolwarm', vmin=-residuals.abs().max(), vmax=residuals.abs().max())
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Temperature', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Test Set - Residuals vs Temperature', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Residual')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path('outputs/figures/residual_vs_temperature.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: outputs/figures/residual_vs_temperature.png")

    # 5. Rolling MAE over time
    test_df_results = test_df_results.sort_values('dteday').reset_index(drop=True)
    test_df_results['rolling_mae'] = test_df_results['abs_error'].rolling(window=24, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_df_results['dteday'], test_df_results['rolling_mae'], linewidth=2, color='steelblue')
    ax.fill_between(test_df_results['dteday'], test_df_results['rolling_mae'], alpha=0.3, color='steelblue')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Rolling MAE (24-hour window)', fontsize=11)
    ax.set_title('Test Set - Rolling MAE Over Time', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path('outputs/figures/rolling_mae_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: outputs/figures/rolling_mae_over_time.png")

    return test_results

def main():
    """Main execution"""
    print(f"\n{'='*70}")
    print("IMPROVED MODEL BUILDING AND COMPARISON")
    print(f"{'='*70}\n")

    # Create output directories
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    Path('outputs/models').mkdir(parents=True, exist_ok=True)
    Path('outputs/figures').mkdir(parents=True, exist_ok=True)

    # Load data splits
    train_df, val_df, test_df = load_splits()

    # Train baseline models
    results_df = train_and_evaluate_models(train_df, val_df, test_df)

    # Save all results
    results_df.to_csv(Path('outputs/metrics/all_results.csv'), index=False)
    print(f"\n[OK] Saved all model results to: outputs/metrics/all_results.csv")
    print(f"\n{results_df.to_string(index=False)}")

    # Tune Gradient Boosting
    tuning_df = tune_gradient_boosting(train_df, val_df)
    tuning_df.to_csv(Path('outputs/metrics/tuning_results.csv'), index=False)
    print(f"\n[OK] Saved tuning results to: outputs/metrics/tuning_results.csv")

    # Select best model
    best_model_data = select_best_model(results_df, train_df, val_df)

    # Final test evaluation
    test_results = test_evaluation(best_model_data, test_df)

    # Summary
    print(f"\n{'='*70}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*70}")

    # Load baseline for comparison
    baseline_df = pd.read_csv(Path('outputs/metrics/baseline_model_results.csv'))
    baseline_f1 = baseline_df[baseline_df['feature_set'] == 'F1']['MAE'].values[0]

    best_val_mae = results_df['MAE'].min()
    final_test_mae = test_results['MAE'].values[0]

    print(f"\nBaseline (F1 on validation): MAE={baseline_f1:.2f}")
    print(f"Best model (validation): MAE={best_val_mae:.2f}")
    print(f"Improvement on validation: {((baseline_f1 - best_val_mae) / baseline_f1 * 100):+.2f}%")
    print(f"\nFinal test performance: MAE={final_test_mae:.2f}")

    print(f"\nAll outputs saved to outputs/")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
