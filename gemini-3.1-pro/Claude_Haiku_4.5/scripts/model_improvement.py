"""
Model improvement and comparison for bike-sharing dataset.
Trains and tunes: Ridge, Random Forest, Gradient Boosting, MLP.
Uses validation set for tuning, final test set for evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import time
import pickle
import warnings

warnings.filterwarnings('ignore')

# Set random seed and configuration
np.random.seed(42)
RANDOM_STATE = 42

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
TRAIN_PATH = OUTPUTS_PATH / "train.csv"
VAL_PATH = OUTPUTS_PATH / "val.csv"
TEST_PATH = OUTPUTS_PATH / "test.csv"
MODELS_PATH = OUTPUTS_PATH / "models"
METRICS_PATH = OUTPUTS_PATH / "metrics"
FIGURES_PATH = OUTPUTS_PATH / "figures"
BENCHMARK_PATH = OUTPUTS_PATH / "benchmark"
LOG_PATH = BENCHMARK_PATH / "experiment_log.txt"

# Create directories
for path in [MODELS_PATH, METRICS_PATH, FIGURES_PATH, BENCHMARK_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Close all figures
plt.close('all')


def log_message(msg):
    """Log message to both console and log file."""
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def load_splits():
    """Load train/val/test splits."""
    log_message(f"\n{'='*70}")
    log_message("MODEL IMPROVEMENT & COMPARISON")
    log_message(f"{'='*70}\n")

    log_message("1. Loading data splits...")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    log_message(f"   Train: {len(train_df)} rows")
    log_message(f"   Val:   {len(val_df)} rows")
    log_message(f"   Test:  {len(test_df)} rows")

    return train_df, val_df, test_df


def define_feature_sets():
    """Define F0 and F1 feature sets."""
    log_message(f"\n2. Defining feature sets...")

    F0 = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr',
          'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    F1 = F0 + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']

    log_message(f"   F0: {len(F0)} features")
    log_message(f"   F1: {len(F1)} features (F0 + cyclical)")

    return F0, F1


def train_evaluate_model(name, model, X_train, y_train, X_val, y_val, feature_set_name):
    """Train and evaluate a model."""
    log_message(f"   {name} ({feature_set_name})...")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return model, mae, rmse, training_time


def train_baseline_models(train_df, val_df, F0, F1):
    """Train baseline models (Ridge, RF, GB, MLP) on both feature sets."""
    log_message(f"\n3. Training baseline models on validation set...")

    y_train = train_df['cnt'].values
    y_val = val_df['cnt'].values

    results = []

    # Ridge Regression
    log_message(f"\n   === Ridge Regression ===")
    for feature_set, feature_set_name in [(F0, 'F0'), (F1, 'F1')]:
        X_train = train_df[feature_set].values
        X_val = val_df[feature_set].values

        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model, mae, rmse, train_time = train_evaluate_model(
            "Ridge", model, X_train, y_train, X_val, y_val, feature_set_name
        )

        log_message(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results.append({
            'model': 'Ridge',
            'feature_set': feature_set_name,
            'split': 'validation',
            'MAE': mae,
            'RMSE': rmse,
            'training_time_seconds': train_time
        })

    # Random Forest Regressor
    log_message(f"\n   === Random Forest ===")
    for feature_set, feature_set_name in [(F0, 'F0'), (F1, 'F1')]:
        X_train = train_df[feature_set].values
        X_val = val_df[feature_set].values

        model = RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=1
        )
        model, mae, rmse, train_time = train_evaluate_model(
            "Random Forest", model, X_train, y_train, X_val, y_val, feature_set_name
        )

        log_message(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results.append({
            'model': 'Random Forest',
            'feature_set': feature_set_name,
            'split': 'validation',
            'MAE': mae,
            'RMSE': rmse,
            'training_time_seconds': train_time
        })

    # Gradient Boosting Regressor
    log_message(f"\n   === Gradient Boosting ===")
    for feature_set, feature_set_name in [(F0, 'F0'), (F1, 'F1')]:
        X_train = train_df[feature_set].values
        X_val = val_df[feature_set].values

        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE
        )
        model, mae, rmse, train_time = train_evaluate_model(
            "Gradient Boosting", model, X_train, y_train, X_val, y_val, feature_set_name
        )

        log_message(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results.append({
            'model': 'Gradient Boosting',
            'feature_set': feature_set_name,
            'split': 'validation',
            'MAE': mae,
            'RMSE': rmse,
            'training_time_seconds': train_time
        })

    # MLP Regressor
    log_message(f"\n   === MLP Regressor ===")
    for feature_set, feature_set_name in [(F0, 'F0'), (F1, 'F1')]:
        X_train = train_df[feature_set].values
        X_val = val_df[feature_set].values

        model = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, learning_rate_init=0.001,
            random_state=RANDOM_STATE, n_iter_no_change=20
        )
        model, mae, rmse, train_time = train_evaluate_model(
            "MLP", model, X_train, y_train, X_val, y_val, feature_set_name
        )

        log_message(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results.append({
            'model': 'MLP',
            'feature_set': feature_set_name,
            'split': 'validation',
            'MAE': mae,
            'RMSE': rmse,
            'training_time_seconds': train_time
        })

    return pd.DataFrame(results)


def tune_models(train_df, val_df, F0, F1):
    """Tune models on validation set."""
    log_message(f"\n4. Tuning models...")

    y_train = train_df['cnt'].values
    y_val = val_df['cnt'].values

    tuning_results = []

    # Ridge tuning: sweep alpha
    log_message(f"\n   === Ridge Alpha Tuning ===")
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for feature_set, feature_set_name in [(F0, 'F0'), (F1, 'F1')]:
        X_train = train_df[feature_set].values
        X_val = val_df[feature_set].values

        best_alpha = None
        best_mae = float('inf')

        for alpha in alphas:
            model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)

            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        log_message(f"   {feature_set_name}: best alpha={best_alpha}, MAE={best_mae:.4f}")

        tuning_results.append({
            'model': 'Ridge',
            'feature_set': feature_set_name,
            'best_param': f"alpha={best_alpha}",
            'best_mae': best_mae
        })

    # Gradient Boosting n_estimators tuning
    log_message(f"\n   === Gradient Boosting n_estimators Tuning ===")
    n_estimators_values = [50, 100, 150, 200, 250]
    gb_tuning_curves = {}

    for feature_set, feature_set_name in [(F0, 'F0'), (F1, 'F1')]:
        X_train = train_df[feature_set].values
        X_val = val_df[feature_set].values

        mae_scores = []
        best_n_est = None
        best_mae = float('inf')

        for n_est in n_estimators_values:
            model = GradientBoostingRegressor(
                n_estimators=n_est, learning_rate=0.1, max_depth=5,
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)

            if mae < best_mae:
                best_mae = mae
                best_n_est = n_est

        log_message(f"   {feature_set_name}: best n_estimators={best_n_est}, MAE={best_mae:.4f}")

        gb_tuning_curves[feature_set_name] = {
            'n_estimators': n_estimators_values,
            'mae_scores': mae_scores,
            'best_n_est': best_n_est,
            'best_mae': best_mae
        }

        tuning_results.append({
            'model': 'Gradient Boosting',
            'feature_set': feature_set_name,
            'best_param': f"n_estimators={best_n_est}",
            'best_mae': best_mae
        })

    return pd.DataFrame(tuning_results), gb_tuning_curves


def plot_gb_tuning_curve(gb_tuning_curves):
    """Plot Gradient Boosting validation MAE vs n_estimators."""
    log_message(f"\n5. Creating Gradient Boosting tuning curve plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for feature_set_name, data in gb_tuning_curves.items():
        ax.plot(data['n_estimators'], data['mae_scores'], marker='o', linewidth=2.5,
               label=f"{feature_set_name} (best={data['best_n_est']}, MAE={data['best_mae']:.2f})",
               markersize=8)

    ax.set_xlabel('Number of Estimators', fontsize=12)
    ax.set_ylabel('Validation MAE', fontsize=12)
    ax.set_title('Gradient Boosting: Validation MAE vs n_estimators', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "validation_curve_gb.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: validation_curve_gb.png")


def save_all_results(baseline_results, tuning_results):
    """Save all model results."""
    log_message(f"\n6. Saving all model results...")

    # Combine baseline results with Linear Regression baseline
    baseline_lr = pd.read_csv(METRICS_PATH / "baseline_model_results.csv")

    all_results = pd.concat([baseline_lr, baseline_results], ignore_index=True)
    all_results.to_csv(METRICS_PATH / "all_results.csv", index=False)

    log_message(f"   Saved: all_results.csv ({len(all_results)} results)")

    # Save tuning results
    tuning_results.to_csv(METRICS_PATH / "tuning_results.csv", index=False)
    log_message(f"   Saved: tuning_results.csv ({len(tuning_results)} tuning results)")


def select_best_model(all_results, train_df, val_df, F0, F1):
    """Select best model based on validation MAE."""
    log_message(f"\n7. Selecting best model based on validation MAE...")

    # Find best model/feature_set combination
    best_row = all_results.loc[all_results['MAE'].idxmin()]
    best_model_name = best_row['model']
    best_feature_set = best_row['feature_set']
    best_mae = best_row['MAE']

    log_message(f"   Best: {best_model_name} + {best_feature_set}")
    log_message(f"   Validation MAE: {best_mae:.4f}")

    # Train best model
    feature_list = F1 if best_feature_set == 'F1' else F0
    X_train = train_df[feature_list].values
    y_train = train_df['cnt'].values

    if best_model_name == 'Ridge':
        # Use best alpha from tuning
        tuning = pd.read_csv(METRICS_PATH / "tuning_results.csv")
        best_tuning = tuning[(tuning['model'] == 'Ridge') & (tuning['feature_set'] == best_feature_set)].iloc[0]
        alpha = float(best_tuning['best_param'].split('=')[1])
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

    elif best_model_name == 'Random Forest':
        model = RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=1
        )

    elif best_model_name == 'Gradient Boosting':
        # Use best n_estimators from tuning
        tuning = pd.read_csv(METRICS_PATH / "tuning_results.csv")
        best_tuning = tuning[(tuning['model'] == 'Gradient Boosting') & (tuning['feature_set'] == best_feature_set)].iloc[0]
        n_est = int(best_tuning['best_param'].split('=')[1])
        model = GradientBoostingRegressor(
            n_estimators=n_est, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE
        )

    elif best_model_name == 'MLP':
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, learning_rate_init=0.001,
            random_state=RANDOM_STATE, n_iter_no_change=20
        )

    elif best_model_name == 'Linear Regression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(n_jobs=1)

    model.fit(X_train, y_train)

    # Save model
    model_path = MODELS_PATH / "final_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'model_name': best_model_name,
            'feature_set': best_feature_set,
            'features': feature_list
        }, f)

    log_message(f"   Saved: final_model.pkl")

    return model, best_model_name, best_feature_set, feature_list


def evaluate_on_test_set(model, test_df, feature_list, model_name, feature_set):
    """Evaluate best model on test set."""
    log_message(f"\n8. Final evaluation on TEST set...")

    X_test = test_df[feature_list].values
    y_test = test_df['cnt'].values

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    log_message(f"   {model_name} ({feature_set})")
    log_message(f"   Test MAE:  {mae:.4f}")
    log_message(f"   Test RMSE: {rmse:.4f}")

    # Save final results
    final_results = pd.DataFrame({
        'model': [model_name],
        'feature_set': [feature_set],
        'split': ['test'],
        'MAE': [mae],
        'RMSE': [rmse]
    })

    final_results.to_csv(METRICS_PATH / "final_model_results.csv", index=False)
    log_message(f"   Saved: final_model_results.csv")

    return y_test, y_pred, mae, rmse


def plot_test_diagnostics(y_test, y_pred, test_df):
    """Create diagnostic plots for test set."""
    log_message(f"\n9. Creating diagnostic plots for test set...")

    residuals = y_test - y_pred

    # Residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
    axes[0].set_xlabel('Residual', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Test Set Residual Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Q-Q plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.sort(np.random.normal(0, residuals.std(), len(residuals)))
    axes[1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.5, s=20, color='steelblue')
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1].set_xlabel('Theoretical Quantiles', fontsize=11)
    axes[1].set_ylabel('Residual Quantiles', fontsize=11)
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "residual_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: residual_distribution.png")

    # MAE by hour
    test_df_temp = test_df.copy()
    test_df_temp['residual'] = residuals
    test_df_temp['abs_error'] = np.abs(residuals)

    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_mae = test_df_temp.groupby('hr')['abs_error'].mean()
    ax.bar(hourly_mae.index, hourly_mae.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Hour of Day (0-23)', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Test Set: Mean Absolute Error by Hour', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "mae_by_hour.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: mae_by_hour.png")

    # MAE by weekday
    fig, ax = plt.subplots(figsize=(12, 6))
    weekday_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                     4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    weekday_mae = test_df_temp.groupby('weekday')['abs_error'].mean()
    weekday_labels = [weekday_names[i] for i in range(7)]
    ax.bar(weekday_labels, [weekday_mae.get(i, 0) for i in range(7)],
           color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Test Set: Mean Absolute Error by Weekday', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "mae_by_weekday.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: mae_by_weekday.png")

    # Residuals vs temperature
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(test_df_temp['temp'], residuals, alpha=0.4, s=20, color='steelblue', edgecolor='none')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Temperature (normalized)', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Test Set: Residuals vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "residual_vs_temperature.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: residual_vs_temperature.png")

    # Rolling MAE over time
    fig, ax = plt.subplots(figsize=(14, 6))
    rolling_mae = pd.Series(np.abs(residuals)).rolling(window=24).mean()
    ax.plot(rolling_mae.values, linewidth=2, color='steelblue', alpha=0.8)
    ax.set_xlabel('Time Index (hours)', fontsize=11)
    ax.set_ylabel('Rolling MAE (24-hour window)', fontsize=11)
    ax.set_title('Test Set: Rolling Mean Absolute Error Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "rolling_mae_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: rolling_mae_over_time.png")


def main():
    """Run complete model improvement pipeline."""
    train_df, val_df, test_df = load_splits()
    F0, F1 = define_feature_sets()

    # Train baseline models
    baseline_results = train_baseline_models(train_df, val_df, F0, F1)

    # Tune models
    tuning_results, gb_tuning_curves = tune_models(train_df, val_df, F0, F1)

    # Plot GB tuning curve
    plot_gb_tuning_curve(gb_tuning_curves)

    # Load all results (including Linear Regression baseline)
    all_results_path = METRICS_PATH / "all_results.csv"
    if all_results_path.exists():
        all_results = pd.read_csv(all_results_path)
    else:
        baseline_lr = pd.read_csv(METRICS_PATH / "baseline_model_results.csv")
        all_results = pd.concat([baseline_lr, baseline_results], ignore_index=True)

    # Save results
    save_all_results(baseline_results, tuning_results)

    # Select and train best model
    model, model_name, feature_set, feature_list = select_best_model(all_results, train_df, val_df, F0, F1)

    # Evaluate on test set
    y_test, y_pred, mae, rmse = evaluate_on_test_set(model, test_df, feature_list, model_name, feature_set)

    # Create diagnostic plots
    plot_test_diagnostics(y_test, y_pred, test_df)

    # Summary
    log_message(f"\n{'='*70}")
    log_message("MODEL COMPARISON RESULTS")
    log_message(f"{'='*70}")
    log_message(f"Best Validation Model: {model_name} + {feature_set}")
    log_message(f"Test MAE: {mae:.4f}")
    log_message(f"Test RMSE: {rmse:.4f}")
    log_message(f"{'='*70}\n")


if __name__ == "__main__":
    main()
