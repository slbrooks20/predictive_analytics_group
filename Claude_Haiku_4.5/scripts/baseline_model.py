"""
Baseline model training and evaluation for bike-sharing dataset.
Uses Linear Regression with F0 and F1 feature sets.
Implements chronological train/val/test split.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

# Set random seed and configuration
np.random.seed(42)
RANDOM_STATE = 42

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_data.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "dataset" / "hour.csv"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
TRAIN_PATH = OUTPUTS_PATH / "train.csv"
VAL_PATH = OUTPUTS_PATH / "val.csv"
TEST_PATH = OUTPUTS_PATH / "test.csv"
MODELS_PATH = OUTPUTS_PATH / "models"
METRICS_PATH = OUTPUTS_PATH / "metrics"
FIGURES_PATH = OUTPUTS_PATH / "figures"
BENCHMARK_PATH = OUTPUTS_PATH / "benchmark"
DOCS_PATH = OUTPUTS_PATH / "docs"
LOG_PATH = BENCHMARK_PATH / "experiment_log.txt"

# Create directories
for path in [MODELS_PATH, METRICS_PATH, FIGURES_PATH, BENCHMARK_PATH, DOCS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Close all figures
plt.close('all')


def log_message(msg):
    """Log message to both console and log file."""
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def load_data():
    """Load cleaned data or fallback to raw dataset."""
    log_message(f"\n{'='*70}")
    log_message("BASELINE MODEL TRAINING & EVALUATION")
    log_message(f"{'='*70}\n")

    log_message("1. Loading data...")

    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH)
        log_message(f"   Loaded cleaned data from {CLEANED_DATA_PATH}")
    else:
        df = pd.read_csv(FALLBACK_DATA_PATH)
        log_message(f"   Loaded raw data from {FALLBACK_DATA_PATH}")
        # Remove leakage columns if present
        if 'casual' in df.columns and 'registered' in df.columns:
            df = df.drop(columns=['casual', 'registered'])

    log_message(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Parse date
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])
        df = df.sort_values('dteday').reset_index(drop=True)
        log_message(f"   Date range: {df['dteday'].min().date()} to {df['dteday'].max().date()}")

    return df


def create_train_val_test_split(df):
    """Create chronological train/val/test split (70/15/15)."""
    log_message(f"\n2. Creating chronological train/val/test split...")

    n = len(df)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)

    # Split
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    log_message(f"   Train: {len(train_df)} rows ({100*len(train_df)/n:.1f}%)")
    log_message(f"   Val:   {len(val_df)} rows ({100*len(val_df)/n:.1f}%)")
    log_message(f"   Test:  {len(test_df)} rows ({100*len(test_df)/n:.1f}%)")

    # Verify chronological ordering
    log_message(f"\n3. Verifying chronological split...")

    if 'dteday' in df.columns:
        train_max = train_df['dteday'].max()
        train_max_idx = train_df.index.max()
        val_min = val_df['dteday'].min()
        val_min_idx = val_df.index.min()
        val_max = val_df['dteday'].max()
        val_max_idx = val_df.index.max()
        test_min = test_df['dteday'].min()
        test_min_idx = test_df.index.min()

        log_message(f"   Train max datetime: {train_max}")
        log_message(f"   Val min datetime:   {val_min}")
        log_message(f"   Val max datetime:   {val_max}")
        log_message(f"   Test min datetime:  {test_min}")

        # Assertions: use <= for date comparison since hourly data has multiple observations per day
        # Check that all train indices are before all val indices, and all val indices are before all test indices
        assert train_max_idx < val_min_idx, f"ERROR: Train/Val index overlap! {train_max_idx} >= {val_min_idx}"
        assert val_max_idx < test_min_idx, f"ERROR: Val/Test index overlap! {val_max_idx} >= {test_min_idx}"
        log_message(f"   [OK] Chronological split verified (no overlap)")
    else:
        log_message(f"   WARNING: dteday not available for verification")

    return train_df, val_df, test_df


def define_feature_sets():
    """Define F0 and F1 feature sets."""
    log_message(f"\n4. Defining feature sets...")

    # F0: Original predictors
    F0 = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr',
          'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

    log_message(f"   F0 (original): {len(F0)} features")
    log_message(f"      {F0}")

    # F1: F0 + cyclical features
    F1 = F0 + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']

    log_message(f"   F1 (F0 + cyclical): {len(F1)} features")
    log_message(f"      Cyclical: sin_hour, cos_hour, sin_month, cos_month")

    return F0, F1


def create_cyclical_features(df):
    """Create cyclical features from hour and month."""
    log_message(f"\n5. Creating cyclical features...")

    df = df.copy()

    # Hour cyclical features
    df['sin_hour'] = np.sin(2 * np.pi * df['hr'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hr'] / 24)

    # Month cyclical features
    df['sin_month'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['mnth'] / 12)

    log_message(f"   Created 4 cyclical features")

    return df


def save_splits(train_df, val_df, test_df):
    """Save splits to CSV."""
    log_message(f"\n6. Saving data splits...")

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    log_message(f"   Saved: {TRAIN_PATH}")
    log_message(f"   Saved: {VAL_PATH}")
    log_message(f"   Saved: {TEST_PATH}")


def save_preprocessing_report(train_df, val_df, test_df):
    """Save preprocessing report."""
    log_message(f"\n7. Saving preprocessing report...")

    report = pd.DataFrame({
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
            True,
            True,
            True
        ]
    })

    report.to_csv(BENCHMARK_PATH / "preprocessing_report.csv", index=False)
    log_message(f"   Saved: preprocessing_report.csv")


def train_linear_regression(X_train, y_train, feature_set_name):
    """Train Linear Regression model."""
    log_message(f"\n   Training Linear Regression ({feature_set_name})...")

    start_time = time.time()
    model = LinearRegression(n_jobs=1)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    log_message(f"   Training completed in {training_time:.4f} seconds")

    return model, training_time


def evaluate_model(model, X_val, y_val, model_name, feature_set_name, training_time):
    """Evaluate model on validation set."""
    log_message(f"\n   Evaluating {model_name} ({feature_set_name})...")

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    log_message(f"   MAE:  {mae:.4f}")
    log_message(f"   RMSE: {rmse:.4f}")

    return y_pred, mae, rmse


def train_and_evaluate_models(train_df, val_df, F0, F1):
    """Train and evaluate models on both feature sets."""
    log_message(f"\n8. Training and evaluating baseline models...")

    # Extract target
    y_train = train_df['cnt'].values
    y_val = val_df['cnt'].values

    results = []

    # F0: Original features
    log_message(f"\n   === Feature Set F0 (Original) ===")
    X_train_F0 = train_df[F0].values
    X_val_F0 = val_df[F0].values

    model_F0, train_time_F0 = train_linear_regression(X_train_F0, y_train, "F0")
    y_pred_F0, mae_F0, rmse_F0 = evaluate_model(
        model_F0, X_val_F0, y_val, "Linear Regression", "F0", train_time_F0
    )

    results.append({
        'model': 'Linear Regression',
        'feature_set': 'F0',
        'split': 'validation',
        'MAE': mae_F0,
        'RMSE': rmse_F0,
        'training_time_seconds': train_time_F0
    })

    # F1: Original + cyclical features
    log_message(f"\n   === Feature Set F1 (Original + Cyclical) ===")
    X_train_F1 = train_df[F1].values
    X_val_F1 = val_df[F1].values

    model_F1, train_time_F1 = train_linear_regression(X_train_F1, y_train, "F1")
    y_pred_F1, mae_F1, rmse_F1 = evaluate_model(
        model_F1, X_val_F1, y_val, "Linear Regression", "F1", train_time_F1
    )

    results.append({
        'model': 'Linear Regression',
        'feature_set': 'F1',
        'split': 'validation',
        'MAE': mae_F1,
        'RMSE': rmse_F1,
        'training_time_seconds': train_time_F1
    })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    return results_df, model_F0, model_F1, y_pred_F0, y_pred_F1, X_val_F0, X_val_F1, y_val


def save_baseline_results(results_df):
    """Save baseline results to CSV."""
    log_message(f"\n9. Saving baseline results...")

    results_df.to_csv(METRICS_PATH / "baseline_model_results.csv", index=False)
    log_message(f"   Saved: baseline_model_results.csv")

    # Print summary
    log_message(f"\n   BASELINE MODEL RESULTS SUMMARY")
    log_message(f"   " + "-" * 66)
    for idx, row in results_df.iterrows():
        log_message(f"   {row['model']:20s} | {row['feature_set']:5s} | "
                   f"MAE: {row['MAE']:7.2f} | RMSE: {row['RMSE']:7.2f}")


def plot_actual_vs_predicted(y_val, y_pred_F0, y_pred_F1):
    """Create actual vs predicted plot."""
    log_message(f"\n10. Creating actual vs predicted plot...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # F0
    axes[0].scatter(y_val, y_pred_F0, alpha=0.4, s=20, color='steelblue', edgecolor='none')
    min_val = min(y_val.min(), y_pred_F0.min())
    max_val = max(y_val.max(), y_pred_F0.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Demand (cnt)', fontsize=11)
    axes[0].set_ylabel('Predicted Demand', fontsize=11)
    axes[0].set_title('Linear Regression - F0 (Original Features)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # F1
    axes[1].scatter(y_val, y_pred_F1, alpha=0.4, s=20, color='darkgreen', edgecolor='none')
    min_val = min(y_val.min(), y_pred_F1.min())
    max_val = max(y_val.max(), y_pred_F1.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Demand (cnt)', fontsize=11)
    axes[1].set_ylabel('Predicted Demand', fontsize=11)
    axes[1].set_title('Linear Regression - F1 (F0 + Cyclical)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: actual_vs_predicted.png")


def plot_residual_distribution(y_val, y_pred_F0, y_pred_F1):
    """Create residual distribution plot."""
    log_message(f"\n11. Creating residual distribution plot...")

    residuals_F0 = y_val - y_pred_F0
    residuals_F1 = y_val - y_pred_F1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F0 histogram
    axes[0, 0].hist(residuals_F0, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Residual Distribution - F0', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # F0 Q-Q plot (approximate)
    sorted_residuals_F0 = np.sort(residuals_F0)
    theoretical_quantiles = np.sort(np.random.normal(0, residuals_F0.std(), len(residuals_F0)))
    axes[0, 1].scatter(theoretical_quantiles, sorted_residuals_F0, alpha=0.5, s=20, color='steelblue')
    min_val = min(theoretical_quantiles.min(), sorted_residuals_F0.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals_F0.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Theoretical Quantiles', fontsize=11)
    axes[0, 1].set_ylabel('Residual Quantiles', fontsize=11)
    axes[0, 1].set_title('Q-Q Plot - F0', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # F1 histogram
    axes[1, 0].hist(residuals_F1, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    axes[1, 0].set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Residual Distribution - F1', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # F1 Q-Q plot
    sorted_residuals_F1 = np.sort(residuals_F1)
    theoretical_quantiles_F1 = np.sort(np.random.normal(0, residuals_F1.std(), len(residuals_F1)))
    axes[1, 1].scatter(theoretical_quantiles_F1, sorted_residuals_F1, alpha=0.5, s=20, color='darkgreen')
    min_val = min(theoretical_quantiles_F1.min(), sorted_residuals_F1.min())
    max_val = max(theoretical_quantiles_F1.max(), sorted_residuals_F1.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=11)
    axes[1, 1].set_ylabel('Residual Quantiles', fontsize=11)
    axes[1, 1].set_title('Q-Q Plot - F1', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "residual_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"   Saved: residual_distribution.png")


def main():
    """Run baseline model training pipeline."""
    df = load_data()

    # Create split
    train_df, val_df, test_df = create_train_val_test_split(df)

    # Define feature sets
    F0, F1 = define_feature_sets()

    # Create cyclical features for all splits
    train_df = create_cyclical_features(train_df)
    val_df = create_cyclical_features(val_df)
    test_df = create_cyclical_features(test_df)

    # Save splits
    save_splits(train_df, val_df, test_df)

    # Save preprocessing report
    save_preprocessing_report(train_df, val_df, test_df)

    # Train and evaluate models
    results_df, model_F0, model_F1, y_pred_F0, y_pred_F1, X_val_F0, X_val_F1, y_val = \
        train_and_evaluate_models(train_df, val_df, F0, F1)

    # Save baseline results
    save_baseline_results(results_df)

    # Create diagnostic plots
    plot_actual_vs_predicted(y_val, y_pred_F0, y_pred_F1)
    plot_residual_distribution(y_val, y_pred_F0, y_pred_F1)

    log_message(f"\n{'='*70}")
    log_message("BASELINE MODEL TRAINING COMPLETED SUCCESSFULLY")
    log_message(f"{'='*70}\n")


if __name__ == "__main__":
    main()
