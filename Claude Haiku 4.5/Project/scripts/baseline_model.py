"""
Baseline Model Building and Evaluation
Builds a Linear Regression baseline with proper train/val/test splits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time

# Set random seed
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(cleaned_path, raw_path):
    """Load dataset"""
    if Path(cleaned_path).exists():
        df = pd.read_csv(cleaned_path)
        print(f"[OK] Loaded cleaned data from: {cleaned_path}")
    else:
        df = pd.read_csv(raw_path)
        print(f"[OK] Loaded raw data from: {raw_path}")

    # Ensure dteday is datetime
    df['dteday'] = pd.to_datetime(df['dteday'])

    return df.sort_values('dteday').reset_index(drop=True)

def create_chronological_split(df, train_pct=0.70, val_pct=0.15, test_pct=0.15):
    """Create chronological train/val/test split with strict separation by date"""
    print(f"\n--- CHRONOLOGICAL SPLIT ---")

    total_rows = len(df)
    train_end = int(total_rows * train_pct)
    val_end = train_end + int(total_rows * val_pct)

    # Get cutoff dates at those row indices
    train_cutoff_date = df.iloc[train_end - 1]['dteday']
    val_cutoff_date = df.iloc[val_end - 1]['dteday']

    # Split strictly by time to ensure no overlap
    train_df = df[df['dteday'] <= train_cutoff_date].copy()
    val_df = df[(df['dteday'] > train_cutoff_date) & (df['dteday'] <= val_cutoff_date)].copy()
    test_df = df[df['dteday'] > val_cutoff_date].copy()

    print(f"Total records: {total_rows}")
    print(f"Train: {len(train_df)} ({len(train_df)/total_rows*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/total_rows*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/total_rows*100:.1f}%)")

    # Verify chronological order
    print(f"\n--- CHRONOLOGICAL ORDER VERIFICATION ---")

    train_max = train_df['dteday'].max()
    val_min = val_df['dteday'].min()
    val_max = val_df['dteday'].max()
    test_min = test_df['dteday'].min()

    print(f"Train max date: {train_max}")
    print(f"Val min date: {val_min}")
    print(f"Val max date: {val_max}")
    print(f"Test min date: {test_min}")

    # Assert strict chronological order
    assert train_max < val_min, f"[ERROR] Train max ({train_max}) >= Val min ({val_min})"
    assert val_max < test_min, f"[ERROR] Val max ({val_max}) >= Test min ({test_min})"

    print(f"[OK] Chronological order verified - strict temporal separation confirmed")

    chronological_pass = True

    return train_df, val_df, test_df, chronological_pass

def create_feature_sets(df):
    """Create F0 and F1 feature sets"""
    print(f"\n--- FEATURE SET DEFINITION ---")

    # F0: Original features
    F0_features = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr',
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

    print(f"F0 features ({len(F0_features)}): {F0_features}")

    # Create cyclical features for F1
    df_copy = df.copy()

    # Hour cyclical features (24-hour cycle)
    df_copy['sin_hour'] = np.sin(2 * np.pi * df_copy['hr'] / 24)
    df_copy['cos_hour'] = np.cos(2 * np.pi * df_copy['hr'] / 24)

    # Month cyclical features (12-month cycle)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['mnth'] / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['mnth'] / 12)

    # F1: F0 + cyclical features
    F1_features = F0_features + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']

    print(f"F1 features ({len(F1_features)}): F0 + [sin_hour, cos_hour, sin_month, cos_month]")

    return df_copy, F0_features, F1_features

def save_splits(train_df, val_df, test_df):
    """Save train/val/test splits to CSV"""
    print(f"\n--- SAVING DATA SPLITS ---")

    outputs_dir = Path('outputs')

    train_path = outputs_dir / 'train.csv'
    val_path = outputs_dir / 'val.csv'
    test_path = outputs_dir / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Saved: {train_path}")
    print(f"[OK] Saved: {val_path}")
    print(f"[OK] Saved: {test_path}")

def fit_scaler(X_train):
    """Fit scaler on training data only"""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def scale_data(scaler, X_train, X_val):
    """Apply fitted scaler to train and validation data"""
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    start_time = time.time()

    model = LinearRegression(n_jobs=1)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    return model, training_time

def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set"""
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return y_pred, mae, rmse

def create_preprocessing_report(train_df, val_df, test_df, chronological_pass):
    """Create preprocessing report"""
    report_data = {
        'Metric': [
            'train_rows',
            'val_rows',
            'test_rows',
            'total_rows',
            'chronological_split_pass',
            'feature_set_F0_defined',
            'feature_set_F1_defined'
        ],
        'Value': [
            str(len(train_df)),
            str(len(val_df)),
            str(len(test_df)),
            str(len(train_df) + len(val_df) + len(test_df)),
            str(chronological_pass),
            'True',
            'True'
        ]
    }

    report_df = pd.DataFrame(report_data)
    return report_df

def plot_actual_vs_predicted(y_val, y_pred_f0, y_pred_f1, output_path):
    """Plot actual vs predicted for both feature sets"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F0 results
    axes[0].scatter(y_val, y_pred_f0, alpha=0.4, s=20, color='steelblue')
    min_val = min(y_val.min(), y_pred_f0.min())
    max_val = max(y_val.max(), y_pred_f0.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Demand', fontsize=11)
    axes[0].set_ylabel('Predicted Demand', fontsize=11)
    axes[0].set_title('F0 Feature Set - Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # F1 results
    axes[1].scatter(y_val, y_pred_f1, alpha=0.4, s=20, color='coral')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Demand', fontsize=11)
    axes[1].set_ylabel('Predicted Demand', fontsize=11)
    axes[1].set_title('F1 Feature Set - Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def plot_residual_distribution(y_val, y_pred_f0, y_pred_f1, output_path):
    """Plot residual distributions for both feature sets"""
    residuals_f0 = y_val - y_pred_f0
    residuals_f1 = y_val - y_pred_f1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F0 histogram
    axes[0, 0].hist(residuals_f0, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 0].set_xlabel('Residual', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('F0 Feature Set - Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # F0 Q-Q like plot
    axes[0, 1].scatter(y_pred_f0, residuals_f0, alpha=0.4, s=20, color='steelblue')
    axes[0, 1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Demand', fontsize=11)
    axes[0, 1].set_ylabel('Residual', fontsize=11)
    axes[0, 1].set_title('F0 Feature Set - Residuals vs Predictions', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # F1 histogram
    axes[1, 0].hist(residuals_f1, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].set_xlabel('Residual', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('F1 Feature Set - Residual Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # F1 Q-Q like plot
    axes[1, 1].scatter(y_pred_f1, residuals_f1, alpha=0.4, s=20, color='coral')
    axes[1, 1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Demand', fontsize=11)
    axes[1, 1].set_ylabel('Residual', fontsize=11)
    axes[1, 1].set_title('F1 Feature Set - Residuals vs Predictions', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def save_baseline_results(results_list):
    """Save baseline model results to CSV"""
    results_df = pd.DataFrame(results_list)

    metrics_dir = Path('outputs/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)

    output_path = metrics_dir / 'baseline_model_results.csv'
    results_df.to_csv(output_path, index=False)

    print(f"[OK] Saved: {output_path}")

    # Print results table
    print(f"\n--- BASELINE MODEL RESULTS ---")
    print(results_df.to_string(index=False))

def main():
    """Main execution"""
    print(f"\n{'='*70}")
    print("BASELINE MODEL BUILDING AND EVALUATION")
    print(f"{'='*70}\n")

    # Paths
    cleaned_path = Path('outputs/cleaned_data.csv')
    raw_path = Path('dataset/hour.csv')
    benchmark_dir = Path('outputs/benchmark')
    figures_dir = Path('outputs/figures')
    metrics_dir = Path('outputs/metrics')
    models_dir = Path('outputs/models')

    # Create directories
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(cleaned_path, raw_path)
    print(f"Dataset shape: {df.shape}")

    # Create chronological split
    train_df, val_df, test_df, chronological_pass = create_chronological_split(df)

    # Create feature sets with cyclical encoding
    df_with_features, F0_features, F1_features = create_feature_sets(df)

    # Update split dataframes with new features
    train_idx = train_df.index
    val_idx = val_df.index
    test_idx = test_df.index

    train_df = df_with_features.iloc[train_idx].copy()
    val_df = df_with_features.iloc[val_idx].copy()
    test_df = df_with_features.iloc[test_idx].copy()

    # Save splits
    save_splits(train_df, val_df, test_df)

    # Create preprocessing report
    preprocessing_report = create_preprocessing_report(train_df, val_df, test_df, chronological_pass)
    preprocessing_report.to_csv(benchmark_dir / 'preprocessing_report.csv', index=False)
    print(f"\n[OK] Saved: {benchmark_dir / 'preprocessing_report.csv'}")

    # Extract features and target
    X_train_f0 = train_df[F0_features].copy()
    X_train_f1 = train_df[F1_features].copy()
    y_train = train_df['cnt'].copy()

    X_val_f0 = val_df[F0_features].copy()
    X_val_f1 = val_df[F1_features].copy()
    y_val = val_df['cnt'].copy()

    # Fit scalers on training data only
    print(f"\n--- SCALING (fitted on training data only) ---")
    scaler_f0 = fit_scaler(X_train_f0)
    scaler_f1 = fit_scaler(X_train_f1)

    # Scale data
    X_train_f0_scaled, X_val_f0_scaled = scale_data(scaler_f0, X_train_f0, X_val_f0)
    X_train_f1_scaled, X_val_f1_scaled = scale_data(scaler_f1, X_train_f1, X_val_f1)

    print(f"[OK] Scalers fit on training set only")

    # Train baseline model with F0
    print(f"\n--- TRAINING LINEAR REGRESSION (F0) ---")
    model_f0, training_time_f0 = train_linear_regression(X_train_f0_scaled, y_train)
    print(f"[OK] Training time: {training_time_f0:.4f} seconds")

    # Evaluate on validation set
    y_pred_f0, mae_f0, rmse_f0 = evaluate_model(model_f0, X_val_f0_scaled, y_val)
    print(f"[OK] Validation MAE: {mae_f0:.2f}")
    print(f"[OK] Validation RMSE: {rmse_f0:.2f}")

    # Train baseline model with F1
    print(f"\n--- TRAINING LINEAR REGRESSION (F1) ---")
    model_f1, training_time_f1 = train_linear_regression(X_train_f1_scaled, y_train)
    print(f"[OK] Training time: {training_time_f1:.4f} seconds")

    # Evaluate on validation set
    y_pred_f1, mae_f1, rmse_f1 = evaluate_model(model_f1, X_val_f1_scaled, y_val)
    print(f"[OK] Validation MAE: {mae_f1:.2f}")
    print(f"[OK] Validation RMSE: {rmse_f1:.2f}")

    # Prepare results
    results = [
        {
            'model': 'Linear Regression',
            'feature_set': 'F0',
            'split': 'validation',
            'MAE': round(mae_f0, 4),
            'RMSE': round(rmse_f0, 4),
            'training_time_seconds': round(training_time_f0, 4)
        },
        {
            'model': 'Linear Regression',
            'feature_set': 'F1',
            'split': 'validation',
            'MAE': round(mae_f1, 4),
            'RMSE': round(rmse_f1, 4),
            'training_time_seconds': round(training_time_f1, 4)
        }
    ]

    # Save results
    save_baseline_results(results)

    # Generate diagnostic plots
    print(f"\n--- GENERATING DIAGNOSTIC PLOTS ---")
    plot_actual_vs_predicted(y_val, y_pred_f0, y_pred_f1, figures_dir / 'actual_vs_predicted.png')
    plot_residual_distribution(y_val, y_pred_f0, y_pred_f1, figures_dir / 'residual_distribution.png')

    # Save models for potential later use
    print(f"\n--- SAVING MODELS ---")
    import pickle

    with open(models_dir / 'baseline_lr_f0.pkl', 'wb') as f:
        pickle.dump({'model': model_f0, 'scaler': scaler_f0}, f)
    print(f"[OK] Saved: {models_dir / 'baseline_lr_f0.pkl'}")

    with open(models_dir / 'baseline_lr_f1.pkl', 'wb') as f:
        pickle.dump({'model': model_f1, 'scaler': scaler_f1}, f)
    print(f"[OK] Saved: {models_dir / 'baseline_lr_f1.pkl'}")

    # Summary
    print(f"\n{'='*70}")
    print("BASELINE MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"\nFeature Set Comparison (Validation Set):")
    print(f"  F0 (Original 11 features):")
    print(f"    MAE: {mae_f0:.2f}")
    print(f"    RMSE: {rmse_f0:.2f}")
    print(f"  F1 (F0 + 4 Cyclical features):")
    print(f"    MAE: {mae_f1:.2f}")
    print(f"    RMSE: {rmse_f1:.2f}")

    improvement_mae = ((mae_f0 - mae_f1) / mae_f0) * 100
    improvement_rmse = ((rmse_f0 - rmse_f1) / rmse_f0) * 100

    print(f"\nImprovement with cyclical features:")
    print(f"  MAE: {improvement_mae:+.2f}%")
    print(f"  RMSE: {improvement_rmse:+.2f}%")

    print(f"\nData Splits:")
    print(f"  Training: {len(train_df)} rows")
    print(f"  Validation: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows (held out)")

    print(f"\nOutputs saved to:")
    print(f"  Splits: outputs/train.csv, outputs/val.csv, outputs/test.csv")
    print(f"  Results: outputs/metrics/baseline_model_results.csv")
    print(f"  Plots: outputs/figures/actual_vs_predicted.png")
    print(f"  Plots: outputs/figures/residual_distribution.png")
    print(f"  Models: outputs/models/baseline_lr_f0.pkl, baseline_lr_f1.pkl")

    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
