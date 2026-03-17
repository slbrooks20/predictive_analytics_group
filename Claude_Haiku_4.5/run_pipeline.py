#!/usr/bin/env python3
"""
Main pipeline runner for complete bike-sharing regression workflow.

Usage:
    python run_pipeline.py

Stages:
    1. Data Ingestion & Validation
    2. Exploratory Data Analysis
    3. Baseline Model Training
    4. Model Improvement & Comparison
    5. Final Test Evaluation

Output:
    - Cleaned dataset with validation reports
    - EDA visualizations and insights
    - Baseline model (Linear Regression) results
    - Improved models (Ridge, RF, GB, MLP) with tuning
    - Final model selection and test evaluation
    - Comprehensive diagnostic plots and metrics
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_pipeline():
    """Execute the complete data processing pipeline."""
    print("=" * 70)
    print("BIKE-SHARING DATASET PROCESSING PIPELINE")
    print("=" * 70)
    print()

    # Step 1: Data Ingestion
    print("Step 1: Data Ingestion & Validation")
    print("-" * 70)
    script_path = PROJECT_ROOT / "scripts" / "data_ingestion.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=PROJECT_ROOT
        )
        print()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Data ingestion failed with return code {e.returncode}")
        sys.exit(1)

    # Step 2: Exploratory Data Analysis
    print("Step 2: Exploratory Data Analysis")
    print("-" * 70)
    script_path = PROJECT_ROOT / "scripts" / "exploratory_data_analysis.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=PROJECT_ROOT
        )
        print()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: EDA failed with return code {e.returncode}")
        sys.exit(1)

    # Step 3: Baseline Model Training
    print("Step 3: Baseline Model Training & Evaluation")
    print("-" * 70)
    script_path = PROJECT_ROOT / "scripts" / "baseline_model.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=PROJECT_ROOT
        )
        print()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Baseline model training failed with return code {e.returncode}")
        sys.exit(1)

    # Step 4: Model Improvement & Comparison
    print("Step 4: Model Improvement & Comparison")
    print("-" * 70)
    script_path = PROJECT_ROOT / "scripts" / "model_improvement.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=PROJECT_ROOT
        )
        print()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Model improvement failed with return code {e.returncode}")
        sys.exit(1)

    # Completion message
    print("=" * 70)
    print("COMPLETE PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 70)
    print()
    print("Key Outputs:")
    print(f"  Data & Splits:")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'train.csv'} (70%)")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'val.csv'} (15%)")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'test.csv'} (15%)")
    print(f"  Model Results:")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'metrics' / 'baseline_model_results.csv'} (baseline)")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'metrics' / 'all_results.csv'} (all models)")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'metrics' / 'final_model_results.csv'} (test evaluation)")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'models' / 'final_model.pkl'} (best model)")
    print(f"  Tuning & Diagnostics:")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'metrics' / 'tuning_results.csv'}")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'figures' / 'validation_curve_gb.png'}")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'figures' / 'mae_by_hour.png'}")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'figures' / 'mae_by_weekday.png'}")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'figures' / 'residual_vs_temperature.png'}")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'figures' / 'rolling_mae_over_time.png'}")
    print(f"  Analysis:")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'docs' / 'eda_summary.txt'}")
    print(f"    - {PROJECT_ROOT / 'outputs' / 'benchmark' / 'experiment_log.txt'}")
    print()


if __name__ == "__main__":
    run_pipeline()
