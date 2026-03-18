"""
run_pipeline.py — Main entry point for the predictive analytics pipeline.
Usage: python run_pipeline.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.data_ingestion import run as run_data_ingestion
from scripts.eda import run as run_eda
from scripts.baseline_model import run as run_baseline
from scripts.model_improvement import run as run_improvement


def main() -> None:
    print("=" * 60)
    print("PREDICTIVE ANALYTICS PIPELINE")
    print("=" * 60)

    # Task 1: Data ingestion, schema checks, missingness handling
    run_data_ingestion()

    # Task 2: Exploratory Data Analysis
    run_eda()

    # Task 3: Baseline model & evaluation harness
    run_baseline()

    # Task 4: Model comparison, tuning & final evaluation
    run_improvement()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
