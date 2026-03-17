"""
Main Pipeline Execution Script
Runs all data ingestion, validation, EDA, baseline, and improved model tasks
"""

import sys
from pathlib import Path

def main():
    """Execute the complete bike-sharing analysis pipeline"""

    print("\n" + "="*70)
    print("BIKE-SHARING REGRESSION ANALYSIS PIPELINE")
    print("="*70 + "\n")

    # Add scripts directory to path
    scripts_dir = Path(__file__).parent / 'scripts'
    sys.path.insert(0, str(scripts_dir))

    # Execute data ingestion and validation
    print("\n[1/4] Running Data Ingestion & Validation...\n")
    import data_ingestion
    data_ingestion.main()

    # Execute EDA
    print("\n[2/4] Running Exploratory Data Analysis (EDA)...\n")
    import eda
    eda.main()

    # Execute baseline model training
    print("\n[3/4] Building Baseline Model & Evaluation Harness...\n")
    import baseline_model
    baseline_model.main()

    # Execute improved model training and comparison
    print("\n[4/4] Building Improved Models & Comparison...\n")
    import improved_models
    improved_models.main()

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
