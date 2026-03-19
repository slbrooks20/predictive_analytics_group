#!/usr/bin/env python3
"""
Main pipeline script to run data ingestion, validation, EDA, baseline modeling, and advanced modeling.
"""

import subprocess
import sys

def main():
    """Execute the data ingestion, validation, EDA, baseline modeling, and advanced modeling pipeline."""
    try:
        # Run the data ingestion and validation script
        result1 = subprocess.run(
            [sys.executable, 'scripts/data_ingestion_validation.py'],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result1.stdout)
        
        if result1.stderr:
            print("Errors:", result1.stderr, file=sys.stderr)
        
        # Run the EDA analysis script
        result2 = subprocess.run(
            [sys.executable, 'scripts/eda_analysis.py'],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result2.stdout)
        
        if result2.stderr:
            print("Errors:", result2.stderr, file=sys.stderr)
        
        # Run the baseline model script
        result3 = subprocess.run(
            [sys.executable, 'scripts/baseline_model.py'],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result3.stdout)
        
        if result3.stderr:
            print("Errors:", result3.stderr, file=sys.stderr)
        
        # Run the advanced modeling script
        result4 = subprocess.run(
            [sys.executable, 'scripts/advanced_modeling.py'],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result4.stdout)
        
        if result4.stderr:
            print("Errors:", result4.stderr, file=sys.stderr)
        
        print("Pipeline completed successfully.")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}", file=sys.stderr)
        print("STDOUT:", e.stdout, file=sys.stderr)
        print("STDERR:", e.stderr, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())