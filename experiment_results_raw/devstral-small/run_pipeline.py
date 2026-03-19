#!/usr/bin/env python3
"""
Main pipeline execution script for bike-sharing data validation.
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def main():
    """Execute the complete pipeline"""
    try:
        # Step 1: Data validation
        from scripts.data_validation import main as validation_main
        validation_main()
        
        # Step 2: EDA analysis
        from scripts.eda_analysis import main as eda_main
        eda_main()
        
        # Step 3: Baseline modeling
        from scripts.baseline_modeling import main as modeling_main
        modeling_main()
        
        # Step 4: Model improvement
        from scripts.model_improvement import main as improvement_main
        improvement_main()
        
        print("\nComplete pipeline executed successfully!")
        return 0
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
