"""
Wrapper for data ingestion module
Allows importing and running data_ingestion from run_pipeline.py
"""

import os
import sys

def run():
    """Run the data ingestion script"""
    # Change to project directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Import and run the main script
    from scripts import data_ingestion
    data_ingestion.main()

if __name__ == '__main__':
    run()
