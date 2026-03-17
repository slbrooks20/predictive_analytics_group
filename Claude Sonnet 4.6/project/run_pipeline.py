"""
run_pipeline.py — Entry point for the Bike-Sharing Regression pipeline.
Usage:  python run_pipeline.py
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_rel_path: str) -> None:
    script_path = os.path.join(BASE_DIR, script_rel_path)
    print(f"\n{'='*70}")
    print(f"  Running: {script_rel_path}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Script failed: {script_rel_path}  (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run_script("scripts/data_ingestion.py")
    run_script("scripts/eda.py")
    run_script("scripts/baseline_model.py")
    run_script("scripts/model_comparison.py")
    print("\nPipeline completed successfully.")
