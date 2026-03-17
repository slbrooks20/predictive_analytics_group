"""
run_pipeline.py — Master entry point for the bike-sharing predictive analytics pipeline.
Run with: python run_pipeline.py
"""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

PIPELINE_STAGES = [
    ("Task 1 — Data Ingestion & Validation", "01_data_ingestion.py"),
    ("Task 2 — Exploratory Data Analysis",   "02_eda.py"),
    ("Task 3 — Baseline Model",              "03_baseline_model.py"),
    ("Task 4 — Model Comparison & Tuning",   "04_model_comparison.py"),
]


def run_stage(label: str, script: str):
    path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {label}")
    print(f"  Script : scripts/{script}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Stage failed: {label} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[OK] {label} completed successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("  BIKE-SHARING REGRESSION PIPELINE")
    print("  Starting pipeline execution …")
    print("=" * 60)

    for label, script in PIPELINE_STAGES:
        run_stage(label, script)

    print("\n" + "=" * 60)
    print("  ALL STAGES COMPLETE")
    print("=" * 60)
