import os
import subprocess
import sys

# Ensure output directories exist (just in case)
directories = [
    'outputs/figures',
    'outputs/metrics',
    'outputs/models',
    'outputs/docs',
    'outputs/benchmark',
    'scripts'
]

for d in directories:
    os.makedirs(d, exist_ok=True)

def run_script(script_path):
    print(f"Executing: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_path}")
        sys.exit(1)

if __name__ == "__main__":
    # Task 1: Data Ingestion and Processing
    run_script(os.path.join('scripts', 'data_processing.py'))
    
    # Task 2: EDA
    run_script(os.path.join('scripts', 'eda.py'))

    # Task 3: Baseline Model Training
    run_script(os.path.join('scripts', 'baseline_model.py'))

    # Task 4: Model Improvement and Tuning
    run_script(os.path.join('scripts', 'model_improvement.py'))
