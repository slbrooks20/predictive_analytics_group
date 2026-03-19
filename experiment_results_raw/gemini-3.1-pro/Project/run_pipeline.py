import os
import subprocess
<<<<<<< HEAD

def main():
    print("Running data preparation pipeline...")
    subprocess.run(["C:\\Users\\elenb\\anaconda3\\envs\\benchmark_311\\python.exe", "scripts/data_prep.py"], check=True)
    
    print("Running EDA pipeline...")
    subprocess.run(["C:\\Users\\elenb\\anaconda3\\envs\\benchmark_311\\python.exe", "scripts/eda.py"], check=True)
    
    print("Running baseline modeling pipeline...")
    subprocess.run(["C:\\Users\\elenb\\anaconda3\\envs\\benchmark_311\\python.exe", "scripts/baseline.py"], check=True)
    
    print("Running model tuning pipeline...")
    subprocess.run(["C:\\Users\\elenb\\anaconda3\\envs\\benchmark_311\\python.exe", "scripts/model_tuning.py"], check=True)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
=======
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
>>>>>>> 1914c2dd4cad679f031794efb9ef4d9b9ba61dc2
