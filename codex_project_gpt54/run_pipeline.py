from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TASK_SCRIPTS = [
    BASE_DIR / "scripts" / "data_ingestion_validation.py",
    BASE_DIR / "scripts" / "eda_analysis.py",
    BASE_DIR / "scripts" / "baseline_model.py",
    BASE_DIR / "scripts" / "model_improvement.py",
]

BASE_ENV = os.environ.copy()
BASE_ENV.update(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
)


if __name__ == "__main__":
    for script_path in TASK_SCRIPTS:
        subprocess.run([sys.executable, str(script_path)], check=True, env=BASE_ENV)
