# MSIN0097 Predictive Group Assignment

This repository is organized into four top-level areas:

- `benchmark_analysis/` contains the benchmark notebook, the benchmark score sheet, and the normalized per-model artifacts used by the notebook.
- `experiment_results_raw/` contains the original raw experiment outputs exactly as collected.
- `experiment_rerun_template/` contains the rerun prompts plus per-model placeholder folders for reproducing the benchmark workflow.
- `Project/` contains the broader assignment deliverables.

The root `requirements.txt` covers Python packages only. CLI installation is handled separately via `install_clis.sh` or the setup notes in `experiment_rerun_template/RERUN_SETUP.md`.

## Viewing the benchmark analysis

If you only want to inspect the benchmark notebook:

1. `cd benchmark_analysis`
2. Install Python dependencies with `python3 -m pip install -r ../requirements.txt`
3. Regenerate the notebook with `python3 generate_benchmark_notebook.py`
4. Open `benchmark_analysis.ipynb`
5. Run all cells before trusting embedded charts or tables

Additional notebook-specific notes live in [benchmark_analysis/README.md](benchmark_analysis/README.md).

## Rerunning the benchmark

The rerun workflow currently has two distinct parts:

1. Install and authenticate the required CLIs.
2. Bootstrap a model folder under `experiment_rerun_template/` with the runnable project scaffold expected by the prompts.

From the repo root, run:

```bash
./prepare_rerun_template.sh
```

That script creates `experiment_rerun_template/<model>/project/` folders with the shared dataset, a starter `run_pipeline.py`, `requirements.txt`, and the expected empty output directories.

Start with [experiment_rerun_template/RERUN_SETUP.md](experiment_rerun_template/RERUN_SETUP.md) before attempting a rerun.
