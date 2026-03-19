# Benchmark Analysis

This folder contains the notebook and artifacts for the benchmark analysis.

## Best Way To Run This

If you download this folder fresh, do this:

1. Open a terminal in this folder.
2. Regenerate the notebook:

```bash
python3 generate_benchmark_notebook.py
```

3. Open `benchmark_analysis.ipynb` in Jupyter or VS Code.
4. Run all cells.
5. Save the notebook if you want the rendered outputs to stay embedded in the `.ipynb`.

That is the correct workflow.

Why this order matters:
- `generate_benchmark_notebook.py` rebuilds the notebook from the current `benchmarks_all.csv` and `outputs/` artifacts.
- opening the notebook without regenerating it can leave you looking at an older notebook structure
- opening the notebook without running cells can leave you looking at stale charts and tables

## What Is In This Folder

- `benchmark_analysis.ipynb`
  - the notebook with the final charts, tables, and commentary
- `generate_benchmark_notebook.py`
  - rebuilds the notebook from source data
- `benchmarks_all.csv`
  - the benchmark score sheet used by the notebook
- `outputs/`
  - per-model artifacts used by the notebook

## Required Python Packages

Install these if your environment does not already have them:

```bash
python3 -m pip install -r requirements.txt
```

The generator and notebook otherwise use only the Python standard library.

## Folder Structure Expected By The Notebook

The notebook expects this structure:

```text
outputs/
├── opus-4.6/
├── haiku-4.5/
├── sonnet-4.6/
├── gemini-3-flash/
├── gemini-3.1-pro/
├── devstral-2/
├── devstral-small/
├── codex-5.3/
└── codex-5.4/
```

Each model folder is expected to contain:

```text
outputs/<model-slug>/
├── figures/
├── metrics/
└── docs/
```

The notebook is built to tolerate some missing files:
- missing artifacts render as placeholders
- alternate filenames are supported for some plots
- incomplete handoff data should not crash the notebook

## What The Notebook Covers

The notebook includes:

- benchmark parsing and cleaning
- overall ranking
- task-by-task comparison
- code quality breakdown
- efficiency analysis
- reliability analysis
- Task 4 output-quality comparison
- CLI tool comparison
- side-by-side plot comparisons
- metrics audit tables
- baseline consistency checks

## Important Interpretation Notes

- Total runtime is summed across tasks.
- Token usage is shown as **peak per-task token percentage**, not a summed total across tasks.
- In the plot comparison grids:
  - red `TASK FAILED` means the model failed Task 4 overall
  - gray `Failed to generate` means the model passed Task 4, but that specific artifact was missing

## If You Want To Export Or Publish

Before exporting to HTML or pushing to GitHub:

1. Run `python3 generate_benchmark_notebook.py`
2. Open `benchmark_analysis.ipynb`
3. Run all cells
4. Save the notebook

If you skip step 3, GitHub or HTML export may still show old charts even though the notebook source was regenerated.
