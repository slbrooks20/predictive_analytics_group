# Experiment Rerun Template

This folder contains the reproducibility scaffold for rerunning the benchmark with the supported CLI agents.

## What is here

- Per-model folders that become runnable after bootstrapping
- Shared task prompt files in `prompts/`
- Setup and authentication instructions in `RERUN_SETUP.md`

## Before you start

1. Install and authenticate the required CLI tools by following `RERUN_SETUP.md`.
2. From the repository root, run:

```bash
./prepare_rerun_template.sh
```

This creates `experiment_rerun_template/<model>/project/` with the standard project structure used in the benchmark, including:

- `dataset/hour.csv`
- `scripts/`
- `outputs/`
- `requirements.txt`
- `run_pipeline.py`

## How to rerun a model

1. Choose the model folder you want to rerun.
2. `cd` into `experiment_rerun_template/<model>/project`
3. Open the matching CLI for that model.
4. Copy and paste the relevant prompt from `prompts/`.
5. Run the generated pipeline and save the CLI log or screenshots for evidence.

## Important notes

- The files in prompts/ are the benchmark task prompts to paste into the active CLI session after you have launched the correct tool and changed into the model's project folder.
- The checked-in model folders are templates; the runnable `project/` subfolders are created by `./prepare_rerun_template.sh`.
- Start with `RERUN_SETUP.md` if you are setting this up from a fresh clone.
