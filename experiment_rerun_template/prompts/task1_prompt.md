# TASK 1 — Dataset Ingestion + Schema Checks + Missingness Handling

## How to use this file

1. Run `./prepare_rerun_template.sh` from the repo root if you have not already done so
2. Open the CLI that matches your assigned model
3. `cd` into the model's `project/` folder
4. Copy everything between the two `---COPY START---` and `---COPY END---` lines below
5. Paste it into the CLI
6. Wait for the model to finish
7. Check that files were created in `scripts/` and `outputs/`
8. Try running the generated code
9. Screenshot/save the CLI log

---

## Prompt to paste

---COPY START---

You are working on a bike-sharing regression dataset with target variable cnt. Complete dataset ingestion, schema checks, and missingness handling in Python.

IMPORTANT RULES — follow these exactly:
- Python version: 3.11
- Allowed libraries ONLY: pandas, numpy, matplotlib, seaborn, scikit-learn
- Set random seed = 42 where relevant
- Do NOT ask the user any questions
- Generate complete, runnable Python scripts
- Save all scripts to: scripts/
- Save all figures to: outputs/figures/
- Save all metrics/results to: outputs/metrics/
- Save all benchmark logs/reports to: outputs/benchmark/
- Save all documentation to: outputs/docs/
- Log major steps and checks to: outputs/benchmark/experiment_log.txt
- Use MAE and RMSE as primary metrics (do NOT use R² as primary)
- The task must be runnable through: python run_pipeline.py

Project structure (already exists — do not recreate dataset/):

project/
├── dataset/hour.csv          ← already exists, do not modify
├── scripts/                  ← write your scripts here
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── docs/
│   └── benchmark/
├── requirements.txt
├── README.md
└── run_pipeline.py

TASK REQUIREMENTS:

1. Load dataset/hour.csv
2. Print and log dataset shape
3. Print column names and data types
4. Identify numeric vs categorical variables
5. Check missing values per column
6. Check for duplicate rows
7. Perform explicit leakage validation:
   - Verify whether casual + registered = cnt
   - If true, remove casual and registered
8. Compute correlation between each feature and cnt
   - Flag any feature with absolute correlation > 0.95
9. Validate that cnt >= 0 for all rows
10. Check for impossible values in count-like fields
11. Save cleaned dataset to: outputs/cleaned_data.csv
12. Save a validation report to: outputs/benchmark/data_validation_report.csv

The validation report must contain at least these rows:
- missing_values_total
- duplicate_rows
- leakage_identity_detected
- leakage_columns_removed
- max_feature_target_corr
- target_nonnegative_check

13. Print a short summary of findings and changes
14. Generate requirements.txt with pinned library versions
15. Generate run_pipeline.py that executes this task

Do NOT train a model in this task.
Do NOT fabricate columns, paths, or dataset properties.

---COPY END---

---

## What to check after the model finishes

- [ ] Script(s) created in `scripts/`
- [ ] `outputs/cleaned_data.csv` exists
- [ ] `outputs/benchmark/data_validation_report.csv` exists
- [ ] `outputs/benchmark/experiment_log.txt` exists
- [ ] `requirements.txt` exists
- [ ] `run_pipeline.py` exists and runs without errors
- [ ] `casual` and `registered` columns are removed from cleaned data
- [ ] No missing values reported (or handled appropriately)
- [ ] Model did NOT ask you any questions
