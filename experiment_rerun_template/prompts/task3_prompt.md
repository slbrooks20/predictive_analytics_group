# TASK 3 — Baseline Model Training + Evaluation Harness

## How to use this file

1. Make sure Tasks 1–2 are complete
2. Open the CLI that matches your assigned model
3. Copy everything between `---COPY START---` and `---COPY END---`
4. Paste it into the CLI
5. Wait for the model to finish
6. Check that metrics and plots were created
7. Screenshot/save the CLI log

---

## Prompt to paste

---COPY START---

You are working on a bike-sharing regression dataset with target variable cnt. Build a baseline model and evaluation harness in Python.

IMPORTANT RULES — follow these exactly:
- Python version: 3.11
- Allowed libraries ONLY: pandas, numpy, matplotlib, seaborn, scikit-learn
- Set random seed = 42 and use random_state=42 for ALL stochastic steps
- Set n_jobs=1 to avoid parallel randomness
- Do NOT ask the user any questions
- Generate complete, runnable Python scripts
- Save all scripts to: scripts/
- Save all figures to: outputs/figures/
- Save all metrics/results to: outputs/metrics/
- Save all models to: outputs/models/
- Save all benchmark logs/reports to: outputs/benchmark/
- Save all documentation to: outputs/docs/
- Log major steps and checks to: outputs/benchmark/experiment_log.txt
- Use MAE and RMSE as primary metrics (do NOT use R² as primary metric)
- The task must be runnable through: python run_pipeline.py

Project structure (already exists):

project/
├── dataset/hour.csv
├── scripts/
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

1. Data preparation:
   - Load the cleaned data from outputs/cleaned_data.csv (or dataset/hour.csv if not available)
   - Create a chronological split: 70% train / 15% validation / 15% test
   - Explicitly verify and assert:
     - max(train_time) < min(validation_time)
     - max(validation_time) < min(test_time)
     - If either fails, raise an error
   - Define two feature sets:
     - F0 = original predictors: hr, weekday, workingday, season, mnth, yr, weathersit, temp, atemp, hum, windspeed
     - F1 = F0 + cyclical features: sin_hour, cos_hour, sin_month, cos_month
   - Save splits to: outputs/train.csv, outputs/val.csv, outputs/test.csv

2. Save a preprocessing report to outputs/benchmark/preprocessing_report.csv with:
   - train_rows, val_rows, test_rows
   - chronological_split_pass (True/False)
   - feature_set_F0_defined, feature_set_F1_defined

3. Train ONE baseline model: Linear Regression
   - Train on training set ONLY
   - Evaluate on validation set
   - Use both F0 and F1 feature sets
   - Compute: MAE, RMSE, training_time_seconds

4. Save baseline results to outputs/metrics/baseline_model_results.csv with columns:
   - model, feature_set, split, MAE, RMSE, training_time_seconds

5. Generate diagnostic plots and save to outputs/figures/:
   - actual_vs_predicted.png
   - residual_distribution.png

6. Update run_pipeline.py to include this task
7. Update requirements.txt if needed

CONSTRAINTS:
- Do NOT tune aggressively — this is a baseline only
- Do NOT use the test set to choose or evaluate the model yet
- Do NOT use classification metrics
- Do NOT fit preprocessing on the full dataset before splitting
- All transforms must be fit on train only, then applied to val/test

---COPY END---

---

## What to check after the model finishes

- [ ] Script(s) created in `scripts/`
- [ ] `outputs/train.csv`, `outputs/val.csv`, `outputs/test.csv` exist
- [ ] `outputs/benchmark/preprocessing_report.csv` exists
- [ ] `outputs/metrics/baseline_model_results.csv` exists with correct columns
- [ ] Chronological split is verified (train dates < val dates < test dates)
- [ ] Baseline uses Linear Regression (not something fancier)
- [ ] MAE and RMSE are reported (not just R²)
- [ ] Test set was NOT used during this task
- [ ] Diagnostic plots exist in `outputs/figures/`
- [ ] `run_pipeline.py` updated and runs cleanly
