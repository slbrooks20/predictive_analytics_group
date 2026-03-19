# TASK 4 — Improving Performance

## How to use this file

1. Make sure Tasks 1–3 are complete
2. Open the CLI that matches your assigned model
3. Copy everything between `---COPY START---` and `---COPY END---`
4. Paste it into the CLI
5. Wait for the model to finish
6. Check that comparison metrics and plots were created
7. Screenshot/save the CLI log

---

## Prompt to paste

---COPY START---

You are working on the same bike-sharing regression task with target variable cnt. Starting from the existing Linear Regression baseline, improve performance using feature engineering, tuning, or model changes.

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

Use the train/val/test splits saved in outputs/ from Task 3.

TASK REQUIREMENTS:

1. Model comparison — train and evaluate ALL of these models:
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - MLP Regressor

2. For each model:
   - Evaluate on BOTH feature sets F0 and F1 (defined in Task 3)
   - Use validation set ONLY for comparison — do NOT use test set
   - Compute: MAE, RMSE, training_time_seconds

3. Save all model results to: outputs/metrics/all_results.csv with columns:
   - model, feature_set, split, MAE, RMSE, training_time_seconds

4. Tuning:
   - Tune Ridge, Random Forest, Gradient Boosting, and MLP
   - For Gradient Boosting, sweep n_estimators values and plot:
     validation MAE vs n_estimators
   - Save figure to: outputs/figures/validation_curve_gb.png
   - Save tuning results to: outputs/metrics/tuning_results.csv

5. Select the best model based on lowest validation MAE
   - Save the best model to: outputs/models/final_model.pkl

6. Final evaluation on TEST set (using best model only):
   - Compute MAE and RMSE
   - Generate and save to outputs/figures/:
     - residual_distribution.png
     - mae_by_hour.png
     - mae_by_weekday.png
     - residual_vs_temperature.png
     - rolling_mae_over_time.png
   - Save final test results to: outputs/metrics/final_model_results.csv with columns:
     - model, feature_set, split, MAE, RMSE

7. Update run_pipeline.py to include this task
8. Update requirements.txt if needed

CONSTRAINTS:
- Do NOT claim improvement without metric evidence
- Do NOT overwrite baseline results — keep them for comparison
- Do NOT use test data during tuning (only for final evaluation)
- Keep the workflow reproducible
- All stochastic steps must use random_state=42

---COPY END---

---

## What to check after the model finishes

- [ ] Script(s) created in `scripts/`
- [ ] `outputs/metrics/all_results.csv` exists with all 4 models × 2 feature sets
- [ ] `outputs/metrics/tuning_results.csv` exists
- [ ] `outputs/metrics/final_model_results.csv` exists (test set results)
- [ ] `outputs/models/final_model.pkl` exists
- [ ] `outputs/figures/validation_curve_gb.png` exists
- [ ] Error analysis plots exist in `outputs/figures/`:
  - [ ] `residual_distribution.png`
  - [ ] `mae_by_hour.png`
  - [ ] `mae_by_weekday.png`
  - [ ] `residual_vs_temperature.png`
  - [ ] `rolling_mae_over_time.png`
- [ ] Baseline results are still preserved (not overwritten)
- [ ] Test set was only used for final evaluation, NOT during tuning
- [ ] Best model is clearly identified with MAE/RMSE evidence
- [ ] `run_pipeline.py` updated and runs the full pipeline end-to-end
