# TASK 2 — EDA and Insight Generation (with plots)

## How to use this file

1. Make sure Task 1 is complete (you need `outputs/cleaned_data.csv`)
2. Open the CLI that matches your assigned model (same session or new one)
3. Copy everything between `---COPY START---` and `---COPY END---`
4. Paste it into the CLI
5. Wait for the model to finish
6. Check that plots were created in `outputs/figures/`
7. Screenshot/save the CLI log

---

## Prompt to paste

---COPY START---

You are working on a bike-sharing regression dataset with target variable cnt. Perform EDA in Python and generate useful modelling insights with plots.

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

Use the cleaned data from Task 1 at outputs/cleaned_data.csv (or load from dataset/hour.csv if that file does not exist).

TASK REQUIREMENTS:

1. Load the data and identify the target variable cnt
2. Analyse target distribution, missingness, outliers, and key feature relationships

3. Create and save these specific plots to outputs/figures/:
   - target_distribution.png — histogram/density of cnt
   - demand_time_series.png — cnt over time
   - heatmap_hour_weekday.png — mean demand by hour × weekday
   - demand_vs_temp.png — cnt vs temperature
   - demand_vs_hum.png — cnt vs humidity
   - demand_vs_windspeed.png — cnt vs windspeed
   - correlation_matrix.png — correlation heatmap for numeric variables

4. Compute and save to outputs/benchmark/eda_tables.csv:
   - Mean demand by hour
   - Mean demand by weekday

5. Write a concise insight summary to outputs/docs/eda_summary.txt that mentions:
   - Demand pattern over time
   - Any rare regimes (e.g., extreme weather categories)
   - Likely modelling risks
   - Any remaining leakage or data quality concerns

6. Update run_pipeline.py to include this task
7. Update requirements.txt if needed

CONSTRAINTS:
- Do NOT do full feature engineering (save that for later tasks)
- Do NOT make causal claims
- Do NOT produce more plots than listed above
- Save all figures to disk — do not just display them

---COPY END---

---

## What to check after the model finishes

- [ ] Script(s) created in `scripts/`
- [ ] All 7 required plots exist in `outputs/figures/`:
  - [ ] `target_distribution.png`
  - [ ] `demand_time_series.png`
  - [ ] `heatmap_hour_weekday.png`
  - [ ] `demand_vs_temp.png`
  - [ ] `demand_vs_hum.png`
  - [ ] `demand_vs_windspeed.png`
  - [ ] `correlation_matrix.png`
- [ ] `outputs/benchmark/eda_tables.csv` exists
- [ ] `outputs/docs/eda_summary.txt` exists
- [ ] `run_pipeline.py` updated to include this task
- [ ] Plots look reasonable (not blank, not garbled)
- [ ] Summary mentions demand patterns, rare regimes, and modelling risks
