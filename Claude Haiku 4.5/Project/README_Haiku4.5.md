<<<<<<< HEAD
# Bike-Sharing Dataset - Data Ingestion & Validation Pipeline

## Overview
This project implements a data ingestion and validation pipeline for the bike-sharing dataset (`hour.csv`). The pipeline loads the dataset, performs schema checks, identifies data quality issues, detects leakage, and saves a cleaned version.

## Project Structure
```
project/
├── dataset/
│   └── hour.csv                    # Raw bike-sharing dataset (17,379 rows x 17 columns)
├── scripts/
│   ├── __init__.py
│   ├── data_ingestion.py          # Main data ingestion and validation script
│   └── data_ingestion_wrapper.py   # Wrapper for pipeline execution
├── outputs/
│   ├── cleaned_data.csv           # Cleaned dataset (17,379 rows x 15 columns)
│   ├── benchmark/
│   │   ├── data_validation_report.csv   # Validation metrics and findings
│   │   └── experiment_log.txt           # Detailed log of all checks
│   ├── figures/                   # Placeholder for visualizations
│   ├── metrics/                   # Placeholder for metrics files
│   ├── models/                    # Placeholder for model files
│   └── docs/                      # Placeholder for documentation
├── requirements.txt               # Python dependencies (pinned versions)
├── run_pipeline.py               # Main pipeline execution script
└── README.md                      # This file
```

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
# Run the complete pipeline
python run_pipeline.py
```

## Data Validation Report

The validation report (`outputs/benchmark/data_validation_report.csv`) contains:

| Metric | Value |
|--------|-------|
| Original Rows | 17,379 |
| Original Columns | 17 |
| Final Rows | 17,379 |
| Final Columns | 15 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Leakage Detected | Yes |
| Columns Removed (Leakage) | 2 |
| Target Non-negative Check | Pass |
| Max Feature-Target Correlation | 0.4048 |
| High Correlation Features (>0.95) | 0 |

## Key Findings

### Data Quality
- **No missing values**: All 17,379 rows have complete data
- **No duplicates**: No duplicate rows detected
- **Non-negative target**: All values of `cnt` (bike rental count) are >= 1

### Data Leakage
**CRITICAL**: The dataset exhibits exact data leakage:
- `casual + registered = cnt` (identity relationship)
- Action taken: **Removed `casual` and `registered` columns** from the cleaned dataset
- These columns directly sum to the target, making them perfect predictors and useless for modeling

### Feature Correlations (Top 10)
1. `temp` (0.4048) - Temperature has the strongest correlation with bike rentals
2. `atemp` (0.4009) - Adjusted temperature
3. `hr` (0.3941) - Hour of day
4. `hum` (-0.3229) - Humidity (negative correlation)
5. `instant` (0.2784) - Record index
6. `yr` (0.2505) - Year
7. `season` (0.1781) - Season
8. `weathersit` (-0.1424) - Weather situation
9. `mnth` (0.1206) - Month
10. `windspeed` (0.0932) - Wind speed

**No features have |correlation| > 0.95**, indicating no additional leakage or multicollinearity issues.

## Variables in Cleaned Dataset

### Target Variable
- `cnt`: Count of bike rentals (target for regression)

### Temporal Features
- `instant`: Record index
- `dteday`: Date (string format)
- `yr`: Year (0: 2011, 1: 2012)
- `season`: Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
- `mnth`: Month (1-12)
- `hr`: Hour of day (0-23)
- `weekday`: Day of week (0-6)
- `holiday`: Holiday flag (0/1)
- `workingday`: Working day flag (0/1)

### Weather Features
- `weathersit`: Weather condition (1-4)
- `temp`: Normalized temperature
- `atemp`: Normalized adjusted temperature
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed

## Metrics

### Primary Metrics (for model evaluation)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

### Secondary Validation Metrics
- Missing values: 0
- Duplicate rows: 0
- Leakage identity: Yes (removed)
- Target non-negative: Pass
- High correlation features: 0

## Output Files

### 1. `outputs/cleaned_data.csv`
Cleaned dataset with:
- 17,379 rows × 15 columns
- Leakage columns (`casual`, `registered`) removed
- All quality checks passed
- Ready for modeling

### 2. `outputs/benchmark/data_validation_report.csv`
Validation metrics in CSV format:
- original_rows, original_columns
- final_rows, final_columns
- missing_values_total, duplicate_rows
- leakage_identity_detected, leakage_columns_removed
- target_nonnegative_check
- max_feature_target_corr, high_corr_features_count

### 3. `outputs/benchmark/experiment_log.txt`
Detailed text log containing:
- All validation checks performed
- Statistical summaries
- Warning flags and errors
- Complete validation report

## Dependencies

All packages are Python 3.11 compatible:
- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.2
- seaborn==0.12.2
- scikit-learn==1.3.0

## EDA (Exploratory Data Analysis) Results

### Key Findings

**Demand Patterns:**
- Peak demand during rush hours: 8-9 AM and 5-6 PM
- Lowest demand during night hours: 3-5 AM
- Mean hourly demand varies from 6.3 (night) to 461.4 (peak hours)
- Strong daily seasonality with clear commuting patterns

**Weekly Patterns:**
- Weekday demand significantly higher than weekend
- More uniform distribution during weekends
- Monday-Friday show consistent high-demand pattern

**Feature Relationships:**
- Temperature has strongest correlation (r=0.405)
- Hour of day is critical feature (r=0.394)
- Humidity shows negative correlation (r=-0.323)
- Most extreme weather conditions are rare (8.2% of data)

### Generated Outputs

**Figures (7 plots):**
- `target_distribution.png` – Distribution and density of bike rentals
- `demand_time_series.png` – Daily demand trend over 2 years
- `heatmap_hour_weekday.png` – Heat map showing demand patterns
- `demand_vs_temp.png` – Scatter plot with temperature correlation
- `demand_vs_hum.png` – Humidity impact on demand
- `demand_vs_windspeed.png` – Wind speed effect on demand
- `correlation_matrix.png` – Feature correlation heatmap

**Summary Tables:**
- `outputs/benchmark/eda_tables.csv` – Mean demand by hour and weekday

**Insights Document:**
- `outputs/docs/eda_summary.txt` – Comprehensive EDA summary with modeling recommendations

### Modeling Recommendations

1. **Temporal Modeling**: Use time-series aware approaches due to strong seasonality
2. **Feature Engineering**: Create hour×weekday interactions for improved accuracy
3. **Data Split**: Use temporal split (not random) for proper train/test separation
4. **Evaluation**: Focus on MAE and RMSE as primary metrics
5. **Handling Skew**: Target distribution is right-skewed; consider transformations
6. **Rare Events**: Extreme weather is underrepresented; use stratified sampling

## Baseline Model Results

### Data Splits (Chronological)
- **Training**: 12,187 rows (70.1%) — Jan 2011 to May 27, 2012
- **Validation**: 2,592 rows (14.9%) — May 28 to Sep 12, 2012
- **Test**: 2,600 rows (15.0%) — Sep 13 to Dec 31, 2012 (held out)

✓ Strict chronological separation verified (no date overlap between splits)

### Feature Sets
- **F0**: 11 original features (hr, weekday, workingday, season, mnth, yr, weathersit, temp, atemp, hum, windspeed)
- **F1**: F0 + 4 cyclical features (sin_hour, cos_hour, sin_month, cos_month)

### Linear Regression Baseline Performance

| Feature Set | MAE | RMSE | Training Time |
|------------|-----|------|---------------|
| F0 | 155.64 | 190.51 | 0.003s |
| F1 | 137.98 | 175.09 | 0.003s |
| **Improvement** | **+11.34%** | **+8.10%** | — |

**Key Insight**: Cyclical features significantly improve baseline performance, suggesting hour-of-day and month seasonality are important for predictions.

### Diagnostic Outputs
- `outputs/figures/actual_vs_predicted.png` — Scatter plots showing prediction accuracy
- `outputs/figures/residual_distribution.png` — Residual analysis (4 subplots)
- `outputs/metrics/baseline_model_results.csv` — Detailed results table
- `outputs/models/baseline_lr_f0.pkl`, `baseline_lr_f1.pkl` — Saved models & scalers

## Improved Model Results

### Model Comparison (Validation Set)

| Model | F0 Features | F1 Features | Best |
|-------|------------|------------|------|
| Ridge | 155.64 | 137.98 | — |
| Random Forest | **47.81** | 48.39 | ✓ |
| Gradient Boosting | 49.39 | 50.52 | — |
| MLP | 58.54 | 56.24 | — |

**Best Model**: Random Forest with F0 features (47.81 MAE)

### Performance Improvement
- **Baseline (Linear Regression F1)**: MAE = 137.98
- **Improved Model (Random Forest F0)**: MAE = 47.81
- **Improvement**: **+65.35%** better (MAE reduction)

### Final Test Set Performance
- **Test MAE**: 50.14
- **Test RMSE**: 75.44
- **Generalization**: Only 2.3% difference from validation (minimal overfitting)

### Gradient Boosting Tuning Results
- Validation curve shows diminishing returns after n_estimators=150
- Optimal n_estimators ≈ 100-150 for this dataset

### Diagnostic Plots Generated
1. **Residual Distribution** – Shows prediction errors are approximately normal
2. **MAE by Hour** – Error varies significantly by hour (lower at peak hours)
3. **MAE by Weekday** – Weekday/weekend patterns visible in error rates
4. **Residuals vs Temperature** – Slight heteroscedasticity at temperature extremes
5. **Rolling MAE Over Time** – Temporal stability of model predictions

## Performance Summary

| Metric | Linear Regression | Random Forest | Improvement |
|--------|------------------|---------------|------------|
| Validation MAE | 137.98 | 47.81 | +65.35% |
| Test MAE | — | 50.14 | — |
| Test RMSE | — | 75.44 | — |

## Key Insights

1. **Random Forest dominates** linear and neural network approaches for this time-series regression task
2. **F0 features outperform F1** for Random Forest (despite F1 being better for linear models), suggesting tree models automatically capture cyclical patterns
3. **Error varies by time** – Hour-of-day has largest impact on prediction accuracy
4. **Model generalizes well** – Small validation/test gap indicates robust model selection
5. **Temperature extremes** are hardest to predict (higher residuals at very high/low temperatures)

## Notes

- **Random seed**: Set to 42 for reproducibility across all stochastic operations
- **Data leakage handling**: Critical columns (casual + registered) removed before train/val/test split
- **Preprocessing**: StandardScaler fit on training data only, applied to validation/test
- **Temporal validation**: Time-series aware split to prevent future information leakage
- **No test set contamination**: Test set used only for final evaluation, never during training/tuning
- **Pipeline status**: Complete end-to-end analysis with data validation, EDA, baseline, and improved modeling
=======
# predictive_analytics_group
UCL MSc BA Predictive Analytics Group Assignment
>>>>>>> 869eeed6afb14b17f7b676f3d639874c6f998cde
