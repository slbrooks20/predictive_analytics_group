# Bike-Sharing Dataset Ingestion & Validation

A complete data ingestion, validation, and cleaning pipeline for the UCI bike-sharing dataset.

## Overview

This project performs comprehensive data validation and cleaning on the bike-sharing hourly dataset with target variable `cnt` (bike rental count).

### Key Features

- **Complete Data Validation**: Missing values, duplicates, data types, and value ranges
- **Leakage Detection**: Identifies and removes leakage variables (casual + registered = cnt)
- **Correlation Analysis**: Computes feature-target correlations and flags high-correlation features
- **Data Quality Checks**: Validates non-negative values and detects impossible values
- **Comprehensive Logging**: Detailed experiment logs and validation reports

## Project Structure

```
project/
├── dataset/
│   └── hour.csv              # Source dataset (17,379 rows × 17 columns)
├── scripts/
│   └── data_ingestion.py     # Main data ingestion and validation script
├── outputs/
│   ├── cleaned_data.csv      # Cleaned dataset after leakage removal
│   ├── figures/              # Generated visualizations (if applicable)
│   ├── metrics/              # Performance metrics and results
│   ├── models/               # Trained models (if applicable)
│   ├── docs/                 # Documentation
│   └── benchmark/
│       ├── data_validation_report.csv   # Validation metrics summary
│       └── experiment_log.txt            # Detailed experiment log
├── requirements.txt          # Python dependencies (pinned versions)
├── run_pipeline.py           # Main pipeline entry point
└── README.md                 # This file
```

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py
```

### Individual Script

```bash
# Run data ingestion directly
python scripts/data_ingestion.py
```

## Requirements

- Python 3.11+
- Dependencies (see `requirements.txt`):
  - pandas 2.2.3
  - numpy 1.26.4
  - matplotlib 3.9.2
  - seaborn 0.13.2
  - scikit-learn 1.5.2

## Data Validation Pipeline

The pipeline performs the following checks in order:

1. **Load Dataset**: Reads `dataset/hour.csv`
2. **Shape & Types**: Prints dataset dimensions and column data types
3. **Variable Classification**: Identifies numeric vs categorical columns
4. **Missing Values**: Checks for null/missing data per column
5. **Duplicate Detection**: Counts duplicate rows
6. **Leakage Validation**: Verifies `casual + registered = cnt` relationship
   - **Finding**: Leakage detected - both columns removed from dataset
7. **Correlation Analysis**: Computes feature-target correlations
   - **Finding**: Max correlation is 0.4048 (temp variable) - no high-correlation features
8. **Target Validation**: Ensures all cnt values ≥ 0
9. **Impossible Values**: Checks count-like fields for negative or non-integer values
10. **Data Cleaning**: Removes leakage columns
11. **Output Generation**: Saves cleaned dataset and validation report

## Key Findings

| Metric | Value |
|--------|-------|
| Original Rows | 17,379 |
| Final Rows | 17,379 |
| Original Columns | 17 |
| Final Columns | 15 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Leakage Detected | Yes (casual, registered) |
| Columns Removed | 2 |
| Max Feature-Target Correlation | 0.4048 |
| Target Non-negative | True |

## Output Files

### `outputs/cleaned_data.csv`
- **Format**: CSV (comma-separated values)
- **Shape**: 17,379 rows × 15 columns
- **Columns**: instant, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, cnt
- **Target Variable**: cnt (bike rental count)

### `outputs/benchmark/data_validation_report.csv`
Summary metrics from the validation pipeline:
- `missing_values_total`: 0
- `duplicate_rows`: 0
- `leakage_identity_detected`: True
- `leakage_columns_removed`: 2
- `max_feature_target_corr`: 0.404772
- `target_nonnegative_check`: True

### `outputs/benchmark/experiment_log.txt`
Detailed step-by-step log of all validation checks and findings.

## Variable Reference

### Target Variable
- **cnt**: Count of total bike rental demand (bikes rented per hour)

### Date/Time Features
- **instant**: Record index
- **dteday**: Date (YYYY-MM-DD format)
- **yr**: Year (0 = 2011, 1 = 2012)
- **mnth**: Month (1-12)
- **hr**: Hour (0-23)

### Categorical Features
- **season**: Season (1=winter, 2=spring, 3=summer, 4=fall)
- **holiday**: 1 if holiday, 0 otherwise
- **weekday**: Day of week (0-6, 0=Sunday)
- **workingday**: 1 if working day, 0 otherwise
- **weathersit**: Weather situation (1=clear, 2=mist, 3=light rain/snow, 4=heavy rain)

### Weather Features
- **temp**: Normalized temperature (0-1 scale)
- **atemp**: Normalized "feels like" temperature (0-1 scale)
- **hum**: Normalized humidity (0-1 scale)
- **windspeed**: Normalized wind speed (0-1 scale)

### Removed Features (Leakage)
- **casual**: Count of casual users (REMOVED - sums to cnt)
- **registered**: Count of registered users (REMOVED - sums to cnt)

## Notes

- All numerical values are normalized to [0, 1] range or represent counts
- The dataset is hourly aggregated bike-sharing data from 2011-2012
- No missing data was found in the original dataset
- Leakage detection ensures `casual` and `registered` do not appear in the cleaned dataset used for modeling
- Random seed is set to 42 for reproducibility

## Author

Generated for Predictive Analytics Group 20-3, T2 2025-26
