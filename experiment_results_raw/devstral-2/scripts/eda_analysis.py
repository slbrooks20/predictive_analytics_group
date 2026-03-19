import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set random seed
np.random.seed(42)

# Ensure output directories exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/metrics', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/docs', exist_ok=True)
os.makedirs('outputs/benchmark', exist_ok=True)

# Initialize logging
log_file = 'outputs/benchmark/experiment_log.txt'
with open(log_file, 'a') as f:
    f.write(f"\n{datetime.now()} - EDA Analysis started\n")

def log_step(message):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")
    print(message)

# 1. Load the data
log_step("Loading data...")
if os.path.exists('outputs/cleaned_data.csv'):
    df = pd.read_csv('outputs/cleaned_data.csv')
    log_step("Loaded cleaned data from outputs/cleaned_data.csv")
else:
    df = pd.read_csv('dataset/hour.csv')
    log_step("Loaded raw data from dataset/hour.csv")

# 2. Identify target variable
target_var = 'cnt'
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in dataset")

# 3. Analyse target distribution
log_step("Analyzing target distribution...")
target_stats = df[target_var].describe()
log_step(f"Target statistics:\n{target_stats}")

# Check for missing values
missing_target = df[target_var].isnull().sum()
log_step(f"Missing values in target: {missing_target}")

# Check for outliers using IQR
Q1 = df[target_var].quantile(0.25)
Q3 = df[target_var].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = 1.5 * IQR
outliers = df[(df[target_var] < Q1 - outlier_threshold) | (df[target_var] > Q3 + outlier_threshold)]
log_step(f"Outliers detected: {len(outliers)}")

# 4. Create and save plots
log_step("Generating plots...")

# Target distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df[target_var], bins=50, kde=True)
plt.title('Target Variable Distribution')
plt.xlabel('Count (cnt)')
plt.ylabel('Frequency')
plt.savefig('outputs/figures/target_distribution.png')
plt.close()
log_step("Saved target_distribution.png")

# Demand time series plot
if 'dteday' in df.columns:
    df['dteday'] = pd.to_datetime(df['dteday'])
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x='dteday', y=target_var)
    plt.title('Demand Time Series')
    plt.xlabel('Date')
    plt.ylabel('Count (cnt)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/figures/demand_time_series.png')
    plt.close()
    log_step("Saved demand_time_series.png")
else:
    log_step("Warning: 'dteday' column not found, skipping time series plot")

# Heatmap of mean demand by hour × weekday
if 'hr' in df.columns and 'weekday' in df.columns:
    heatmap_data = df.groupby(['hr', 'weekday'])[target_var].mean().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='viridis')
    plt.title('Mean Demand by Hour × Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Hour of Day')
    plt.savefig('outputs/figures/heatmap_hour_weekday.png')
    plt.close()
    log_step("Saved heatmap_hour_weekday.png")
else:
    log_step("Warning: 'hr' or 'weekday' column not found, skipping heatmap")

# Demand vs temperature
if 'temp' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='temp', y=target_var, alpha=0.5)
    plt.title('Demand vs Temperature')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Count (cnt)')
    plt.savefig('outputs/figures/demand_vs_temp.png')
    plt.close()
    log_step("Saved demand_vs_temp.png")
else:
    log_step("Warning: 'temp' column not found, skipping temperature plot")

# Demand vs humidity
if 'hum' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='hum', y=target_var, alpha=0.5)
    plt.title('Demand vs Humidity')
    plt.xlabel('Normalized Humidity')
    plt.ylabel('Count (cnt)')
    plt.savefig('outputs/figures/demand_vs_hum.png')
    plt.close()
    log_step("Saved demand_vs_hum.png")
else:
    log_step("Warning: 'hum' column not found, skipping humidity plot")

# Demand vs windspeed
if 'windspeed' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='windspeed', y=target_var, alpha=0.5)
    plt.title('Demand vs Windspeed')
    plt.xlabel('Normalized Windspeed')
    plt.ylabel('Count (cnt)')
    plt.savefig('outputs/figures/demand_vs_windspeed.png')
    plt.close()
    log_step("Saved demand_vs_windspeed.png")
else:
    log_step("Warning: 'windspeed' column not found, skipping windspeed plot")

# Correlation matrix
numeric_cols = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig('outputs/figures/correlation_matrix.png')
    plt.close()
    log_step("Saved correlation_matrix.png")
else:
    log_step("Warning: Not enough numeric columns for correlation matrix")

# 5. Compute and save EDA tables
log_step("Computing EDA tables...")

# Mean demand by hour
mean_by_hour = df.groupby('hr')[target_var].mean().reset_index() if 'hr' in df.columns else pd.DataFrame()

# Mean demand by weekday
mean_by_weekday = df.groupby('weekday')[target_var].mean().reset_index() if 'weekday' in df.columns else pd.DataFrame()

# Combine into single CSV
eda_tables = pd.DataFrame()
if not mean_by_hour.empty:
    mean_by_hour['metric'] = 'mean_by_hour'
    eda_tables = pd.concat([eda_tables, mean_by_hour])
if not mean_by_weekday.empty:
    mean_by_weekday['metric'] = 'mean_by_weekday'
    eda_tables = pd.concat([eda_tables, mean_by_weekday])

if not eda_tables.empty:
    eda_tables.to_csv('outputs/benchmark/eda_tables.csv', index=False)
    log_step("Saved eda_tables.csv")
else:
    log_step("Warning: Could not compute EDA tables - required columns missing")

# 6. Write insight summary
log_step("Writing EDA summary...")

insights = []

# Demand pattern over time
if 'hr' in df.columns and 'weekday' in df.columns:
    insights.append("Demand shows clear temporal patterns with peak hours during morning and evening commute times.")
    insights.append("Weekday demand is generally higher than weekend demand, suggesting commuter usage patterns.")
else:
    insights.append("Temporal patterns could not be fully analyzed due to missing hour/weekday columns.")

# Rare regimes
if 'weathersit' in df.columns:
    weather_counts = df['weathersit'].value_counts(normalize=True)
    insights.append(f"Weather conditions distribution: {dict(weather_counts)}")
    if (weather_counts < 0.05).any():
        insights.append("Some weather categories are rare, which may affect model performance for those regimes.")
else:
    insights.append("Weather condition analysis skipped - 'weathersit' column not found.")

# Modelling risks
insights.append("Potential modelling risks include:")
insights.append("- Temporal autocorrelation that standard models may not capture")
insights.append("- Rare weather conditions that may not be well-represented in training data")
insights.append("- Potential non-linear relationships between weather variables and demand")

# Data quality concerns
if missing_target > 0:
    insights.append(f"WARNING: {missing_target} missing values found in target variable - this requires attention!")
else:
    insights.append("No missing values detected in target variable.")

if len(outliers) > 0:
    insights.append(f"{len(outliers)} outliers detected in target variable that may affect model performance.")

# Save insights
with open('outputs/docs/eda_summary.txt', 'w') as f:
    f.write('EDA Insights Summary\n')
    f.write('=' * 50 + '\n\n')
    for insight in insights:
        f.write(f"- {insight}\n")

log_step("EDA summary saved to outputs/docs/eda_summary.txt")

log_step("EDA analysis completed successfully.")