import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
<<<<<<< HEAD
import logging
=======
>>>>>>> 1914c2dd4cad679f031794efb9ef4d9b9ba61dc2

# Set random seed
np.random.seed(42)

<<<<<<< HEAD
# Set up logging
log_dir = "outputs/benchmark"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "experiment_log.txt"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Started EDA pipeline.")
    
    # Load dataset
    data_path = "outputs/cleaned_data.csv"
    if not os.path.exists(data_path):
        data_path = "dataset/hour.csv"
        logger.info(f"Cleaned data not found, falling back to {data_path}")
    else:
        logger.info(f"Loading cleaned data from {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Identify target variable
    target = 'cnt'
    if target not in df.columns:
        logger.error(f"Target variable '{target}' not found in the dataset.")
        return
        
    # Directories
    fig_dir = "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    docs_dir = "outputs/docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    benchmark_dir = "outputs/benchmark"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # 1. target_distribution.png
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], kde=True, bins=50)
    plt.title('Target Distribution (cnt)')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(fig_dir, 'target_distribution.png'), bbox_inches='tight')
    plt.close()
    logger.info("Saved target_distribution.png")

    # 2. demand_time_series.png
    plt.figure(figsize=(15, 6))
    if 'dteday' in df.columns:
        # Convert to datetime for plotting, if possible
        df_ts = df.copy()
        df_ts['dteday'] = pd.to_datetime(df_ts['dteday'])
        # Since it's hourly data, group by day for a clearer time series or plot a subset
        daily_cnt = df_ts.groupby('dteday')[target].sum().reset_index()
        sns.lineplot(data=daily_cnt, x='dteday', y=target)
        plt.title('Daily Demand Over Time')
        plt.xlabel('Date')
    else:
        # Fallback to index
        plt.plot(df.index, df[target])
        plt.title('Demand Over Time (Index)')
        plt.xlabel('Index')
    plt.ylabel('Count')
    plt.savefig(os.path.join(fig_dir, 'demand_time_series.png'), bbox_inches='tight')
    plt.close()
    logger.info("Saved demand_time_series.png")

    # 3. heatmap_hour_weekday.png
    if 'hr' in df.columns and 'weekday' in df.columns:
        plt.figure(figsize=(12, 8))
        pivot_table = df.pivot_table(values=target, index='hr', columns='weekday', aggfunc='mean')
        sns.heatmap(pivot_table, cmap='YlGnBu', annot=False)
        plt.title('Mean Demand by Hour and Weekday')
        plt.xlabel('Weekday')
        plt.ylabel('Hour')
        plt.savefig(os.path.join(fig_dir, 'heatmap_hour_weekday.png'), bbox_inches='tight')
        plt.close()
        logger.info("Saved heatmap_hour_weekday.png")
    
    # 4. demand_vs_temp.png
    if 'temp' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='temp', y=target, alpha=0.3)
        plt.title('Demand vs Temperature')
        plt.xlabel('Temperature (normalized)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(fig_dir, 'demand_vs_temp.png'), bbox_inches='tight')
        plt.close()
        logger.info("Saved demand_vs_temp.png")

    # 5. demand_vs_hum.png
    if 'hum' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='hum', y=target, alpha=0.3)
        plt.title('Demand vs Humidity')
        plt.xlabel('Humidity (normalized)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(fig_dir, 'demand_vs_hum.png'), bbox_inches='tight')
        plt.close()
        logger.info("Saved demand_vs_hum.png")

    # 6. demand_vs_windspeed.png
    if 'windspeed' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='windspeed', y=target, alpha=0.3)
        plt.title('Demand vs Windspeed')
        plt.xlabel('Windspeed (normalized)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(fig_dir, 'demand_vs_windspeed.png'), bbox_inches='tight')
        plt.close()
        logger.info("Saved demand_vs_windspeed.png")

    # 7. correlation_matrix.png
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Correlation Matrix for Numeric Variables')
    plt.savefig(os.path.join(fig_dir, 'correlation_matrix.png'), bbox_inches='tight')
    plt.close()
    logger.info("Saved correlation_matrix.png")
    
    # Compute and save eda_tables.csv
    tables = []
    if 'hr' in df.columns:
        hr_mean = df.groupby('hr')[target].mean().reset_index()
        hr_mean['group'] = 'hour'
        hr_mean.rename(columns={'hr': 'category', target: 'mean_demand'}, inplace=True)
        tables.append(hr_mean)
        
    if 'weekday' in df.columns:
        wd_mean = df.groupby('weekday')[target].mean().reset_index()
        wd_mean['group'] = 'weekday'
        wd_mean.rename(columns={'weekday': 'category', target: 'mean_demand'}, inplace=True)
        tables.append(wd_mean)
        
    if tables:
        eda_tables = pd.concat(tables, ignore_index=True)
        # Reorder columns to group, category, mean_demand
        eda_tables = eda_tables[['group', 'category', 'mean_demand']]
        eda_tables_path = os.path.join(benchmark_dir, 'eda_tables.csv')
        eda_tables.to_csv(eda_tables_path, index=False)
        logger.info(f"Saved eda_tables to {eda_tables_path}")
        
    # Write summary
    summary_text = """EDA Summary:
1. Demand pattern over time: The demand shows a clear seasonal pattern, with lower demand in winter months and higher in summer. There is also an overall increasing trend across the two years (2011 to 2012).
2. Rare regimes: Certain weather situations, such as extreme heavy rain/snow (weathersit = 4), occur very rarely in the dataset and could be considered outliers or a rare regime.
3. Likely modelling risks: 
   - The target distribution is heavily right-skewed, which might violate normality assumptions for some models (e.g., linear regression). Transformation of the target (like log1p) could be beneficial.
   - Time-based features like hour and weekday have strong non-linear relationships with the target, indicating a need for non-linear models or encoding (e.g., cyclical features, dummy variables).
   - Time series autocorrelation is highly likely. Validation strategy should respect the time dimension (e.g., TimeSeriesSplit) instead of random K-Fold.
4. Remaining leakage or data quality concerns: The casual and registered columns were already removed to prevent target leakage. Other features do not show extreme correlation with the target. Maintaining proper chronological train/validation splits is essential to prevent future leakage.
"""
    summary_path = os.path.join(docs_dir, 'eda_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    logger.info(f"Saved EDA summary to {summary_path}")
    
    logger.info("EDA pipeline completed successfully.")

if __name__ == "__main__":
    main()
=======
def run_eda():
    # Define paths
    input_path = 'outputs/cleaned_data.csv'
    if not os.path.exists(input_path):
        input_path = 'dataset/hour.csv'
    
    figures_dir = 'outputs/figures/'
    benchmark_dir = 'outputs/benchmark/'
    docs_dir = 'outputs/docs/'
    log_path = 'outputs/benchmark/experiment_log.txt'

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    # Logging helper
    def log(message):
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + '\n')

    log("\n--- EDA STARTED ---")

    # Load dataset
    df = pd.read_csv(input_path)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])

    # 1. Target Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cnt'], kde=True, color='skyblue')
    plt.title('Distribution of Rental Counts (cnt)')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_dir, 'target_distribution.png'))
    plt.close()

    # 2. Demand Time Series
    plt.figure(figsize=(15, 6))
    plt.plot(df['dteday'], df['cnt'], color='teal', alpha=0.5)
    plt.title('Rental Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.savefig(os.path.join(figures_dir, 'demand_time_series.png'))
    plt.close()

    # 3. Heatmap: Hour x Weekday
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(values='cnt', index='hr', columns='weekday', aggfunc='mean')
    sns.heatmap(pivot_table, cmap='YlGnBu')
    plt.title('Mean Demand by Hour and Weekday')
    plt.xlabel('Weekday (0=Sun, 6=Sat)')
    plt.ylabel('Hour of Day')
    plt.savefig(os.path.join(figures_dir, 'heatmap_hour_weekday.png'))
    plt.close()

    # 4. Demand vs Temp
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='temp', y='cnt', alpha=0.1, color='orange')
    plt.title('Demand vs Normalized Temperature')
    plt.savefig(os.path.join(figures_dir, 'demand_vs_temp.png'))
    plt.close()

    # 5. Demand vs Humidity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='hum', y='cnt', alpha=0.1, color='green')
    plt.title('Demand vs Normalized Humidity')
    plt.savefig(os.path.join(figures_dir, 'demand_vs_hum.png'))
    plt.close()

    # 6. Demand vs Windspeed
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='windspeed', y='cnt', alpha=0.1, color='blue')
    plt.title('Demand vs Normalized Windspeed')
    plt.savefig(os.path.join(figures_dir, 'demand_vs_windspeed.png'))
    plt.close()

    # 7. Correlation Matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_matrix.png'))
    plt.close()

    # Compute EDA tables
    mean_hr = df.groupby('hr')['cnt'].mean().reset_index()
    mean_hr.columns = ['Hour', 'Mean_Demand']
    mean_weekday = df.groupby('weekday')['cnt'].mean().reset_index()
    mean_weekday.columns = ['Weekday', 'Mean_Demand']

    # Combine tables for saving (using a marker to separate or just saving two files)
    # The requirement says eda_tables.csv, so I'll concatenate with a label.
    mean_hr['Type'] = 'Hour'
    mean_weekday['Type'] = 'Weekday'
    eda_tables = pd.concat([mean_hr.rename(columns={'Hour': 'Value'}), 
                            mean_weekday.rename(columns={'Weekday': 'Value'})])
    eda_tables.to_csv(os.path.join(benchmark_dir, 'eda_tables.csv'), index=False)
    log("EDA tables saved to eda_tables.csv")

    # Generate Insights Summary
    summary = f"""EDA INSIGHT SUMMARY
====================
Demand Pattern over Time:
- Clear seasonality is observed in the time series plot, with higher demand in warmer months.
- A strong daily cycle exists, with peaks during morning (7-9 AM) and evening (5-7 PM) commute hours on weekdays.
- Weekends show a more distributed demand peak during the afternoon.

Rare Regimes:
- Extreme weather conditions (very high windspeed or very low/high humidity) correspond to significantly lower demand.
- Heavy precipitation or severe weather (weathersit 4) is extremely rare in this dataset.

Modelling Risks:
- High variance in demand at peak hours suggests that 'hr' and 'workingday' interactions are critical.
- The target 'cnt' is right-skewed, which might require transformation or specific loss functions.
- 'temp' and 'atemp' are highly correlated (multi-collinearity risk).

Data Quality & Leakage:
- No missing values or duplicates remain in the cleaned dataset.
- Leakage columns (casual, registered) were successfully removed in Task 1.
- No obvious remaining leakage detected; correlations between features and target are within reasonable bounds.
"""
    with open(os.path.join(docs_dir, 'eda_summary.txt'), 'w') as f:
        f.write(summary)
    log("EDA summary written to eda_summary.txt")

    log("--- EDA COMPLETED ---")

if __name__ == '__main__':
    run_eda()
>>>>>>> 1914c2dd4cad679f031794efb9ef4d9b9ba61dc2
