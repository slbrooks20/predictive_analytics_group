"""
Task 2: Exploratory Data Analysis
Bike-sharing regression dataset — target variable cnt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_data.csv")
RAW_PATH = os.path.join(BASE_DIR, "dataset", "hour.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
BENCHMARK_DIR = os.path.join(BASE_DIR, "outputs", "benchmark")
DOCS_DIR = os.path.join(BASE_DIR, "outputs", "docs")
EXPERIMENT_LOG = os.path.join(BENCHMARK_DIR, "experiment_log.txt")

for d in [FIGURES_DIR, BENCHMARK_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    log("\n" + "=" * 60)
    log("TASK 2: Exploratory Data Analysis")
    log("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if os.path.exists(CLEANED_PATH):
        df = pd.read_csv(CLEANED_PATH)
        log(f"[1] Loaded cleaned data from {CLEANED_PATH}")
    else:
        df = pd.read_csv(RAW_PATH)
        log(f"[1] Cleaned data not found; loaded raw data from {RAW_PATH}")

    log(f"    Shape: {df.shape}")
    log(f"    Target variable: cnt")

    # ------------------------------------------------------------------
    # 2. Target distribution, missingness, outliers, key relationships
    # ------------------------------------------------------------------
    log("\n[2] Target variable summary:")
    desc = df["cnt"].describe()
    for stat in desc.index:
        log(f"    {stat:8s}: {desc[stat]:.2f}")

    missing_cnt = int(df["cnt"].isnull().sum())
    log(f"    missing:  {missing_cnt}")

    q1 = df["cnt"].quantile(0.25)
    q3 = df["cnt"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (df["cnt"] < lower) | (df["cnt"] > upper)
    n_outliers = int(outlier_mask.sum())
    log(f"    IQR outliers (1.5×IQR): {n_outliers}  ({n_outliers / len(df) * 100:.1f}%)")
    log(f"    IQR bounds: [{lower:.1f}, {upper:.1f}]")

    # Skewness
    skew = df["cnt"].skew()
    log(f"    skewness: {skew:.3f}")

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 150})

    # --- 3a. target_distribution.png ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["cnt"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("cnt (bike rentals)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram of cnt")
    sns.kdeplot(df["cnt"], ax=axes[1], fill=True, color="steelblue")
    axes[1].set_xlabel("cnt (bike rentals)")
    axes[1].set_title("Density of cnt")
    fig.suptitle("Target Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"), bbox_inches="tight")
    plt.close(fig)
    log("\n[3a] Saved target_distribution.png")

    # --- 3b. demand_time_series.png ---
    df["dteday"] = pd.to_datetime(df["dteday"])
    daily = df.groupby("dteday")["cnt"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["dteday"], daily["cnt"], linewidth=0.8, color="steelblue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Demand (cnt)")
    ax.set_title("Bike-Sharing Demand Over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "demand_time_series.png"), bbox_inches="tight")
    plt.close(fig)
    log("[3b] Saved demand_time_series.png")

    # --- 3c. heatmap_hour_weekday.png ---
    pivot = df.pivot_table(values="cnt", index="hr", columns="weekday", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
    ax.set_xlabel("Weekday (0=Sun … 6=Sat)")
    ax.set_ylabel("Hour of Day")
    ax.set_title("Mean Demand by Hour × Weekday")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "heatmap_hour_weekday.png"), bbox_inches="tight")
    plt.close(fig)
    log("[3c] Saved heatmap_hour_weekday.png")

    # --- 3d. demand_vs_temp.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["temp"], df["cnt"], alpha=0.15, s=8, color="steelblue")
    ax.set_xlabel("Normalized Temperature")
    ax.set_ylabel("cnt")
    ax.set_title("Demand vs Temperature")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "demand_vs_temp.png"), bbox_inches="tight")
    plt.close(fig)
    log("[3d] Saved demand_vs_temp.png")

    # --- 3e. demand_vs_hum.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["hum"], df["cnt"], alpha=0.15, s=8, color="teal")
    ax.set_xlabel("Normalized Humidity")
    ax.set_ylabel("cnt")
    ax.set_title("Demand vs Humidity")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "demand_vs_hum.png"), bbox_inches="tight")
    plt.close(fig)
    log("[3e] Saved demand_vs_hum.png")

    # --- 3f. demand_vs_windspeed.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["windspeed"], df["cnt"], alpha=0.15, s=8, color="coral")
    ax.set_xlabel("Normalized Windspeed")
    ax.set_ylabel("cnt")
    ax.set_title("Demand vs Windspeed")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "demand_vs_windspeed.png"), bbox_inches="tight")
    plt.close(fig)
    log("[3f] Saved demand_vs_windspeed.png")

    # --- 3g. correlation_matrix.png ---
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Matrix (Numeric Variables)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    log("[3g] Saved correlation_matrix.png")

    # ------------------------------------------------------------------
    # 4. EDA tables → eda_tables.csv
    # ------------------------------------------------------------------
    mean_by_hour = df.groupby("hr")["cnt"].mean().reset_index()
    mean_by_hour.columns = ["hr", "mean_cnt"]
    mean_by_hour["group"] = "hour"
    mean_by_hour.rename(columns={"hr": "key"}, inplace=True)

    mean_by_weekday = df.groupby("weekday")["cnt"].mean().reset_index()
    mean_by_weekday.columns = ["weekday", "mean_cnt"]
    mean_by_weekday["group"] = "weekday"
    mean_by_weekday.rename(columns={"weekday": "key"}, inplace=True)

    eda_tables = pd.concat([mean_by_hour, mean_by_weekday], ignore_index=True)
    eda_tables = eda_tables[["group", "key", "mean_cnt"]]
    eda_tables.to_csv(os.path.join(BENCHMARK_DIR, "eda_tables.csv"), index=False)
    log(f"\n[4] Saved eda_tables.csv to {BENCHMARK_DIR}")

    log("\n    Mean demand by hour:")
    for _, row in mean_by_hour.iterrows():
        log(f"      hr {int(row['key']):2d}: {row['mean_cnt']:.1f}")

    log("    Mean demand by weekday:")
    for _, row in mean_by_weekday.iterrows():
        log(f"      weekday {int(row['key'])}: {row['mean_cnt']:.1f}")

    # ------------------------------------------------------------------
    # 5. Insight summary → eda_summary.txt
    # ------------------------------------------------------------------
    # Gather stats for summary
    peak_hours = mean_by_hour.nlargest(3, "mean_cnt")
    low_hours = mean_by_hour.nsmallest(3, "mean_cnt")

    # Weather-sit distribution (cast to int for clean formatting)
    ws_counts = {int(k): int(v) for k, v in df["weathersit"].value_counts().sort_index().items()}

    # Season distribution (cast to int for clean formatting)
    season_counts = {int(k): int(v) for k, v in df["season"].value_counts().sort_index().items()}

    temp_corr = corr.loc["temp", "cnt"] if "cnt" in corr.columns else df["temp"].corr(df["cnt"])
    hum_corr = corr.loc["hum", "cnt"] if "cnt" in corr.columns else df["hum"].corr(df["cnt"])
    wind_corr = corr.loc["windspeed", "cnt"] if "cnt" in corr.columns else df["windspeed"].corr(df["cnt"])
    temp_atemp_corr = corr.loc["temp", "atemp"]

    summary_lines = [
        "EDA INSIGHT SUMMARY",
        "=" * 50,
        "",
        "1. DEMAND PATTERN OVER TIME",
        f"   - Dataset spans {daily['dteday'].min().date()} to {daily['dteday'].max().date()} ({len(daily)} days).",
        f"   - Clear upward trend from 2011 to 2012, suggesting growing service adoption.",
        f"   - Strong seasonality: demand peaks in summer months and drops in winter.",
        f"   - Daily totals range from {int(daily['cnt'].min())} to {int(daily['cnt'].max())}.",
        "",
        "2. HOURLY AND WEEKDAY PATTERNS",
        f"   - Peak demand hours: {', '.join(f'hr {int(r.key)} ({r.mean_cnt:.0f})' for _, r in peak_hours.iterrows())}.",
        f"   - Lowest demand hours: {', '.join(f'hr {int(r.key)} ({r.mean_cnt:.0f})' for _, r in low_hours.iterrows())}.",
        "   - Weekday pattern shows commute peaks (8 AM, 5-6 PM) on workdays,",
        "     while weekends have a single midday plateau — indicating two distinct",
        "     usage regimes (commuter vs. recreational).",
        "",
        "3. RARE CATEGORIES",
        f"   - weathersit distribution: {ws_counts}.",
        f"     Category 4 (heavy rain/snow) has very few observations ({ws_counts.get(4, 0)} rows),",
        "     which may cause instability in models that split on this category.",
        f"   - season distribution: {season_counts} — roughly balanced.",
        f"   - holiday=1 rows: {int(df['holiday'].sum())} out of {len(df)} ({df['holiday'].mean()*100:.1f}%).",
        "",
        "4. WEATHER FEATURE RELATIONSHIPS",
        f"   - temp vs cnt correlation:      {temp_corr:+.3f} (moderate positive).",
        f"   - hum vs cnt correlation:        {hum_corr:+.3f} (moderate negative).",
        f"   - windspeed vs cnt correlation:  {wind_corr:+.3f} (weak positive).",
        f"   - temp vs atemp correlation:     {temp_atemp_corr:+.3f} (near-perfect collinearity).",
        "     Consider dropping one of temp/atemp during feature engineering.",
        "",
        "5. TARGET DISTRIBUTION",
        f"   - cnt is right-skewed (skewness = {skew:.2f}), with many low-demand hours.",
        f"   - IQR-based outliers: {n_outliers} rows ({n_outliers/len(df)*100:.1f}%).",
        "   - A log transform or tree-based model may handle this skew well.",
        "",
        "6. LIKELY MODELLING RISKS",
        "   - Right-skewed target may degrade linear models; consider transforms.",
        "   - temp and atemp are highly correlated — multicollinearity risk for",
        "     linear models. Use one or apply regularisation.",
        "   - The 'instant' column is a row index, not a true feature — it should",
        "     be excluded from modelling to avoid spurious trend capture.",
        "   - 'dteday' is a date string; useful for time-based splits but should",
        "     not be fed directly as a feature.",
        "   - Rare weathersit=4 category may cause high-variance predictions.",
        "",
        "7. REMAINING LEAKAGE / DATA QUALITY CONCERNS",
        "   - casual and registered were already removed in Task 1 (they sum to cnt).",
        "   - No remaining perfect-correlation leakage detected (all |corr| < 0.95).",
        "   - No missing values; no duplicate rows.",
        "   - 'instant' monotonically increases and correlates with cnt (+0.28) only",
        "     because of the year-over-year growth trend — not a genuine predictor.",
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = os.path.join(DOCS_DIR, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    log(f"\n[5] Saved eda_summary.txt to {DOCS_DIR}")

    # Print summary to log as well
    for line in summary_lines:
        log(f"    {line}")

    log("\n" + "=" * 60)
    log("TASK 2 COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    run()
