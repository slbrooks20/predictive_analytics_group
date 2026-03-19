from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def ensure_directories(project_root: Path) -> dict[str, Path]:
    outputs_root = project_root / "outputs"
    paths = {
        "figures": outputs_root / "figures",
        "metrics": outputs_root / "metrics",
        "docs": outputs_root / "docs",
        "benchmark": outputs_root / "benchmark",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def detect_count_like_fields(df: pd.DataFrame) -> list[str]:
    count_like = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        is_integer_like = np.all(np.isclose(series.values, np.round(series.values)))
        if is_integer_like:
            count_like.append(col)
    return count_like


def run_pipeline() -> None:
    np.random.seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "dataset" / "hour.csv"
    output_paths = ensure_directories(project_root)

    log_path = output_paths["benchmark"] / "experiment_log.txt"
    log_path.write_text("", encoding="utf-8")
    write_log(log_path, "Started ingestion and validation pipeline.")
    write_log(log_path, f"Random seed set to {RANDOM_SEED}.")

    df = pd.read_csv(data_path)
    write_log(log_path, f"Loaded dataset from {data_path}.")
    write_log(log_path, f"Dataset shape: {df.shape}.")

    print(f"Dataset shape: {df.shape}")
    print("\nColumn names and data types:")
    print(df.dtypes.to_string())

    write_log(log_path, "Printed schema (column names and dtypes).")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    print(f"\nNumeric variables ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical variables ({len(categorical_cols)}): {categorical_cols}")

    write_log(log_path, f"Identified numeric variables: {numeric_cols}")
    write_log(log_path, f"Identified categorical variables: {categorical_cols}")

    missing_per_col = df.isna().sum()
    missing_total = int(missing_per_col.sum())
    duplicate_rows = int(df.duplicated().sum())

    print("\nMissing values per column:")
    print(missing_per_col.to_string())
    print(f"\nDuplicate rows: {duplicate_rows}")

    write_log(log_path, f"Computed missing values per column. Total missing: {missing_total}")
    write_log(log_path, f"Duplicate rows detected: {duplicate_rows}")

    leakage_detected = False
    leakage_columns_removed = ""
    if {"casual", "registered", "cnt"}.issubset(df.columns):
        leakage_detected = bool((df["casual"] + df["registered"] == df["cnt"]).all())
        write_log(log_path, f"Leakage identity check (casual + registered == cnt): {leakage_detected}")
        if leakage_detected:
            df = df.drop(columns=["casual", "registered"])
            leakage_columns_removed = "casual,registered"
            write_log(log_path, "Removed leakage columns: casual, registered.")

    corr_map = {}
    for col in df.columns:
        if col == "cnt":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            corr_map[col] = float(df[col].corr(df["cnt"]))

    corr_df = pd.DataFrame(
        {"feature": list(corr_map.keys()), "correlation_with_cnt": list(corr_map.values())}
    )
    corr_df["abs_correlation_with_cnt"] = corr_df["correlation_with_cnt"].abs()
    corr_df = corr_df.sort_values("abs_correlation_with_cnt", ascending=False)

    corr_output_path = output_paths["metrics"] / "feature_target_correlations.csv"
    corr_df.to_csv(corr_output_path, index=False)
    write_log(log_path, f"Saved feature-target correlations to {corr_output_path}.")

    high_corr_features = corr_df.loc[corr_df["abs_correlation_with_cnt"] > 0.95, "feature"].tolist()
    max_feature_target_corr = float(corr_df["abs_correlation_with_cnt"].max()) if not corr_df.empty else np.nan
    write_log(log_path, f"Features with |corr| > 0.95: {high_corr_features}")

    target_nonnegative_check = bool((df["cnt"] >= 0).all())
    write_log(log_path, f"Target non-negative check (cnt >= 0): {target_nonnegative_check}")

    count_like_fields = detect_count_like_fields(df)
    impossible_count_like = {}
    for col in count_like_fields:
        negative_count = int((df[col] < 0).sum())
        if negative_count > 0:
            impossible_count_like[col] = negative_count
    impossible_count_like_issue_count = int(sum(impossible_count_like.values()))
    write_log(log_path, f"Count-like fields checked: {count_like_fields}")
    write_log(log_path, f"Impossible values in count-like fields (negative counts): {impossible_count_like}")

    cleaned_data_path = project_root / "outputs" / "cleaned_data.csv"
    df.to_csv(cleaned_data_path, index=False)
    write_log(log_path, f"Saved cleaned dataset to {cleaned_data_path}.")

    validation_rows = [
        {
            "check": "missing_values_total",
            "value": missing_total,
            "details": "Total missing values across all columns",
        },
        {
            "check": "duplicate_rows",
            "value": duplicate_rows,
            "details": "Number of fully duplicated rows",
        },
        {
            "check": "leakage_identity_detected",
            "value": leakage_detected,
            "details": "Whether casual + registered equals cnt for all rows",
        },
        {
            "check": "leakage_columns_removed",
            "value": leakage_columns_removed,
            "details": "Columns removed due to leakage rule",
        },
        {
            "check": "max_feature_target_corr",
            "value": max_feature_target_corr,
            "details": "Maximum absolute correlation between features and cnt",
        },
        {
            "check": "target_nonnegative_check",
            "value": target_nonnegative_check,
            "details": "All cnt values are non-negative",
        },
        {
            "check": "impossible_count_like_values",
            "value": impossible_count_like_issue_count,
            "details": f"Total negative entries across count-like fields: {impossible_count_like}",
        },
        {
            "check": "high_correlation_features_gt_0_95",
            "value": ",".join(high_corr_features),
            "details": "Features with absolute correlation greater than 0.95",
        },
    ]
    validation_report = pd.DataFrame(validation_rows)
    validation_report_path = output_paths["benchmark"] / "data_validation_report.csv"
    validation_report.to_csv(validation_report_path, index=False)
    write_log(log_path, f"Saved validation report to {validation_report_path}.")

    summary = [
        "Summary of findings:",
        f"- Shape: {df.shape}",
        f"- Total missing values: {missing_total}",
        f"- Duplicate rows: {duplicate_rows}",
        f"- Leakage detected: {leakage_detected}",
        f"- Leakage columns removed: {leakage_columns_removed or 'None'}",
        f"- Max |feature-cnt correlation|: {max_feature_target_corr:.4f}" if not np.isnan(max_feature_target_corr) else "- Max |feature-cnt correlation|: NaN",
        f"- cnt non-negative check: {target_nonnegative_check}",
        f"- Impossible count-like values detected: {impossible_count_like_issue_count}",
    ]
    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    doc_path = output_paths["docs"] / "data_ingestion_summary.md"
    doc_path.write_text(summary_text + "\n", encoding="utf-8")
    write_log(log_path, f"Saved summary documentation to {doc_path}.")
    write_log(log_path, "Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
