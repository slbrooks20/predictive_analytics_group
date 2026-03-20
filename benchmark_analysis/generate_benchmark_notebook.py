import json
from pathlib import Path
from textwrap import dedent


def to_source(text: str):
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


cells = [
    markdown_cell(
        """
        # CLI Coding Agent Benchmark Analysis

        This notebook parses the semi-structured benchmark score sheet, normalizes benchmark artifacts from each model directory, and builds a comparison dashboard across quality, reliability, efficiency, and output quality.

        The notebook is intentionally resilient to incomplete team handoff data:

        - Missing plot or metrics files render as placeholders instead of failing.
        - Nested artifact layouts such as `outputs/codex-5.3/outputs/metrics` are normalized automatically.
        - CSV files with merge-conflict markers are cleaned before parsing where possible.
        """
    ),
    code_cell(
        """
        import csv
        import io
        import math
        import re
        import textwrap
        from pathlib import Path

        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display

        sns.set_theme(style="whitegrid", palette="Set2")
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10

        ROOT = Path(".")
        BENCHMARK_CSV = ROOT / "benchmarks_all.csv"
        OUTPUTS_DIR = ROOT / "outputs"

        MODEL_SPECS = [
            {"order": 1, "raw": "Opus 4.6", "short": "Opus 4.6", "cli_tool": "Claude Code", "model_family": "Claude", "slug": "opus-4.6"},
            {"order": 2, "raw": "Claude Haiku 4.5 (Claude Code)", "short": "Haiku 4.5", "cli_tool": "Claude Code", "model_family": "Claude", "slug": "haiku-4.5"},
            {"order": 3, "raw": "Claude Sonnet 4.6 (Claude Code) — Run 2", "short": "Sonnet 4.6", "cli_tool": "Claude Code", "model_family": "Claude", "slug": "sonnet-4.6"},
            {"order": 4, "raw": "gemini-3-flash-preview (Gemini CLI v0.33.2)", "short": "Gemini 3 Flash", "cli_tool": "Gemini CLI", "model_family": "Gemini", "slug": "gemini-3-flash"},
            {"order": 5, "raw": "gemini-3.1-pro-preview (Gemini CLI v0.33.2)", "short": "Gemini 3.1 Pro", "cli_tool": "Gemini CLI", "model_family": "Gemini", "slug": "gemini-3.1-pro"},
            {"order": 6, "raw": "devstral-2 (Mistral Vibe v2.4.2)", "short": "Devstral 2", "cli_tool": "Mistral Vibe", "model_family": "Mistral", "slug": "devstral-2"},
            {"order": 7, "raw": "devstral-small (Mistral Vibe v2.4.2)", "short": "Devstral Small", "cli_tool": "Mistral Vibe", "model_family": "Mistral", "slug": "devstral-small"},
            {"order": 8, "raw": "Codex 5.3", "short": "Codex 5.3", "cli_tool": "Codex", "model_family": "Codex", "slug": "codex-5.3"},
            {"order": 9, "raw": "Codex 5.4", "short": "Codex 5.4", "cli_tool": "Codex", "model_family": "Codex", "slug": "codex-5.4"},
        ]

        MODEL_META_BY_RAW = {spec["raw"]: spec for spec in MODEL_SPECS}
        MODEL_META_BY_SLUG = {spec["slug"]: spec for spec in MODEL_SPECS}
        MODEL_ORDER = [spec["short"] for spec in MODEL_SPECS]
        MODEL_SLUG_ORDER = [spec["slug"] for spec in MODEL_SPECS]

        TASK_ORDER = [
            "Task 1: Data Ingestion",
            "Task 2: EDA",
            "Task 3: Baseline Model",
            "Task 4: Improvements",
        ]
        TASK_CODE_MAP = {
            "Task 1: Data Ingestion": "T1",
            "Task 2: EDA": "T2",
            "Task 3: Baseline Model": "T3",
            "Task 4: Improvements": "T4",
        }

        CLI_COLORS = {
            "Claude Code": "#4C78A8",
            "Gemini CLI": "#59A14F",
            "Mistral Vibe": "#F28E2B",
            "Codex": "#9C6ADE",
        }

        EXPECTED_COLUMNS = [
            "Task",
            "VER (0/1)",
            "SR (0/1)",
            "Spec (X/10)",
            "Spec %",
            "Error Type",
            "Leakage (count)",
            "Metric OK",
            "Split Disc.",
            "Cold-Start",
            "Seed %",
            "Deps",
            "CQ1 (/20)",
            "CQ2 (/20)",
            "CQ3 (/20)",
            "CQ4 (/20)",
            "CQ5 (/20)",
            "Code Total",
            "Prompts",
            "Re-prompts",
            "Self-corrections",
            "Time (min)",
            "Token %",
            "Hallucinations",
            "Fabrications",
            "Safety Issues",
            "Verdict",
            "Key Findings",
        ]

        CQ_COLUMNS = ["CQ1", "CQ2", "CQ3", "CQ4", "CQ5"]
        """
    ),
    code_cell(
        """
        def non_empty_cells(row):
            return [cell for cell in row if str(cell).strip()]


        def is_blank_row(row):
            return len(non_empty_cells(row)) == 0


        def normalize_algorithm_name(value):
            if pd.isna(value):
                return np.nan
            text = str(value).strip()
            text = text.replace("_", " ")
            text = re.sub(r"(?<!^)([A-Z])", r" \\1", text).strip()
            normalized = re.sub(r"\\s+", " ", text)
            canonical = normalized.replace(" ", "")
            mapping = {
                "RF": "Random Forest",
                "RandomForest": "Random Forest",
                "RandomForestRegressor": "Random Forest",
                "GradientBoosting": "Gradient Boosting",
                "GradientBoostingRegressor": "Gradient Boosting",
                "GB": "Gradient Boosting",
                "LinearRegression": "Linear Regression",
                "Linear Regression": "Linear Regression",
                "MLP": "MLP",
                "MLPRegressor": "MLP",
            }
            return mapping.get(canonical, mapping.get(normalized, normalized))


        def shorten(text, width=90):
            if pd.isna(text):
                return ""
            text = str(text).strip()
            if len(text) <= width:
                return text
            return text[: width - 1].rstrip() + "…"


        def sum_or_nan(series):
            valid = pd.Series(series).dropna()
            return valid.sum() if not valid.empty else np.nan


        def max_or_nan(series):
            valid = pd.Series(series).dropna()
            return valid.max() if not valid.empty else np.nan


        def parse_time_to_minutes(value):
            if pd.isna(value):
                return np.nan
            text = str(value).strip()
            if not text:
                return np.nan
            if re.fullmatch(r"\\d+(?:\\.\\d+)?", text):
                return float(text)

            minute_match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(?:m|min|mins|minute|minutes)\\b", text, flags=re.I)
            second_match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(?:s|sec|secs|second|seconds)\\b", text, flags=re.I)
            colon_match = re.fullmatch(r"(\\d+):(\\d+(?:\\.\\d+)?)", text)

            total = 0.0
            matched = False
            if minute_match:
                total += float(minute_match.group(1))
                matched = True
            if second_match:
                total += float(second_match.group(1)) / 60.0
                matched = True
            if matched:
                return total
            if colon_match:
                minutes = float(colon_match.group(1))
                seconds = float(colon_match.group(2))
                return minutes + seconds / 60.0
            return np.nan


        def parse_token_pct(value):
            if pd.isna(value):
                return np.nan
            text = str(value).strip()
            if not text:
                return np.nan
            used_match = re.search(r"(\\d+(?:\\.\\d+)?)%\\s*used", text, flags=re.I)
            if used_match:
                return float(used_match.group(1))
            consumed_match = re.search(r"(\\d+(?:\\.\\d+)?)%\\s*consumed", text, flags=re.I)
            if consumed_match:
                return float(consumed_match.group(1))
            first_match = re.search(r"(\\d+(?:\\.\\d+)?)%", text)
            return float(first_match.group(1)) if first_match else np.nan


        def looks_like_header(row):
            return len(row) >= 2 and row[0].strip() == "Task" and row[1].strip() == "VER (0/1)"


        def looks_like_combined_header(row):
            return len(row) >= 2 and row[0].strip() in MODEL_META_BY_RAW and row[1].strip() == "VER (0/1)"


        def looks_like_model_label(row):
            return row and row[0].strip() in MODEL_META_BY_RAW and len(non_empty_cells(row)) == 1


        def parse_benchmark_csv(path):
            rows = list(csv.reader(open(path, encoding="utf-8-sig", newline="")))
            records = []
            current_model = None
            header = None

            for row in rows:
                row = [(cell or "").strip() for cell in row]
                if is_blank_row(row):
                    continue

                first = row[0]

                if first.startswith("BENCHMARK SCORE SHEET"):
                    continue
                if first.startswith("TOTALS"):
                    continue
                if looks_like_combined_header(row):
                    current_model = first
                    header = ["Task"] + row[1:len(EXPECTED_COLUMNS)]
                    continue
                if looks_like_model_label(row):
                    current_model = first
                    header = None
                    continue
                if looks_like_header(row):
                    header = row[:len(EXPECTED_COLUMNS)]
                    continue
                if first.startswith("Task "):
                    if current_model is None or header is None:
                        raise ValueError(f"Encountered task row before header/model context: {row}")
                    padded = (row + [""] * len(header))[:len(header)]
                    record = dict(zip(header, padded))
                    record["model"] = current_model
                    records.append(record)

            df = pd.DataFrame(records)
            rename_map = {
                "Task": "task",
                "VER (0/1)": "VER",
                "SR (0/1)": "SR",
                "Spec (X/10)": "spec",
                "Spec %": "spec_pct",
                "Error Type": "error_type",
                "Leakage (count)": "leakage_count",
                "Metric OK": "metric_ok",
                "Split Disc.": "split_disc",
                "Cold-Start": "cold_start",
                "Seed %": "seed_pct",
                "Deps": "deps",
                "CQ1 (/20)": "CQ1",
                "CQ2 (/20)": "CQ2",
                "CQ3 (/20)": "CQ3",
                "CQ4 (/20)": "CQ4",
                "CQ5 (/20)": "CQ5",
                "Code Total": "CQ_total",
                "Prompts": "prompts",
                "Re-prompts": "re_prompts",
                "Self-corrections": "self_corrections",
                "Time (min)": "time_raw",
                "Token %": "token_raw",
                "Hallucinations": "hallucinations",
                "Fabrications": "fabrications",
                "Safety Issues": "safety_issues",
                "Verdict": "verdict",
                "Key Findings": "key_findings",
            }
            df = df.rename(columns=rename_map)

            numeric_columns = [
                "VER",
                "SR",
                "spec",
                "spec_pct",
                "leakage_count",
                "CQ1",
                "CQ2",
                "CQ3",
                "CQ4",
                "CQ5",
                "CQ_total",
                "prompts",
                "re_prompts",
                "self_corrections",
                "hallucinations",
                "fabrications",
                "safety_issues",
            ]
            for column in numeric_columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

            df["time_min"] = df["time_raw"].apply(parse_time_to_minutes)
            df["token_pct"] = df["token_raw"].apply(parse_token_pct)
            df["CQ_total"] = df["CQ_total"].fillna(df[CQ_COLUMNS].sum(axis=1, min_count=1))
            df["error_type"] = df["error_type"].replace({"": "N/A"}).fillna("N/A")
            df["verdict"] = df["verdict"].fillna("").str.extract(r"(PASS|FAIL)", expand=False).fillna(df["verdict"])
            df["task_code"] = df["task"].map(TASK_CODE_MAP)
            df["task_short"] = df["task"].str.replace(r"^Task \\d+:\\s*", "", regex=True)

            meta = pd.DataFrame(MODEL_SPECS).rename(columns={"raw": "model"})
            df = df.merge(meta, on="model", how="left")
            df["model_order"] = df["order"]
            df["task_order"] = df["task"].map({task: idx for idx, task in enumerate(TASK_ORDER, start=1)})
            df = df.sort_values(["model_order", "task_order"]).reset_index(drop=True)
            return df


        def first_existing(paths):
            for path in paths:
                if path and Path(path).exists():
                    return Path(path)
            return None


        def get_model_paths(slug):
            base = OUTPUTS_DIR / slug
            nested = base / "outputs"
            return {
                "root": base if base.exists() else None,
                "figures": first_existing([base / "figures", nested / "figures"]),
                "metrics": first_existing([base / "metrics", nested / "metrics"]),
                "docs": first_existing([base / "docs", nested / "docs"]),
            }


        def resolve_git_conflict_text(text):
            if "<<<<<<<" not in text or "=======" not in text or ">>>>>>>" not in text:
                return text
            resolved = []
            state = "normal"
            for line in text.splitlines():
                if line.startswith("<<<<<<<"):
                    state = "keep_first"
                    continue
                if line.startswith("=======") and state == "keep_first":
                    state = "skip_second"
                    continue
                if line.startswith(">>>>>>>") and state == "skip_second":
                    state = "normal"
                    continue
                if state in {"normal", "keep_first"}:
                    resolved.append(line)
            return "\\n".join(resolved)


        def safe_read_csv(path):
            if path is None or not Path(path).exists():
                return pd.DataFrame()
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            cleaned = resolve_git_conflict_text(text).strip()
            if not cleaned:
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(cleaned))


        def safe_read_text(path):
            if path is None or not Path(path).exists():
                return None
            return Path(path).read_text(encoding="utf-8", errors="ignore")


        def clean_metric_dataframe(df):
            if df is None or df.empty:
                return pd.DataFrame()
            frame = df.copy()
            frame = frame.dropna(how="all")
            if "MAE" not in frame.columns and "val_MAE" in frame.columns:
                frame["MAE"] = frame["val_MAE"]
            if "RMSE" not in frame.columns and "val_RMSE" in frame.columns:
                frame["RMSE"] = frame["val_RMSE"]
            if "model" in frame.columns:
                frame = frame[~frame["model"].astype(str).str.startswith(("<<<<<<<", "=======", ">>>>>>>"), na=False)]
                frame["model_norm"] = frame["model"].apply(normalize_algorithm_name)
            if "split" in frame.columns:
                frame["split"] = frame["split"].astype(str).str.strip().str.lower()
            if "feature_set" in frame.columns:
                frame["feature_set"] = frame["feature_set"].astype(str).str.strip()
            for column in ["MAE", "RMSE", "training_time_seconds", "n_estimators"]:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame.reset_index(drop=True)


        def load_metric_csv(slug, filename):
            paths = get_model_paths(slug)
            metrics_dir = paths["metrics"]
            if metrics_dir is None:
                return pd.DataFrame()
            return clean_metric_dataframe(safe_read_csv(metrics_dir / filename))


        def load_doc_text(slug, filename):
            paths = get_model_paths(slug)
            docs_dir = paths["docs"]
            if docs_dir is None:
                return None
            return safe_read_text(docs_dir / filename)


        def build_artifact_inventory():
            rows = []
            for spec in MODEL_SPECS:
                paths = get_model_paths(spec["slug"])
                figures_dir = paths["figures"]
                metrics_dir = paths["metrics"]
                docs_dir = paths["docs"]
                figure_count = len(list(figures_dir.glob("*.png"))) if figures_dir else 0
                metric_count = len(list(metrics_dir.glob("*.csv"))) if metrics_dir else 0
                summary_present = bool(docs_dir and (docs_dir / "eda_summary.txt").exists())
                rows.append(
                    {
                        "Model": spec["short"],
                        "CLI Tool": spec["cli_tool"],
                        "Figures Dir": str(figures_dir) if figures_dir else "missing",
                        "Metrics Dir": str(metrics_dir) if metrics_dir else "missing",
                        "Docs Dir": str(docs_dir) if docs_dir else "missing",
                        "PNG Files": figure_count,
                        "Metric CSVs": metric_count,
                        "EDA Summary": summary_present,
                    }
                )
            return pd.DataFrame(rows)


        def add_bar_labels(ax, fmt="{:.1f}", rotation=0):
            for container in ax.containers:
                labels = []
                for bar in container:
                    height = bar.get_height()
                    if np.isnan(height):
                        labels.append("")
                    else:
                        labels.append(fmt.format(height))
                ax.bar_label(container, labels=labels, padding=3, rotation=rotation, fontsize=9)


        def placeholder_axes(ax, title, message="Missing artifact", facecolor="#f3f3f3", edgecolor="#bdbdbd", textcolor="dimgray"):
            ax.axis("off")
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                fontsize=12,
                color=textcolor,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": facecolor, "edgecolor": edgecolor},
            )




        def resolve_figure_path(slug, candidates):
            figure_dir = get_model_paths(slug)["figures"]
            if figure_dir is None:
                return None
            if isinstance(candidates, str):
                candidates = [candidates]
            for candidate in candidates:
                path = figure_dir / candidate
                if path.exists():
                    return path
            return None


        def compare_plot_grid(candidates, title, grid=(3, 3)):
            rows, cols = grid
            fig, axes = plt.subplots(rows, cols, figsize=(20, 18))
            fig.suptitle(title, fontsize=18, y=0.99)
            axes = np.array(axes).reshape(rows, cols)
            t4_verdict_lookup = {}
            if "benchmark_df" in globals():
                t4_rows = benchmark_df[benchmark_df["task_code"] == "T4"][["slug", "verdict"]].drop_duplicates()
                t4_verdict_lookup = dict(zip(t4_rows["slug"], t4_rows["verdict"]))
            for ax, spec in zip(axes.flat, MODEL_SPECS):
                path = resolve_figure_path(spec["slug"], candidates)
                if path is None:
                    if t4_verdict_lookup.get(spec["slug"]) == "FAIL":
                        placeholder_axes(
                            ax,
                            spec["short"],
                            "TASK FAILED",
                            facecolor="#fde2e4",
                            edgecolor="#d62839",
                            textcolor="#9d0208",
                        )
                    else:
                        placeholder_axes(ax, spec["short"], "Failed to generate")
                    continue
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.set_title(spec["short"], fontsize=11)
                ax.axis("off")
            for ax in axes.flat[len(MODEL_SPECS):]:
                ax.axis("off")
            plt.tight_layout()
            plt.show()


        def build_output_cost_df(benchmark_df, plot_specs):
            rows = []
            for spec in MODEL_SPECS:
                model_rows = benchmark_df[benchmark_df["slug"] == spec["slug"]]
                for plot_spec in plot_specs:
                    task_row = model_rows[model_rows["task_code"] == plot_spec["task_code"]]
                    task_record = task_row.iloc[0] if not task_row.empty else None
                    plot_path = resolve_figure_path(spec["slug"], plot_spec["candidates"])
                    rows.append(
                        {
                            "model": spec["short"],
                            "slug": spec["slug"],
                            "plot_label": plot_spec["label"],
                            "task_code": plot_spec["task_code"],
                            "token_pct": task_record["token_pct"] if task_record is not None else np.nan,
                            "time_min": task_record["time_min"] if task_record is not None else np.nan,
                            "quality_proxy": task_record["CQ_total"] if task_record is not None else np.nan,
                            "verdict": task_record["verdict"] if task_record is not None else np.nan,
                            "has_output": plot_path is not None,
                        }
                    )
            return pd.DataFrame(rows)


        def annotate_staggered_points(ax, df, x_col, y_col, label_col):
            offsets = [(6, 8), (6, -10), (-18, 8), (-18, -10), (10, 14), (10, -14)]
            for idx, (_, row) in enumerate(df.iterrows()):
                dx, dy = offsets[idx % len(offsets)]
                ax.annotate(
                    row[label_col],
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )


        def parse_baseline_metrics_from_text(text):
            result = {"baseline_f0_mae": np.nan, "baseline_f1_mae": np.nan}
            if not text:
                return result
            f0 = re.search(r"LR\\s*F0\\s*val\\s*MAE\\s*=\\s*([0-9.]+)", text, flags=re.I)
            f1 = re.search(r"LR\\s*F1\\s*val\\s*MAE\\s*=\\s*([0-9.]+)", text, flags=re.I)
            if f0:
                result["baseline_f0_mae"] = float(f0.group(1))
            if f1:
                result["baseline_f1_mae"] = float(f1.group(1))
            return result


        def baseline_summary(slug, task3_text=None):
            df = load_metric_csv(slug, "baseline_model_results.csv")
            result = {"baseline_f0_mae": np.nan, "baseline_f1_mae": np.nan, "source": "benchmark text"}
            if not df.empty and {"feature_set", "MAE"}.issubset(df.columns):
                result["source"] = "artifact"
                if "split" in df.columns:
                    validation_rows = df[df["split"].isin(["validation", "val"])]
                    if not validation_rows.empty:
                        df = validation_rows
                for feature in ["F0", "F1"]:
                    rows = df[df["feature_set"] == feature]
                    if not rows.empty:
                        result[f"baseline_{feature.lower()}_mae"] = rows["MAE"].iloc[0]
                return result
            return {**result, **parse_baseline_metrics_from_text(task3_text)}


        def extract_t4_from_benchmark_text(text):
            result = {
                "selected_model": np.nan,
                "selected_feature_set": np.nan,
                "selected_val_mae": np.nan,
                "test_mae": np.nan,
                "best_validation_model": np.nan,
                "best_validation_feature_set": np.nan,
                "best_validation_mae": np.nan,
                "selection_correct": np.nan,
                "notes": shorten(text, 140),
                "source": "benchmark text",
            }
            if not text:
                return result

            incorrect_match = re.search(
                r"Best val model was\\s*(RF|GB|MLP|Ridge|RandomForest|GradientBoosting)\\s*(F\\d)\\s*\\(MAE=?\\s*([0-9.]+)\\).*?but\\s*(RF|GB|MLP|Ridge|RandomForest|GradientBoosting)\\s*(F\\d)\\s*\\(val MAE=?\\s*([0-9.]+)\\)\\s*selected as final",
                text,
                flags=re.I,
            )
            if incorrect_match:
                result["best_validation_model"] = normalize_algorithm_name(incorrect_match.group(1))
                result["best_validation_feature_set"] = incorrect_match.group(2)
                result["best_validation_mae"] = float(incorrect_match.group(3))
                result["selected_model"] = normalize_algorithm_name(incorrect_match.group(4))
                result["selected_feature_set"] = incorrect_match.group(5)
                result["selected_val_mae"] = float(incorrect_match.group(6))
                result["selection_correct"] = False

            best_val_match = re.search(
                r"Best val:\\s*(RF|GB|MLP|Ridge|RandomForest|GradientBoosting|Random Forest|Gradient Boosting)\\s*(F\\d)\\s*MAE=?\\s*([0-9.]+)",
                text,
                flags=re.I,
            )
            if best_val_match:
                model = normalize_algorithm_name(best_val_match.group(1))
                feature = best_val_match.group(2)
                mae = float(best_val_match.group(3))
                result["best_validation_model"] = model
                result["best_validation_feature_set"] = feature
                result["best_validation_mae"] = mae
                if pd.isna(result["selected_model"]):
                    result["selected_model"] = model
                    result["selected_feature_set"] = feature
                    result["selected_val_mae"] = mae

            selected_match = re.search(
                r"(RF|GB|MLP|Ridge|RandomForest|GradientBoosting|Random Forest|Gradient Boosting)\\s*(F\\d)\\s*selected\\s*\\(val\\s*MAE\\s*([0-9.]+)\\)",
                text,
                flags=re.I,
            )
            if selected_match:
                result["selected_model"] = normalize_algorithm_name(selected_match.group(1))
                result["selected_feature_set"] = selected_match.group(2)
                result["selected_val_mae"] = float(selected_match.group(3))

            compact_match = re.search(
                r"(RF|GB|MLP|Ridge|RandomForest|GradientBoosting|Random Forest|Gradient Boosting)\\s*(?:\\([^)]*\\)\\s*)?best(?:\\s+\\w+)?\\s*\\(MAE=?\\s*([0-9.]+)\\s*val,\\s*([0-9.]+)\\s*test\\)",
                text,
                flags=re.I,
            )
            if compact_match:
                model = normalize_algorithm_name(compact_match.group(1))
                result["selected_model"] = model
                result["selected_val_mae"] = float(compact_match.group(2))
                result["test_mae"] = float(compact_match.group(3))
                if pd.isna(result["best_validation_model"]):
                    result["best_validation_model"] = model
                    result["best_validation_mae"] = float(compact_match.group(2))

            final_test_match = re.search(
                r"Final test:\\s*(RF|GB|MLP|Ridge|RandomForest|GradientBoosting|Random Forest|Gradient Boosting)\\s*(F\\d)?\\s*MAE=?\\s*([0-9.]+)",
                text,
                flags=re.I,
            )
            if final_test_match:
                result["selected_model"] = normalize_algorithm_name(final_test_match.group(1))
                if final_test_match.group(2):
                    result["selected_feature_set"] = final_test_match.group(2)
                result["test_mae"] = float(final_test_match.group(3))

            test_only_match = re.search(r"test\\s*MAE\\s*([0-9.]+)", text, flags=re.I)
            if test_only_match and pd.isna(result["test_mae"]):
                result["test_mae"] = float(test_only_match.group(1))

            if "correctly selected" in text.lower():
                result["selection_correct"] = True
            if "selection incorrect" in text.lower() or "type 3" in text.lower():
                result["selection_correct"] = False
            if "crashed" in text.lower() and pd.isna(result["selected_model"]):
                result["selection_correct"] = False

            return result


        def t4_summary_from_artifacts(slug):
            final_df = load_metric_csv(slug, "final_model_results.csv")
            all_df = load_metric_csv(slug, "all_results.csv")
            result = {
                "selected_model": np.nan,
                "selected_feature_set": np.nan,
                "selected_val_mae": np.nan,
                "test_mae": np.nan,
                "best_validation_model": np.nan,
                "best_validation_feature_set": np.nan,
                "best_validation_mae": np.nan,
                "selection_correct": np.nan,
                "notes": "",
                "source": "artifact",
            }

            if not final_df.empty and {"model_norm", "feature_set", "MAE"}.issubset(final_df.columns):
                test_rows = final_df.copy()
                if "split" in test_rows.columns:
                    test_rows = test_rows[test_rows["split"].isin(["test", "testing"])]
                if not test_rows.empty:
                    row = test_rows.iloc[0]
                    result["selected_model"] = row["model_norm"]
                    result["selected_feature_set"] = row.get("feature_set", np.nan)
                    result["test_mae"] = row.get("MAE", np.nan)

            if not all_df.empty and {"model_norm", "feature_set", "MAE"}.issubset(all_df.columns):
                val_rows = all_df.copy()
                if "split" in val_rows.columns:
                    val_rows = val_rows[val_rows["split"].isin(["validation", "val"])]
                if "model_norm" in val_rows.columns:
                    val_rows = val_rows[~val_rows["model_norm"].eq("Linear Regression")]
                if not val_rows.empty:
                    best = val_rows.sort_values("MAE").iloc[0]
                    result["best_validation_model"] = best["model_norm"]
                    result["best_validation_feature_set"] = best.get("feature_set", np.nan)
                    result["best_validation_mae"] = best.get("MAE", np.nan)
                    selected_match = val_rows[
                        (val_rows["model_norm"] == result["selected_model"])
                        & (val_rows["feature_set"] == result["selected_feature_set"])
                    ]
                    if not selected_match.empty:
                        result["selected_val_mae"] = selected_match.sort_values("MAE").iloc[0]["MAE"]
                    if pd.notna(result["selected_model"]):
                        result["selection_correct"] = bool(
                            result["selected_model"] == result["best_validation_model"]
                            and result["selected_feature_set"] == result["best_validation_feature_set"]
                        )
            return result


        def build_t4_quality_table(benchmark_df):
            rows = []
            t4_rows = benchmark_df[benchmark_df["task_code"] == "T4"].set_index("slug")
            for spec in MODEL_SPECS:
                benchmark_row = t4_rows.loc[spec["slug"]] if spec["slug"] in t4_rows.index else None
                artifact_summary = t4_summary_from_artifacts(spec["slug"])
                benchmark_summary = extract_t4_from_benchmark_text(
                    benchmark_row["key_findings"] if benchmark_row is not None else None
                )
                summary = artifact_summary.copy()
                for key, value in benchmark_summary.items():
                    if pd.notna(value) and value != "":
                        summary[key] = value
                if (
                    pd.isna(benchmark_summary["selection_correct"])
                    and any(
                        pd.notna(benchmark_summary[key])
                        for key in ["selected_model", "selected_feature_set", "selected_val_mae", "test_mae"]
                    )
                ):
                    summary["selection_correct"] = np.nan
                if pd.notna(benchmark_summary["best_validation_model"]) and pd.isna(benchmark_summary["best_validation_feature_set"]):
                    summary["best_validation_feature_set"] = np.nan
                summary["source"] = (
                    "benchmark text"
                    if any(
                        pd.notna(benchmark_summary[key])
                        for key in [
                            "selected_model",
                            "selected_feature_set",
                            "selected_val_mae",
                            "test_mae",
                            "best_validation_model",
                            "best_validation_mae",
                            "selection_correct",
                        ]
                    )
                    else artifact_summary["source"]
                )
                if (
                    pd.notna(benchmark_summary["selected_val_mae"])
                    and pd.isna(benchmark_summary["best_validation_mae"])
                    and benchmark_summary["selection_correct"] is not False
                ):
                    summary["best_validation_model"] = summary["selected_model"]
                    summary["best_validation_feature_set"] = summary["selected_feature_set"]
                    summary["best_validation_mae"] = summary["selected_val_mae"]
                    summary["selection_correct"] = True
                rows.append(
                    {
                        "Model": spec["short"],
                        "Selected Model": summary["selected_model"],
                        "Feature Set": summary["selected_feature_set"],
                        "Correct Selection": summary["selection_correct"],
                        "Selected Val MAE": summary["selected_val_mae"],
                        "Test MAE": summary["test_mae"],
                        "Best Validation": (
                            f"{summary['best_validation_model']} {summary['best_validation_feature_set']}"
                            if pd.notna(summary["best_validation_model"])
                            else np.nan
                        ),
                        "Best Val MAE": summary["best_validation_mae"],
                        "Source": summary["source"],
                        "Notes": summary["notes"],
                    }
                )
            return pd.DataFrame(rows)


        def summary_topic_flags(text):
            if not text:
                return {"demand_patterns": False, "rare_regimes": False, "modelling_risks": False, "data_quality": False}
            lowered = text.lower()
            return {
                "demand_patterns": any(term in lowered for term in ["peak", "demand pattern", "rush", "hour", "weekday"]),
                "rare_regimes": any(term in lowered for term in ["rare", "weathersit=4", "heavy rain", "regime"]),
                "modelling_risks": any(term in lowered for term in ["risk", "leakage", "multicollinearity", "heteroscedasticity"]),
                "data_quality": any(term in lowered for term in ["missing", "duplicate", "quality", "clean", "schema"]),
            }


        def pareto_frontier(df, quality_col="avg_CQ", minimize_cols=("total_time_min", "peak_token_pct")):
            frame = df.dropna(subset=[quality_col, *minimize_cols]).reset_index(drop=True)
            frontier = []
            for idx, row in frame.iterrows():
                dominated = False
                for jdx, other in frame.iterrows():
                    if idx == jdx:
                        continue
                    no_worse = (
                        other[quality_col] >= row[quality_col]
                        and all(other[col] <= row[col] for col in minimize_cols)
                    )
                    strictly_better = (
                        other[quality_col] > row[quality_col]
                        or any(other[col] < row[col] for col in minimize_cols)
                    )
                    if no_worse and strictly_better:
                        dominated = True
                        break
                frontier.append(not dominated)
            frame["pareto_frontier"] = frontier
            return frame


        def frontier_curve(df, x_col, y_col, maximize_y=True):
            frame = df.dropna(subset=[x_col, y_col]).sort_values([x_col, y_col], ascending=[True, not maximize_y]).copy()
            if frame.empty:
                return frame
            frontier_rows = []
            best_y = -np.inf if maximize_y else np.inf
            for _, row in frame.iterrows():
                y_value = row[y_col]
                is_better = y_value > best_y if maximize_y else y_value < best_y
                if is_better:
                    frontier_rows.append(row)
                    best_y = y_value
            return pd.DataFrame(frontier_rows)
        """
    ),
    markdown_cell(
        """
        ## 1. Data Loading and Cleaning

        This benchmark starts from messy evidence, not a tidy table. The score sheet is semi-structured, artifact folders are uneven, and some teams handed off more complete outputs than others.

        That makes this first section more than bookkeeping. It establishes what the notebook can trust, what had to be normalized, and where missing files should be interpreted as a real benchmark gap rather than a notebook bug.
        """
    ),
    code_cell(
        """
        benchmark_df = parse_benchmark_csv(BENCHMARK_CSV)
        artifact_inventory = build_artifact_inventory()

        display(Markdown("### Clean Benchmark DataFrame"))
        display(benchmark_df[
            [
                "model",
                "short",
                "task_code",
                "VER",
                "SR",
                "spec",
                "spec_pct",
                "error_type",
                "CQ1",
                "CQ2",
                "CQ3",
                "CQ4",
                "CQ5",
                "CQ_total",
                "re_prompts",
                "self_corrections",
                "time_min",
                "token_pct",
                "verdict",
            ]
        ])

        display(Markdown("### Artifact Inventory"))
        display(artifact_inventory)

        duplicate_pairs = [
            ("Devstral 2", "Codex 5.3"),
            ("Devstral Small", "Codex 5.4"),
        ]
        duplicate_rows = []
        compare_cols = ["task_code", "spec_pct", "CQ_total", "SR", "re_prompts", "self_corrections"]
        for left, right in duplicate_pairs:
            left_df = benchmark_df[benchmark_df["short"] == left][compare_cols].reset_index(drop=True)
            right_df = benchmark_df[benchmark_df["short"] == right][compare_cols].reset_index(drop=True)
            duplicate_rows.append(
                {
                    "Left": left,
                    "Right": right,
                    "Benchmark Rows Identical": left_df.equals(right_df),
                }
            )

        display(Markdown("### Potential Scorecard Duplication Check"))
        display(pd.DataFrame(duplicate_rows))
        """
    ),
    markdown_cell(
        """
        ## 2. Overall Ranking

        This section answers the headline question most readers will ask first: **which models were actually strongest overall once quality, specification compliance, and completion are considered together?**

        The first chart is the cleanest leaderboard in the notebook. It centers average **Code Quality (CQ)**, but the inline labels keep specification compliance and runtime visible so a high score cannot hide an expensive or incomplete run.

        The second chart matters because rankings alone flatten trade-offs. Models closer to the **top-left** are doing the hard thing: staying strong while also staying fast.

        The pass/fail matrix is the guardrail. A model can look impressive on averages and still hide one decisive failure, especially late in the pipeline.
        """
    ),
    code_cell(
        """
        model_summary = (
            benchmark_df.groupby(["short", "cli_tool", "model_family", "model_order"], as_index=False)
            .agg(
                avg_CQ=("CQ_total", "mean"),
                avg_spec=("spec_pct", "mean"),
                avg_SR=("SR", "mean"),
                total_time_min=("time_min", sum_or_nan),
                peak_token_pct=("token_pct", max_or_nan),
                total_re_prompts=("re_prompts", "sum"),
                total_self_corrections=("self_corrections", "sum"),
                overall_verdict=("verdict", lambda s: "PASS" if (s == "PASS").all() else "FAIL"),
            )
            .sort_values(["avg_CQ", "avg_spec", "avg_SR", "total_time_min"], ascending=[False, False, False, True])
            .reset_index(drop=True)
        )

        cq_plot = model_summary.sort_values(["avg_CQ", "avg_spec", "avg_SR", "total_time_min"], ascending=[False, False, False, True]).reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.1, 1.0]})

        cq_y = np.arange(len(cq_plot))
        cq_colors = cq_plot["cli_tool"].map(CLI_COLORS).tolist()
        axes[0].barh(cq_y, cq_plot["avg_CQ"], color=cq_colors)
        axes[0].set_yticks(cq_y, labels=cq_plot["short"])
        axes[0].invert_yaxis()
        axes[0].set_title("Overall Leaderboard: Average Code Quality")
        axes[0].set_xlabel("Average CQ")
        axes[0].set_ylabel("")
        for patch, (_, row) in zip(axes[0].patches, cq_plot.iterrows()):
            label = f"{row['avg_CQ']:.1f} CQ | {row['avg_spec']:.1f}% spec"
            if pd.notna(row["total_time_min"]):
                label += f"\\n{row['total_time_min']:.1f} min"
            axes[0].text(
                patch.get_width() + 0.6,
                patch.get_y() + patch.get_height() / 2,
                label,
                va="center",
                fontsize=8.5,
            )
        axes[0].set_xlim(0, max(118, cq_plot["avg_CQ"].max() + 20))
        axes[0].grid(axis="x", alpha=0.18)

        scatter_df = model_summary.dropna(subset=["total_time_min"]).copy()
        frontier_speed = frontier_curve(scatter_df, "total_time_min", "avg_CQ", maximize_y=True)
        marker_map = {"PASS": "o", "FAIL": "X"}
        for verdict, marker in marker_map.items():
            subset = scatter_df[scatter_df["overall_verdict"] == verdict]
            axes[1].scatter(
                subset["total_time_min"],
                subset["avg_CQ"],
                c=subset["cli_tool"].map(CLI_COLORS),
                s=110,
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.95,
            )
        if len(frontier_speed) >= 2:
            axes[1].plot(frontier_speed["total_time_min"], frontier_speed["avg_CQ"], color="#222222", linestyle="--", linewidth=1.6)
        annotate_staggered_points(axes[1], scatter_df.sort_values("total_time_min"), "total_time_min", "avg_CQ", "short")
        axes[1].set_title("Quality vs Speed Frontier")
        axes[1].set_xlabel("Total Time (min)")
        axes[1].set_ylabel("Average CQ")
        axes[1].grid(alpha=0.22)
        axes[1].set_xlim(left=max(0, scatter_df["total_time_min"].min() - 1))
        axes[1].set_ylim(80, 101)

        cli_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=tool, markerfacecolor=color, markeredgecolor="black", markersize=8)
            for tool, color in CLI_COLORS.items()
        ]
        verdict_handles = [
            plt.Line2D([0], [0], marker=marker, color="#444444", linestyle="None", label=verdict, markersize=8)
            for verdict, marker in marker_map.items()
        ]
        axes[1].legend(handles=cli_handles + verdict_handles, frameon=False, loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.show()

        verdict_matrix = (
            benchmark_df.assign(status=benchmark_df["verdict"].fillna("MISSING"))
            .pivot(index="short", columns="task_code", values="status")
            .reindex(MODEL_ORDER)
            .reindex(columns=["T1", "T2", "T3", "T4"])
        )
        verdict_numeric = verdict_matrix.apply(lambda col: col.map({"FAIL": 0.0, "MISSING": 0.5, "PASS": 1.0}))

        display(Markdown("### Pass / Fail Matrix"))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            verdict_numeric,
            annot=verdict_matrix,
            fmt="",
            cmap=sns.color_palette(["#f8d7da", "#f1f3f5", "#d8f3dc"], as_cmap=True),
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title("Pass / Fail Matrix")
        ax.set_xlabel("Task")
        ax.set_ylabel("Model")
        plt.tight_layout()
        plt.show()

        best_model = cq_plot.iloc[0]
        fastest_model = scatter_df.sort_values("total_time_min").iloc[0]
        fail_models = model_summary.loc[model_summary["overall_verdict"].eq("FAIL"), "short"].tolist()
        missing_time_models = model_summary.loc[model_summary["total_time_min"].isna(), "short"].tolist()
        display(
            Markdown(
                "\\n".join(
                    [
                        "### What This Ranking Says",
                        "",
                        f"- **Top quality:** {best_model['short']} leads on average Code Quality (CQ) at {best_model['avg_CQ']:.1f}.",
                        f"- **Fastest full run:** {fastest_model['short']} finished in {fastest_model['total_time_min']:.2f} minutes.",
                        f"- **Hard failures:** {', '.join(fail_models) if fail_models else 'None'}.",
                        f"- **Missing time data:** {', '.join(missing_time_models) if missing_time_models else 'None'}.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 3. Task-by-Task Comparison

        The benchmark is staged on purpose, so averages only tell part of the story. This section asks: **where does performance actually break down across the workflow?**

        The heatmap is useful for separating two very different behaviors: models that are steady throughout, and models that look strong early but wobble once the work becomes more judgment-heavy.

        The dedicated Task 4 chart is the real stress test. That is the point where the benchmark stops rewarding neat setup work and starts rewarding correct decisions under ambiguity.
        """
    ),
    code_cell(
        """
        heatmap_df = (
            benchmark_df.pivot(index="short", columns="task_code", values="CQ_total")
            .reindex(MODEL_ORDER)
            .reindex(columns=["T1", "T2", "T3", "T4"])
        )
        hardest_tasks = (
            benchmark_df.groupby(["task_code", "task"], as_index=False)["CQ_total"]
            .mean()
            .rename(columns={"CQ_total": "avg_CQ"})
            .sort_values("task_code")
        )
        task4_plot = (
            benchmark_df[benchmark_df["task_code"] == "T4"]
            .sort_values("CQ_total", ascending=True)
            .reset_index(drop=True)
        )

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=axes[0])
        axes[0].set_title("Task Scores by Model")
        axes[0].set_xlabel("Task")
        axes[0].set_ylabel("Model")

        sns.barplot(data=hardest_tasks, x="task_code", y="avg_CQ", color="#4C78A8", ax=axes[1])
        axes[1].set_title("Average Difficulty by Task")
        axes[1].set_xlabel("Task")
        axes[1].set_ylabel("Average CQ")
        add_bar_labels(axes[1], fmt="{:.1f}")
        axes[1].set_ylim(0, 105)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        task4_y = np.arange(len(task4_plot))
        task4_colors = task4_plot["cli_tool"].map(CLI_COLORS).tolist()
        ax.barh(task4_y, task4_plot["CQ_total"], color=task4_colors)
        ax.set_yticks(task4_y, labels=task4_plot["short"])
        ax.invert_yaxis()
        ax.set_title("Task 4 Is the Decision Bottleneck")
        ax.set_xlabel("Task 4 CQ Total")
        ax.set_ylabel("")
        for patch, (_, row) in zip(ax.patches, task4_plot.iterrows()):
            ax.text(
                patch.get_width() + 0.3,
                patch.get_y() + patch.get_height() / 2,
                row["error_type"] if row["error_type"] != "N/A" else row["verdict"],
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        plt.show()

        display(Markdown("### Hardest Tasks by Average Code Quality (CQ)"))
        display(hardest_tasks.sort_values("avg_CQ"))

        hardest_row = hardest_tasks.sort_values("avg_CQ").iloc[0]
        easiest_row = hardest_tasks.sort_values("avg_CQ", ascending=False).iloc[0]
        display(
            Markdown(
                "\\n".join(
                    [
                        "### Story From the Task Breakdown",
                        "",
                        f"- **Hardest task:** {hardest_row['task']} at {hardest_row['avg_CQ']:.1f} average Code Quality (CQ).",
                        f"- **Easiest task:** {easiest_row['task']} at {easiest_row['avg_CQ']:.1f} average Code Quality (CQ).",
                        "- **What this means:** the benchmark is not mainly separating models on setup or plotting. The separation happens when they have to compare options, keep track of validation evidence, and commit to the right final choice.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 4. Code Quality Breakdown

        A single **Code Quality (CQ)** total is useful for ranking, but not for diagnosis. This section opens that total up so the reader can see *why* a model dropped points.

        In practice, this is where the benchmark becomes more revealing. Many models can produce plausible code; fewer are consistently strong in interpretation, evaluation discipline, and comparison logic.

        The weakest-area chart is the shortcut view. It highlights the one category that most limited each model, which is often more informative than the overall average.
        """
    ),
    code_cell(
        """
        cq_breakdown = (
            benchmark_df.groupby(["short", "model_order"], as_index=False)[CQ_COLUMNS]
            .mean()
            .sort_values("model_order")
        )

        weakest_cq = pd.DataFrame(
            {
                "Model": cq_breakdown["short"],
                "Weakest CQ Subcategory": cq_breakdown[CQ_COLUMNS].idxmin(axis=1),
                "Average Score": cq_breakdown[CQ_COLUMNS].min(axis=1),
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.heatmap(
            cq_breakdown.set_index("short")[CQ_COLUMNS],
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=0.5,
            ax=axes[0],
        )
        axes[0].set_title("Where Each Model Drops Points")
        axes[0].set_xlabel("CQ Subcategory")
        axes[0].set_ylabel("Model")

        weakest_plot = weakest_cq.sort_values(["Average Score", "Model"], ascending=[True, True]).reset_index(drop=True)
        weakest_palette = dict(zip(CQ_COLUMNS, sns.color_palette("muted", n_colors=len(CQ_COLUMNS))))
        weakest_y = np.arange(len(weakest_plot))
        weakest_colors = weakest_plot["Weakest CQ Subcategory"].map(weakest_palette).tolist()
        axes[1].barh(weakest_y, weakest_plot["Average Score"], color=weakest_colors)
        axes[1].set_yticks(weakest_y, labels=weakest_plot["Model"])
        axes[1].invert_yaxis()
        axes[1].axvline(20, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Each Model's Weakest CQ Area")
        axes[1].set_xlabel("Average Score in Weakest CQ Subcategory")
        axes[1].set_ylabel("")
        weakest_handles = [
            plt.Line2D([0], [0], marker="s", color=color, linestyle="None", markersize=8, label=label)
            for label, color in weakest_palette.items()
        ]
        axes[1].legend(handles=weakest_handles, title="Weakest Area", bbox_to_anchor=(1.02, 1), loc="upper left")
        for patch, (_, row) in zip(axes[1].patches, weakest_plot.iterrows()):
            axes[1].text(
                patch.get_width() + 0.1,
                patch.get_y() + patch.get_height() / 2,
                f"{row['Average Score']:.2f}".rstrip("0").rstrip("."),
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        plt.show()

        display(Markdown("### Weakest Code Quality (CQ) Area per Model"))
        display(weakest_cq)

        weakest_counts = weakest_cq["Weakest CQ Subcategory"].value_counts()
        display(
            Markdown(
                "\\n".join(
                    [
                        "### Story From the Code Quality (CQ) Breakdown",
                        "",
                        f"- **Most common weakness:** {weakest_counts.index[0]} appears most often as the lowest sub-score.",
                        "- **What this means:** the weaker runs are usually not failing because they cannot write code. They are failing because the code does not stay disciplined when the task requires evaluation judgment, comparison logic, or careful output handling.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 5. Efficiency Analysis

        Strong outputs matter less if they require disproportionate runtime or token budget. This section asks a more practical question: **which models are genuinely efficient, and which ones only look good if cost is ignored?**

        The frontier chart is the anchor. Points on the dashed frontier represent the best visible trade-offs between runtime and quality in this benchmark.

        The ranked bars underneath split the cost story in two. Runtime is summed across tasks, but token usage is shown as the **peak per-task token percentage** for each model. That avoids the misleading move of adding percentages across tasks as if they were cumulative totals.
        """
    ),
    code_cell(
        """
        timed_models = model_summary.dropna(subset=["total_time_min"]).copy()
        token_models = model_summary.dropna(subset=["peak_token_pct"]).copy()
        frontier_df = pareto_frontier(timed_models)
        frontier_models = set(frontier_df.loc[frontier_df["pareto_frontier"], "short"])

        fig, axes = plt.subplots(1, 3, figsize=(21, 7), gridspec_kw={"width_ratios": [1.0, 0.9, 0.9]})

        scatter_df = timed_models.dropna(subset=["avg_CQ", "peak_token_pct"]).copy()
        frontier_speed = frontier_curve(scatter_df, "total_time_min", "avg_CQ", maximize_y=True)
        for verdict, marker in {"PASS": "o", "FAIL": "X"}.items():
            subset = scatter_df[scatter_df["overall_verdict"] == verdict]
            axes[0].scatter(
                subset["total_time_min"],
                subset["avg_CQ"],
                c=subset["cli_tool"].map(CLI_COLORS),
                s=120,
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.95,
            )
        if len(frontier_speed) >= 2:
            axes[0].plot(frontier_speed["total_time_min"], frontier_speed["avg_CQ"], linestyle="--", color="#222222", linewidth=1.6)
        annotate_staggered_points(axes[0], scatter_df.sort_values("total_time_min"), "total_time_min", "avg_CQ", "short")
        axes[0].set_title("Useful Trade-off: Quality vs Runtime")
        axes[0].set_xlabel("Total Time (min)")
        axes[0].set_ylabel("Average CQ")
        axes[0].grid(alpha=0.22)
        axes[0].set_ylim(80, 101)

        time_plot_df = timed_models[["short", "total_time_min", "cli_tool"]].sort_values("total_time_min", ascending=True).reset_index(drop=True)
        time_y = np.arange(len(time_plot_df))
        time_colors = time_plot_df["cli_tool"].map(CLI_COLORS).tolist()
        axes[1].barh(time_y, time_plot_df["total_time_min"], color=time_colors)
        axes[1].set_yticks(time_y, labels=time_plot_df["short"])
        axes[1].invert_yaxis()
        axes[1].set_title("Who Is Actually Fast")
        axes[1].set_xlabel("Total Time (min)")
        axes[1].set_ylabel("")
        for patch, (_, row) in zip(axes[1].patches, time_plot_df.iterrows()):
            axes[1].text(
                patch.get_width() + 0.08,
                patch.get_y() + patch.get_height() / 2,
                f"{row['total_time_min']:.2f} min",
                va="center",
                fontsize=9,
            )
        axes[1].grid(axis="x", alpha=0.18)

        token_plot_df = token_models[["short", "peak_token_pct", "cli_tool"]].sort_values("peak_token_pct", ascending=True).reset_index(drop=True)
        token_y = np.arange(len(token_plot_df))
        token_colors = token_plot_df["cli_tool"].map(CLI_COLORS).tolist()
        axes[2].barh(token_y, token_plot_df["peak_token_pct"], color=token_colors)
        axes[2].set_yticks(token_y, labels=token_plot_df["short"])
        axes[2].invert_yaxis()
        axes[2].set_title("Who Peaks Highest on Token Usage")
        axes[2].set_xlabel("Peak Token % Used")
        axes[2].set_ylabel("")
        for patch, (_, row) in zip(axes[2].patches, token_plot_df.iterrows()):
            axes[2].text(
                patch.get_width() + 0.08,
                patch.get_y() + patch.get_height() / 2,
                f"{row['peak_token_pct']:.1f}%",
                va="center",
                fontsize=9,
            )
        axes[2].grid(axis="x", alpha=0.18)

        plt.tight_layout()
        plt.show()

        display(Markdown("### Models on the Efficiency Frontier"))
        display(pd.DataFrame({"Model": sorted(frontier_models)}))
        missing_time = sorted(set(model_summary["short"]) - set(timed_models["short"]))
        display(
            Markdown(
                "\\n".join(
                    [
                        "### Story From the Efficiency View",
                        "",
                        f"- **Frontier models:** {', '.join(sorted(frontier_models)) if frontier_models else 'None'}.",
                        f"- **Time data still missing:** {', '.join(missing_time) if missing_time else 'None'}.",
                        "- **How to read it:** the best models here are not just the fastest ones. They are the ones that stay near the top of the quality distribution without paying a disproportionate runtime or token penalty.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 6. Reliability Analysis

        Reliability is not the same thing as raw quality. This section asks: **which models stay on track on their own, and which ones only finish once a human steps in?**

        The heatmap shows where human rescue clustered across the pipeline.

        The intervention chart compresses that story into something easier to read: self-correction is a positive signal, while repeated human re-prompts are a warning that the run is fragile.
        """
    ),
    code_cell(
        """
        reprompt_matrix = (
            benchmark_df.pivot_table(index="short", columns="task_code", values="re_prompts", aggfunc="sum")
            .reindex(MODEL_ORDER)
            .reindex(columns=["T1", "T2", "T3", "T4"])
            .fillna(0)
        )

        self_corr_df = model_summary.sort_values("total_self_corrections", ascending=False)
        intervention_df = self_corr_df[["short", "total_re_prompts", "total_self_corrections"]].melt(
            id_vars="short",
            var_name="Intervention Type",
            value_name="Count",
        )
        intervention_df["Intervention Type"] = intervention_df["Intervention Type"].map(
            {
                "total_re_prompts": "Human Re-prompts",
                "total_self_corrections": "Self-corrections",
            }
        )
        intervention_df = intervention_df.groupby(["short", "Intervention Type"], as_index=False)["Count"].sum()
        intervention_df = intervention_df[intervention_df["Count"] > 0]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.heatmap(reprompt_matrix, annot=True, fmt=".0f", cmap="OrRd", cbar=False, linewidths=0.5, ax=axes[0])
        axes[0].set_title("Where Human Re-prompts Were Needed")
        axes[0].set_xlabel("Task")
        axes[0].set_ylabel("Model")

        sns.barplot(
            data=intervention_df,
            y="short",
            x="Count",
            hue="Intervention Type",
            palette={"Human Re-prompts": "#E15759", "Self-corrections": "#4C78A8"},
            ax=axes[1],
        )
        axes[1].set_title("Models That Needed Intervention")
        axes[1].set_xlabel("Count")
        axes[1].set_ylabel("")
        axes[1].legend(title="", loc="lower right")
        axes[1].grid(axis="x", alpha=0.18)
        plt.tight_layout()
        plt.show()

        error_table = (
            benchmark_df.loc[benchmark_df["error_type"].ne("N/A"), ["short", "task_code", "error_type", "key_findings"]]
            .rename(columns={"short": "Model", "task_code": "Task", "error_type": "Error Type", "key_findings": "Notes"})
        )
        if error_table.empty:
            display(Markdown("### Error Types"))
            display(Markdown("No explicit error types were recorded in the parsed benchmark rows."))
        else:
            error_table["Notes"] = error_table["Notes"].apply(lambda value: shorten(value, 120))
            display(Markdown("### Error Types"))
            display(error_table)

        no_reprompt_models = model_summary.loc[model_summary["total_re_prompts"].eq(0), "short"].tolist()
        display(
            Markdown(
                "### Zero-Re-prompt Completion\\n"
                + (" , ".join(no_reprompt_models) if no_reprompt_models else "No model completed all tasks without re-prompts.")
            )
        )
        display(
            Markdown(
                "\\n".join(
                    [
                        "### Story From the Reliability Section",
                        "",
                        "- **Self-correction is rare:** only a small subset of models cleaned up their own mistakes without extra human input.",
                        "- **Why Task 4 matters here too:** once the job shifts from producing code to choosing the final pipeline, models that looked stable earlier start needing rescue.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 7. Output Quality Comparison

        This is the most outcome-oriented section in the notebook. It asks the question that matters most at the end of a long run: **when models reached the final machine learning (ML) stage, did they actually choose and evaluate the right pipeline?**

        The first chart compares the final test **Mean Absolute Error (MAE)** of the pipeline each model actually selected.

        The second chart is there to stop the reader from over-crediting complicated workflows. If a final pipeline barely improves on a simple baseline, that matters.

        The selection-gap chart is the audit view. It catches the especially important case where a model evaluated strong candidates, but still finalized the wrong one.
        """
    ),
    code_cell(
        """
        t4_quality_df = build_t4_quality_table(benchmark_df)
        display(Markdown("### Task 4 Final Model Selection Summary"))
        display(t4_quality_df)

        plot_df = t4_quality_df.dropna(subset=["Test MAE"]).sort_values("Test MAE")
        if plot_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            placeholder_axes(ax, "Task 4 Test MAE", "No final model results available")
            plt.show()
        else:
            plot_order = plot_df["Model"].tolist()
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            sns.barplot(data=plot_df, y="Model", x="Test MAE", order=plot_order, color="#76B7B2", ax=axes[0])
            axes[0].set_title("Final Test MAE by Selected Pipeline")
            axes[0].set_xlabel("Test MAE (lower is better)")
            axes[0].set_ylabel("")
            for patch, (_, row) in zip(axes[0].patches, plot_df.iterrows()):
                axes[0].text(
                    patch.get_width() + 0.12,
                    patch.get_y() + patch.get_height() / 2,
                    f"{row['Test MAE']:.2f}",
                    va="center",
                    fontsize=9,
                )

            baseline_rows = []
            task3_rows = benchmark_df[benchmark_df["task_code"] == "T3"].set_index("slug")
            for spec in MODEL_SPECS:
                task3_text = task3_rows.loc[spec["slug"], "key_findings"] if spec["slug"] in task3_rows.index else None
                baseline = baseline_summary(spec["slug"], task3_text)
                final_row = t4_quality_df.loc[t4_quality_df["Model"] == spec["short"]]
                final_test_mae = final_row["Test MAE"].iloc[0] if not final_row.empty else np.nan
                baseline_candidates = [baseline["baseline_f0_mae"], baseline["baseline_f1_mae"]]
                baseline_valid = [value for value in baseline_candidates if pd.notna(value)]
                baseline_best = min(baseline_valid) if baseline_valid else np.nan
                baseline_rows.append(
                    {
                        "Model": spec["short"],
                        "Best Baseline Validation MAE": baseline_best,
                        "Final Test MAE": final_test_mae,
                    }
                )
            baseline_compare = pd.DataFrame(baseline_rows)
            baseline_compare["Absolute MAE Gain"] = (
                baseline_compare["Best Baseline Validation MAE"] - baseline_compare["Final Test MAE"]
            )
            improvement_plot = baseline_compare.dropna(subset=["Absolute MAE Gain"]).sort_values("Absolute MAE Gain", ascending=False)
            improvement_order = improvement_plot["Model"].tolist()
            sns.barplot(
                data=improvement_plot,
                y="Model",
                x="Absolute MAE Gain",
                order=improvement_order,
                color="#59A14F",
                ax=axes[1],
            )
            axes[1].set_title("How Much Better the Final Pipeline Is Than Baseline")
            axes[1].set_xlabel("Absolute MAE Improvement vs Best LR Baseline")
            axes[1].set_ylabel("")
            for patch, (_, row) in zip(axes[1].patches, improvement_plot.iterrows()):
                axes[1].text(
                    patch.get_width() + 0.12,
                    patch.get_y() + patch.get_height() / 2,
                    f"{row['Absolute MAE Gain']:.1f}",
                    va="center",
                    fontsize=9,
                )
            plt.tight_layout()
            plt.show()

        selection_gap_plot = t4_quality_df.copy()
        selection_gap_plot["Selection Gap"] = (
            selection_gap_plot["Selected Val MAE"] - selection_gap_plot["Best Val MAE"]
        )
        selection_gap_plot["Selection Status"] = selection_gap_plot["Correct Selection"].map(
            {True: "Correct", False: "Wrong / Failed"}
        ).fillna("Unknown")
        selection_gap_sorted = selection_gap_plot.sort_values("Selection Gap", ascending=False, na_position="last")
        selection_gap_order = selection_gap_sorted["Model"].tolist()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=selection_gap_sorted,
            y="Model",
            x="Selection Gap",
            order=selection_gap_order,
            hue="Selection Status",
            palette={"Correct": "#59A14F", "Wrong / Failed": "#E15759", "Unknown": "#9D9D9D"},
            ax=ax,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Model-selection Mistakes in Task 4")
        ax.set_xlabel("Selected Val MAE - Best Val MAE")
        ax.set_ylabel("")
        ax.legend(title="")
        plt.tight_layout()
        plt.show()

        notable_findings = (
            benchmark_df.groupby("short")["key_findings"]
            .apply(lambda s: shorten(" | ".join(s.dropna().astype(str).tolist()), 160))
            .reset_index(name="Notable Findings")
            .rename(columns={"short": "Model"})
        )
        display(Markdown("### Notable Key Findings by Model"))
        display(notable_findings)

        selection_errors = selection_gap_plot.loc[
            selection_gap_plot["Selection Status"].eq("Wrong / Failed") & selection_gap_plot["Selection Gap"].notna(),
            ["Model", "Selection Gap"]
        ]
        display(
            Markdown(
                "\\n".join(
                    [
                        "### Story From Output Quality",
                        "",
                        "- **Task 4 decides the real winner:** clean early-stage work is not enough if the final selection step is wrong, incomplete, or brittle.",
                        f"- **Biggest explicit selection miss:** {selection_errors.sort_values('Selection Gap', ascending=False).iloc[0]['Model']}."
                        if not selection_errors.empty else "- **Explicit selection misses:** none found in the parsed Task 4 outputs.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 8. CLI Tool Comparison

        The models were not all run inside the same toolchain, so this section rolls results up by **Command-Line Interface (CLI)** ecosystem rather than by individual model.

        That makes this a different kind of comparison. Instead of asking which single model won, it asks whether the surrounding CLI environment seems to help or hurt benchmark performance overall.

        This section should be read as directional context, not proof of causality. There are too few runs per ecosystem, and the artifacts were not all produced under equally controlled conditions.
        """
    ),
    code_cell(
        """
        cli_summary = (
            model_summary.groupby("cli_tool", as_index=False)
            .agg(
                avg_CQ=("avg_CQ", "mean"),
                avg_time_min=("total_time_min", "mean"),
                avg_peak_token_pct=("peak_token_pct", "mean"),
            )
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = [
            ("avg_CQ", "Average CQ by CLI Tool", "Average CQ"),
            ("avg_time_min", "Average Total Time by CLI Tool", "Minutes"),
            ("avg_peak_token_pct", "Average Peak Token Usage by CLI Tool", "Peak Token %"),
        ]

        for ax, (column, title, ylabel) in zip(axes, metrics):
            sns.barplot(data=cli_summary, x="cli_tool", y=column, palette=CLI_COLORS, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=25)
            add_bar_labels(ax, fmt="{:.1f}")

        plt.tight_layout()
        plt.show()

        display(cli_summary.sort_values("avg_CQ", ascending=False))
        display(
            Markdown(
                "\\n".join(
                    [
                        "### Story From the Command-Line Interface (CLI) Comparison",
                        "",
                        f"- **Top average quality ecosystems:** {', '.join(cli_summary.sort_values('avg_CQ', ascending=False)['cli_tool'].head(2).tolist())}.",
                        "- **Limitation:** these rollups are descriptive only. The sample is small, the handoff quality is uneven, and the CLI effect is not isolated from the model effect.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 9. Summary Dashboard

        This dashboard is the compressed decision view for a reader who does not want to parse every chart individually.

        It brings ranking, efficiency, intervention burden, and final-task notes into one place so the trade-offs are visible at a glance.

        If the rest of the notebook is the argument, this table is the verdict sheet.
        """
    ),
    code_cell(
        """
        task4_notes = (
            benchmark_df[benchmark_df["task_code"] == "T4"][["short", "key_findings"]]
            .assign(key_findings=lambda df: df["key_findings"].apply(lambda text: shorten(text, 100)))
            .rename(columns={"short": "short", "key_findings": "notable"})
        )

        dashboard = (
            model_summary.merge(task4_notes, on="short", how="left")
            .sort_values(["avg_CQ", "avg_spec", "avg_SR", "total_time_min"], ascending=[False, False, False, True])
            .reset_index(drop=True)
        )
        dashboard.insert(0, "Rank", np.arange(1, len(dashboard) + 1))
        dashboard = dashboard.rename(
            columns={
                "short": "Model",
                "avg_CQ": "Avg CQ",
                "avg_spec": "Avg Spec",
                "total_time_min": "Total Time (min)",
                "peak_token_pct": "Peak Token %",
                "total_re_prompts": "Re-prompts",
                "overall_verdict": "Verdict",
                "notable": "Notable",
            }
        )
        display(dashboard)

        hardest_task = (
            benchmark_df.groupby("task_short")["CQ_total"].mean().sort_values().index[0]
            if not benchmark_df.empty else "Unknown"
        )
        zero_reprompts = model_summary.loc[model_summary["total_re_prompts"].eq(0), "short"].tolist()
        ranked_models = model_summary.sort_values(["avg_CQ", "avg_spec", "avg_SR", "total_time_min"], ascending=[False, False, False, True])
        top_pass = ranked_models.loc[ranked_models["overall_verdict"].eq("PASS")].iloc[0] if not ranked_models.loc[ranked_models["overall_verdict"].eq("PASS")].empty else None
        fastest_timed = ranked_models.loc[ranked_models["total_time_min"].notna()].sort_values("total_time_min").iloc[0] if not ranked_models.loc[ranked_models["total_time_min"].notna()].empty else None
        caution_models = ranked_models.loc[ranked_models["overall_verdict"].eq("FAIL"), "short"].tolist()
        takeaways = "\\n".join(
            [
                "### Key Takeaways",
                "",
                f"- **Hardest stage:** {hardest_task} remains the main failure point, largely because model selection and final evaluation logic are easy to hardcode incorrectly.",
                f"- **Highest-ranked passing model:** {top_pass['short']}." if top_pass is not None else "- **Highest-ranked passing model:** None.",
                f"- **Fastest timed run:** {fastest_timed['short']} at {fastest_timed['total_time_min']:.2f} minutes." if fastest_timed is not None else "- **Fastest timed run:** No time data available.",
                f"- **Needs caution:** {', '.join(caution_models)}." if caution_models else "- **Needs caution:** None.",
                f"- **Zero re-prompts:** {', '.join(zero_reprompts) if zero_reprompts else 'None'}.",
            ]
        )
        display(Markdown(takeaways))
        """
    ),
    markdown_cell(
        """
        ## 10. Conclusions

        The benchmark is not just a ranking of raw quality. It is a benchmark of whether an agent can stay on-spec across a full ML pipeline without introducing silent errors at the end.

        The conclusions here should be read as usage guidance rather than a single winner-take-all verdict. The best choice depends on what failure you are willing to tolerate: slowness, higher token spend, human intervention, or the risk of a bad final decision.
        """
    ),
    code_cell(
        """
        ranked_models = model_summary.sort_values(["avg_CQ", "avg_spec", "avg_SR", "total_time_min"], ascending=[False, False, False, True]).copy()
        passing_models = ranked_models[ranked_models["overall_verdict"].eq("PASS")].copy()
        zero_reprompt_passes = passing_models[passing_models["total_re_prompts"].eq(0)].copy()
        fastest_pass = passing_models.dropna(subset=["total_time_min"]).sort_values("total_time_min").iloc[0] if not passing_models.dropna(subset=["total_time_min"]).empty else None
        safest_pass = passing_models.iloc[0] if not passing_models.empty else None
        caution_models = ranked_models.loc[ranked_models["overall_verdict"].eq("FAIL"), "short"].tolist()
        low_intervention = zero_reprompt_passes["short"].head(3).tolist()

        conclusion_lines = [
            "### Conclusion Notes",
            "",
            f"- **Safest current pick:** {safest_pass['short']} leads the passing models on the benchmark ranking." if safest_pass is not None else "- **Safest current pick:** None.",
            f"- **Fastest passing run with recorded time:** {fastest_pass['short']} at {fastest_pass['total_time_min']:.2f} minutes." if fastest_pass is not None else "- **Fastest passing run with recorded time:** No recorded time data.",
            f"- **Low-intervention options:** {', '.join(low_intervention)}." if low_intervention else "- **Low-intervention options:** None recorded.",
            f"- **Models that still need caution on this benchmark:** {', '.join(caution_models)}." if caution_models else "- **Models that still need caution on this benchmark:** None.",
        ]
        display(Markdown("\\n".join(conclusion_lines)))
        """
    ),
    markdown_cell(
        """
        ## 11. Cost to Produce Specific Outputs

        This section zooms in on a more concrete question: **how much token budget did a model spend to produce a specific deliverable, and what level of task quality came with that spend?**

        The benchmark does not score individual figures directly, so these charts use **task Code Quality (CQ)** as a proxy for the quality of the output bundle produced in that task.

        These plots are meant to be read row by row. Models are sorted by token spend within each output type. The marker shows whether the artifact exists, and the annotation next to it shows the task-level quality signal that accompanied that spend.
        """
    ),
    code_cell(
        """
        output_story_specs = [
            {"label": "EDA heatmap", "task_code": "T2", "candidates": ["heatmap_hour_weekday.png"]},
            {"label": "Baseline diagnostic scatter", "task_code": "T3", "candidates": ["actual_vs_predicted.png", "actual_vs_predicted_F1.png", "actual_vs_predicted_F0.png"]},
            {"label": "Task 4 error-analysis plot", "task_code": "T4", "candidates": ["mae_by_hour.png"]},
            {"label": "Task 4 GB validation curve", "task_code": "T4", "candidates": ["validation_curve_gb.png"]},
        ]

        output_cost_df = build_output_cost_df(benchmark_df, output_story_specs)

        fig, axes = plt.subplots(2, 2, figsize=(18, 11))
        axes = axes.flatten()

        for idx, (ax, plot_spec) in enumerate(zip(axes, output_story_specs)):
            sub = (
                output_cost_df[output_cost_df["plot_label"] == plot_spec["label"]]
                .dropna(subset=["token_pct", "quality_proxy"])
                .sort_values(["token_pct", "quality_proxy", "model"], na_position="last")
                .reset_index(drop=True)
            )
            plot_sub = sub.copy()
            plot_sub["Model"] = plot_sub["model"]
            plot_sub["Status"] = plot_sub["has_output"].map({True: "Output present", False: "Failed to generate"})
            plot_sub = plot_sub.sort_values(["token_pct", "quality_proxy", "Model"], ascending=[True, False, True]).reset_index(drop=True)
            plot_sub["ypos"] = np.arange(len(plot_sub))[::-1]

            for _, row in plot_sub.iterrows():
                ax.hlines(row["ypos"], 0, row["token_pct"], color="#d9d9d9", linewidth=1.1, zorder=1)

            generated = plot_sub[plot_sub["has_output"]]
            missing = plot_sub[~plot_sub["has_output"]]

            if not generated.empty:
                ax.scatter(
                    generated["token_pct"],
                    generated["ypos"],
                    c=generated["quality_proxy"],
                    cmap="viridis",
                    vmin=50,
                    vmax=100,
                    s=95,
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=3,
                    label="Output present",
                )
            if not missing.empty:
                ax.scatter(
                    missing["token_pct"],
                    missing["ypos"],
                    color="#E15759",
                    marker="X",
                    s=110,
                    linewidth=1.8,
                    zorder=4,
                    label="Failed to generate",
                )

            for _, row in plot_sub.iterrows():
                quality_label = f"CQ {row['quality_proxy']:.0f}"
                ax.text(
                    row["token_pct"] + 0.45,
                    row["ypos"],
                    quality_label,
                    va="center",
                    fontsize=8.5,
                    color="#3b3b3b",
                )

            ax.set_title(plot_spec["label"])
            ax.set_xlabel("Token % spent in task")
            ax.set_ylabel("")
            ax.set_yticks(plot_sub["ypos"])
            ax.set_yticklabels(plot_sub["Model"])
            ax.set_xlim(left=0, right=max(8, plot_sub["token_pct"].max() + 6))
            ax.grid(axis="x", alpha=0.22)
            ax.grid(axis="y", alpha=0.08)
            if idx == 0:
                ax.legend(frameon=False, loc="lower right", fontsize=8)

        plt.tight_layout()
        plt.show()

        output_cost_summary = (
            output_cost_df.assign(output_status=output_cost_df["has_output"].map({True: "Generated", False: "Missing"}))
            [["model", "plot_label", "token_pct", "quality_proxy", "output_status"]]
            .rename(columns={"model": "Model", "plot_label": "Output", "token_pct": "Token %", "quality_proxy": "Task CQ Proxy", "output_status": "Output Status"})
        )
        display(output_cost_summary)

        display(
            Markdown(
                "\\n".join(
                    [
                        "### Reading These Output Cost Charts",
                        "",
                        "- **X-axis:** token budget spent on the task that produced the output.",
                        "- **Rows:** models, ordered from lower to higher token spend within that output type.",
                        "- **Marker color / label:** the task Code Quality (CQ) score that came with that spend.",
                        "- **Red X:** the expected output file was missing even though the task still consumed tokens. This flags incomplete delivery, not necessarily a full task failure.",
                    ]
                )
            )
        )
        """
    ),
    markdown_cell(
        """
        ## 12. Output Quality Comparison: Plots Side by Side

        The charts above summarize quality numerically; this section lets the reader inspect the artifacts directly.

        These image grids are especially useful when two models land near each other numerically but feel very different in practice. They let the reader compare clarity, labeling, density, and polish directly.

        The placeholders are intentionally specific:

        - Red tile: `TASK FAILED`. The model failed Task 4 itself, so the missing artifact is part of a broader breakdown in the benchmark run.
        - Gray tile: `Failed to generate`. The model passed Task 4 overall, but this particular artifact was missing, so the failure is narrower than a full task collapse.

        That distinction matters. It separates “the whole final stage broke” from “the run mostly worked, but one required deliverable never materialized.”
        """
    ),
    code_cell(
        """
        plot_specs = [
            (["target_distribution.png"], "Target Distribution Across Models"),
            (["heatmap_hour_weekday.png"], "Hour x Weekday Heatmap Across Models"),
            (["correlation_matrix.png"], "Correlation Matrix Across Models"),
            (["actual_vs_predicted.png", "actual_vs_predicted_F1.png", "actual_vs_predicted_F0.png"], "Actual vs Predicted Across Models"),
            (["residual_distribution.png", "residual_distribution_final.png", "residual_distribution_F1.png", "residual_distribution_F0.png"], "Residual Distribution Across Models"),
            (["validation_curve_gb.png"], "Gradient Boosting Validation Curve Across Models"),
            (["mae_by_hour.png"], "MAE by Hour Across Models"),
            (["rolling_mae_over_time.png"], "Rolling MAE Over Time Across Models"),
        ]

        for candidates, title in plot_specs:
            compare_plot_grid(candidates, title)
        """
    ),
    markdown_cell(
        """
        ## 13. Output Quality Comparison: EDA Summaries

        The **Exploratory Data Analysis (EDA)** summary text is one of the clearest places where “looks complete” and “is actually useful” part ways.

        This section checks two things at once: whether each model covered the required topics, and whether the prose reads like analysis rather than a dump of observations.
        """
    ),
    code_cell(
        """
        summary_rows = []
        for spec in MODEL_SPECS:
            summary_text = load_doc_text(spec["slug"], "eda_summary.txt")
            flags = summary_topic_flags(summary_text)
            summary_rows.append(
                {
                    "Model": spec["short"],
                    "Has Summary File": summary_text is not None,
                    "Demand Patterns": flags["demand_patterns"],
                    "Rare Regimes": flags["rare_regimes"],
                    "Modelling Risks": flags["modelling_risks"],
                    "Data Quality": flags["data_quality"],
                }
            )
        summary_eval = pd.DataFrame(summary_rows)
        display(summary_eval)

        for spec in MODEL_SPECS:
            summary_text = load_doc_text(spec["slug"], "eda_summary.txt")
            preview = "NOT FOUND" if summary_text is None else summary_text[:700].strip()
            display(Markdown(f"### {spec['short']}"))
            display(Markdown(f"```text\\n{preview}\\n```"))
        """
    ),
    markdown_cell(
        """
        ## 14. Output Quality Comparison: Metrics

        The saved metrics files are the closest thing this notebook has to an audit trail. They show what the models actually evaluated, saved, and carried forward.

        This matters because the benchmark notes can describe intent, but the metrics files show what was really recorded on disk, especially around tuning depth and final model selection.

        ## 15. Baseline Metrics Consistency

        Because all agents train on the same benchmark dataset, baseline linear-regression metrics should be nearly identical.

        Any large deviation here is a red flag. It usually means the split logic, preprocessing, or cyclical feature engineering drifted in a way that should not have happened.
        """
    ),
    code_cell(
        """
        metric_summary_rows = []
        baseline_rows = []

        t3_lookup = benchmark_df[benchmark_df["task_code"] == "T3"].set_index("slug")

        for spec in MODEL_SPECS:
            all_results = load_metric_csv(spec["slug"], "all_results.csv")
            tuning_results = load_metric_csv(spec["slug"], "tuning_results.csv")
            final_results = load_metric_csv(spec["slug"], "final_model_results.csv")

            validation_rows = all_results[all_results["split"].eq("validation")] if "split" in all_results.columns else all_results
            selected_model = np.nan
            if not validation_rows.empty and {"model_norm", "MAE"}.issubset(validation_rows.columns):
                best_row = validation_rows.sort_values("MAE").iloc[0]
                selected_model = f"{best_row['model_norm']} {best_row.get('feature_set', '')}".strip()

            final_test_mae = np.nan
            if not final_results.empty and "MAE" in final_results.columns:
                final_test_mae = final_results["MAE"].dropna().iloc[0] if not final_results["MAE"].dropna().empty else np.nan

            metric_summary_rows.append(
                {
                    "Model": spec["short"],
                    "Selected From all_results.csv": selected_model,
                    "Final Test MAE": final_test_mae,
                    "Model x Feature Rows": len(validation_rows) if not validation_rows.empty else 0,
                    "Tuning Rows": len(tuning_results) if not tuning_results.empty else 0,
                }
            )

            task3_text = t3_lookup.loc[spec["slug"], "key_findings"] if spec["slug"] in t3_lookup.index else None
            baseline = baseline_summary(spec["slug"], task3_text)
            baseline_rows.append(
                {
                    "Model": spec["short"],
                    "LR F0 Val MAE": baseline["baseline_f0_mae"],
                    "LR F1 Val MAE": baseline["baseline_f1_mae"],
                    "F0 Delta vs 155.29": baseline["baseline_f0_mae"] - 155.29 if pd.notna(baseline["baseline_f0_mae"]) else np.nan,
                    "F1 Delta vs 137.55": baseline["baseline_f1_mae"] - 137.55 if pd.notna(baseline["baseline_f1_mae"]) else np.nan,
                    "Source": baseline["source"],
                }
            )

        metric_summary_df = pd.DataFrame(metric_summary_rows)
        baseline_consistency_df = pd.DataFrame(baseline_rows)

        display(Markdown("### Metric File Comparison"))
        display(metric_summary_df)

        mae_plot_df = metric_summary_df.dropna(subset=["Final Test MAE"]).sort_values("Final Test MAE")
        if not mae_plot_df.empty:
            mae_plot_order = mae_plot_df["Model"].tolist()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=mae_plot_df, y="Model", x="Final Test MAE", order=mae_plot_order, hue="Model", palette="Set2", legend=False, ax=ax)
            ax.set_title("Final Test MAE from final_model_results.csv")
            ax.set_xlabel("Final Test MAE")
            ax.set_ylabel("")
            for patch, (_, row) in zip(ax.patches, mae_plot_df.iterrows()):
                ax.text(
                    patch.get_width() + 0.12,
                    patch.get_y() + patch.get_height() / 2,
                    f"{row['Final Test MAE']:.2f}",
                    va="center",
                    fontsize=9,
                )
            plt.tight_layout()
            plt.show()

        metric_leader = metric_summary_df.dropna(subset=["Final Test MAE"]).sort_values("Final Test MAE").head(1)
        if not metric_leader.empty:
            leader_row = metric_leader.iloc[0]
            display(
                Markdown(
                    "\\n".join(
                        [
                            "### What the Metric Files Show",
                            "",
                            f"- **Best recorded final test Mean Absolute Error (MAE):** {leader_row['Model']} at {leader_row['Final Test MAE']:.2f}.",
                            "- **How to read this table:** `all_results.csv` shows which validation candidates were actually recorded, while `final_model_results.csv` shows what each run ultimately shipped as the final pipeline.",
                            "- **Why this matters:** the metrics files are the hard evidence layer. When they disagree with a narrative summary, the files should usually be trusted first.",
                        ]
                    )
                )
            )

        display(Markdown("### Baseline Consistency Check"))
        display(baseline_consistency_df)

        baseline_eval = baseline_consistency_df.copy()
        baseline_eval["abs_f0_delta"] = baseline_eval["F0 Delta vs 155.29"].abs()
        baseline_eval["abs_f1_delta"] = baseline_eval["F1 Delta vs 137.55"].abs()
        clustered_models = baseline_eval[
            baseline_eval["abs_f0_delta"].fillna(np.inf).lt(0.01)
            & baseline_eval["abs_f1_delta"].fillna(np.inf).lt(0.01)
        ]["Model"].tolist()
        f1_outliers = baseline_eval.sort_values("abs_f1_delta", ascending=False)
        biggest_f1_outlier = f1_outliers.iloc[0] if not f1_outliers.empty else None
        second_f1_outlier = f1_outliers.iloc[1] if len(f1_outliers) > 1 else None

        baseline_notes = [
            "### How to Read the Baseline Consistency Table",
            "",
            "- **Expected pattern:** because every model is solving the same baseline task on the same dataset, these validation scores should be nearly identical.",
            f"- **Tight consistency cluster:** {', '.join(clustered_models)}." if clustered_models else "- **Tight consistency cluster:** none.",
        ]
        if biggest_f1_outlier is not None:
            baseline_notes.append(
                f"- **Largest deviation:** {biggest_f1_outlier['Model']} is furthest from the shared baseline on the F1 metric ({biggest_f1_outlier['F1 Delta vs 137.55']:+.3f})."
            )
        if second_f1_outlier is not None and second_f1_outlier["abs_f1_delta"] > 0.05:
            baseline_notes.append(
                f"- **Next notable deviation:** {second_f1_outlier['Model']} is the next largest offset ({second_f1_outlier['F1 Delta vs 137.55']:+.3f})."
            )
        baseline_notes.extend(
            [
                "- **Interpretation:** small deviations near zero are normal rounding noise. Larger gaps usually point to changed split logic, preprocessing drift, or feature-engineering mistakes.",
                "- **Important caveat:** Opus 4.6 shows a dramatically different baseline artifact. That is too large to treat as normal variation, so it should be read as a non-comparable baseline output or overwritten artifact rather than a genuine apples-to-apples baseline result.",
            ]
        )
        display(Markdown("\\n".join(baseline_notes)))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path("benchmark_analysis") / "benchmark_analysis.ipynb"
output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {output_path}")
