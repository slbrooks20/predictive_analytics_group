#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PROJECT="$ROOT_DIR/Project"
TEMPLATE_ROOT="$ROOT_DIR/experiment_rerun_template"

MODELS=(
  "Claude_Haiku_4.5"
  "Claude_Sonnet_4.6"
  "Codex-5.3"
  "Codex-5.4"
  "Opus-4.6"
  "devstral-2"
  "devstral-small"
  "gemini-3.1-pro"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
fi

if [[ ! -f "$SOURCE_PROJECT/dataset/hour.csv" ]]; then
  echo "Error: source dataset not found at $SOURCE_PROJECT/dataset/hour.csv" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  model_root="$TEMPLATE_ROOT/$model"
  project_root="$model_root/project"
  mkdir -p "$model_root"

  mkdir -p \
    "$project_root/dataset" \
    "$project_root/scripts" \
    "$project_root/outputs/figures" \
    "$project_root/outputs/metrics" \
    "$project_root/outputs/models" \
    "$project_root/outputs/docs" \
    "$project_root/outputs/benchmark"

  cp "$SOURCE_PROJECT/dataset/hour.csv" "$project_root/dataset/hour.csv"
  cp "$ROOT_DIR/requirements.txt" "$project_root/requirements.txt"
  cp "$SOURCE_PROJECT/run_pipeline.py" "$project_root/run_pipeline.py"

  echo "Prepared $project_root"
done

echo
echo "Rerun template bootstrap complete."
echo "Next steps:"
echo "1. cd into experiment_rerun_template/<model>/project"
echo "2. launch the matching CLI"
echo "3. paste the task prompts in order"
