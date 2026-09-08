#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "The project environment does not exist yet."
  echo "Run ${PROJECT_ROOT}/scripts/setup_environment.sh first."
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "[1/5] Rebuilding the midfield analysis and three figures..."
"${PYTHON_PATH}" src/analyse_midfield.py

echo "[2/5] Running automated data and statistical tests..."
"${PYTHON_PATH}" -m unittest discover -s tests -v

echo "[3/5] Executing the complete notebook from a clean kernel..."
PATH="${PROJECT_ROOT}/.venv/bin:${PATH}" "${PROJECT_ROOT}/.venv/bin/jupyter-nbconvert" \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=300 \
  notebooks/objective1_midfield_analysis.ipynb

echo "[4/5] Checking the notebook for saved execution errors..."
"${PYTHON_PATH}" - <<'PY'
import json
from pathlib import Path

notebook = json.loads(Path("notebooks/objective1_midfield_analysis.ipynb").read_text())
errors = [
    output
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
if errors:
    raise SystemExit(f"Notebook contains {len(errors)} saved execution error(s)")
print("Notebook executed successfully with no saved errors.")
PY

echo "[5/5] Checking all required deliverables..."
required_outputs=(
  "data/processed/midfield_population.csv"
  "data/processed/midfield_sample_seed2026.csv"
  "data/processed/descriptive_statistics.csv"
  "data/processed/inferential_results.csv"
  "data/processed/assumption_tests.csv"
  "data/processed/data_quality_summary.csv"
  "data/processed/sample_design.csv"
  "data/processed/analysis_summary.json"
  "figures/01_sample_design.png"
  "figures/02_tackles_distribution.png"
  "figures/03_mean_difference_ci.png"
)

for output in "${required_outputs[@]}"; do
  if [[ ! -s "${output}" ]]; then
    echo "Missing or empty output: ${output}"
    exit 1
  fi
done

echo
echo "SUCCESS: analysis, tests, executed notebook and all deliverable checks passed."
