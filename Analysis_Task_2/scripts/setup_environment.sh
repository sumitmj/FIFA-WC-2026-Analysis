#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "Creating the project environment with ${PYTHON_COMMAND}..."
"${PYTHON_COMMAND}" -m venv "${VENV_PATH}"

echo "Installing the project scientific environment..."
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
if [[ -f "${PROJECT_ROOT}/requirements-lock.txt" ]]; then
  "${VENV_PATH}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements-lock.txt"
else
  "${VENV_PATH}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

echo
"${VENV_PATH}/bin/python" --version
"${VENV_PATH}/bin/python" -c "import numpy, pandas, scipy, matplotlib, seaborn; print('Scientific stack imported successfully')"
echo
echo "Environment ready: ${VENV_PATH}"
echo "Run the complete project with:"
echo "  ${PROJECT_ROOT}/scripts/run_all.sh"
