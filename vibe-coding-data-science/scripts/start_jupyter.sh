#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-$PWD/.jupyter/config}"
export JUPYTER_DATA_DIR="${JUPYTER_DATA_DIR:-$PWD/.jupyter/data}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$PWD/.jupyter/runtime}"
export IPYTHONDIR="${IPYTHONDIR:-$PWD/.ipython}"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f ../.venv/bin/activate ]; then
  source ../.venv/bin/activate
else
  echo "No virtual environment found. Run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt"
  exit 1
fi

jupyter lab
