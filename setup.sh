#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="my_env"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.14}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found in PATH"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo "[setup] requirements.txt not found"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[setup] root=${ROOT_DIR}"
echo "[setup] env=${ENV_NAME}"
echo "[setup] python=${PYTHON_VERSION}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] removing existing conda env ${ENV_NAME}"
  conda env remove -n "${ENV_NAME}" -y
fi

echo "[setup] creating conda env ${ENV_NAME}"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "[setup] done"
echo "[setup] entering conda env: ${ENV_NAME}"
exec "${SHELL:-/bin/bash}" --noprofile --norc -i
