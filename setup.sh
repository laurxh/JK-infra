#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-my_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found in PATH"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
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

# Install vLLM (includes torch and all dependencies)
echo "[setup] installing vLLM..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install vllm

# Install decision service dependencies
echo "[setup] installing decision service deps..."
conda run -n "${ENV_NAME}" python -m pip install httpx pydantic

echo "[setup] done"
