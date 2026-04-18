#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-my_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.14}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-1}"

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

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

# flash-attn build step imports torch; install torch first and install flash-attn
# without build isolation to avoid ModuleNotFoundError: torch during wheel prep.
TMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${TMP_REQUIREMENTS}"' EXIT

TORCH_SPEC="$(awk '
  /^[[:space:]]*#/ {next}
  /^[[:space:]]*$/ {next}
  /^torch([[:space:]]*[=<>!~].*)?$/ {print; exit}
' "${ROOT_DIR}/requirements.txt")"

if [[ -n "${TORCH_SPEC}" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install "${TORCH_SPEC}"
fi

awk '
  /^[[:space:]]*#/ {next}
  /^[[:space:]]*$/ {next}
  /^torch([[:space:]]*[=<>!~].*)?$/ {next}
  /^flash-attn([[:space:]]*[=<>!~].*)?$/ {next}
  {print}
' "${ROOT_DIR}/requirements.txt" > "${TMP_REQUIREMENTS}"

if [[ -s "${TMP_REQUIREMENTS}" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -r "${TMP_REQUIREMENTS}"
fi

FLASH_ATTN_SPEC="$(awk '
  /^[[:space:]]*#/ {next}
  /^[[:space:]]*$/ {next}
  /^flash-attn([[:space:]]*[=<>!~].*)?$/ {print; exit}
' "${ROOT_DIR}/requirements.txt")"

if [[ -n "${FLASH_ATTN_SPEC}" ]]; then
  echo "[setup] installing flash-attn build deps"
  conda run -n "${ENV_NAME}" python -m pip install psutil
  echo "[setup] installing flash-attn without build isolation"
  GPU_CC=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  fi

  if [[ -n "${GPU_CC}" ]]; then
    echo "[setup] flash-attn arch=${GPU_CC} max_jobs=${FLASH_ATTN_MAX_JOBS} nvcc_threads=${FLASH_ATTN_NVCC_THREADS}"
    MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
    NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
    TORCH_CUDA_ARCH_LIST="${GPU_CC}" \
      conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation "${FLASH_ATTN_SPEC}"
  else
    echo "[setup] flash-attn arch=auto max_jobs=${FLASH_ATTN_MAX_JOBS} nvcc_threads=${FLASH_ATTN_NVCC_THREADS}"
    MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
    NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
      conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation "${FLASH_ATTN_SPEC}"
  fi
fi

echo "[setup] done"
echo "[setup] entering conda env: ${ENV_NAME}"
conda activate "${ENV_NAME}"
exec "${SHELL:-/bin/bash}" --noprofile --norc -i
