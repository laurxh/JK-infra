#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="0.0.0.0"
PORT="${CONTESTANT_PORT:-9000}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_MODEL_LEN="8192"
MAX_NUM_SEQS="64"
MAX_NUM_BATCHED_TOKENS="32768"
GPU_MEMORY_UTILIZATION="0.97"
QUEUE_TIMEOUT="600"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[run] MODEL_PATH is required"
  echo "[run] example: MODEL_PATH=/path/to/Qwen3-32B CONTESTANT_PORT=9000 bash run.sh"
  exit 1
fi

echo "[run] model_path=${MODEL_PATH}"
echo "[run] host=${HOST} port=${PORT}"
echo "[run] max_num_seqs=${MAX_NUM_SEQS} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[run] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"

export HOST
export CONTESTANT_PORT="${PORT}"
export MODEL_PATH
export MAX_MODEL_LEN
export MAX_NUM_SEQS
export MAX_NUM_BATCHED_TOKENS
export GPU_MEMORY_UTILIZATION
export QUEUE_TIMEOUT
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/src/server_qwen3_32b.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --queue-timeout "${QUEUE_TIMEOUT}"

"${PYTHON_BIN}" -m decision_service.app
