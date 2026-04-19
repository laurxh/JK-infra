#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-my_env}"

# ---------- Activate conda env ----------
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}" 2>/dev/null || true
  echo "[run] activated conda env: ${ENV_NAME}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------- Platform-injected env vars ----------
MODEL_PATH="${MODEL_PATH:-/mnt/model/Qwen3-32B}"
CONTESTANT_PORT="${CONTESTANT_PORT:-9000}"
CONFIG_PATH="${CONFIG_PATH:-}"
PLATFORM_URL="${PLATFORM_URL:-http://127.0.0.1:8003}"

echo "[run] MODEL_PATH=${MODEL_PATH}"
echo "[run] CONTESTANT_PORT=${CONTESTANT_PORT}"
echo "[run] PLATFORM_URL=${PLATFORM_URL}"
echo "[run] CONFIG_PATH=${CONFIG_PATH}"

# ---------- Export for child processes ----------
export MODEL_PATH CONTESTANT_PORT CONFIG_PATH PLATFORM_URL
export INFERENCE_URL="http://localhost:${CONTESTANT_PORT}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"

# ---------- 1. Start vLLM inference engine (background) ----------
echo "[run] Starting vLLM inference engine on port ${CONTESTANT_PORT}..."
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
TP_SIZE=4
if [ "$GPU_COUNT" -lt 4 ]; then TP_SIZE=$GPU_COUNT; fi

nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port "${CONTESTANT_PORT}" \
  --host 0.0.0.0 \
  --enable-prefix-caching \
  > /tmp/vllm.log 2>&1 &
ENGINE_PID=$!
echo "[run] vLLM PID: ${ENGINE_PID}"

# ---------- 2. Wait for engine health ----------
echo "[run] Waiting for vLLM at ${INFERENCE_URL} ..."
for i in $(seq 1 55); do
  if curl -s -o /dev/null -w "%{http_code}" "${INFERENCE_URL}/health" 2>/dev/null | grep -q "200"; then
    echo "[run] vLLM ready (${i}s)"
    break
  fi
  if [ "$i" -eq 55 ]; then
    echo "[run] ERROR: vLLM not ready after 55s"
    tail -20 /tmp/vllm.log
    exit 1
  fi
  sleep 1
done

# ---------- 3. Start bidder (decision client) ----------
echo "[run] Starting bidder..."
nohup "${PYTHON_BIN}" "${ROOT_DIR}/submission/bidder.py" \
  > /tmp/bidder.log 2>&1 &
BIDDER_PID=$!
echo "[run] Bidder PID: ${BIDDER_PID}"

# ---------- 4. Wait ----------
trap 'echo "[run] Shutting down..."; kill ${BIDDER_PID} ${ENGINE_PID} 2>/dev/null; exit 0' SIGTERM SIGINT
wait ${BIDDER_PID}
