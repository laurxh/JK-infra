#!/bin/bash
# ============================================================
#  比赛提交入口 — 平台执行 bash run.sh 启动整个系统
#  环境变量由平台注入：
#    CONTESTANT_PORT  选手 HTTP 服务端口（默认 9000）
#    MODEL_PATH       模型权重目录
#    CONFIG_PATH      比赛配置 JSON 路径
#    PLATFORM_URL     评测平台地址
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 如果有 setup.sh 创建的虚拟环境，先激活 ----------
if [ -f /tmp/contestant_env/bin/activate ]; then
    source /tmp/contestant_env/bin/activate
fi

# ---------- 默认值 ----------
export INFERENCE_URL="${INFERENCE_URL:-http://127.0.0.1:8000}"
export TOKEN="${TOKEN:-team_jk_token}"
export TEAM_NAME="${TEAM_NAME:-team_jk}"

# ---------- 1. 启动推理引擎（队友的代码） ----------
echo "[run.sh] Starting inference engine..."

# >>>>>> TODO: 替换为队友的实际启动命令 <<<<<<
# 示例（vLLM）：
#   python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_PATH" --tensor-parallel-size 4 \
#     --port 8000 &
# 示例（自研 nanovllm）：
#   python -m engine.server --model-path "$MODEL_PATH" --port 8000 &
echo "[run.sh] WARNING: 推理引擎启动命令未配置，请编辑 run.sh 第 30 行"
# INFERENCE_PID=$!
# >>>>>> END TODO <<<<<<

# ---------- 2. 等推理引擎就绪（最多 55 秒） ----------
echo "[run.sh] Waiting for inference engine at $INFERENCE_URL ..."
for i in $(seq 1 55); do
    if curl -s -o /dev/null -w "%{http_code}" "$INFERENCE_URL/health" | grep -q "200"; then
        echo "[run.sh] Inference engine ready (${i}s)"
        break
    fi
    if [ "$i" -eq 55 ]; then
        echo "[run.sh] ERROR: Inference engine not ready after 55s"
        exit 1
    fi
    sleep 1
done

# ---------- 3. 启动决策服务 ----------
echo "[run.sh] Starting decision service..."
export PYTHONPATH="${SCRIPT_DIR}/.."
python -m decision_service.app &
DECISION_PID=$!
echo "[run.sh] Decision service started (PID=$DECISION_PID)"

# ---------- 4. 等待退出 ----------
# 平台用 Ctrl+C / SIGTERM 停止整个服务
trap "echo '[run.sh] Shutting down...'; kill $DECISION_PID 2>/dev/null; exit 0" SIGTERM SIGINT

wait $DECISION_PID
