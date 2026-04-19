#!/bin/bash

# VLLM 推理服务启动脚本
# 使用方法: bash run.sh
# 平台调用时会传入环境变量

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "VLLM 推理服务启动 - Pipeline Mode"
echo "=========================================="

# 解析环境变量
export CONTESTANT_PORT="${CONTESTANT_PORT:-9000}"
export MODEL_PATH="${MODEL_PATH:-/mnt/model/Qwen3-32B}"
export CONFIG_PATH="${CONFIG_PATH:-/mnt/config/contest.json}"
export PLATFORM_URL="${PLATFORM_URL:-http://10.0.0.1:8003}"
export INFERENCE_URL="${INFERENCE_URL:-http://localhost:${CONTESTANT_PORT}}"

echo "环境变量:"
echo "  CONTESTANT_PORT: $CONTESTANT_PORT"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  CONFIG_PATH: $CONFIG_PATH"
echo "  PLATFORM_URL: $PLATFORM_URL"
echo "  INFERENCE_URL: $INFERENCE_URL"
echo "=========================================="

# 激活虚拟环境
source "${SCRIPT_DIR}/venv/bin/activate"

# 配置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRIPT_DIR}/.cache/vllm"
export VLLM_RPC_BASE_PATH="${SCRIPT_DIR}/tmp"
export FLASHINFER_WORKSPACE_DIR="${SCRIPT_DIR}/.cache/flashinfer"

# 计算 GPU 数量
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
echo "检测到 GPU 数量: $GPU_COUNT"

# 根据 GPU 数量设置 tensor-parallel-size
if [ "$GPU_COUNT" -ge 4 ]; then
    TP_SIZE=4
elif [ "$GPU_COUNT" -ge 2 ]; then
    TP_SIZE=2
else
    TP_SIZE=1
fi
echo "使用 Tensor Parallel Size: $TP_SIZE"

# ==========================================
# Step 1: 启动 vLLM 推理服务 (端口 CONTESTANT_PORT)
# ==========================================
echo ""
echo "Step 1: 启动 vLLM 推理服务..."
echo "监听端口: $CONTESTANT_PORT"
echo "模型路径: $MODEL_PATH"
echo "=========================================="

start_vllm() {
    nohup vllm serve "$MODEL_PATH" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization 0.78 \
        --max-model-len 4096 \
        --port $CONTESTANT_PORT \
        --host 0.0.0.0 \
        > /tmp/vllm_${CONTESTANT_PORT}.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM 进程 PID: $VLLM_PID"
    echo $VLLM_PID > /tmp/vllm_${CONTESTANT_PORT}.pid
    return $VLLM_PID
}

VLLM_PID=""

# 启动 vLLM 服务，带重试
MAX_RETRIES=3
RETRY_COUNT=0

while [ -z "$VLLM_PID" ] || ! kill -0 "$VLLM_PID" 2>/dev/null; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ] && [ $MAX_RETRIES -gt 0 ]; then
        echo "[ERROR] vLLM 启动失败，已达到最大重试次数 ($MAX_RETRIES)"
        echo "日志:"
        tail -50 /tmp/vllm_${CONTESTANT_PORT}.log 2>/dev/null || true
        exit 1
    fi

    if [ $RETRY_COUNT -gt 0 ]; then
        echo "[WARN] vLLM 启动失败，${RETRY_COUNT}/${MAX_RETRIES}，等待 5 秒后重试..."
        sleep 5
    fi

    echo "[INFO] 启动 vLLM 服务 (尝试 $((RETRY_COUNT+1))/${MAX_RETRIES})..."
    start_vllm
    VLLM_PID=$(cat /tmp/vllm_${CONTESTANT_PORT}.pid 2>/dev/null)
    RETRY_COUNT=$((RETRY_COUNT+1))

    if [ -n "$VLLM_PID" ]; then
        echo "[INFO] 等待 vLLM 服务启动 (最多300秒)..."
        MAX_WAIT=300
        WAIT_COUNT=0

        while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
            if curl -s "http://localhost:${CONTESTANT_PORT}/health" > /dev/null 2>&1; then
                echo "✅ vLLM 服务启动成功! (耗时: ${WAIT_COUNT}秒)"
                break
            fi
            sleep 1
            WAIT_COUNT=$((WAIT_COUNT + 1))
            if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
                echo -n "."
            fi
        done

        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "[ERROR] vLLM 服务启动超时!"
            echo "日志:"
            tail -50 /tmp/vllm_${CONTESTANT_PORT}.log 2>/dev/null || true
            VLLM_PID=""
        fi
    fi
done

echo ""

# ==========================================
# Step 2: 启动选手服务 bidder.py (端口 CONTESTANT_PORT)
# ==========================================

echo "Step 2: 启动选手服务..."
echo "监听端口: $CONTESTANT_PORT"
echo "=========================================="

# 检查 bidder.py 是否存在
if [ -f "${SCRIPT_DIR}/bidder.py" ]; then
    start_bidder() {
        PYTHONUNBUFFERED=1 nohup python -u "${SCRIPT_DIR}/bidder.py" \
            > /tmp/bidder_${CONTESTANT_PORT}.log 2>&1 &

        BIDDER_PID=$!
        echo "选手服务进程 PID: $BIDDER_PID"
        echo $BIDDER_PID > /tmp/bidder_${CONTESTANT_PORT}.pid
        return $BIDDER_PID
    }

    start_bidder
    sleep 2

    # 检查 bidder 是否成功启动
    BIDDER_PID=$(cat /tmp/bidder_${CONTESTANT_PORT}.pid 2>/dev/null)
    if [ -n "$BIDDER_PID" ] && kill -0 "$BIDDER_PID" 2>/dev/null; then
        echo "✅ 选手服务已启动"
        echo ""
        echo "[INFO] 实时显示 bidder 日志 (Ctrl+C 退出日志但服务继续运行):"
        tail -f /tmp/bidder_${CONTESTANT_PORT}.log &
        TAIL_PID=$!
    else
        echo "[WARN] 选手服务启动可能失败，3秒后检查日志..."
        sleep 3
        tail -10 /tmp/bidder_${CONTESTANT_PORT}.log 2>/dev/null || true
    fi
else
    echo "⚠️  bidder.py 不存在，跳过选手服务启动"
    echo "   如需启动选手服务，请创建 bidder.py"
    BIDDER_PID=""
fi

echo ""
echo "=========================================="
echo "Pipeline 启动完成!"
echo "=========================================="
echo "组件状态:"
echo "  ✅ vLLM 推理服务: http://localhost:${CONTESTANT_PORT}"
echo "  ✅ API 文档: http://localhost:${CONTESTANT_PORT}/docs"
if [ -n "$BIDDER_PID" ]; then
    echo "  ✅ 选手服务: http://localhost:$CONTESTANT_PORT"
fi
echo "=========================================="

# ==========================================
# 主循环：保持服务运行，bidder 退出后自动重启
# ==========================================

cleanup() {
    echo "[INFO] 收到中断信号，开始清理..."
    if [ -n "$TAIL_PID" ]; then
        kill $TAIL_PID 2>/dev/null || true
    fi
    if [ -n "$BIDDER_PID" ]; then
        kill $BIDDER_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

TAIL_PID=""

# 保持脚本运行，监控 bidder 进程
while true; do
    # 检查 bidder 进程是否存在
    BIDDER_PID=$(cat /tmp/bidder_${CONTESTANT_PORT}.pid 2>/dev/null)

    if [ -z "$BIDDER_PID" ] || ! kill -0 "$BIDDER_PID" 2>/dev/null; then
        echo "[WARN] bidder 进程已退出，尝试重新启动..."

        # 停止旧的 tail 进程
        if [ -n "$TAIL_PID" ]; then
            kill $TAIL_PID 2>/dev/null || true
            TAIL_PID=""
        fi

        if [ -f "${SCRIPT_DIR}/bidder.py" ]; then
            # 检查 vLLM 是否还在运行
            VLLM_PID=$(cat /tmp/vllm_${CONTESTANT_PORT}.pid 2>/dev/null)
            if [ -z "$VLLM_PID" ] || ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "[ERROR] vLLM 服务已退出，pipeline 无法继续运行"
                exit 1
            fi

            # 重启 bidder
            PYTHONUNBUFFERED=1 nohup python -u "${SCRIPT_DIR}/bidder.py" \
                > /tmp/bidder_${CONTESTANT_PORT}.log 2>&1 &

            BIDDER_PID=$!
            echo $BIDDER_PID > /tmp/bidder_${CONTESTANT_PORT}.pid
            echo "[INFO] bidder 已重启，PID: $BIDDER_PID"

            # 重新启动 tail -f 显示日志
            tail -f /tmp/bidder_${CONTESTANT_PORT}.log &
            TAIL_PID=$!
            sleep 3
        else
            echo "[WARN] bidder.py 不存在，无法重启"
            break
        fi
    fi

    # 每 30 秒检查一次
    sleep 30
done
