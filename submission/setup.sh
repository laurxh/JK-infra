#!/bin/bash

# VLLM 环境安装脚本
# 使用方法: bash setup.sh
# 平台: Ubuntu 22.04 + Python 3.12 + CUDA 12.8 + 4x RTX 4090

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "VLLM 环境构建"
echo "=========================================="

# 配置镜像源
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRIPT_DIR}/.cache/vllm"
export FLASHINFER_WORKSPACE_DIR="${SCRIPT_DIR}/.cache/flashinfer"

# 创建必要目录
echo "创建目录结构..."
mkdir -p "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/.cache/vllm"
mkdir -p "${SCRIPT_DIR}/.cache/huggingface"
mkdir -p "${SCRIPT_DIR}/.cache/flashinfer"
mkdir -p "${SCRIPT_DIR}/tmp"

# 创建虚拟环境
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "创建虚拟环境..."
    python3.12 -m venv "${SCRIPT_DIR}/venv"
fi

# 激活虚拟环境
source "${SCRIPT_DIR}/venv/bin/activate"

# 升级pip
echo "升级pip..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch
echo "安装 PyTorch..."
pip install torch==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 直接安装 vllm，让其自动处理所有依赖
echo "安装 vllm (自动处理依赖)..."
pip install vllm==0.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# bidder 依赖
echo "安装 bidder 依赖..."
pip install httpx -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=========================================="
echo "环境安装完成!"
echo "Python: $(python --version)"
echo "Torch: $(pip show torch | grep Version)"
echo "VLLM: $(pip show vllm | grep Version)"
echo "=========================================="
echo "下一步: bash run.sh"
echo "=========================================="
