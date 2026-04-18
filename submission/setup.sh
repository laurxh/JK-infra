#!/bin/bash
# ============================================================
#  环境安装脚本 — 平台首次运行时执行 bash setup.sh
#  安装完成后的环境会被缓存，后续不再重复执行
#  系统预装 python3.12、pip、uv
# ============================================================
set -e

echo "[setup.sh] Creating virtual environment..."
python3.12 -m venv /tmp/contestant_env
source /tmp/contestant_env/bin/activate

echo "[setup.sh] Installing decision service dependencies..."
pip install -r decision_service/requirements.txt

echo "[setup.sh] Installing inference engine dependencies..."
# server_qwen3_32b.py 只依赖 nanovllm（仓库内的本地包）和标准库
# nanovllm 的依赖（torch 等）应该已在平台环境预装
# 如果需要额外依赖，在这里加：
# pip install torch  # 平台通常预装

echo "[setup.sh] Done."
