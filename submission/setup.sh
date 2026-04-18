#!/bin/bash
# ============================================================
#  环境安装脚本 — 平台首次运行时执行 bash setup.sh
#  安装完成后的环境会被缓存，后续不再重复执行
# ============================================================
set -e

echo "[setup.sh] Creating virtual environment..."
python3.12 -m venv /tmp/contestant_env
source /tmp/contestant_env/bin/activate

echo "[setup.sh] Installing decision service dependencies..."
pip install -r decision_service/requirements.txt

# >>>>>> TODO: 队友的推理引擎依赖 <<<<<<
# pip install -r engine/requirements.txt
# pip install vllm  # 如果用 vLLM
echo "[setup.sh] WARNING: 推理引擎依赖未配置，请编辑 setup.sh"
# >>>>>> END TODO <<<<<<

echo "[setup.sh] Done."
