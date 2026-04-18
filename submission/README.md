# 提交包集成指南

## 队友需要做的

### 1. 编辑 `submission/run.sh` 第 30 行

把推理引擎的启动命令填进去，例如：

```bash
python -m engine.server --model-path "$MODEL_PATH" --port 8000 &
INFERENCE_PID=$!
```

### 2. 编辑 `submission/setup.sh`

把推理引擎的依赖安装加进去：

```bash
pip install -r engine/requirements.txt
```

### 3. 推理引擎需提供的 5 个 HTTP 接口

详见 `docs/inference-contract.md`（v1.0 定稿）：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 返回 200 表示就绪 |
| `/status` | GET | 返回 `{running: {decode_tokens_remaining, task_count}, waiting: {...}}` |
| `/generate` | POST | 接收 `{request_id, prompt, max_tokens, temperature, ...}` 返回 `{id, text}` |
| `/loglikelihood` | POST | 接收 `{request_id, prompt, continuation}` 返回 `{id, accuracy}` |
| `/loglikelihood_rolling` | POST | 接收 `{request_id, prompt}` 返回 `{id, accuracy}` |

**关键点**：
- engine **不需要**套 chat template，prompt 已由决策服务渲染好
- `request_id` 由决策服务生成并在响应中回显
- 同一 `request_id` 重复请求直接返回上次结果（幂等）

### 4. 本地联调

```bash
# 终端 1：启动推理引擎
python -m engine.server --model-path /path/to/Qwen3-32B --port 8000

# 终端 2：启动 mock 平台
cd ubiservice && python bin/start.py

# 终端 3：启动决策服务
export PLATFORM_URL=http://127.0.0.1:8003 INFERENCE_URL=http://127.0.0.1:8000
export TOKEN=test_token TEAM_NAME=test_team CONFIG_PATH=ubiservice/config/defination_base.json
PYTHONPATH=. python -m decision_service.app
```

没有 GPU 时可用 stub 推理服务替代引擎：
```bash
PYTHONPATH=. python -m decision_service.stub_inference
```

### 5. 提交包结构

```
submission.tar.gz
├── setup.sh                    # 环境安装
├── run.sh                      # 入口脚本
├── decision_service/           # 决策服务代码
│   ├── app.py
│   ├── config.py
│   ├── schemas.py
│   ├── decision_engine.py
│   ├── ...
│   └── requirements.txt
├── engine/                     # 队友的推理引擎代码
│   └── ...
└── docs/
    └── inference-contract.md   # 接口契约
```
