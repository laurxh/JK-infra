# 提交包集成指南

## 当前状态

- ✅ `run.sh` 已填入真实启动命令（`python server_qwen3_32b.py --model-path $MODEL_PATH --port 8000`）
- ✅ `setup.sh` 已配置（决策服务依赖 + 推理引擎依赖说明）
- ✅ 接口已对齐 `server_qwen3_32b.py` 的真实 API

## 接口概览

| 接口 | 方法 | 请求 | 响应 |
|---|---|---|---|
| `/health` | GET | — | `{status, uptime}` |
| `/stats` | GET | — | `{status, queue_stats: {running: {task_count, decode_tokens_remaining}, waiting: {...}}}` |
| `/generate` | POST | `{ID, prompt, max_tokens, temperature, top_p, top_k, stop}` | `{ID, text}` |
| `/loglikelihood` | POST | `{ID, prompt, eval_continuation, eval_request_type}` | `{ID, accuracy}` |
| `/loglikelihood_rolling` | POST | `{ID, prompt, eval_request_type}` | `{ID, accuracy}` |

## 本地联调

```bash
# 终端 1：启动推理引擎（需要 GPU）
python server_qwen3_32b.py --model-path /path/to/Qwen3-32B --port 8000

# 终端 2：启动 mock 平台（需要 Redis）
cd ubiservice && python bin/start.py

# 终端 3：启动决策服务
export PLATFORM_URL=http://127.0.0.1:8003 INFERENCE_URL=http://127.0.0.1:8000
export TOKEN=test_token TEAM_NAME=test_team CONFIG_PATH=/tmp/contest_config.json
PYTHONPATH=. python -m decision_service.app
```

没有 GPU 时用 stub 推理服务：
```bash
PYTHONPATH=. python -m decision_service.stub_inference
```

## 提交包结构

```
submission.tar.gz 内容（平台要求）：
├── setup.sh                    # 环境安装
├── run.sh                      # 入口脚本（启动引擎 + 决策服务）

仓库根目录（run.sh cd .. 后的工作目录）：
├── server_qwen3_32b.py         # 推理引擎（队友）
├── nanovllm/                   # 推理引擎依赖（仓库内）
├── decision_service/           # 决策服务
│   ├── app.py                  # 入口
│   ├── admission.py            # 候选池 + selector + 引擎门控
│   ├── candidate_pool.py       # 分桶 + reward 排序
│   ├── execution.py            # 并发推理 + submit + deadline watcher
│   ├── inference_client.py     # 对齐 server_qwen3_32b.py 的 HTTP client
│   ├── prompt_renderer.py      # chat template 渲染
│   ├── config.py / schemas.py  # 配置 + 数据结构
│   └── requirements.txt
└── docs/
    └── decision-engine-playbook.md  # 项目纲领
```
