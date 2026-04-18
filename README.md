# JK Infra 服务说明

## 启动服务

使用默认本地模型路径启动：

```bash
python server_qwen3_32b.py --port 8000
```

显式指定模型路径启动：

```bash
python server_qwen3_32b.py \
  --model-path /data1/luxuhao/workspace/jk-infra/models/qwen/Qwen3-32B \
  --port 8000
```

带更多显式引擎参数的启动示例：

```bash
python server_qwen3_32b.py \
  --model-path /data1/luxuhao/workspace/jk-infra/models/qwen/Qwen3-32B \
  --devices 2,3,4,5 \
  --port 8000 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.95
```

服务启动后会暴露以下接口：

- `GET /health`
- `GET /stats`
- `POST /generate`
- `POST /loglikelihood`
- `POST /loglikelihood_rolling`

先用下面的命令检查服务是否正常：

```bash
curl http://127.0.0.1:8000/health
```

## 接口说明

### `GET /health`

健康检查接口。

返回示例：

```json
{
  "status": "ok",
  "uptime": 123.45
}
```

### `GET /stats`

返回当前引擎队列状态。

返回示例：

```json
{
  "status": "ok",
  "queue_stats": {
    "running": {
      "task_count": 2,
      "decode_tokens_remaining": 128
    },
    "waiting": {
      "task_count": 5,
      "compute_tokens_remaining": 4096
    }
  }
}
```

### `POST /generate`

文本生成接口。

支持的请求字段：

- `ID`
- `prompt`
- `temperature`
- `max_tokens` 或 `max_gen_toks`
- `top_p`
- `top_k`
- `until` 或 `stop`
- `ignore_eos`

文本输入示例：

```bash
curl -sS -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "ID": 1,
    "prompt": "Context: Tom has 3 apples and gets 2 more.\nQuestion: How many apples does he have now?\nAnswer:",
    "temperature": 0.0,
    "max_gen_toks": 64,
    "top_p": 1.0,
    "top_k": 1,
    "until": ["\n\n"]
  }'
```

返回示例：

```json
{
  "ID": 1,
  "text": " 5"
}
```

说明：

- 这是生成接口，会生成新 token。
- 返回里只包含请求 `ID` 和生成出的 `text`。
- `until` 只有在生成文本中实际出现对应字符串时才会截断。

### `POST /loglikelihood`

计算 `log P(eval_continuation | prompt)`，不会生成 token。

平台请求格式：

```json
{
  "ID": 0,
  "prompt": "Context text...",
  "eval_request_type": "loglikelihood",
  "eval_continuation": "choice A",
  "eval_gen_kwargs": null
}
```

调用示例：

```bash
curl -sS -X POST http://127.0.0.1:8000/loglikelihood \
  -H 'Content-Type: application/json' \
  -d '{
    "ID": 0,
    "prompt": "The capital of France is",
    "eval_request_type": "loglikelihood",
    "eval_continuation": " Paris",
    "eval_gen_kwargs": null
  }'
```

返回：

```json
{
  "ID": 0,
  "accuracy": -3.27
}
```

说明：

- `accuracy` 就是当前这一个候选答案的总 logprob。
- 如果一道选择题有多个候选，平台应针对每个候选分别发一次请求，再对各候选的 `accuracy` 做 `argmax`。
- 这个接口不会生成 token。

### `POST /loglikelihood_rolling`

计算整段 `prompt` 的 rolling log-likelihood，不会生成 token。

请求格式：

```json
{
  "ID": 0,
  "prompt": "A long document text...",
  "eval_request_type": "loglikelihood_rolling",
  "eval_continuation": null,
  "eval_gen_kwargs": null
}
```

调用示例：

```bash
curl -sS -X POST http://127.0.0.1:8000/loglikelihood_rolling \
  -H 'Content-Type: application/json' \
  -d '{
    "ID": 0,
    "prompt": "A long document text...",
    "eval_request_type": "loglikelihood_rolling",
    "eval_continuation": null,
    "eval_gen_kwargs": null
  }'
```

返回：

```json
{
  "ID": 0,
  "accuracy": -123.45
}
```

说明：

- `accuracy` 是整段文本的 rolling log-likelihood 总和。
- 这个接口不会生成 token。

## 评测脚本

统一三类任务的评测脚本：

- `benchmark_eval_jk_infra_with_service.py`

支持的任务类型：

- `generate_until`
- `loglikelihood`
- `loglikelihood_rolling`

运行全部支持的任务：

```bash
python benchmark_eval_jk_infra_with_service.py \
  --data-file JK-infra-eval_data/data/delivery/athlete_main_800_available_protocol_close_v1/full_task_pool.jsonl \
  --task-types all \
  --concurrency 8 \
  --output athlete_main_results.jsonl
```

只运行生成任务：

```bash
python benchmark_eval_jk_infra_with_service.py \
  --data-file JK-infra-eval_data/data/delivery/athlete_main_800_available_protocol_close_v1/full_task_pool.jsonl \
  --task-types generate_until \
  --output generate_results.jsonl
```

只运行 likelihood 相关任务：

```bash
python benchmark_eval_jk_infra_with_service.py \
  --data-file JK-infra-eval_data/data/delivery/athlete_main_800_available_protocol_close_v1/full_task_pool.jsonl \
  --task-types loglikelihood,loglikelihood_rolling \
  --output likelihood_results.jsonl
```
