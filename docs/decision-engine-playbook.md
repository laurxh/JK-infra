# 决策引擎纲领（Decision Engine Playbook）

> **这是什么**：跨会话持久化的项目知识库。每次开新会话，先读这个文件对齐上下文。
> **怎么用**：新会话开头说"读一下 `docs/decision-engine-playbook.md`"，AI 就能接手。
> **谁改它**：每次决策逻辑有实质性变更时更新。小 bugfix 不用。

---

## 1. 系统全景

```
平台 (PLATFORM_URL)         决策服务 (decision_service/)           推理引擎 (server_qwen3_32b.py)
                     ┌─────────────────────────────────┐
  /query ──────────> │ A. QueryHarvester (去重, N并发)   │
                     │         ↓ overview_queue         │
                     │ B. AdmissionController           │
                     │    CandidatePool + Selector 循环  │
  /ask   <────────── │    → pick_best → /ask             │
                     │         ↓ exec_queue             │
                     │ C. ExecutionScheduler             │
                     │    PromptRenderer → InferClient   │──> POST /generate
  /submit <───────── │    asyncio.gather → submit        │──> POST /loglikelihood
                     │    _deadline_watcher (600s 兜底)  │──> POST /loglikelihood_rolling
                     │ EngineStatsPoller (200ms)         │──> GET  /stats
                     └─────────────────────────────────┘    GET  /health
```

**关键文件**：
- `admission.py` — 候选池 + selector 循环 + 引擎门控 + profile 选择，~120 行
- `candidate_pool.py` — 分桶 + reward 排序，~60 行
- `execution.py` — 并发推理 + submit + deadline watcher，~120 行
- `config.py` — 所有超参，`DecisionConfig` dataclass
- `schemas.py` — 数据结构
- `inference_client.py` — 对齐队友 `server_qwen3_32b.py` 的 5 个 HTTP 接口
- `prompt_renderer.py` — chat template 渲染（engine 不套 template）
- `decision_engine.py` — 旧版决策逻辑（目前未被 admission 使用，保留备用）

---

## 2. 当前决策逻辑（v1 — 2026-04-18 候选池版）

### 2.1 Selector 循环

```
QueryHarvester → overview_queue → AdmissionController._drain_queue_to_pool()
                                         ↓
                                   CandidatePool（去重, TTL 60s 清理）
                                         ↓
                                   Selector 循环（100ms 一轮）：
                                     1. purge_stale()
                                     2. 检查引擎门控：GET /stats → stats.total_tasks < max_engine_tasks
                                     3. pick_best(max=5, slots=available)
                                        → 优先级：loglikelihood > loglikelihood_rolling > generate_until
                                        → 同类型按 reward 降序
                                     4. 选 profile → /ask → exec_queue
```

### 2.2 决策规则

| 任务类型 | 决策 | Profile |
|---|---|---|
| loglikelihood | 几乎无条件接（prefill 0.1-0.2s） | chat_no_think |
| loglikelihood_rolling | 引擎不忙时接 | raw |
| generate_until + 宽 SLA(≥6s) + 高 reward | 接 | chat_think |
| generate_until + 中 SLA(≥4s) | 接 | chat_no_think |
| generate_until + 紧 SLA(<4s) | 接但轻量 | raw |
| 任何类型 + 引擎满（stats.total_tasks ≥ max_engine_tasks） | 不接（留在池里等） | — |
| 任何类型 + 候选过期（>60s） | 清理 | — |

### 2.3 关键常数

| 参数 | 值 | 位置 | 可信度 |
|---|---|---|---|
| `max_engine_tasks` | 16 (default) | config.py | **必须在真机标定**——队友 server 用 `--max-num-seqs 64`，但实际并发取决于 KV cache |
| `high_reward_threshold` | 1000.0 | config.py | 拍的，需要看真实 reward 分布 |
| `staleness_s` | 60.0 | candidate_pool.py | 保守值（任务 TTL 300s，60s 内决定） |
| `max_picks_per_round` | 5 | config.py | 一轮最多 ask 5 个 |
| `default_output_estimate` | 256 | config.py | 粗估，/ask 后用真实 max_gen_toks 修正 |
| `think_token_multiplier` | 3.0 | config.py | 拍的 |
| `DEADLINE_MARGIN_S` | 30 | execution.py | 绝对超时前 30s 强制 submit |

### 2.4 Profile 选择

**不依赖 eval_task_name**（该字段不可靠）。基于强特征：

| 条件 | Profile |
|---|---|
| request_type = loglikelihood | chat_no_think |
| request_type = loglikelihood_rolling | raw |
| generate + SLA ≥ 6s + reward ≥ HIGH_REWARD_THRESHOLD | chat_think |
| generate + SLA ≥ 4s | chat_no_think |
| generate + SLA < 4s | raw |

### 2.5 关键特性

- **多 message 并发**：loglikelihood 多选题 4 个 message 用 `asyncio.gather` 并发推理
- **600s 兜底**：`_deadline_watcher` 每 5s 检查，剩余 <30s 强制 submit 避免 -2×R_i
- **引擎门控**：`stats.total_tasks ≥ max_engine_tasks` 时暂停接新单。只用 engine 自己的 /stats 做权威来源，不加 inflight.count（避免双重计数——engine 里的 task 和 inflight 有重叠）

---

## 3. 已知问题与改进方向

### 3.1 ✅ 已解决（P0 — 2026-04-18）

| 问题 | 解决方案 |
|---|---|
| ~~逐条判断，没有候选池~~ | 新增 CandidatePool，按 request_type 优先 + reward 排序 |
| ~~token 估计全用 256~~ | admission 阶段用 DEFAULT_OUTPUT_ESTIMATE，后续可 /ask 后修正 |
| ~~profile 靠 task_name 硬编码~~ | 改为基于 SLA + reward 强特征选 profile |
| ~~多 message 串行~~ | asyncio.gather 并发推理 |
| ~~600s 超时无兜底~~ | deadline_watcher 每 5s 检查，<30s 强制 submit |

### 3.2 ✅ 已解决（P1 — 2026-04-18）

| 问题 | 解决方案 |
|---|---|
| ~~/ask 后 inflight token 不修正~~ | admission.py 提取真实 max_gen_toks 回写 inflight_registry |
| ~~正确率冷启动全 0.5~~ | history_store 按 request_type 分初值（ll=0.7, gen=0.5, llr=0.6），键改为 (request_type, profile) |
| ~~HIGH_REWARD_THRESHOLD 硬编码~~ | 搬到 DecisionConfig，连同 max_picks_per_round、default_output_estimate |
| **MAX_ENGINE_TASKS 需要标定** | 默认 16，不知道真实上限。真机上用不同值跑，观察 /status 的 waiting.task_count 是否长期 >0 |

### 3.3 🟢 P2 — 后续优化

| 问题 | 改法 |
|---|---|
| **generate + 严 SLA 该不该直接 drop** | 目前还是会接（用 raw profile）；可能应该直接 drop Diamond/Stellar/Glorious/Supreme 的 generate |
| **动态调整 max_engine_tasks** | 问队友在 /stats 里加返回 `max_num_seqs`（engine 启动参数），省去手动配置；或根据 waiting.task_count 长期 >0 自动收紧 |
| **prefix cache 感知** | 相同 prompt 前缀的 loglikelihood 多选题提示 engine 做 KV 复用 |
| **task_name 统计闭环** | 收集 (task_name, correctness) 统计表，发现正确率规律后可以回灌决策 |
| **observe 队列** | 对 SLA 宽松但当前引擎忙的任务，暂存等引擎空闲再接 |

---

## 4. 推理引擎接口（以 `server_qwen3_32b.py` 为准）

启动：`python server_qwen3_32b.py --model-path $MODEL_PATH --port 8000 [--max-num-seqs 64] [--gpu-memory-utilization 0.95]`

### GET /health

```json
{"status": "ok", "uptime": 123.45}
```

### GET /stats

```json
{
  "status": "ok",
  "queue_stats": {
    "running": {"task_count": 2, "decode_tokens_remaining": 128},
    "waiting": {"task_count": 5, "compute_tokens_remaining": 4096}
  }
}
```

- `total_tasks = running.task_count + waiting.task_count`（message 级，不是 platform task 级）
- 决策服务 200ms 轮询一次

### POST /generate

请求：
```json
{"ID": 1, "prompt": "...", "temperature": 0.0, "max_tokens": 256, "top_p": 1.0, "top_k": 1, "stop": ["\n\n"]}
```
- `max_tokens` 或 `max_gen_toks` 都接受
- `stop` 或 `until` 都接受

响应：`{"ID": 1, "text": " 5"}`

### POST /loglikelihood

请求：
```json
{"ID": 0, "prompt": "...", "eval_continuation": " Paris", "eval_request_type": "loglikelihood"}
```

响应：`{"ID": 0, "accuracy": -3.27}`

### POST /loglikelihood_rolling

请求：
```json
{"ID": 0, "prompt": "...", "eval_request_type": "loglikelihood_rolling"}
```

响应：`{"ID": 0, "accuracy": -123.45}`

### 关键约定

- **ID 字段**：请求用 `ID`，响应回显 `ID`。`inference_client.py` 内部 normalize 成 `id`
- **engine 不套 chat template**：决策侧用 `tokenizer.apply_chat_template(tokenize=False)` 渲染后塞 prompt
- **prefill 优先调度**：loglikelihood 类任务抢占 generate 的 decode
- **engine 超时**：`--queue-timeout 600`（默认），超时返回 504

---

## 5. 调参手册

### 怎么调

1. 跑一轮 mock 测试或真平台，收集 `stdout` 的 JSON 日志
2. 用 `grep '^{' decision.log | python3 -c "import sys,json; [print(json.loads(l)['actual_ttft_s'], json.loads(l)['sla_met']) for l in sys.stdin if 'submit' in l]"` 看 SLA 命中率
3. 按 §5.1 的表格逐个调

### 5.1 调什么

| 你观察到 | 该调 | 方向 |
|---|---|---|
| 引擎经常空闲 | `MAX_ENGINE_TASKS` ↑ 或 `query_concurrency` ↑ | 多接 |
| 接受率太低 | `MAX_ENGINE_TASKS` ↑ | 放开门控 |
| generate 经常超 SLA | `HIGH_REWARD_THRESHOLD` ↑（少用 think）或 generate + 紧 SLA 直接 drop | 更保守 |
| loglikelihood 正确率低 | 检查 chat template 渲染是否正确 | 调试 |
| thinking 任务正确率不高 | `think_token_multiplier` ↑ 或检查 chat template `enable_thinking` | 给更多 token |
| 频繁被 deadline watcher 强制 submit | `DEADLINE_MARGIN_S` ↑ 或少接 generate | 更保守 |

---

## 6. 变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-04-18 | v0 初版上线 | 三层架构 + fast/slow path + profile 选择 |
| 2026-04-18 | 加 null 字段防御 | ubiservice mock 返回 eval_task_name=null |
| 2026-04-18 | **v1 候选池版** | 候选池+分桶+引擎门控 替代逐条公式；多message并发；600s兜底watcher；profile不靠task_name |
| 2026-04-18 | **P1 完成** | /ask后修正inflight token；冷启动按request_type分初值；超参配置化 |
| 2026-04-18 | **对齐队友真实 API** | /stats(非/status)、ID(非request_id)、eval_continuation(非continuation)；修复双重计数 |
| 2026-04-18 | **Playbook 全面修订** | 统一到队友 server_qwen3_32b.py 的真实接口；修 run.sh/setup.sh |
