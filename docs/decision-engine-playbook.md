# 决策引擎纲领（Decision Engine Playbook）

> **这是什么**：跨会话持久化的项目知识库。每次开新会话，先读这个文件对齐上下文。
> **怎么用**：新会话开头说"读一下 `docs/decision-engine-playbook.md`"，AI 就能接手。
> **谁改它**：每次决策逻辑有实质性变更时更新。小 bugfix 不用。

---

## 1. 系统全景

```
平台 (ubiservice)          决策服务 (decision_service/)           推理引擎 (队友)
                     ┌─────────────────────────────────┐
  /query ──────────> │ A. QueryHarvester (去重, N并发)   │
                     │         ↓ overview_queue         │
                     │ B. AdmissionController           │
                     │    DecisionEngine.decide()       │
  /ask   <────────── │    → ask_now / drop              │
                     │         ↓ exec_queue             │
                     │ C. ExecutionScheduler             │
                     │    PromptRenderer → InferClient   │──> /generate
  /submit <───────── │    填 message → submit            │──> /loglikelihood
                     │                                   │──> /loglikelihood_rolling
                     │ EngineStatsPoller (200ms)         │──> /status
                     └─────────────────────────────────┘    /health
```

**关键文件**：
- `decision_engine.py` — 决策核心，纯函数，~100 行
- `admission.py` — 把决策结果执行出去（/ask），~80 行
- `execution.py` — 推理 + 提交，~100 行
- `config.py` — 超参，`DecisionConfig` dataclass
- `schemas.py` — 所有数据结构
- `docs/inference-contract.md` — 与推理引擎的接口契约 v1.0

---

## 2. 当前决策逻辑（v1 — 2026-04-18 候选池版）

### 2.1 架构变化：从"逐条判断"到"攒池挑最好"

```
QueryHarvester → overview_queue → AdmissionController._drain_queue_to_pool()
                                         ↓
                                   CandidatePool（去重, TTL 60s 清理）
                                         ↓
                                   Selector 循环（100ms 一轮）：
                                     1. purge_stale()
                                     2. 检查引擎门控：stats.total_tasks + inflight.count < MAX_ENGINE_TASKS(8)
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
| 任何类型 + 引擎满（≥8 tasks + inflight） | 不接（留在池里等） | — |
| 任何类型 + 候选过期（>60s） | 清理 | — |

### 2.3 关键常数

| 参数 | 值 | 位置 | 可信度 |
|---|---|---|---|
| `MAX_ENGINE_TASKS` | 16 (default) | config.py `max_engine_tasks` | **必须在真机标定**——取决于 KV cache / 模型 / sequence 长度 |
| `HIGH_REWARD_THRESHOLD` | 1000.0 | admission.py | 拍的，需要看真实 reward 分布 |
| `staleness_s` | 60.0 | candidate_pool.py | 保守值（TTL 300s，60s 内决定） |
| `MAX_PICKS_PER_ROUND` | 5 | admission.py | 一轮最多 ask 5 个 |
| `DEFAULT_OUTPUT_ESTIMATE` | 256 | admission.py | 粗估，/ask 后可用真实 max_gen_toks 修正 |
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
- **引擎门控**：`stats.total_tasks + inflight.count ≥ MAX_ENGINE_TASKS` 时暂停接新单

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

### 3.2 🟡 P1 — 显著影响得分

| 问题 | 现状 | 改法 |
|---|---|---|
| **/ask 后修正 inflight token** | admission 用默认 256 估 | /ask 后从 messages 提取真实 max_gen_toks，更新 inflight |
| **正确率冷启动** | history_store 全部 0.5 | 按 request_type 给不同初值；或前 N 个任务无条件接 |
| **HIGH_REWARD_THRESHOLD 不准** | 硬编码 1000 | 需要看真实 reward 分布后标定 |
| **MAX_ENGINE_TASKS 需要标定** | 默认 16，不知道真实上限 | 真机上用不同值跑，观察 /status 的 waiting.task_count 是否长期 >0（说明塞太多了） |

### 3.3 🟢 P2 — 后续优化

| 问题 | 改法 |
|---|---|
| **generate + 严 SLA 该不该直接 drop** | 目前还是会接（用 raw profile）；可能应该直接 drop Diamond/Stellar/Glorious/Supreme 的 generate |
| **动态调整 MAX_ENGINE_TASKS** | 根据 /status 观察到的实际并发峰值自适应 |
| **prefix cache 感知** | 相同 prompt 前缀的 loglikelihood 多选题提示 engine 做 KV 复用 |
| **task_name 统计闭环** | 收集 (task_name, correctness) 统计表，发现正确率规律后可以回灌决策 |
| **observe 队列** | 对 SLA 宽松但当前引擎忙的任务，暂存等引擎空闲再接 |

---

## 4. 与推理引擎的接口

详见 `docs/inference-contract.md`（v1.0 定稿）。核心要点：

- 5 个 HTTP 接口：`/health`, `/status`, `/generate`, `/loglikelihood`, `/loglikelihood_rolling`
- engine 不套 chat template，决策侧用 `tokenizer.apply_chat_template(tokenize=False)` 渲染 prompt 字符串
- `request_id` 幂等（同 id 返回上次结果）
- engine 是 prefill 优先调度（logprob 任务抢占 generate 的 decode）

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
