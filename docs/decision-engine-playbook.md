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

## 2. 当前决策逻辑（v0 — 2026-04-18）

### 2.1 两条路径

| 路径 | 任务类型 | 逻辑 |
|---|---|---|
| **fast-path** | loglikelihood / loglikelihood_rolling | 无条件接（prefill 优先调度，0.1-0.2s 完成，任何 SLA 都稳过） |
| **slow-path** | generate_until | 估 finish_time → 对比 SLA → 选最优 profile → 检查 EV |

### 2.2 slow-path 公式

```
backlog_s = (running.decode_tokens_remaining / throughput) × inflation
          + waiting.compute_tokens_remaining / prefill_tok_per_s
          + inflight.total_estimated_output_tokens / throughput × inflation × 0.5

self_s = estimate_output_tokens(profile) / throughput × inflation

finish_s = backlog_s + self_s

accept if finish_s × safety_margin ≤ sla_ttft AND expected_reward > min_ev
```

### 2.3 当前超参

| 参数 | 值 | 含义 | 可信度 |
|---|---|---|---|
| `safety_margin` | 1.2 | 完成时间乘这个再比 SLA | 拍的，需要真数据调 |
| `min_ev` | 0.1 | 期望收益低于此就 drop | 拍的 |
| `decode_tok_per_s` | 750.0 | 整体 decode 吞吐 baseline | 来自队友口述，需要实测 |
| `prefill_tok_per_s` | 18000.0 | prefill 吞吐 | 拍的 |
| `logprob_traffic_ratio` | 0.3 | decode 被 prefill 抢占的膨胀系数 | 拍的 |
| `think_token_multiplier` | 3.0 | thinking mode 输出 token 倍数 | 拍的 |
| `DEFAULT_MAX_GEN_TOKS` | 256 | 所有 generate 任务的输出 token 估计 | **粗糙**——短回答高估，长推理低估 |
| 正确率初始值 | 0.5 | 所有 (task, profile) 冷启动 | **粗糙**——前 50 个决策不准 |

### 2.4 Profile 选择

| eval_task_name 包含 | 候选 profile 列表 |
|---|---|
| `gsm8k`, `math`, `minerva_math`, `mathqa`, `asdiv` | [chat_think, chat_no_think, raw] |
| 其他 | [chat_no_think, raw] |

选最高 EV 的那个。SLA 太紧时 chat_think 会被过滤掉，自动降级。

---

## 3. 已知问题与改进方向

### 3.1 🔴 P0 — 必须在真引擎联调前修

| 问题 | 现状 | 改法 |
|---|---|---|
| **token 估计全用 256** | `_estimate_output_tokens` 不区分任务 | /ask 拿到 prompt 后可以用 tokenizer 估 prompt 长度；按 `eval_task_name` 维护一张经验值表；对 `max_gen_toks` 大的 sampling profile 做修正 |
| **多 message 串行** | execution.py 里逐 message 串行调推理 | loglikelihood 多选题 4 个 message 可以 `asyncio.gather` 并发发出去 |
| **600s 绝对超时兜底缺失** | 没有 deadline watcher | ExecutionScheduler 要跑一个后台协程，对 `inflight.get_overdue_task_ids()` 做 best-effort submit |

### 3.2 🟡 P1 — 显著影响得分

| 问题 | 现状 | 改法 |
|---|---|---|
| **正确率冷启动** | 全部 0.5 | 按 request_type 给不同初值（loglikelihood 可给 0.7，generate 给 0.5）；或前 N 个任务无条件接（warmup phase） |
| **不认识的 task_name** | 全走 [chat_no_think, raw] | 增加更多映射（开发文档提到：数学推理、常识问答、语言理解、知识推理、语言建模）；或对不认识的先用 chat_no_think 试，看正确率自学习 |
| **inflight 重叠系数** | 硬编码 `0.5` | 需要理解 engine /status 与 inflight 的真实重叠关系：engine 的 `decode_tokens_remaining` 是否已经包含我们提交但还在排队的任务？如果是，这个 0.5 应该去掉 |
| **没有 SLA 等级偏好** | 有空间就接 | Diamond+ 的 SLA 对 generate 任务可能根本不可行（0.5-2s），应该直接 drop 而不是算一遍公式再 drop |

### 3.3 🟢 P2 — 后续优化

| 问题 | 改法 |
|---|---|
| **批量挑题** | 从 overview_queue 一次取 K 个，挑 EV/cost 最优的若干个接 |
| **observe 队列** | 对 SLA 宽松但当前 backlog 重的任务，暂存观察，等 backlog 降下来再接 |
| **prefix cache 感知** | 如果推理引擎支持 prefix sharing，相同 prompt 前缀的 loglikelihood 多选题可以提示 engine 做 KV 复用 |
| **动态 inflation** | `logprob_traffic_ratio` 用最近 N 秒的实际 logprob/generate 比例替代常数 |

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
| SLA 命中率低 | `safety_margin` ↑ 或 `DEFAULT_MAX_GEN_TOKS` ↑ | 更保守 |
| 接受率太低（很多 drop） | `safety_margin` ↓ 或 `min_ev` ↓ | 更激进 |
| generate 任务经常超时 | `logprob_traffic_ratio` ↑ | decode 被抢占更多 |
| thinking 任务正确率不高 | `think_token_multiplier` ↑ 或检查 chat template 渲染 | 给更多 token / 检查拼装 |
| engine 经常空闲 | `query_concurrency` ↑ + `safety_margin` ↓ | 多拉题多接 |

---

## 6. 变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-04-18 | v0 初版上线 | 三层架构 + fast/slow path + profile 选择 |
| 2026-04-18 | 加 null 字段防御 | ubiservice mock 返回 eval_task_name=null |
| | | |
