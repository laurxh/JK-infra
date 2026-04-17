# Athlete Main 800 Available Protocol Close V1

这是一份给运动员侧联调用的干净数据包，只包含可见给 `/query` 与 `/ask` 的两类数据：

- `overview_pool.jsonl`
- `full_task_pool.jsonl`

当前性质：

- 共 `800` 条 task
- `generate_until` 共 `300` 条
- `loglikelihood` 共 `500` 条
- benchmark 构成为 `SVAMP / SciQ / CommonsenseQA / ARC`
- benchmark 分布为 `SVAMP 300 / SciQ 175 / CommonsenseQA 175 / ARC 150`
- 最大 prompt 长度为 `3762` 字符
- `MMLU` 当前未纳入，因为本地 raw 目录里仍是 Git LFS pointer
- `loglikelihood_rolling` 当前未纳入，因为 rolling 语料尚未落地

注意：

- 本目录**不包含** `oracle_pool.jsonl`、`oracle_meta.jsonl`
- 本目录**不包含** `gold`、`difficulty_units`、`w_sla` 等裁判侧信息
- `target_reward` / `target_sla` / `eval_request_type` / `eval_sampling_param` 保留，是官方协议本来就会给运动员侧的概要字段
