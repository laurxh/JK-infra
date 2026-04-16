# Athlete SVAMP Generate 100 V1

这是一份给运动员侧联调使用的干净数据包，只包含可见给 `/query` 与 `/ask` 的两类数据：

- `overview_pool.jsonl`
- `full_task_pool.jsonl`

当前性质：

- 共 `100` 条 task
- 全部来自 `SVAMP`
- 全部是 `generate_until`
- prompt 长度均小于 `4000` 字符
- `sampling profile` 在四档上均匀分布
- `target_sla` 在八档上近似均匀分布

注意：

- 本目录**不包含** `oracle_pool.jsonl`、`oracle_meta.jsonl`
- 本目录**不包含** `gold`、`difficulty_units`、`w_sla` 等裁判侧信息
- `target_reward` / `target_sla` / `eval_request_type` / `eval_sampling_param` 保留，是官方协议本来就会给运动员侧的概要字段
