from __future__ import annotations
import json
import os
from dataclasses import dataclass, field


@dataclass
class DecisionConfig:
    platform_url: str
    inference_url: str
    token: str
    team_name: str
    model_name: str
    model_path: str
    contestant_port: int
    duration_s: int

    sla_levels: dict[str, float]
    sampling_params: dict[str, dict]

    safety_margin: float = 1.2
    min_ev: float = 0.1
    decode_tok_per_s: float = 750.0
    prefill_tok_per_s: float = 18000.0
    logprob_traffic_ratio: float = 0.3
    stats_poll_interval_s: float = 0.2
    query_concurrency: int = 3
    max_submit_retries: int = 2
    query_backoff_s: float = 0.5
    health_poll_interval_s: float = 1.0
    health_timeout_s: float = 55.0
    think_token_multiplier: float = 3.0
    max_engine_tasks: int = 64              # matches engine --max-num-seqs 64
    high_reward_threshold: float = 1000.0   # reward above this → eligible for thinking profile
    max_picks_per_round: int = 5            # max candidates to /ask per selector round
    default_output_estimate: int = 256      # pre-ask output token estimate for generate_until

    def sla_ttft(self, sla_name: str) -> float:
        return self.sla_levels[sla_name]

    def sampling(self, param_name: str) -> dict:
        return self.sampling_params.get(param_name, {})


def load_config() -> DecisionConfig:
    config_path = os.environ.get("CONFIG_PATH", "")
    contest: dict = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            contest = json.load(f)

    sla_levels = {
        name: level.get("max_latency") or level.get("ttft_avg") or 10.0
        for name, level in contest.get("sla_levels", {}).items()
    }

    return DecisionConfig(
        platform_url=os.environ.get("PLATFORM_URL", contest.get("platform_url", "http://127.0.0.1:8003")),
        inference_url=os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000"),
        token=os.environ.get("TOKEN", "default_token"),
        team_name=os.environ.get("TEAM_NAME", "team_jk"),
        model_name=contest.get("model_name", "Qwen3-32B"),
        model_path=os.environ.get("MODEL_PATH", contest.get("model_path", "")),
        contestant_port=int(os.environ.get("CONTESTANT_PORT", contest.get("contestant_port", 9000))),
        duration_s=contest.get("duration_s", 3600),
        sla_levels=sla_levels,
        sampling_params=contest.get("sampling_params", {}),
    )
