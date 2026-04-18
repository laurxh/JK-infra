import os
import json

from decision_service.config import load_config, DecisionConfig


def test_load_config_from_env_and_file(tmp_path):
    contest_cfg = {
        "platform_url": "http://10.0.0.1:8003",
        "model_name": "Qwen3-32B",
        "model_path": "/mnt/model/Qwen3-32B",
        "contestant_port": 9000,
        "duration_s": 3600,
        "sla_levels": {
            "Bronze": {"ttft_avg": 10.0}, "Silver": {"ttft_avg": 8.0},
            "Gold": {"ttft_avg": 6.0}, "Platinum": {"ttft_avg": 4.0},
            "Diamond": {"ttft_avg": 2.0}, "Stellar": {"ttft_avg": 1.5},
            "Glorious": {"ttft_avg": 0.8}, "Supreme": {"ttft_avg": 0.5},
        },
        "sampling_params": {
            "Deterministic": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                              "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
            "Normal": {"temperature": 0.1, "top_p": 0.9, "top_k": 50,
                       "repetition_penalty": 1.1, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        },
    }
    cfg_path = tmp_path / "contest.json"
    cfg_path.write_text(json.dumps(contest_cfg))

    env = {
        "PLATFORM_URL": "http://10.0.0.1:8003", "INFERENCE_URL": "http://127.0.0.1:8000",
        "CONTESTANT_PORT": "9000", "CONFIG_PATH": str(cfg_path),
        "TOKEN": "test_token", "TEAM_NAME": "test_team",
    }
    for k, v in env.items():
        os.environ[k] = v
    try:
        cfg = load_config()
        assert cfg.platform_url == "http://10.0.0.1:8003"
        assert cfg.inference_url == "http://127.0.0.1:8000"
        assert cfg.token == "test_token"
        assert cfg.team_name == "test_team"
        assert cfg.sla_ttft("Gold") == 6.0
        assert cfg.sla_ttft("Supreme") == 0.5
        assert cfg.sampling("Deterministic")["temperature"] == 0.0
        assert cfg.safety_margin == 1.2
    finally:
        for k in env:
            os.environ.pop(k, None)
