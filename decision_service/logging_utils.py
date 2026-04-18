from __future__ import annotations
import json
import logging
import sys
import time
from decision_service.schemas import (
    TaskOverview, AdmissionDecision, EngineStats,
)


def setup_logging():
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def log_decision(overview: TaskOverview, decision: AdmissionDecision, stats: EngineStats | None):
    record = {
        "event": "admission_decision",
        "ts": time.time(),
        "task_id": overview.task_id,
        "eval_request_type": overview.eval_request_type,
        "eval_task_name": overview.eval_task_name,
        "target_sla": overview.target_sla,
        "target_reward": overview.target_reward,
        "action": decision.action.value,
        "profile": decision.profile.value if decision.profile else None,
        "reason": decision.reason,
        "estimated_finish_s": decision.estimated_finish_s,
        "expected_reward": decision.expected_reward,
    }
    if stats:
        record["stats"] = {
            "running_decode_remaining": stats.running.decode_tokens_remaining,
            "running_task_count": stats.running.task_count,
            "waiting_compute_remaining": stats.waiting.compute_tokens_remaining,
            "waiting_task_count": stats.waiting.task_count,
        }
    print(json.dumps(record), file=sys.stdout, flush=True)


def log_submit(task_id: int, sla_met: bool, actual_ttft_s: float, profile: str | None):
    record = {
        "event": "submit",
        "ts": time.time(),
        "task_id": task_id,
        "sla_met": sla_met,
        "actual_ttft_s": actual_ttft_s,
        "profile": profile,
    }
    print(json.dumps(record), file=sys.stdout, flush=True)
