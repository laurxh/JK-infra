from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class ExecutionProfile(str, Enum):
    RAW = "raw"
    CHAT_NO_THINK = "chat_no_think"
    CHAT_THINK = "chat_think"


class AdmissionAction(str, Enum):
    ASK_NOW = "ask_now"
    DROP = "drop"


@dataclass
class TaskOverview:
    """Fields from platform /query response."""
    task_id: int
    target_sla: str
    target_reward: float
    max_winners: int
    eval_task_name: str
    eval_request_type: str
    eval_sampling_param: str
    eval_timeout_s: int
    queried_at: float = field(default_factory=time.monotonic)


@dataclass
class RunningStats:
    decode_tokens_remaining: int
    task_count: int


@dataclass
class WaitingStats:
    compute_tokens_remaining: int
    task_count: int


@dataclass
class EngineStats:
    running: RunningStats
    waiting: WaitingStats
    polled_at: float = field(default_factory=time.monotonic)

    @property
    def total_tasks(self) -> int:
        return self.running.task_count + self.waiting.task_count


@dataclass
class InflightTask:
    task_id: int
    task_data: dict[str, Any]
    profile: ExecutionProfile
    estimated_output_tokens: int
    sla_ttft: float
    ask_time: float
    absolute_deadline: float


@dataclass
class AdmissionDecision:
    action: AdmissionAction
    profile: ExecutionProfile | None
    reason: str
    estimated_finish_s: float | None = None
    expected_reward: float | None = None
