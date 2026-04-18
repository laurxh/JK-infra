from decision_service.schemas import (
    TaskOverview, EngineStats, RunningStats, WaitingStats,
    ExecutionProfile, AdmissionAction, AdmissionDecision,
)


def test_engine_stats_total_tasks(idle_stats, busy_stats):
    assert idle_stats.total_tasks == 0
    assert busy_stats.total_tasks == 8


def test_task_overview_fields(gen_overview):
    assert gen_overview.task_id == 1001
    assert gen_overview.eval_request_type == "generate_until"
    assert gen_overview.queried_at > 0


def test_execution_profile_values():
    assert ExecutionProfile.RAW.value == "raw"
    assert ExecutionProfile.CHAT_NO_THINK.value == "chat_no_think"
    assert ExecutionProfile.CHAT_THINK.value == "chat_think"


def test_admission_decision():
    d = AdmissionDecision(
        action=AdmissionAction.ASK_NOW, profile=ExecutionProfile.CHAT_NO_THINK,
        reason="fast_path", estimated_finish_s=0.15, expected_reward=0.8,
    )
    assert d.action == AdmissionAction.ASK_NOW
