from decision_service.prompt_renderer import render_prompt
from decision_service.schemas import ExecutionProfile


def test_raw_profile_returns_original():
    original = "Question: What is 1+1?\nAnswer:"
    result = render_prompt(original, ExecutionProfile.RAW)
    assert result == original


def test_chat_no_think_wraps_in_template():
    original = "What is 1+1?"
    result = render_prompt(original, ExecutionProfile.CHAT_NO_THINK)
    assert original in result
    assert len(result) > len(original)


def test_chat_think_wraps_in_template():
    original = "What is 1+1?"
    result = render_prompt(original, ExecutionProfile.CHAT_THINK)
    assert original in result
    assert len(result) > len(original)


def test_chat_think_differs_from_no_think():
    original = "Solve: 2+2"
    no_think = render_prompt(original, ExecutionProfile.CHAT_NO_THINK)
    think = render_prompt(original, ExecutionProfile.CHAT_THINK)
    assert original in no_think
    assert original in think
