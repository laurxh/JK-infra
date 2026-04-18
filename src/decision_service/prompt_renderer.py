from __future__ import annotations
import logging
from decision_service.schemas import ExecutionProfile

logger = logging.getLogger(__name__)

_tokenizer = None
_tokenizer_load_attempted = False


def _get_tokenizer():
    global _tokenizer, _tokenizer_load_attempted
    if _tokenizer_load_attempted:
        return _tokenizer
    _tokenizer_load_attempted = True
    try:
        from transformers import AutoTokenizer
        import os
        model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3-32B")
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Loaded tokenizer from %s", model_path)
    except Exception as e:
        logger.warning("Could not load tokenizer: %s. Chat template rendering will use fallback.", e)
    return _tokenizer


def _fallback_chat_wrap(prompt: str, enable_thinking: bool) -> str:
    think_tag = "<think>\n" if enable_thinking else ""
    return (
        f"<|im_start|>system\nYou are a helpful assistant.{' /think' if not enable_thinking else ''}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_tag}"
    )


def render_prompt(prompt: str, profile: ExecutionProfile) -> str:
    if profile == ExecutionProfile.RAW:
        return prompt

    enable_thinking = profile == ExecutionProfile.CHAT_THINK
    tokenizer = _get_tokenizer()

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            return rendered
        except Exception as e:
            logger.warning("apply_chat_template failed: %s, using fallback", e)

    return _fallback_chat_wrap(prompt, enable_thinking)
