from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = False
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[list[int]] = field(default_factory=list)

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be >= 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k >= 0, "top_k must be >= 0"
