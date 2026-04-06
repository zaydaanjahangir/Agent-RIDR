from dataclasses import dataclass, field
from typing import Any, Literal

import torch

Channel1Status = Literal["ok", "unavailable"]
Channel1Decision = Literal["safe", "attack"]


@dataclass(frozen=True)
class PromptBuildResult:
    """
    Output of the prompt builder.

    instruction_token_end is exclusive.
    """

    full_text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    instruction_token_start: int
    instruction_token_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.instruction_token_end <= self.instruction_token_start:
            raise ValueError(
                "instruction_token_end must be greater than instruction_token_start"
            )

        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        if self.attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, seq_len]")

        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")

        seq_len = self.input_ids.shape[1]
        if self.instruction_token_end > seq_len:
            raise ValueError("instruction token span exceeds sequence length")

    @classmethod
    def example(cls) -> "PromptBuildResult":
        return cls(
            full_text="<system> Analyze sentiment </system><user> This movie is great </user>",
            input_ids=[[101, 200, 300, 400]],  # mock example
            attention_mask=[[1, 1, 1, 1]],
            instruction_token_start=1,
            instruction_token_end=3,
            metadata={"note": "example only, not real tokenization"},
        )


@dataclass(frozen=True)
class Channel1Config:
    """
    Static detector configuration.
    """

    model_id: str
    important_heads: list[tuple[int, int]]
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.important_heads:
            raise ValueError("important_heads must not be empty")

        for layer, head in self.important_heads:
            if layer < 0 or head < 0:
                raise ValueError("important head indices must be non-negative")

        if not (0 <= self.threshold <= 1):
            raise ValueError("threshold must be between 0 and 1")

    @classmethod
    def example(cls) -> "Channel1Config":
        return cls(
            model_id="Qwen2-1.5B-Instruct",
            important_heads=[
                (10, 6),
                (11, 0),
                (11, 2),
                (11, 8),
            ],
            threshold=0.25,
            metadata={"source": "paper default-style example"},
        )


@dataclass(frozen=True)
class Channel1Result:
    """
    Output of Channel 1 detector.
    """

    status: Channel1Status
    focus_score: float | None
    decision: Channel1Decision | None
    threshold: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return self.status == "ok"

    @classmethod
    def example_success(cls) -> "Channel1Result":
        return cls(
            status="ok",
            focus_score=0.18,
            decision="attack",
            threshold=0.25,
            metadata={"num_heads_used": 3},
        )

    @classmethod
    def example_unavailable(cls) -> "Channel1Result":
        return cls(
            status="unavailable",
            focus_score=None,
            decision=None,
            threshold=None,
            metadata={"reason": "attention_not_available"},
        )
