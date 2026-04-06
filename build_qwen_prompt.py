from types import Channel1Config, Channel1Result, PromptBuildResult
from typing import Any


def build_qwen_prompt(
    instruction: str,
    data: str,
    tokenizer: Any,
) -> PromptBuildResult:
    """
    Build a Qwen-compatible prompt from a trusted instruction and untrusted data.

    Responsibilities:
    - construct the exact serialized prompt text for the supported Qwen path
    - tokenize that prompt into model-ready tensors
    - identify the contiguous token span corresponding to the trusted instruction
    - return all prompt-building artifacts needed by Channel 1 scoring

    Args:
        instruction:
            Trusted instruction segment I.
        data:
            Untrusted user/data segment D.
        tokenizer:
            Tokenizer for the supported Qwen model/runtime.

    Returns:
        PromptBuildResult containing:
        - full serialized prompt text
        - input_ids tensor
        - attention_mask tensor
        - instruction token start/end indices

    Raises:
        ValueError:
            If the instruction token span cannot be determined or is invalid.

    Notes:
        This function assumes the trusted instruction appears as one contiguous
        token span in the final serialized prompt for the first implementation.
    """
