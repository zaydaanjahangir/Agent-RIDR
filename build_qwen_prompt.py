from typing import Any

import torch

from .types import PromptBuildResult


def build_qwen_prompt(
    instruction: str,
    data: str,
    tokenizer: Any,
) -> PromptBuildResult:
    """
    Build a Qwen-compatible prompt and return the contiguous token span
    corresponding to the trusted instruction in the final tokenized prompt.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    if not isinstance(data, str) or not data.strip():
        raise ValueError("data must be a non-empty string")

    full_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": data},
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids: torch.Tensor = full_tokens["input_ids"]
    attention_mask: torch.Tensor = full_tokens["attention_mask"]

    prefix_messages = [
        {"role": "system", "content": ""},
    ]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix_plus_instruction_messages = [
        {"role": "system", "content": instruction},
    ]
    prefix_plus_instruction_text = tokenizer.apply_chat_template(
        prefix_plus_instruction_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix_tokens = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    prefix_plus_instruction_tokens = tokenizer(
        prefix_plus_instruction_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    instruction_token_start = int(prefix_tokens.shape[1])
    instruction_token_end = int(prefix_plus_instruction_tokens.shape[1])

    result = PromptBuildResult(
        full_text=full_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        instruction_token_start=instruction_token_start,
        instruction_token_end=instruction_token_end,
        metadata={
            "model_family": "qwen",
            "prompt_format": "system_user_chat_template",
        },
    )
    result.validate()
    return result
