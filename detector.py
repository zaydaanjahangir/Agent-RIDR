from typing import Any

from .types import Channel1Config, Channel1Result, PromptBuildResult


def score_channel1(
    model: Any,
    prompt: PromptBuildResult,
    config: Channel1Config,
) -> Channel1Result:
    """
    Run Channel 1 detection for one prompt instance using precomputed important heads.
    """
    try:
        prompt.validate()
        config.validate()
    except Exception as e:
        return Channel1Result(
            status="unavailable",
            focus_score=None,
            decision=None,
            threshold=None,
            metadata={"reason": "invalid_input", "error": str(e)},
        )

    try:
        outputs = model(
            input_ids=prompt.input_ids,
            attention_mask=prompt.attention_mask,
            output_attentions=True,
        )
    except Exception as e:
        return Channel1Result(
            status="unavailable",
            focus_score=None,
            decision=None,
            threshold=None,
            metadata={"reason": "model_forward_failed", "error": str(e)},
        )

    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        return Channel1Result(
            status="unavailable",
            focus_score=None,
            decision=None,
            threshold=None,
            metadata={"reason": "attention_not_available"},
        )

    per_head_scores: list[float] = []
    skipped_heads: list[dict[str, Any]] = []

    start = prompt.instruction_token_start
    end = prompt.instruction_token_end

    requested_heads = len(config.important_heads)

    for layer, head in config.important_heads:
        if layer < 0 or layer >= len(attentions):
            skipped_heads.append(
                {"layer": layer, "head": head, "reason": "invalid_layer"}
            )
            continue

        layer_tensor = attentions[layer]
        if layer_tensor.ndim != 4:
            skipped_heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "reason": "invalid_tensor_shape",
                    "ndim": layer_tensor.ndim,
                }
            )
            continue

        num_heads = int(layer_tensor.shape[1])
        if head < 0 or head >= num_heads:
            skipped_heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "reason": "invalid_head_index",
                    "num_heads": num_heads,
                }
            )
            continue

        head_map = layer_tensor[0, head]  # [seq_len, seq_len]
        last_token_row = head_map[-1]  # [seq_len]

        if end > last_token_row.shape[0]:
            return Channel1Result(
                status="unavailable",
                focus_score=None,
                decision=None,
                threshold=None,
                metadata={
                    "reason": "instruction_span_out_of_bounds",
                    "instruction_token_start": start,
                    "instruction_token_end": end,
                    "seq_len": int(last_token_row.shape[0]),
                    "num_heads_requested": requested_heads,
                    "num_heads_used": len(per_head_scores),
                    "skipped_heads": skipped_heads,
                },
            )

        attn_to_instruction = last_token_row[start:end].sum().item()
        per_head_scores.append(float(attn_to_instruction))

    num_heads_used = len(per_head_scores)

    if num_heads_used == 0:
        return Channel1Result(
            status="unavailable",
            focus_score=None,
            decision=None,
            threshold=None,
            metadata={
                "reason": "no_valid_heads",
                "num_heads_requested": requested_heads,
                "num_heads_used": 0,
                "skipped_heads": skipped_heads,
            },
        )

    min_heads_required = max(1, requested_heads // 2)
    is_reliable = num_heads_used >= min_heads_required

    focus_score = float(sum(per_head_scores) / num_heads_used)

    if config.threshold is None:
        decision = None
    else:
        decision = "attack" if focus_score < config.threshold else "safe"

    return Channel1Result(
        status="ok",
        focus_score=focus_score,
        decision=decision,
        threshold=config.threshold,
        metadata={
            "model_id": config.model_id,
            "num_heads_requested": requested_heads,
            "num_heads_used": num_heads_used,
            "min_heads_required": min_heads_required,
            "is_reliable": is_reliable,
            "reliability_ratio": num_heads_used / requested_heads
            if requested_heads > 0
            else 0.0,
            "instruction_token_start": start,
            "instruction_token_end": end,
            "skipped_heads": skipped_heads,
        },
    )
