from typing import Any

from .types import Channel1Config, Channel1Result, PromptBuildResult


def score_channel1(
    model: Any,
    prompt: PromptBuildResult,
    config: Channel1Config,
) -> Channel1Result:
    """
    Run Channel 1 detection for one prompt instance using precomputed important heads.

    Responsibilities:
    - run a forward pass with attention outputs enabled
    - extract per-layer, per-head attention maps
    - compute attention from the last input token to the instruction token span
      for each important head
    - average those per-head scores into the focus score FS
    - compare FS against the configured threshold
    - return either a valid result or an unavailable result

    Args:
        model:
            Supported causal language model exposing per-layer, per-head attentions.
        prompt:
            PromptBuildResult produced by the prompt builder.
        config:
            Static Channel 1 detector configuration, including important heads
            and threshold.

    Returns:
        Channel1Result:
        - status="ok" with focus_score / decision / threshold populated, or
        - status="unavailable" if Channel 1 assumptions are not satisfied.

    Notes:
        This implements the paper's runtime detection path:
        compute the focus score by averaging instruction-attention over important
        heads, then compare against threshold t. :contentReference[oaicite:0]{index=0}
        The decision rule is reject when FS < t. :contentReference[oaicite:1]{index=1}
    """
