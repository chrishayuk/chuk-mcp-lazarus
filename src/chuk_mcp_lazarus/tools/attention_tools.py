"""
Single-model attention inspection tools: attention_pattern, attention_heads.

These tools expose the model's attention patterns for interpretability.
They reuse _compute_attention_weights from _compare.py (manual Q*K
computation with RoPE) since MLX's fused attention kernel doesn't
expose intermediate attention weights.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import mlx.core as mx
from pydantic import BaseModel, Field

from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class AttendedPosition(BaseModel):
    """A token position and its attention weight."""

    position: int
    token: str
    weight: float


class HeadPattern(BaseModel):
    """Attention pattern for a single head at one layer."""

    head: int
    attention_weights: list[float]
    top_attended: list[AttendedPosition]


class LayerAttentionPattern(BaseModel):
    """Attention patterns for all heads at one layer."""

    layer: int
    num_heads: int
    heads: list[HeadPattern]


class AttentionPatternResult(BaseModel):
    """Result from attention_pattern."""

    prompt: str
    token_position: int
    token_text: str
    tokens: list[str]
    num_layers_analyzed: int
    patterns: list[LayerAttentionPattern]


class HeadAnalysis(BaseModel):
    """Entropy and focus analysis for a single head."""

    layer: int
    head: int
    entropy: float
    max_attention: float
    top_attended_positions: list[AttendedPosition]


class AttentionHeadsResult(BaseModel):
    """Result from attention_heads."""

    prompt: str
    tokens: list[str]
    num_heads_analyzed: int
    heads: list[HeadAnalysis]
    summary: dict = Field(..., description="Most focused and most diffuse heads.")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def attention_pattern(
    prompt: str,
    layers: list[int] | None = None,
    token_position: int = -1,
    top_k: int = 5,
) -> dict:
    """
    Extract attention patterns at specified layers for a single model.

    Shows which tokens the specified position attends to, per head.
    Critical for understanding information routing: "what is token X
    looking at?"

    Args:
        prompt:         Input text.
        layers:         Layer indices to analyze. None = sample 3 layers.
        token_position: Which token's attention to show (-1 = last, default).
        top_k:          Number of top-attended positions per head (default: 5).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "attention_pattern",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        layers = [0, num_layers // 2, num_layers - 1]

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "attention_pattern",
        )

    if top_k < 1 or top_k > 100:
        return make_error(
            ToolError.INVALID_INPUT,
            f"top_k must be 1-100, got {top_k}.",
            "attention_pattern",
        )

    try:
        result = await asyncio.to_thread(
            _attention_pattern_impl,
            state.model,
            state.config,
            state.tokenizer,
            prompt,
            sorted(layers),
            token_position,
            top_k,
        )
        return result

    except Exception as e:
        logger.exception("attention_pattern failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "attention_pattern")


def _attention_pattern_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    token_position: int,
    top_k: int,
) -> dict:
    """Compute attention patterns using manual Q*K projection."""
    from .._compare import _compute_attention_weights

    # Get token labels
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    num_tokens = len(token_ids)

    # Resolve position
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    token_text = tokens[pos]

    # Compute attention weights
    attn_weights = _compute_attention_weights(model, config, tokenizer, prompt, layers)

    patterns = []
    for layer_idx in layers:
        w = attn_weights.get(layer_idx)
        if w is None:
            continue

        # Shape: [batch, heads, seq, seq] -> [heads, seq, seq]
        if w.ndim == 4:
            w = w[0]

        num_heads = w.shape[0]
        head_patterns = []

        for head_idx in range(num_heads):
            # Attention from token_position to all other tokens
            head_weights = w[head_idx, pos, :].tolist()

            # Top-k attended positions
            indices = sorted(range(len(head_weights)), key=lambda i: head_weights[i], reverse=True)[
                :top_k
            ]
            top_attended = [
                AttendedPosition(
                    position=i,
                    token=tokens[i] if i < len(tokens) else f"[{i}]",
                    weight=round(head_weights[i], 6),
                )
                for i in indices
            ]

            head_patterns.append(
                HeadPattern(
                    head=head_idx,
                    attention_weights=[round(w, 6) for w in head_weights],
                    top_attended=top_attended,
                )
            )

        patterns.append(
            LayerAttentionPattern(
                layer=layer_idx,
                num_heads=num_heads,
                heads=head_patterns,
            )
        )

    result = AttentionPatternResult(
        prompt=prompt,
        token_position=token_position,
        token_text=token_text,
        tokens=tokens,
        num_layers_analyzed=len(patterns),
        patterns=patterns,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def attention_heads(
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 5,
) -> dict:
    """
    Analyze attention head entropy and focus across layers.

    High entropy = diffuse attention (head looks everywhere equally).
    Low entropy = focused attention (head concentrates on specific tokens).
    Helps identify which heads are "doing work" vs attending uniformly.

    Args:
        prompt: Input text.
        layers: Layer indices to analyze. None = sample 3 layers.
        top_k:  Number of top-attended positions per head (default: 5).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "attention_heads",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        layers = [0, num_layers // 2, num_layers - 1]

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "attention_heads",
        )

    if top_k < 1 or top_k > 100:
        return make_error(
            ToolError.INVALID_INPUT,
            f"top_k must be 1-100, got {top_k}.",
            "attention_heads",
        )

    try:
        result = await asyncio.to_thread(
            _attention_heads_impl,
            state.model,
            state.config,
            state.tokenizer,
            prompt,
            sorted(layers),
            top_k,
        )
        return result

    except Exception as e:
        logger.exception("attention_heads failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "attention_heads")


def _attention_heads_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    top_k: int,
) -> dict:
    """Compute per-head entropy and focus analysis."""
    from .._compare import _compute_attention_weights

    # Get token labels
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Compute attention weights
    attn_weights = _compute_attention_weights(model, config, tokenizer, prompt, layers)

    heads = []
    for layer_idx in layers:
        w = attn_weights.get(layer_idx)
        if w is None:
            continue

        if w.ndim == 4:
            w = w[0]

        num_heads = w.shape[0]
        seq_len = w.shape[-1]
        max_entropy = math.log(seq_len) if seq_len > 1 else 1.0

        for head_idx in range(num_heads):
            # Use last token's attention for analysis
            head_weights = w[head_idx, -1, :]
            head_weights_list = head_weights.tolist()

            # Entropy: -sum(p * log(p))
            eps = 1e-10
            log_w = mx.log(mx.clip(head_weights, eps, 1.0))
            entropy = float(-mx.sum(head_weights * log_w))
            # Normalize by max entropy for interpretability
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            max_attn = float(mx.max(head_weights))

            # Top-k attended positions
            indices = sorted(
                range(len(head_weights_list)), key=lambda i: head_weights_list[i], reverse=True
            )[:top_k]
            top_attended = [
                AttendedPosition(
                    position=i,
                    token=tokens[i] if i < len(tokens) else f"[{i}]",
                    weight=round(head_weights_list[i], 6),
                )
                for i in indices
            ]

            heads.append(
                HeadAnalysis(
                    layer=layer_idx,
                    head=head_idx,
                    entropy=round(normalized_entropy, 6),
                    max_attention=round(max_attn, 6),
                    top_attended_positions=top_attended,
                )
            )

    # Summary: most focused (lowest entropy) and most diffuse (highest entropy)
    sorted_by_entropy = sorted(heads, key=lambda h: h.entropy)
    most_focused = [
        {"layer": h.layer, "head": h.head, "entropy": h.entropy} for h in sorted_by_entropy[:5]
    ]
    most_diffuse = [
        {"layer": h.layer, "head": h.head, "entropy": h.entropy} for h in sorted_by_entropy[-5:]
    ]

    result = AttentionHeadsResult(
        prompt=prompt,
        tokens=tokens,
        num_heads_analyzed=len(heads),
        heads=heads,
        summary={
            "most_focused_heads": most_focused,
            "most_diffuse_heads": most_diffuse,
        },
    )
    return result.model_dump()
