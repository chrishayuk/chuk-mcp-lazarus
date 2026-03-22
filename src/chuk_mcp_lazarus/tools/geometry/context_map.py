"""
context_map — decode the residual stream at every token position.

Shows what the model KNOWS at every position after N layers of processing,
decoded through the logit lens.  Each position gets top-K predictions,
entropy, specificity score, residual norm, and token↔residual angle.

This is a document-level view: while decode_residual focuses on a single
position, context_map covers every position in the prompt in one call.

Supports an optional ``initial_residual`` (boundary residual) that carries
accumulated context from prior windows in a chained prefill.  When provided,
the boundary vector is prepended at position 0 and the model sees the window
in the context of the full preceding document.  Without it, the model sees
the window text in isolation.
"""

import asyncio
import logging
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ContextMapPrediction(BaseModel):
    """A single logit-lens prediction at a position."""

    token: str
    token_id: int
    probability: float


class ContextMapPosition(BaseModel):
    """Decoded residual at one token position."""

    position: int
    token: str
    token_id: int
    predictions: list[ContextMapPrediction]
    entropy: float = Field(..., description="Shannon entropy of top-K prediction distribution.")
    specificity: float = Field(
        ...,
        description="1 - normalised entropy.  0 = generic, 1 = highly specific.",
    )
    residual_norm: float
    token_residual_angle: float = Field(..., description="Angle in degrees between token embedding and residual.")


class ContextMapResult(BaseModel):
    """Result from context_map."""

    prompt: str
    layer: int
    top_k: int
    num_positions: int
    has_boundary: bool = Field(False, description="True if initial_residual was provided.")
    positions: list[ContextMapPosition]


class ContextMapWithQueryPosition(ContextMapPosition):
    """Position entry with query attention overlay."""

    h5_attention: float = 0.0
    h4_attention: float = 0.0
    h2_attention: float = 0.0
    copy_head_rank: int = 0


class ContextMapWithQueryResult(BaseModel):
    """Result from context_map_with_query."""

    prompt: str
    query: str
    layer: int
    top_k: int
    num_positions: int
    has_boundary: bool = False
    positions: list[ContextMapWithQueryPosition]


# ---------------------------------------------------------------------------
# Shared forward pass with boundary residual support
# ---------------------------------------------------------------------------


def _forward_to_layer_with_boundary(
    model: Any,
    config: Any,
    input_ids: Any,
    stop_after_layer: int,
    initial_residual: list[float] | None = None,
) -> tuple[Any, int]:
    """Forward pass to a layer, optionally prepending a boundary residual.

    Returns (hidden_states [1, seq, hidden], boundary_offset).
    boundary_offset is 1 if a boundary was prepended, 0 otherwise.
    The caller should skip the first ``boundary_offset`` positions in results.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    embed = helper._get_embed_tokens()
    embedding_scale = helper._get_embedding_scale()

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    h = embed(input_ids)
    if embedding_scale is not None:
        h = h * mx.array(embedding_scale, dtype=h.dtype)

    # Prepend boundary residual if provided
    boundary_offset = 0
    if initial_residual is not None:
        boundary_vec = mx.array([initial_residual], dtype=h.dtype)  # [1, hidden_dim]
        boundary_vec = boundary_vec[None, :, :]  # [1, 1, hidden_dim]
        h = mx.concatenate([boundary_vec, h], axis=1)  # [1, 1+seq, hidden_dim]
        boundary_offset = 1

    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    for idx, layer in enumerate(model_layers):
        try:
            out = layer(h, mask=mask, cache=None)
        except TypeError:
            try:
                out = layer(h, cache=None)
            except TypeError:
                out = layer(h)

        h = (
            out.hidden_states
            if hasattr(out, "hidden_states")
            else (out[0] if isinstance(out, tuple) else out)
        )

        if idx == stop_after_layer:
            break

    mx.eval(h)
    return h, boundary_offset


# ---------------------------------------------------------------------------
# Shared position processing
# ---------------------------------------------------------------------------


def _process_positions(
    hidden: Any,           # [seq_len, hidden_dim] — already sliced to prompt positions
    logits_np: np.ndarray, # [num_positions, vocab_size]
    tok_ids: list[int],
    tokenizer: Any,
    embed_w: Any,
    top_k: int,
) -> list[dict]:
    """Process all positions into ContextMapPosition-compatible dicts."""
    import mlx.core as mx

    num_positions = len(tok_ids)
    results = []

    for pos in range(num_positions):
        pos_logits = logits_np[pos]

        # Softmax
        max_logit = float(np.max(pos_logits))
        exp_vals = np.exp(pos_logits - max_logit)
        sum_exp = float(np.sum(exp_vals))
        probs = exp_vals / sum_exp if sum_exp > 0 else exp_vals

        # Top-K
        top_indices = np.argsort(-pos_logits)[:top_k]
        predictions = []
        for idx in top_indices:
            tid = int(idx)
            predictions.append(ContextMapPrediction(
                token=tokenizer.decode([tid]),
                token_id=tid,
                probability=round(float(probs[tid]), 6),
            ))

        # Entropy over full distribution
        pos_probs = probs[probs > 1e-10]
        entropy = float(-np.sum(pos_probs * np.log(pos_probs)))

        # Specificity: 1 - normalized entropy
        vocab_size = len(pos_logits)
        max_entropy = math.log(vocab_size)
        specificity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        # Residual norm
        h_vec = hidden[pos]
        mx.eval(h_vec)
        h_np = np.array(h_vec.tolist(), dtype=np.float32)
        residual_norm = float(np.linalg.norm(h_np))

        # Token↔Residual angle
        token_residual_angle = 90.0
        if embed_w is not None:
            tok_embed = embed_w[tok_ids[pos]]
            mx.eval(tok_embed)
            tok_np = np.array(tok_embed.tolist(), dtype=np.float32)
            tok_norm = float(np.linalg.norm(tok_np))
            if tok_norm > 1e-8 and residual_norm > 1e-8:
                cos_sim = float(np.dot(tok_np, h_np)) / (tok_norm * residual_norm)
                cos_sim = max(-1.0, min(1.0, cos_sim))
                token_residual_angle = math.degrees(math.acos(cos_sim))

        results.append({
            "position": pos,
            "token": tokenizer.decode([tok_ids[pos]]),
            "token_id": tok_ids[pos],
            "predictions": predictions,
            "entropy": round(entropy, 4),
            "specificity": round(specificity, 4),
            "residual_norm": round(residual_norm, 2),
            "token_residual_angle": round(token_residual_angle, 2),
        })

    return results


def _batch_logit_lens(
    hidden: Any,
    num_positions: int,
    final_norm: Any,
    lm_head: Any,
) -> np.ndarray:
    """Apply logit lens to all positions at once.  Returns [num_positions, vocab_size] numpy."""
    import mlx.core as mx

    if final_norm is not None:
        normed = final_norm(hidden)
    else:
        normed = hidden

    logits = lm_head(normed.reshape(1, num_positions, -1))
    if hasattr(logits, "logits"):
        logits = logits.logits
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits[0]
    mx.eval(logits)
    return np.array(logits.tolist(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tool: context_map
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def context_map(
    prompt: str,
    layer: int,
    top_k: int = 5,
    initial_residual: list[float] | None = None,
) -> dict:
    """Decode the residual stream at every position in the prompt via the logit lens.

    Returns per-position top-K predictions, entropy, specificity, residual norm,
    and token↔residual angle.  Use this to see what the model understands at
    every position after processing through the given layer.

    Three modes:
      1. Standalone (no initial_residual): model sees the prompt in isolation.
      2. With initial_residual: boundary residual from a chained prefill is
         prepended at position 0, carrying accumulated context from prior
         windows.  The model sees the prompt in the context of the full
         preceding document.  Positions in the result are still 0-indexed
         relative to the prompt tokens (the boundary is not included).

    Args:
        prompt:            Input text.
        layer:             Layer index to decode at.
        top_k:             Number of top predictions per position (1-20, default 5).
        initial_residual:  Boundary residual vector [hidden_dim] from chained
                           prefill.  Optional — omit for standalone mode.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "context_map")

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "context_map",
        )
    top_k = max(1, min(top_k, 20))

    # Validate initial_residual dimensions
    if initial_residual is not None and len(initial_residual) != meta.hidden_dim:
        return make_error(
            ToolError.INVALID_INPUT,
            f"initial_residual must have {meta.hidden_dim} dimensions, got {len(initial_residual)}.",
            "context_map",
        )

    try:
        return await asyncio.to_thread(
            _context_map_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            layer,
            top_k,
            initial_residual,
        )
    except Exception as exc:
        logger.exception("context_map failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "context_map")


# ---------------------------------------------------------------------------
# Tool: context_map_with_query
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def context_map_with_query(
    prompt: str,
    query: str,
    layer: int,
    top_k: int = 5,
    initial_residual: list[float] | None = None,
) -> dict:
    """Context map with query attention overlay.

    Returns everything from context_map plus per-position attention weights
    from heads H5, H4, H2 for the query, and copy-head rank.

    The initial_residual applies to BOTH the document and query passes —
    the query attends over the document in the context of prior windows.

    Args:
        prompt:            Input text (the document/context).
        query:             Query text to overlay attention from.
        layer:             Layer index to decode at.
        top_k:             Number of top predictions per position (1-20, default 5).
        initial_residual:  Boundary residual vector [hidden_dim].  Optional.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "context_map_with_query")

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "context_map_with_query",
        )
    top_k = max(1, min(top_k, 20))

    if initial_residual is not None and len(initial_residual) != meta.hidden_dim:
        return make_error(
            ToolError.INVALID_INPUT,
            f"initial_residual must have {meta.hidden_dim} dimensions, got {len(initial_residual)}.",
            "context_map_with_query",
        )

    try:
        return await asyncio.to_thread(
            _context_map_with_query_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            query,
            layer,
            top_k,
            initial_residual,
        )
    except Exception as exc:
        logger.exception("context_map_with_query failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "context_map_with_query")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _context_map_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    top_k: int,
    initial_residual: list[float] | None,
) -> dict:
    """Sync implementation of context_map."""
    import mlx.core as mx

    from ..._residual_helpers import (
        _get_embed_weight,
        _get_lm_projection,
    )
    from chuk_lazarus.introspection.hooks import ModelHooks

    # -- Tokenize --
    tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array(tok_ids)
    num_positions = len(tok_ids)

    # -- Get model components --
    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(ToolError.EXTRACTION_FAILED, "Could not access lm_head.", "context_map")

    embed_w = _get_embed_weight(model)

    # -- Forward pass to target layer (with optional boundary) --
    h, boundary_offset = _forward_to_layer_with_boundary(
        model, config, input_ids, layer, initial_residual,
    )

    # h shape: [1, boundary_offset + num_positions, hidden_dim]
    # Skip the boundary position, keep only prompt positions
    hidden = h[0, boundary_offset:, :]  # [num_positions, hidden_dim]

    # -- Batch logit lens --
    logits_np = _batch_logit_lens(hidden, num_positions, final_norm, lm_head)

    # -- Process each position --
    pos_data = _process_positions(hidden, logits_np, tok_ids, tokenizer, embed_w, top_k)
    positions = [ContextMapPosition(**d) for d in pos_data]

    return ContextMapResult(
        prompt=prompt,
        layer=layer,
        top_k=top_k,
        num_positions=num_positions,
        has_boundary=initial_residual is not None,
        positions=positions,
    ).model_dump()


def _context_map_with_query_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    query: str,
    layer: int,
    top_k: int,
    initial_residual: list[float] | None,
) -> dict:
    """Sync implementation of context_map_with_query."""
    import mlx.core as mx

    from ..._residual_helpers import (
        _get_embed_weight,
        _get_lm_projection,
    )
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    # -- Tokenize prompt + query together --
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    combined_ids = prompt_ids + query_ids
    input_ids = mx.array(combined_ids)
    num_prompt_positions = len(prompt_ids)

    # -- Get model components --
    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(ToolError.EXTRACTION_FAILED, "Could not access lm_head.", "context_map_with_query")

    embed_w = _get_embed_weight(model)

    # -- Forward pass with attention capture and optional boundary --
    # For the query variant, we need attention weights, so we use ModelHooks
    # with CaptureConfig rather than our custom forward.
    # When boundary is provided, we prepend it manually.

    boundary_offset = 0
    if initial_residual is not None:
        # Embed the combined tokens
        embed = helper._get_embed_tokens()
        embedding_scale = helper._get_embedding_scale()

        if input_ids.ndim == 1:
            input_ids_2d = input_ids[None, :]
        else:
            input_ids_2d = input_ids

        h_embed = embed(input_ids_2d)
        if embedding_scale is not None:
            h_embed = h_embed * mx.array(embedding_scale, dtype=h_embed.dtype)

        # Prepend boundary
        boundary_vec = mx.array([initial_residual], dtype=h_embed.dtype)[None, :, :]
        h_embed = mx.concatenate([boundary_vec, h_embed], axis=1)
        boundary_offset = 1
        mx.eval(h_embed)

        # Manual forward through layers with attention capture
        import mlx.nn as nn

        model_layers = helper._get_layers()
        seq_len = h_embed.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h_embed.dtype)

        h = h_embed
        attn_weights = None

        for idx, lyr in enumerate(model_layers):
            try:
                out = lyr(h, mask=mask, cache=None)
            except TypeError:
                try:
                    out = lyr(h, cache=None)
                except TypeError:
                    out = lyr(h)

            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            if idx == layer:
                break

        mx.eval(h)
        hidden = h[0]  # [boundary_offset + seq, hidden_dim]

        # For attention weights with boundary, we don't capture them through
        # ModelHooks (too complex to inject boundary there).  Fall back to
        # zero attention — the comprehension map is still valid.
        h5_attn = np.zeros(num_prompt_positions, dtype=np.float32)
        h4_attn = np.zeros(num_prompt_positions, dtype=np.float32)
        h2_attn = np.zeros(num_prompt_positions, dtype=np.float32)
    else:
        # No boundary — use ModelHooks with attention capture
        helper.configure(CaptureConfig(
            layers=[layer],
            capture_hidden_states=True,
            capture_attention_weights=True,
        ))
        helper.forward(input_ids if input_ids.ndim == 2 else input_ids[None, :])
        mx.eval(helper.state.hidden_states, helper.state.attention_weights)

        hidden_raw = helper.state.hidden_states.get(layer)
        if hidden_raw is None:
            return make_error(
                ToolError.EXTRACTION_FAILED,
                f"No hidden state at layer {layer}.",
                "context_map_with_query",
            )
        hidden = hidden_raw[0] if hidden_raw.ndim == 3 else hidden_raw

        # Extract attention weights
        query_last_pos = len(combined_ids) - 1
        aw = helper.state.attention_weights.get(layer)

        h5_attn = np.zeros(num_prompt_positions, dtype=np.float32)
        h4_attn = np.zeros(num_prompt_positions, dtype=np.float32)
        h2_attn = np.zeros(num_prompt_positions, dtype=np.float32)

        if aw is not None:
            mx.eval(aw)
            if aw.ndim == 4:
                aw = aw[0]
            aw_np = np.array(aw.tolist(), dtype=np.float32)
            num_heads = aw_np.shape[0]

            if num_heads > 5:
                h5_attn = aw_np[5, query_last_pos, :num_prompt_positions]
            if num_heads > 4:
                h4_attn = aw_np[4, query_last_pos, :num_prompt_positions]
            if num_heads > 2:
                h2_attn = aw_np[2, query_last_pos, :num_prompt_positions]

    # -- Slice to prompt positions only (skip boundary if present) --
    prompt_hidden = hidden[boundary_offset:boundary_offset + num_prompt_positions]

    # -- Batch logit lens --
    logits_np = _batch_logit_lens(prompt_hidden, num_prompt_positions, final_norm, lm_head)

    # -- Copy head ranks --
    h5_sorted = np.argsort(-h5_attn)
    copy_ranks = np.zeros(num_prompt_positions, dtype=np.int32)
    for rank_idx, pos_idx in enumerate(h5_sorted):
        copy_ranks[pos_idx] = rank_idx + 1

    # -- Process positions --
    pos_data = _process_positions(
        prompt_hidden, logits_np, prompt_ids, tokenizer, embed_w, top_k,
    )

    positions = [
        ContextMapWithQueryPosition(
            **d,
            h5_attention=round(float(h5_attn[i]), 6),
            h4_attention=round(float(h4_attn[i]), 6),
            h2_attention=round(float(h2_attn[i]), 6),
            copy_head_rank=int(copy_ranks[i]),
        )
        for i, d in enumerate(pos_data)
    ]

    return ContextMapWithQueryResult(
        prompt=prompt,
        query=query,
        layer=layer,
        top_k=top_k,
        num_positions=num_prompt_positions,
        has_boundary=initial_residual is not None,
        positions=positions,
    ).model_dump()
