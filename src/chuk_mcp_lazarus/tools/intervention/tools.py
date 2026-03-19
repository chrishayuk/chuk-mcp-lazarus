"""
Component intervention tool: surgical causal intervention on attention, FFN, or heads.

Runs two forward passes -- clean and intervened -- and compares top-k predictions,
KL divergence, and whether the top-1 prediction changed.  Proves causality by
showing that zeroing/scaling a specific component changes model output.
"""

import asyncio
import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str
    token_id: int
    probability: float


class InterventionResult(BaseModel):
    """Result from component_intervention."""

    prompt: str
    layer: int
    component: str
    intervention: str
    scale_factor: float
    head: int | None = None
    token_position: int
    token_text: str
    original_top_k: list[TokenPrediction]
    intervened_top_k: list[TokenPrediction]
    kl_divergence: float = Field(..., description="KL(original || intervened).")
    target_delta: float = Field(
        ..., description="Change in original top-1's logit after intervention."
    )
    original_top1: str
    intervened_top1: str
    top1_changed: bool
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool: component_intervention
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def component_intervention(
    prompt: str,
    layer: int,
    component: str,
    intervention: str = "zero",
    scale_factor: float = 0.0,
    head: int | None = None,
    top_k: int = 10,
    token_position: int = -1,
) -> dict:
    """
    Surgically intervene on a model component and measure the effect.

    Runs two forward passes: clean and intervened. At the target layer,
    zero or scale the attention output, FFN output, or a single attention
    head, then compare predictions.

    Proves causality: if zeroing attention at layer 2 destroys the correct
    prediction, that layer's attention is causally necessary.

    Args:
        prompt:         Input text.
        layer:          Layer index to intervene on.
        component:      "attention", "ffn", or "head".
        intervention:   "zero" (sets scale_factor=0) or "scale".
        scale_factor:   Multiplier for the component (used with "scale").
        head:           Head index (required when component="head").
        top_k:          Number of top predictions to return (1-50).
        token_position: Token position to analyze (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "component_intervention",
        )

    if component not in ("attention", "ffn", "head"):
        return make_error(
            ToolError.INVALID_INPUT,
            f"component must be 'attention', 'ffn', or 'head', got {component!r}.",
            "component_intervention",
        )

    if intervention not in ("zero", "scale"):
        return make_error(
            ToolError.INVALID_INPUT,
            f"intervention must be 'zero' or 'scale', got {intervention!r}.",
            "component_intervention",
        )

    if intervention == "zero":
        scale_factor = 0.0

    if component == "head" and head is None:
        return make_error(
            ToolError.INVALID_INPUT,
            "head index is required when component='head'.",
            "component_intervention",
        )

    num_heads = state.metadata.num_attention_heads
    if component == "head" and head is not None:
        if head < 0 or head >= num_heads:
            return make_error(
                ToolError.INVALID_INPUT,
                f"head {head} out of range [0, {num_heads - 1}].",
                "component_intervention",
            )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "component_intervention",
        )

    if top_k < 1 or top_k > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            f"top_k must be 1-50, got {top_k}.",
            "component_intervention",
        )

    try:
        result = await asyncio.to_thread(
            _component_intervention_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            component,
            intervention,
            scale_factor,
            head,
            top_k,
            token_position,
        )
        return result
    except ValueError as exc:
        return make_error(ToolError.INVALID_INPUT, str(exc), "component_intervention")
    except Exception as e:
        logger.exception("component_intervention failed")
        return make_error(ToolError.INTERVENTION_FAILED, str(e), "component_intervention")


# ---------------------------------------------------------------------------
# Sync implementation
# ---------------------------------------------------------------------------


def _run_forward_with_intervention(
    model: Any,
    config: Any,
    input_ids: mx.array,
    target_layer: int,
    component: str,
    scale_factor: float,
    head: int | None,
    metadata: Any,
) -> mx.array:
    """Run forward pass with intervention at target_layer.

    Returns the final hidden state after all layers.
    """
    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import _has_four_norms, _has_sublayers

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    embed = helper._get_embed_tokens()
    embedding_scale = helper._get_embedding_scale()

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    h = embed(input_ids)
    if embedding_scale is not None:
        scale = mx.array(embedding_scale, dtype=h.dtype)
        h = h * scale

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    num_heads = metadata.num_attention_heads
    num_kv_heads = metadata.num_kv_heads or num_heads
    head_dim = metadata.head_dim or (metadata.hidden_dim // num_heads)

    for layer_idx, layer in enumerate(model_layers):
        if layer_idx == target_layer and _has_sublayers(layer):
            # Decompose and apply intervention
            normed = layer.input_layernorm(h)

            # Compute attention output
            try:
                attn_out, _ = layer.self_attn(normed, mask=mask, cache=None)
            except TypeError:
                try:
                    attn_out, _ = layer.self_attn(normed, cache=None)
                except (TypeError, ValueError):
                    attn_result = layer.self_attn(normed)
                    attn_out = attn_result[0] if isinstance(attn_result, tuple) else attn_result

            if hasattr(layer, "dropout") and layer.dropout is not None:
                attn_out = layer.dropout(attn_out)

            # Apply attention intervention
            if component == "attention":
                attn_out = attn_out * scale_factor
            elif component == "head":
                assert head is not None
                attn_out = _intervene_head(
                    layer,
                    normed,
                    attn_out,
                    mask,
                    head,
                    scale_factor,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )

            if _has_four_norms(layer):
                attn_normed = layer.post_attention_layernorm(attn_out)
                h_post_attn = h + attn_normed
                ffn_input = layer.pre_feedforward_layernorm(h_post_attn)
                ffn_out_raw = layer.mlp(ffn_input)
                if component == "ffn":
                    ffn_out_raw = ffn_out_raw * scale_factor
                ffn_normed = layer.post_feedforward_layernorm(ffn_out_raw)
                h = h_post_attn + ffn_normed
            else:
                h_post_attn = h + attn_out
                normed2 = layer.post_attention_layernorm(h_post_attn)
                ffn_out = layer.mlp(normed2)
                if component == "ffn":
                    ffn_out = ffn_out * scale_factor
                if hasattr(layer, "dropout") and layer.dropout is not None:
                    ffn_out = layer.dropout(ffn_out)
                h = h_post_attn + ffn_out
        else:
            # Standard forward
            try:
                layer_out = layer(h, mask=mask, cache=None)
            except TypeError:
                try:
                    layer_out = layer(h, cache=None)
                except TypeError:
                    layer_out = layer(h)
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

    return h


def _intervene_head(
    layer: Any,
    normed: mx.array,
    original_attn_out: mx.array,
    mask: mx.array,
    head: int,
    scale_factor: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> mx.array:
    """Apply intervention to a single attention head.

    Decomposes attention into per-head outputs via o_proj weight slicing,
    scales the target head, and reconstructs.
    """
    attn = layer.self_attn

    queries = attn.q_proj(normed)
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    batch_size = normed.shape[0]
    seq_len = normed.shape[1]

    queries = queries.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        queries = attn.q_norm(queries)
    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        keys = attn.k_norm(keys)
    if hasattr(attn, "rope") and attn.rope is not None:
        queries = attn.rope(queries)
        keys = attn.rope(keys)
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    attn_scale = getattr(attn, "scale", head_dim**-0.5)
    context = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn_scale,
        mask=mask,
    )
    mx.eval(context)

    # Per-head output through o_proj slices
    o_weight = attn.o_proj.weight

    # Reconstruct with intervention on target head
    head_outputs = []
    for h_i in range(num_heads):
        head_ctx = context[:, h_i, :, :]  # [batch, seq, head_dim]
        w_slice = o_weight[:, h_i * head_dim : (h_i + 1) * head_dim]
        head_out = head_ctx @ w_slice.T  # [batch, seq, hidden_dim]
        if h_i == head:
            head_out = head_out * scale_factor
        head_outputs.append(head_out)

    # Sum all head outputs + bias
    result = head_outputs[0]
    for ho in head_outputs[1:]:
        result = result + ho

    if hasattr(attn.o_proj, "bias") and attn.o_proj.bias is not None:
        result = result + attn.o_proj.bias

    return result


def _component_intervention_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    metadata: Any,
    prompt: str,
    layer: int,
    component: str,
    intervention: str,
    scale_factor: float,
    head: int | None,
    top_k: int,
    token_position: int,
) -> dict:
    """Sync implementation of component_intervention."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    from ..._residual_helpers import _get_lm_projection, _norm_project

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = input_ids.shape[-1]

    # Resolve position
    from ..._serialize import to_pylist

    ids_list = to_pylist(input_ids)
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    token_text = tokenizer.decode([ids_list[pos]])

    # Clean forward pass
    hooks = ModelHooks(model, model_config=config)
    num_layers = metadata.num_layers
    hooks.configure(
        CaptureConfig(
            layers=[num_layers - 1],
            capture_hidden_states=True,
        )
    )
    hooks.forward(input_ids)

    lm_head = _get_lm_projection(model)
    final_norm = ModelHooks(model, model_config=config)._get_final_norm()

    h_clean = hooks.state.hidden_states.get(num_layers - 1)
    if h_clean is None:
        raise RuntimeError(f"No hidden state at layer {num_layers - 1}")

    from ..._residual_helpers import _extract_position

    vec_clean = _extract_position(h_clean, token_position)
    clean_logits = _norm_project(final_norm, lm_head, vec_clean)
    mx.eval(clean_logits)

    # Intervention forward pass
    h_intervened = _run_forward_with_intervention(
        model,
        config,
        input_ids,
        layer,
        component,
        scale_factor,
        head,
        metadata,
    )
    vec_int = _extract_position(h_intervened, token_position)
    int_logits = _norm_project(final_norm, lm_head, vec_int)
    mx.eval(int_logits)

    # Compute probabilities
    clean_probs = mx.softmax(clean_logits, axis=-1)
    int_probs = mx.softmax(int_logits, axis=-1)
    mx.eval(clean_probs, int_probs)

    # Original top-k
    clean_sorted = mx.argsort(clean_logits)[::-1]
    mx.eval(clean_sorted)
    clean_sorted_list = to_pylist(clean_sorted)

    original_top_k = []
    for tid in clean_sorted_list[:top_k]:
        original_top_k.append(
            TokenPrediction(
                token=tokenizer.decode([tid]),
                token_id=tid,
                probability=round(float(clean_probs[tid]), 6),
            )
        )

    # Intervened top-k
    int_sorted = mx.argsort(int_logits)[::-1]
    mx.eval(int_sorted)
    int_sorted_list = to_pylist(int_sorted)

    intervened_top_k = []
    for tid in int_sorted_list[:top_k]:
        intervened_top_k.append(
            TokenPrediction(
                token=tokenizer.decode([tid]),
                token_id=tid,
                probability=round(float(int_probs[tid]), 6),
            )
        )

    # KL divergence: KL(clean || intervened) = sum(clean * log(clean / intervened))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    clean_p = clean_probs + eps
    int_p = int_probs + eps
    kl = float(mx.sum(clean_p * mx.log(clean_p / int_p)))
    kl = max(0.0, kl)  # Numerical safety

    # Target delta: change in original top-1's logit
    orig_top1_id = clean_sorted_list[0]
    target_delta = float(int_logits[orig_top1_id]) - float(clean_logits[orig_top1_id])

    original_top1 = original_top_k[0].token if original_top_k else ""
    intervened_top1 = intervened_top_k[0].token if intervened_top_k else ""
    top1_changed = original_top1 != intervened_top1

    summary: dict[str, Any] = {
        "original_top1_probability": original_top_k[0].probability if original_top_k else 0.0,
        "intervened_top1_probability": intervened_top_k[0].probability if intervened_top_k else 0.0,
        "probability_change": round(
            (intervened_top_k[0].probability if intervened_top_k else 0.0)
            - (original_top_k[0].probability if original_top_k else 0.0),
            6,
        ),
        "kl_divergence": round(kl, 6),
        "effect_magnitude": "strong" if kl > 1.0 else "moderate" if kl > 0.1 else "weak",
    }

    result = InterventionResult(
        prompt=prompt,
        layer=layer,
        component=component,
        intervention=intervention,
        scale_factor=scale_factor,
        head=head,
        token_position=token_position,
        token_text=token_text,
        original_top_k=original_top_k,
        intervened_top_k=intervened_top_k,
        kl_divergence=round(kl, 6),
        target_delta=round(target_delta, 6),
        original_top1=original_top1,
        intervened_top1=intervened_top1,
        top1_changed=top1_changed,
        summary=summary,
    )
    return result.model_dump(exclude_none=True)
