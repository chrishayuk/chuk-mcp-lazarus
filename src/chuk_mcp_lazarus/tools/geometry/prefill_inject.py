"""
prefill_inject — partial forward pass and vector injection test.

Tools:
    prefill_to_layer — Run the model to a given layer, return hidden state.
    kv_inject_test   — Replace factual KV attention with a 12-byte signal and
                       compare output distributions.

The injection formula used by kv_inject_test:

    h_new[pos] = h[pos] + (coefficient - dot(h[pos], e_unit)) * e_unit

This sets the residual stream's component in the target token's unembedding
direction to exactly ``coefficient``, matching what a copy head would have
written via attention.  If the KL divergence between full-attention and
injected outputs is near zero, the factual content is genuinely 1-dimensional
and compressible to (token_id, coefficient) — 12 bytes.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import _resolve_token_to_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class PrefillResult(BaseModel):
    """Result from prefill_to_layer."""

    prompt: str
    layer: int
    position: int
    num_tokens: int
    hidden_state: list[float] = Field(
        ..., description="Residual stream at (layer, position) [hidden_dim]."
    )
    hidden_norm: float
    top_raw_logits: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Top tokens by raw dot product with unembedding (no final norm applied). "
            "Shows logit-lens prediction at this depth."
        ),
    )


class TokenProbEntry(BaseModel):
    """A token with its probability under both full and injected distributions."""

    token: str
    token_id: int
    full_prob: float
    injected_prob: float
    delta: float


class KvInjectTestResult(BaseModel):
    """Result from kv_inject_test."""

    prompt: str
    token: str
    token_id: int
    coefficient: float
    inject_layer: int
    position: int
    full_target_prob: float = Field(
        ..., description="Target token probability under full attention (baseline)."
    )
    injected_target_prob: float = Field(
        ..., description="Target token probability after vec injection."
    )
    target_prob_delta: float
    kl_divergence: float = Field(
        ..., description="KL(full || injected). Near zero = injection is faithful."
    )
    top_k_comparison: list[TokenProbEntry]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_position(num_tokens: int, position: int) -> int:
    pos = position if position >= 0 else num_tokens + position
    return max(0, min(pos, num_tokens - 1))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _kl_divergence_np(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) from raw logits."""
    p = _softmax_np(p_logits)
    q = np.maximum(_softmax_np(q_logits), 1e-10)
    p = np.maximum(p, 1e-10)
    return float(np.sum(p * np.log(p / q)))


def _top_k_tokens(logits_np: np.ndarray, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    probs = _softmax_np(logits_np)
    order = np.argsort(-probs)
    return [
        {
            "token": tokenizer.decode([int(order[i])]),
            "token_id": int(order[i]),
            "probability": round(float(probs[order[i]]), 6),
        }
        for i in range(min(k, len(probs)))
    ]


def _run_forward_to_layer(
    model: Any,
    config: Any,
    input_ids: Any,
    stop_after_layer: int,
) -> Any:
    """Forward pass from embedding through stop_after_layer. Returns h [1, seq, hidden_dim]."""
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

    seq_len = input_ids.shape[1]
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
    return h


def _run_forward_from_layer(
    model: Any,
    config: Any,
    h: Any,
    start_layer: int,
    final_norm: Any,
    lm_head: Any,
    pos: int,
) -> np.ndarray:
    """Continue forward from start_layer and return numpy logits [vocab] at pos."""
    import mlx.core as mx
    import mlx.nn as nn

    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()

    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    for idx in range(start_layer, len(model_layers)):
        layer = model_layers[idx]
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

    vec = h[0, pos, :]
    if final_norm is not None:
        vec = final_norm(vec.reshape(1, 1, -1))[0, 0]
    out = lm_head(vec.reshape(1, 1, -1))
    if hasattr(out, "logits"):
        logits = out.logits[0, 0]
    elif isinstance(out, tuple):
        logits = out[0][0, 0]
    else:
        logits = out[0, 0]
    mx.eval(logits)
    return np.array(logits.tolist(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tool: prefill_to_layer
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def prefill_to_layer(
    prompt: str,
    layer: int,
    position: int = -1,
    top_k_tokens: int = 5,
) -> dict:
    """Run the model forward to a given layer and return the hidden state.

    Returns the residual stream vector at (layer, position) after that
    layer's computation has completed.  Useful for inspecting what the
    model has computed at a given depth, or for verifying the bare-query
    residual is orthogonal to the answer space before injection.

    The top_raw_logits field provides a logit-lens view at this depth:
    what the model would predict if it projected from here without the
    final layer norm.

    Args:
        prompt:       Input text.
        layer:        Layer to stop at (layer N's output is returned).
        position:     Token position to extract (-1 = last token).
        top_k_tokens: Number of top raw logit tokens to return.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "prefill_to_layer"
        )

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "prefill_to_layer",
        )

    try:
        return await asyncio.to_thread(
            _prefill_to_layer_impl,
            state.model,
            state.config,
            state.tokenizer,
            prompt,
            layer,
            position,
            max(1, min(top_k_tokens, 20)),
        )
    except Exception as exc:
        logger.exception("prefill_to_layer failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "prefill_to_layer")


def _prefill_to_layer_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    position: int,
    top_k_tokens: int,
) -> dict:
    import mlx.core as mx

    from ..residual_tools import _get_lm_projection

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    h = _run_forward_to_layer(model, config, input_ids, layer)
    vec = h[0, pos, :]
    mx.eval(vec)
    vec_np = np.array(vec.tolist(), dtype=np.float32)

    lm_head = _get_lm_projection(model)
    top_raw: list[dict[str, Any]] = []
    if lm_head is not None:
        raw_logits = lm_head(vec.reshape(1, 1, -1))
        if hasattr(raw_logits, "logits"):
            raw_logits = raw_logits.logits[0, 0]
        elif isinstance(raw_logits, tuple):
            raw_logits = raw_logits[0][0, 0]
        else:
            raw_logits = raw_logits[0, 0]
        mx.eval(raw_logits)
        top_raw = _top_k_tokens(
            np.array(raw_logits.tolist(), dtype=np.float32), tokenizer, top_k_tokens
        )

    return PrefillResult(
        prompt=prompt,
        layer=layer,
        position=position,
        num_tokens=num_tokens,
        hidden_state=vec_np.tolist(),
        hidden_norm=round(float(np.linalg.norm(vec_np)), 4),
        top_raw_logits=top_raw,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: kv_inject_test
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def kv_inject_test(
    prompt: str,
    token: str,
    coefficient: float,
    inject_layer: int,
    position: int = -1,
    top_k: int = 10,
) -> dict:
    """Injection test: replace KV attention output with a 12-byte signal.

    Proves that the factual content carried by the copy circuit is
    1-dimensional.  Instead of running the full KV attention mechanism,
    the tool sets the residual stream's component in the target token's
    unembedding direction to exactly ``coefficient`` at ``inject_layer``.

    Injection formula:
        h[pos] += (coefficient - dot(h[pos], e_unit)) * e_unit

    Two forward passes are compared:
    - Full:     normal model inference (all layers, full KV cache).
    - Injected: forward to inject_layer, apply formula, continue to output.

    Low KL divergence means the injection is faithful — the (token_id,
    coefficient) pair carries all the factual signal that the copy head
    would have written via attention.

    Use the coefficient from extract_attention_output (the top projection
    value) and inject_layer = copy_layer + 1 (one layer after the copy head).

    Args:
        prompt:       Input text (the full factual prompt).
        token:        Fact token to inject (e.g. "Paris").
        coefficient:  Scalar from extract_attention_output content projection.
        inject_layer: Layer at which to inject (typically copy_layer + 1).
        position:     Token position to inject at (-1 = last token).
        top_k:        Tokens to include in the comparison table.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "kv_inject_test")

    meta = state.metadata
    if inject_layer < 0 or inject_layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"inject_layer {inject_layer} out of [0, {meta.num_layers - 1}].",
            "kv_inject_test",
        )

    try:
        return await asyncio.to_thread(
            _kv_inject_test_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            token,
            coefficient,
            inject_layer,
            position,
            max(1, min(top_k, 50)),
        )
    except Exception as exc:
        logger.exception("kv_inject_test failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "kv_inject_test")


def _kv_inject_test_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    token: str,
    coefficient: float,
    inject_layer: int,
    position: int,
    top_k: int,
) -> dict:
    import mlx.core as mx

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..residual_tools import (
        _get_lm_projection,
        _get_unembed_vector,
        _norm_project,
        _run_decomposition_forward,
    )

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    last_layer = meta.num_layers - 1

    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        raise ValueError("Could not access the language model head.")

    token_id = _resolve_token_to_id(tokenizer, token)
    if token_id is None:
        raise ValueError(f"Cannot encode {token!r} to a single token ID.")
    token_text = tokenizer.decode([token_id])

    u = _get_unembed_vector(model, token_id)
    if u is None:
        raise ValueError("Could not extract unembedding vector.")
    mx.eval(u)
    u_np = np.array(u.tolist(), dtype=np.float32)
    u_norm = float(np.linalg.norm(u_np))
    if u_norm < 1e-8:
        raise ValueError(f"Unembedding vector for {token!r} has near-zero norm.")
    e_unit_np = u_np / u_norm

    # ── Full forward pass ─────────────────────────────────────────────────
    decomp_full = _run_decomposition_forward(model, config, input_ids, [last_layer])
    mx.eval(*decomp_full["hidden_states"].values())

    full_logits_mx = _norm_project(
        final_norm, lm_head, decomp_full["hidden_states"][last_layer][0, pos, :]
    )
    mx.eval(full_logits_mx)
    full_logits_np = np.array(full_logits_mx.tolist(), dtype=np.float32)

    # ── Injected forward pass ─────────────────────────────────────────────
    if inject_layer == 0:
        # Inject into embedding output before any transformer layer
        embed = helper._get_embed_tokens()
        embedding_scale = helper._get_embedding_scale()
        h_pre = embed(input_ids)
        if embedding_scale is not None:
            h_pre = h_pre * mx.array(embedding_scale, dtype=h_pre.dtype)
        mx.eval(h_pre)
    else:
        h_pre = _run_forward_to_layer(model, config, input_ids, inject_layer - 1)

    # Apply injection: set e_unit component to coefficient
    h_pos_np = np.array(h_pre[0, pos, :].tolist(), dtype=np.float32)
    current_component = float(np.dot(h_pos_np, e_unit_np))
    h_pos_np = h_pos_np + (coefficient - current_component) * e_unit_np

    h_pre_full_np = np.array(h_pre[0].tolist(), dtype=np.float32)  # [seq, hidden]
    h_pre_full_np[pos] = h_pos_np
    h_injected_mx = mx.array(h_pre_full_np[None])  # [1, seq, hidden]
    mx.eval(h_injected_mx)

    injected_logits_np = _run_forward_from_layer(
        model, config, h_injected_mx, inject_layer, final_norm, lm_head, pos
    )

    # ── Compare ───────────────────────────────────────────────────────────
    kl = _kl_divergence_np(full_logits_np, injected_logits_np)
    full_probs = _softmax_np(full_logits_np)
    inj_probs = _softmax_np(injected_logits_np)

    full_target_prob = float(full_probs[token_id])
    inj_target_prob = float(inj_probs[token_id])

    union_ids = list(
        dict.fromkeys(list(np.argsort(-full_probs)[:top_k]) + list(np.argsort(-inj_probs)[:top_k]))
    )[:top_k]

    comparison = [
        TokenProbEntry(
            token=tokenizer.decode([tid]),
            token_id=tid,
            full_prob=round(float(full_probs[tid]), 6),
            injected_prob=round(float(inj_probs[tid]), 6),
            delta=round(float(inj_probs[tid]) - float(full_probs[tid]), 6),
        )
        for tid in union_ids
    ]

    summary: dict[str, Any] = {
        "kl_divergence": round(kl, 6),
        "statistically_indistinguishable": kl < 0.001,
        "target_token": token_text,
        "target_token_id": token_id,
        "full_target_prob": round(full_target_prob, 6),
        "injected_target_prob": round(inj_target_prob, 6),
        "target_prob_delta": round(inj_target_prob - full_target_prob, 6),
        "component_before_injection": round(current_component, 4),
        "component_after_injection": round(coefficient, 4),
        "interpretation": (
            "12 bytes = full KV cache for this fact"
            if kl < 0.001
            else f"KL={kl:.4f} — verify copy head and coefficient (use batch_dla_scan + extract_attention_output)"
        ),
    }

    return KvInjectTestResult(
        prompt=prompt,
        token=token_text,
        token_id=token_id,
        coefficient=coefficient,
        inject_layer=inject_layer,
        position=position,
        full_target_prob=round(full_target_prob, 6),
        injected_target_prob=round(inj_target_prob, 6),
        target_prob_delta=round(inj_target_prob - full_target_prob, 6),
        kl_divergence=round(kl, 6),
        top_k_comparison=comparison,
        summary=summary,
    ).model_dump()
