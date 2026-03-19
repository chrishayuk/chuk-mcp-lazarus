"""
Shared injection helpers for geometry tools.

Extracted from ``inject_residual.py`` so that sibling geometry tools
(branch_and_collapse, subspace_surgery) can import shared code without
creating tool-to-tool dependencies (Principle 4).
"""

from typing import Any

import numpy as np
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str
    token_id: int
    probability: float


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax in numpy."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) from logits, computed in numpy."""
    p = _softmax_np(p_logits)
    q = _softmax_np(q_logits)
    q = np.maximum(q, 1e-10)
    p = np.maximum(p, 1e-10)
    return float(np.sum(p * np.log(p / q)))


def _top_k_from_logits(
    logits_np: np.ndarray, tokenizer: Any, k: int = 10
) -> tuple[list[TokenPrediction], int, str, float]:
    """Extract top-k predictions from numpy logits.

    Returns (top_k_list, top1_id, top1_token, top1_prob).
    """
    probs = _softmax_np(logits_np)
    order = np.argsort(-probs)
    entries: list[TokenPrediction] = []
    for i in range(min(k, len(probs))):
        tid = int(order[i])
        entries.append(
            TokenPrediction(
                token=tokenizer.decode([tid]),
                token_id=tid,
                probability=round(float(probs[tid]), 6),
            )
        )
    top1_id = int(order[0])
    top1_token = tokenizer.decode([top1_id])
    top1_prob = float(probs[top1_id])
    return entries, top1_id, top1_token, top1_prob


# ---------------------------------------------------------------------------
# Forward pass with injection
# ---------------------------------------------------------------------------


def _run_forward_with_injection(
    model: Any,
    config: Any,
    input_ids: Any,
    target_layer: int,
    target_position: int,
    injection_vec: Any,
    metadata: Any,
) -> Any:
    """Run forward pass, replacing hidden state at target_layer/position.

    After layer target_layer completes normally, the hidden state at
    target_position is overwritten with injection_vec.  Remaining layers
    continue from this modified state.

    Returns the final hidden state [1, seq, hidden_dim].
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
        scale = mx.array(embedding_scale, dtype=h.dtype)
        h = h * scale

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    for layer_idx, layer_module in enumerate(model_layers):
        # Standard layer forward
        try:
            layer_out = layer_module(h, mask=mask, cache=None)
        except TypeError:
            try:
                layer_out = layer_module(h, cache=None)
            except TypeError:
                layer_out = layer_module(h)

        if hasattr(layer_out, "hidden_states"):
            h = layer_out.hidden_states
        elif isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out

        # After target layer, inject
        if layer_idx == target_layer:
            if injection_vec.ndim == 3:
                # All-position mode: replace entire hidden state tensor
                h = injection_vec
                new_seq_len = h.shape[1]
                if new_seq_len != seq_len:
                    mask = nn.MultiHeadAttention.create_additive_causal_mask(new_seq_len)
                    mask = mask.astype(h.dtype)
                    seq_len = new_seq_len
            else:
                # Single-position mode: splice at target position
                pos = target_position
                if pos < 0:
                    pos = seq_len + pos
                pos = max(0, min(pos, seq_len - 1))
                before = h[:, :pos, :]
                after = h[:, pos + 1 :, :]
                injected = injection_vec.reshape(1, 1, -1)
                h = mx.concatenate([before, injected, after], axis=1)

    return h


# ---------------------------------------------------------------------------
# Generation from hidden state
# ---------------------------------------------------------------------------


def _generate_from_hidden(
    model: Any,
    tokenizer: Any,
    final_hidden: Any,
    final_norm: Any,
    lm_head: Any,
    input_ids: Any,
    position: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """Generate text starting from a modified hidden state.

    First token comes from the injected hidden state projected through
    norm + lm_head.  Subsequent tokens use normal model forward passes.
    """
    import mlx.core as mx

    from ..._residual_helpers import _extract_position, _norm_project

    # First token from injected hidden state
    vec = _extract_position(final_hidden, position)
    first_logits = _norm_project(final_norm, lm_head, vec)
    mx.eval(first_logits)

    if temperature <= 0:
        first_id = int(mx.argmax(first_logits).item())
    else:
        probs = mx.softmax(first_logits / temperature)
        first_id = int(mx.random.categorical(mx.log(probs)).item())

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if first_id == eos_token_id:
        return tokenizer.decode([first_id], skip_special_tokens=True), 1

    generated_ids = [first_id]

    # Subsequent tokens: normal model forward passes
    if input_ids.ndim == 1:
        flat_ids = input_ids
    else:
        flat_ids = input_ids.reshape(-1)
    current_ids = mx.concatenate([flat_ids, mx.array([first_id])])

    for _ in range(max_new_tokens - 1):
        logits = model(current_ids[None, :] if current_ids.ndim == 1 else current_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        if hasattr(logits, "logits"):
            logits = logits.logits
        next_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]

        if temperature <= 0:
            next_id = int(mx.argmax(next_logits).item())
        else:
            probs = mx.softmax(next_logits / temperature)
            next_id = int(mx.random.categorical(mx.log(probs)).item())

        if next_id == eos_token_id:
            break
        generated_ids.append(next_id)
        current_ids = mx.concatenate([current_ids.reshape(-1), mx.array([next_id])])

    return (
        tokenizer.decode(generated_ids, skip_special_tokens=True),
        len(generated_ids),
    )
