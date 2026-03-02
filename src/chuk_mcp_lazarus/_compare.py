"""
Shared comparison kernels for two-model analysis.

Pure functions, no MCP imports. Each kernel takes two models and
returns structured results as plain Python dicts/lists.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


# ---------------------------------------------------------------------------
# Weight Divergence
# ---------------------------------------------------------------------------

def get_layer_weights(model: Any, layer_idx: int) -> dict[str, mx.array]:
    """Extract weight tensors for a specific layer."""
    layer = model.model.layers[layer_idx]
    weights = {}

    # Attention weights
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, name):
                weights[f"attn_{name[0]}"] = getattr(attn, name).weight

    # MLP weights
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(mlp, name):
                short = name.replace("_proj", "")
                weights[f"mlp_{short}"] = getattr(mlp, name).weight

    return weights


def weight_divergence(
    model_a: Any,
    model_b: Any,
    layers: list[int],
) -> list[dict]:
    """Compute per-layer, per-component weight divergence.

    Returns list of dicts with: layer, component, frobenius_norm_diff, cosine_similarity.
    """
    results = []

    for layer_idx in layers:
        weights_a = get_layer_weights(model_a, layer_idx)
        weights_b = get_layer_weights(model_b, layer_idx)

        for component in weights_a:
            if component not in weights_b:
                continue

            w_a = weights_a[component]
            w_b = weights_b[component]

            if w_a.shape != w_b.shape:
                continue

            diff = w_b - w_a

            # Frobenius norm (normalized by base norm)
            base_norm = float(mx.sqrt(mx.sum(w_a * w_a)))
            diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
            normalized_diff = diff_norm / (base_norm + 1e-8)

            # Cosine similarity
            flat_a = w_a.reshape(-1)
            flat_b = w_b.reshape(-1)
            dot = float(mx.sum(flat_a * flat_b))
            norm_a = float(mx.sqrt(mx.sum(flat_a * flat_a)))
            norm_b = float(mx.sqrt(mx.sum(flat_b * flat_b)))
            cos_sim = dot / (norm_a * norm_b + 1e-8)

            results.append({
                "layer": layer_idx,
                "component": component,
                "frobenius_norm_diff": round(normalized_diff, 6),
                "cosine_similarity": round(min(max(cos_sim, -1.0), 1.0), 6),
            })

    return results


# ---------------------------------------------------------------------------
# Activation Divergence
# ---------------------------------------------------------------------------

def _get_hidden_states(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    token_position: int = -1,
) -> dict[int, mx.array]:
    """Capture hidden states at specified layers for one prompt."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))

    hooks = ModelHooks(model, model_config=config)
    hooks.configure(
        CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
        )
    )
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)

    hidden = {}
    for layer_idx in layers:
        h = hooks.state.hidden_states.get(layer_idx)
        if h is not None:
            # Handle shapes: [hidden], [seq, hidden], [batch, seq, hidden]
            if h.ndim == 1:
                hidden[layer_idx] = h
            elif h.ndim == 2:
                hidden[layer_idx] = h[token_position]
            elif h.ndim == 3:
                hidden[layer_idx] = h[0, token_position]

    return hidden


def activation_divergence(
    model_a: Any,
    config_a: Any,
    model_b: Any,
    config_b: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    token_position: int = -1,
) -> list[dict]:
    """Compare hidden-state activations between two models across prompts.

    Returns list of dicts with: layer, prompt, cosine_similarity, l2_distance, relative_l2.
    """
    results = []

    for prompt in prompts:
        hidden_a = _get_hidden_states(model_a, config_a, tokenizer, prompt, layers, token_position)
        hidden_b = _get_hidden_states(model_b, config_b, tokenizer, prompt, layers, token_position)

        for layer_idx in layers:
            h_a = hidden_a.get(layer_idx)
            h_b = hidden_b.get(layer_idx)

            if h_a is None or h_b is None:
                continue

            # Cosine similarity
            dot = float(mx.sum(h_a * h_b))
            norm_a = float(mx.sqrt(mx.sum(h_a * h_a)))
            norm_b = float(mx.sqrt(mx.sum(h_b * h_b)))
            cos_sim = dot / (norm_a * norm_b + 1e-8)

            # L2 distance
            diff = h_b - h_a
            l2_dist = float(mx.sqrt(mx.sum(diff * diff)))

            # Relative L2 (normalized by average magnitude)
            avg_norm = (norm_a + norm_b) / 2
            rel_l2 = l2_dist / (avg_norm + 1e-8)

            results.append({
                "layer": layer_idx,
                "prompt": prompt,
                "cosine_similarity": round(min(max(cos_sim, -1.0), 1.0), 6),
                "l2_distance": round(l2_dist, 4),
                "relative_l2": round(rel_l2, 6),
            })

    return results


# ---------------------------------------------------------------------------
# Attention Divergence
# ---------------------------------------------------------------------------

def _js_divergence(p: mx.array, q: mx.array, eps: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    p = mx.clip(p, eps, 1.0)
    q = mx.clip(q, eps, 1.0)
    p = p / mx.sum(p, axis=-1, keepdims=True)
    q = q / mx.sum(q, axis=-1, keepdims=True)
    m = (p + q) / 2
    kl_pm = mx.sum(p * mx.log(p / m), axis=-1)
    kl_qm = mx.sum(q * mx.log(q / m), axis=-1)
    return float(mx.mean((kl_pm + kl_qm) / 2))


def _compute_attention_weights(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
) -> dict[int, mx.array]:
    """Compute attention weights by manually projecting Q and K.

    MLX's mx.fast.scaled_dot_product_attention is a fused kernel that
    doesn't expose intermediate attention weights. We capture hidden
    states at the layer *before* each target layer (the input to the
    target layer), then project through Q/K to compute attention weights.
    """
    from chuk_lazarus.introspection.hooks import (
        CaptureConfig, ModelHooks, PositionSelection,
    )

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))

    # We need hidden states from layers *before* the target layers.
    # Hidden state at layer N is the OUTPUT of layer N.
    # The INPUT to layer N is the output of layer N-1 (or embeddings for layer 0).
    # Capture both the predecessor layers and layer 0's pre-layer state.
    capture_layers = set()
    for l in layers:
        if l > 0:
            capture_layers.add(l - 1)
        capture_layers.add(l)
    capture_layers = sorted(capture_layers)

    hooks = ModelHooks(model, model_config=config)
    hooks.configure(
        CaptureConfig(
            layers=capture_layers,
            capture_hidden_states=True,
            # Must capture ALL positions — attention needs the full sequence
            # to produce meaningful [seq, seq] attention weight matrices.
            # Default (LAST) gives seq_len=1, making softmax trivially 1.0.
            positions=PositionSelection.ALL,
        )
    )
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)

    attn_weights = {}
    num_heads = getattr(config, "num_attention_heads", 8)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 2560) // num_heads)
    scale = head_dim ** -0.5

    for layer_idx in layers:
        # Get the INPUT to this layer = output of previous layer
        if layer_idx > 0:
            h = hooks.state.hidden_states.get(layer_idx - 1)
        else:
            # For layer 0, use the embedding output.
            # We approximate by using the layer 0 output (not ideal but close).
            h = hooks.state.hidden_states.get(0)

        if h is None:
            continue

        # Ensure [batch, seq, hidden] shape
        if h.ndim == 1:
            h = h[None, None, :]
        elif h.ndim == 2:
            h = h[None, :]

        try:
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn

            # Apply input norm (pre-attention layernorm)
            if hasattr(layer, "input_layernorm"):
                h_normed = layer.input_layernorm(h)
            else:
                h_normed = h

            # Project Q and K
            queries = attn.q_proj(h_normed)
            keys = attn.k_proj(h_normed)

            batch_size, seq_len = h_normed.shape[0], h_normed.shape[1]

            # Reshape to [batch, seq, heads, head_dim]
            queries = queries.reshape(batch_size, seq_len, num_heads, head_dim)
            keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim)

            # Transpose to [batch, heads, seq, head_dim]
            queries = queries.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)

            # Apply Q/K norms if present (Gemma-style)
            if hasattr(attn, "q_norm"):
                queries = attn.q_norm(queries)
            if hasattr(attn, "k_norm"):
                keys = attn.k_norm(keys)

            # Apply RoPE positional encoding (critical for meaningful attention)
            if hasattr(attn, "rope") and attn.rope is not None:
                queries = attn.rope(queries)
                keys = attn.rope(keys)

            # Repeat KV heads for GQA
            if hasattr(attn, "n_rep") and attn.n_rep > 1:
                keys = mx.repeat(keys, attn.n_rep, axis=1)
            elif num_kv_heads < num_heads:
                n_rep = num_heads // num_kv_heads
                keys = mx.repeat(keys, n_rep, axis=1)

            # Compute attention scores: [batch, heads, seq, seq]
            attn_scale = getattr(attn, "scale", scale)
            scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * attn_scale

            # Apply causal mask
            causal = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
            scores = scores + causal.astype(scores.dtype)

            weights = mx.softmax(scores, axis=-1)
            mx.eval(weights)

            attn_weights[layer_idx] = weights  # [batch, heads, seq, seq]

        except Exception:
            # Skip layers where manual computation fails
            continue

    return attn_weights


def attention_divergence(
    model_a: Any,
    config_a: Any,
    model_b: Any,
    config_b: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
) -> list[dict]:
    """Compare attention patterns between two models per head.

    Returns list of dicts with: layer, head, js_divergence, cosine_similarity.
    """
    attn_a = _compute_attention_weights(model_a, config_a, tokenizer, prompt, layers)
    attn_b = _compute_attention_weights(model_b, config_b, tokenizer, prompt, layers)

    results = []

    for layer_idx in layers:
        w_a = attn_a.get(layer_idx)
        w_b = attn_b.get(layer_idx)

        if w_a is None or w_b is None:
            continue

        # Attention weights shape: [batch, heads, seq, seq] or [heads, seq, seq]
        if w_a.ndim == 4:
            w_a = w_a[0]  # [heads, seq, seq]
        if w_b.ndim == 4:
            w_b = w_b[0]

        num_heads = w_a.shape[0]

        for head_idx in range(num_heads):
            # Last token's attention distribution
            head_a = w_a[head_idx, -1, :]
            head_b = w_b[head_idx, -1, :]

            js_div = _js_divergence(head_a[None, :], head_b[None, :])

            # Cosine similarity
            dot = float(mx.sum(head_a * head_b))
            norm_a = float(mx.sqrt(mx.sum(head_a * head_a)))
            norm_b = float(mx.sqrt(mx.sum(head_b * head_b)))
            cos_sim = dot / (norm_a * norm_b + 1e-8)

            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "js_divergence": round(js_div, 6),
                "cosine_similarity": round(min(max(cos_sim, -1.0), 1.0), 6),
            })

    return results
