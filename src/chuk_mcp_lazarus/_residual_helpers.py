"""
Shared residual-stream helpers.

Functions extracted from ``tools/residual_tools.py`` so that multiple
tool subpackages can import them without creating tool-to-tool
dependencies (Principle 4: tools never import each other)
"""

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ._serialize import to_pylist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer introspection
# ---------------------------------------------------------------------------


def _has_sublayers(layer: Any) -> bool:
    """Check if a layer has the standard transformer sub-components."""
    return (
        hasattr(layer, "self_attn")
        and hasattr(layer, "input_layernorm")
        and hasattr(layer, "post_attention_layernorm")
        and hasattr(layer, "mlp")
    )


def _has_four_norms(layer: Any) -> bool:
    """Check if a layer uses the Gemma-style 4-norm pattern.

    Gemma 3 (and potentially other models) apply post-norms to
    component outputs before the residual add, and use a separate
    pre_feedforward_layernorm before the MLP.
    """
    return hasattr(layer, "pre_feedforward_layernorm") and hasattr(
        layer, "post_feedforward_layernorm"
    )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _l2_norm(vec: mx.array) -> float:
    """Compute L2 norm of a vector."""
    return float(mx.sqrt(mx.sum(vec * vec)))


def _extract_position(tensor: mx.array, position: int) -> mx.array:
    """Extract a single position from a [batch, seq, hidden] or [seq, hidden] tensor."""
    if tensor.ndim == 3:
        return tensor[0, position, :]
    elif tensor.ndim == 2:
        return tensor[position, :]
    return tensor


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _compute_clustering_scores(
    labels: list[str],
    similarity_matrix: list[list[float]],
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Compute within-cluster, between-cluster similarity, and separation score.

    Returns (within_sim, between_sim, separation_score).
    between_sim keys are "label1 vs label2" strings (JSON-safe).
    """
    label_groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        label_groups.setdefault(label, []).append(i)

    # Within-cluster
    within_sim: dict[str, float] = {}
    for label, indices in label_groups.items():
        if len(indices) < 2:
            within_sim[label] = 1.0
            continue
        sims = []
        for i in indices:
            for j in indices:
                if i < j:
                    sims.append(similarity_matrix[i][j])
        within_sim[label] = round(sum(sims) / len(sims), 6) if sims else 1.0

    # Between-cluster
    between_sim: dict[str, float] = {}
    unique_labels = sorted(label_groups.keys())
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            if i < j:
                sims = []
                for idx1 in label_groups[l1]:
                    for idx2 in label_groups[l2]:
                        sims.append(similarity_matrix[idx1][idx2])
                key = f"{l1} vs {l2}"
                between_sim[key] = round(sum(sims) / len(sims), 6) if sims else 0.0

    avg_within = sum(within_sim.values()) / len(within_sim) if within_sim else 0.0
    avg_between = sum(between_sim.values()) / len(between_sim) if between_sim else 0.0
    separation = round(avg_within - avg_between, 6)

    return within_sim, between_sim, separation


# ---------------------------------------------------------------------------
# LM head / unembedding helpers
# ---------------------------------------------------------------------------


def _get_lm_projection(model: Any) -> Any:
    """Get the correct logit projection function.

    For tied-embedding models, the model's lm_head attribute may have a
    different vocab size than embed_tokens (e.g. Gemma 3 4B-it has 262,144
    in lm_head but 262,208 in embed_tokens).  The backbone uses
    embed_tokens.as_linear() for tied embeddings, so we must do the same.
    """
    # Check if model uses tied embeddings with as_linear
    tie = getattr(model, "tie_word_embeddings", False)
    inner = getattr(model, "model", model)
    embed = getattr(inner, "embed_tokens", None)

    if tie and embed is not None and hasattr(embed, "as_linear"):
        return embed.as_linear

    # Fallback: explicit lm_head
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head

    # Last resort: from ModelHooks
    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model)
    return helper._get_lm_head()


def _project_to_logits(lm_head: Any, vec: mx.array) -> mx.array:
    """Project a [hidden_dim] vector through the LM head → [vocab_size] logits."""
    out = lm_head(vec.reshape(1, 1, -1))
    if hasattr(out, "logits"):
        return out.logits[0, 0]
    if isinstance(out, tuple):
        return out[0][0, 0]
    return out[0, 0]


def _get_embed_weight(model: Any) -> mx.array | None:
    """Return the raw embedding weight matrix as an ``mx.array``.

    Handles chuk-lazarus wrappers where ``embed_tokens.weight`` may be
    an ``nn.Embedding`` object rather than a plain array.  In that case
    the actual array lives at ``embed_tokens.weight.weight``.
    """
    inner = getattr(model, "model", model)
    embed = getattr(inner, "embed_tokens", None)
    if embed is None:
        return None

    w = getattr(embed, "weight", None)
    if w is None:
        return None

    # Plain mx.array
    if isinstance(w, mx.array):
        return w

    # nn.Embedding wrapper — real array is one level deeper
    ww = getattr(w, "weight", None)
    if isinstance(ww, mx.array):
        return ww

    return None


def _get_unembed_vector(model: Any, token_id: int) -> mx.array | None:
    """Get the unembedding vector for a specific token.

    Returns the row of the output projection matrix for *token_id*.
    For tied-embedding models this is ``embed_tokens.weight[token_id]``;
    for separate lm_head models it is ``lm_head.weight[token_id]``.
    Falls back to projecting a one-hot through lm_head if no direct
    weight access is possible.

    Returns a ``[hidden_dim]`` vector, or ``None`` on failure.
    """
    # Try tied embeddings first
    tie = getattr(model, "tie_word_embeddings", False)
    embed_w = _get_embed_weight(model)

    if tie and embed_w is not None:
        return embed_w[token_id]

    # Try explicit lm_head weight
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm = model.lm_head
        if hasattr(lm, "weight") and isinstance(lm.weight, mx.array):
            return lm.weight[token_id]

    # Try embedding weight
    if embed_w is not None:
        return embed_w[token_id]

    # No direct weight access available — return None and let caller
    # produce a clear error rather than allocating a huge matrix.
    return None


def _resolve_target_token(
    tokenizer: Any,
    full_logits: mx.array,
    target_token: str | None,
) -> tuple[int, str]:
    """Resolve a target-token string to ``(token_id, decoded_text)``.

    * If *target_token* is ``None``, returns the model's top-1 prediction.
    * Otherwise tries both the bare string and a space-prefixed variant,
      picks whichever has the higher model logit.

    Raises ``ValueError`` if the token cannot be encoded.
    """
    if target_token is None:
        tid = int(mx.argmax(full_logits).item())
        return tid, tokenizer.decode([tid])

    candidates: list[int] = []
    for variant in (target_token, " " + target_token):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            candidates.append(ids[0])

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if not unique:
        raise ValueError(f"Could not encode target token {target_token!r}")

    tid = max(unique, key=lambda t: float(full_logits[t].item()))
    return tid, tokenizer.decode([tid])


def _norm_project(final_norm: Any, lm_head: Any, vec: mx.array) -> mx.array:
    """Apply final norm then project through lm_head.  Returns [vocab] logits."""
    if final_norm is not None:
        normed = final_norm(vec.reshape(1, 1, -1))[0, 0]
    else:
        normed = vec
    return _project_to_logits(lm_head, normed)


# ---------------------------------------------------------------------------
# Decomposition forward pass
# ---------------------------------------------------------------------------


def _run_decomposition_forward(
    model: Any,
    config: Any,
    input_ids: mx.array,
    layers: list[int],
) -> dict:
    """
    Manual forward pass capturing per-layer attention and FFN sub-outputs.

    For TransformerBlock layers (which have self_attn, input_layernorm,
    post_attention_layernorm, mlp), we run sub-components separately to
    get individual contributions.  For other block types, we run the full
    block and set attn/ffn to None (caller falls back to 50/50 split).

    Returns dict with keys:
        embeddings:    mx.array
        hidden_states: {layer_idx: mx.array}
        prev_hidden:   {layer_idx: mx.array}  (state entering each layer)
        attn_outputs:  {layer_idx: mx.array | None}
        ffn_outputs:   {layer_idx: mx.array | None}
    """
    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    embed = helper._get_embed_tokens()
    embedding_scale = helper._get_embedding_scale()

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    # Embeddings — cast scale to the embedding dtype to match
    # the backbone's own forward pass (e.g. Gemma casts sqrt(hidden_size)
    # to bfloat16, which rounds 50.596 → 50.5; we must match this).
    h = embed(input_ids)
    if embedding_scale is not None:
        scale = mx.array(embedding_scale, dtype=h.dtype)
        h = h * scale
    embeddings = mx.stop_gradient(h)

    # Causal mask
    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    layers_set = set(layers)
    max_layer = max(layers) if layers else 0

    hidden_states: dict[int, mx.array] = {}
    prev_hiddens: dict[int, mx.array] = {}
    attn_outputs: dict[int, mx.array | None] = {}
    ffn_outputs: dict[int, mx.array | None] = {}

    for layer_idx, layer in enumerate(model_layers):
        if layer_idx > max_layer:
            break

        should_capture = layer_idx in layers_set

        if should_capture:
            prev_hiddens[layer_idx] = mx.stop_gradient(h)

        if should_capture and _has_sublayers(layer):
            # --- Decomposed forward (mirrors the layer's __call__) ---
            normed = layer.input_layernorm(h)

            # self_attn: try with mask, fall back without (SSM layers)
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

            if _has_four_norms(layer):
                # Gemma-style: post-norm on outputs, separate pre-FFN norm
                attn_normed = layer.post_attention_layernorm(attn_out)
                h_post_attn = h + attn_normed

                ffn_input = layer.pre_feedforward_layernorm(h_post_attn)
                ffn_out_raw = layer.mlp(ffn_input)
                ffn_normed = layer.post_feedforward_layernorm(ffn_out_raw)
                h = h_post_attn + ffn_normed

                attn_outputs[layer_idx] = mx.stop_gradient(attn_normed)
                ffn_outputs[layer_idx] = mx.stop_gradient(ffn_normed)
            else:
                # Standard: pre-norm only
                h_post_attn = h + attn_out

                normed2 = layer.post_attention_layernorm(h_post_attn)
                ffn_out = layer.mlp(normed2)

                if hasattr(layer, "dropout") and layer.dropout is not None:
                    ffn_out = layer.dropout(ffn_out)

                h = h_post_attn + ffn_out

                attn_outputs[layer_idx] = mx.stop_gradient(attn_out)
                ffn_outputs[layer_idx] = mx.stop_gradient(ffn_out)

            hidden_states[layer_idx] = mx.stop_gradient(h)
        else:
            # --- Standard forward (cannot decompose) ---
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

            if should_capture:
                hidden_states[layer_idx] = mx.stop_gradient(h)
                attn_outputs[layer_idx] = None
                ffn_outputs[layer_idx] = None

    return {
        "embeddings": embeddings,
        "hidden_states": hidden_states,
        "prev_hidden": prev_hiddens,
        "attn_outputs": attn_outputs,
        "ffn_outputs": ffn_outputs,
    }


# ---------------------------------------------------------------------------
# Tokenizer helpers (extracted from activation_tools.py)
# ---------------------------------------------------------------------------


def _tokenize(tokenizer: Any, prompt: str) -> mx.array:
    """Encode a prompt and return MLX token IDs."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    return mx.array(ids)


def _token_text(tokenizer: Any, token_ids: mx.array, position: int) -> str:
    """Decode a single token at the given position."""
    ids_list = to_pylist(token_ids)
    if isinstance(ids_list[0], list):
        ids_list = ids_list[0]
    idx = position if position >= 0 else len(ids_list) + position
    idx = max(0, min(idx, len(ids_list) - 1))
    return tokenizer.decode([ids_list[idx]])
