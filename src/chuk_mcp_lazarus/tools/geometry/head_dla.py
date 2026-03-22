"""
head_dla — per-head Direct Logit Attribution and attention output analysis.

Tools:
    compute_dla              — DLA scalar for a single (layer, head, token) triple.
    batch_dla_scan           — Full layers × heads DLA matrix in one forward pass.
    extract_attention_output — Head output vector + content projection onto vocab.
    get_token_embedding      — Token input and unembedding vectors.
    extract_k_vector         — K vector for a specific KV head (addressing space).
    extract_q_vector         — Q vector for a specific query head (addressing space).

All DLA computations use raw (un-normalised) Direct Logit Attribution.
This is exact for head decomposition because o_proj is linear: the sum
of per-head DLA equals the layer's total raw attention DLA.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import _cosine_sim, _resolve_token_to_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class DlaHeadEntry(BaseModel):
    """Per-head DLA contribution at a single layer."""

    head: int
    dla: float = Field(..., description="Raw DLA: dot(head_output, unembed_vector).")
    fraction_of_layer: float = Field(
        ..., description="This head's |DLA| as fraction of total layer |DLA|."
    )
    top_token: str = Field(..., description="Token this head's output points toward most.")


class DlaLayerEntry(BaseModel):
    """DLA breakdown for all heads at a single layer."""

    layer: int
    is_decomposable: bool = Field(
        ..., description="False for non-standard blocks (Mamba, sliding-window, etc.)."
    )
    heads: list[DlaHeadEntry]
    layer_total_dla: float = Field(
        ..., description="Sum of per-head DLA values (signed). Equals layer raw DLA."
    )


class HotCell(BaseModel):
    """A (layer, head) pair with significant DLA, ranked by |DLA|."""

    layer: int
    head: int
    dla: float
    top_token: str
    fraction_of_layer: float
    abs_rank: int = Field(..., description="Rank by |DLA| across all scanned cells (1 = highest).")


class BatchDlaScanResult(BaseModel):
    """Result from batch_dla_scan."""

    prompt: str
    target_token: str
    target_token_id: int
    position: int
    num_layers_scanned: int
    num_heads: int
    layers: list[DlaLayerEntry]
    hot_cells: list[HotCell] = Field(..., description="Top cells ranked by |DLA|.")
    summary: dict[str, Any]


class ComputeDlaResult(BaseModel):
    """Result from compute_dla."""

    prompt: str
    layer: int
    head: int
    target_token: str
    target_token_id: int
    position: int
    dla: float = Field(..., description="Raw DLA: dot(head_output, unembed_vector).")
    fraction_of_layer: float
    top_token: str = Field(..., description="Token this head's output points toward most.")
    head_output_norm: float


class AttentionOutputResult(BaseModel):
    """Result from extract_attention_output."""

    prompt: str
    layer: int
    head: int
    position: int
    vector: list[float] = Field(..., description="Head output in hidden space [hidden_dim].")
    vector_norm: float
    top_projections: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Top-N projections onto token unembedding directions. "
            "Each entry: {token, token_id, coefficient, fraction}."
        ),
    )
    dimensionality: dict[str, Any] = Field(
        ...,
        description=(
            "Fraction of total |projection| mass captured by top-1/2/5 directions. "
            "dims_for_99pct <= 2 supports the 1-dimensional content claim."
        ),
    )


class EmbeddingResult(BaseModel):
    """Result from get_token_embedding."""

    token: str
    token_id: int
    unembedding: list[float] = Field(
        ...,
        description=(
            "Unembedding (output projection) vector [hidden_dim]. "
            "Used for DLA and content projection."
        ),
    )
    input_embedding: list[float] | None = Field(
        None,
        description=(
            "Input embedding vector [hidden_dim]. "
            "For tied-embedding models this equals unembedding."
        ),
    )
    embeddings_tied: bool
    unembedding_norm: float
    input_embedding_norm: float | None = None
    cosine_similarity: float | None = Field(
        None, description="Cosine similarity between input and unembedding (if both available)."
    )


class KVectorResult(BaseModel):
    """Result from extract_k_vector."""

    prompt: str
    layer: int
    kv_head: int
    position: int
    k_vector: list[float] = Field(..., description="K vector [head_dim] after RoPE.")
    k_norm: float
    num_kv_heads: int
    head_dim: int


class QVectorResult(BaseModel):
    """Result from extract_q_vector."""

    prompt: str
    layer: int
    head: int
    position: int
    q_vector: list[float] = Field(..., description="Q vector [head_dim] after RoPE.")
    q_norm: float
    num_heads: int
    head_dim: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_sublayers(layer: Any) -> bool:
    """True for standard transformer blocks with self_attn + mlp."""
    return hasattr(layer, "self_attn") and hasattr(layer, "input_layernorm")


def _resolve_position(num_tokens: int, position: int) -> int:
    """Resolve a negative position to an absolute index, clamped to [0, n-1]."""
    pos = position if position >= 0 else num_tokens + position
    return max(0, min(pos, num_tokens - 1))


def _compute_per_head_attention(
    target_layer: Any,
    prev_h: Any,  # mx.array [1, seq, hidden_dim]
    mask: Any,
    pos: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[Any, Any]:
    """Run attention sub-layer and return (context, o_weight).

    context: [1, num_heads, seq, head_dim] — SDPA output.
    o_weight: [hidden_dim, num_heads * head_dim] — output projection.
    """
    import mlx.core as mx

    attn = target_layer.self_attn
    seq_len = prev_h.shape[1]
    batch_size = prev_h.shape[0]

    normed = target_layer.input_layernorm(prev_h)
    queries = attn.q_proj(normed)
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

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
        queries, keys, values, scale=attn_scale, mask=mask
    )
    mx.eval(context)
    return context, attn.o_proj.weight


def _per_head_dla(
    context: Any,
    o_weight: Any,
    unembed_u: Any,
    pos: int,
    num_heads: int,
    head_dim: int,
    tokenizer: Any,
    lm_head: Any,
) -> list[tuple[float, str]]:
    """Compute (raw_DLA, top_token) for each head at position pos.

    Returns list of (dla_scalar, top_token_str) length num_heads.
    """
    import mlx.core as mx

    head_vecs: list[Any] = []
    for h in range(num_heads):
        head_ctx = context[0, h, pos, :]
        w_slice = o_weight[:, h * head_dim : (h + 1) * head_dim]
        head_vecs.append(head_ctx @ w_slice.T)

    mx.eval(*head_vecs)

    dla_vals = [float((hv * unembed_u).sum().item()) for hv in head_vecs]

    head_stack = mx.stack(head_vecs)
    all_logits = lm_head(head_stack.reshape(1, num_heads, -1))
    if hasattr(all_logits, "logits"):
        all_logits = all_logits.logits
    elif isinstance(all_logits, tuple):
        all_logits = all_logits[0]
    all_logits = all_logits[0]
    top_ids = mx.argmax(all_logits, axis=1)
    mx.eval(top_ids)
    _raw_ids: Any = top_ids.tolist() if hasattr(top_ids, "tolist") else top_ids
    top_ids_list: list[int] = [int(v) for v in _raw_ids]

    return [(dla_vals[h], tokenizer.decode([int(top_ids_list[h])])) for h in range(num_heads)]


# ---------------------------------------------------------------------------
# Tool: compute_dla
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def compute_dla(
    prompt: str,
    layer: int,
    head: int,
    target_token: str | None = None,
    position: int = -1,
) -> dict:
    """Compute Direct Logit Attribution (DLA) for a single (layer, head) pair.

    Projects the attention head's output vector onto the target token's
    unembedding direction.  Positive DLA means this head promotes the
    token; negative means it suppresses it.

    Use batch_dla_scan first to identify which heads have high DLA,
    then drill into specific cells with this tool.

    Args:
        prompt:       Input text.
        layer:        Layer index.
        head:         Attention head index.
        target_token: Token to attribute (None = model's top-1 prediction).
        position:     Token position to analyse (-1 = last token).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "compute_dla")

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "compute_dla",
        )
    if head < 0 or head >= meta.num_attention_heads:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Head {head} out of [0, {meta.num_attention_heads - 1}].",
            "compute_dla",
        )

    try:
        return await asyncio.to_thread(
            _compute_dla_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            head,
            target_token,
            position,
        )
    except Exception as exc:
        logger.exception("compute_dla failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "compute_dla")


def _compute_dla_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    head: int,
    target_token: str | None,
    position: int,
) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import (
        _get_lm_projection,
        _get_unembed_vector,
        _norm_project,
        _resolve_target_token,
        _run_decomposition_forward,
    )

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    num_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_heads
    head_dim = meta.head_dim or (meta.hidden_dim // num_heads)
    last_layer = meta.num_layers - 1

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        raise ValueError("Could not access the language model head.")

    decomp = _run_decomposition_forward(model, config, input_ids, sorted({layer, last_layer}))
    mx.eval(*decomp["prev_hidden"].values(), *decomp["hidden_states"].values())

    full_logits = _norm_project(final_norm, lm_head, decomp["hidden_states"][last_layer][0, pos, :])
    mx.eval(full_logits)
    target_id, target_text = _resolve_target_token(tokenizer, full_logits, target_token)

    u = _get_unembed_vector(model, target_id)
    if u is None:
        raise ValueError("Could not extract unembedding vector.")
    mx.eval(u)

    target_layer_module = model_layers[layer]
    if not _has_sublayers(target_layer_module):
        raise ValueError(f"Layer {layer} does not have a decomposable attention sub-layer.")

    seq_len = input_ids.shape[1]
    prev_h = decomp["prev_hidden"][layer]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(prev_h.dtype)

    context, o_weight = _compute_per_head_attention(
        target_layer_module, prev_h, mask, pos, num_heads, num_kv_heads, head_dim
    )
    all_dla = _per_head_dla(context, o_weight, u, pos, num_heads, head_dim, tokenizer, lm_head)
    layer_total_abs = sum(abs(v) for v, _ in all_dla)
    target_dla, top_token = all_dla[head]
    fraction = abs(target_dla) / layer_total_abs if layer_total_abs > 1e-8 else 0.0

    head_ctx = context[0, head, pos, :]
    w_slice = o_weight[:, head * head_dim : (head + 1) * head_dim]
    head_out = head_ctx @ w_slice.T
    mx.eval(head_out)
    head_norm = float(np.linalg.norm(np.array(head_out.tolist(), dtype=np.float32)))

    return ComputeDlaResult(
        prompt=prompt,
        layer=layer,
        head=head,
        target_token=target_text,
        target_token_id=target_id,
        position=position,
        dla=round(target_dla, 6),
        fraction_of_layer=round(fraction, 6),
        top_token=top_token,
        head_output_norm=round(head_norm, 4),
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: batch_dla_scan
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def batch_dla_scan(
    prompt: str,
    target_token: str | None = None,
    layers: list[int] | None = None,
    position: int = -1,
    top_k_cells: int = 5,
) -> dict:
    """Scan all (layer, head) pairs for DLA contribution to a target token.

    Runs a single forward pass then recomputes per-head attention at each
    layer to build the full layers × heads DLA matrix.  Identifies copy
    heads — the sparse set responsible for factual retrieval.

    The result is the heatmap data for Layer 1 of copy-circuit analysis.
    Click a hot cell to drill into extract_attention_output.

    Args:
        prompt:       Input text.
        target_token: Token to attribute (None = model's top-1 prediction).
        layers:       Layers to scan (None = all layers).
        position:     Token position to analyse (-1 = last token).
        top_k_cells:  Number of hot cells to return ranked by |DLA|.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "batch_dla_scan")

    meta = state.metadata
    num_layers = meta.num_layers
    resolved_layers: list[int] = (
        list(range(num_layers)) if layers is None else [int(x) for x in layers]
    )

    oob = [lay for lay in resolved_layers if lay < 0 or lay >= num_layers]
    if oob:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {oob} out of [0, {num_layers - 1}].",
            "batch_dla_scan",
        )
    if not resolved_layers:
        return make_error(ToolError.INVALID_INPUT, "layers must not be empty.", "batch_dla_scan")

    try:
        return await asyncio.to_thread(
            _batch_dla_scan_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            target_token,
            resolved_layers,
            position,
            max(1, min(top_k_cells, 20)),
        )
    except Exception as exc:
        logger.exception("batch_dla_scan failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "batch_dla_scan")


def _batch_dla_scan_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    target_token: str | None,
    layers: list[int],
    position: int,
    top_k_cells: int,
) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import (
        _get_lm_projection,
        _get_unembed_vector,
        _norm_project,
        _resolve_target_token,
        _run_decomposition_forward,
    )

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    num_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_heads
    head_dim = meta.head_dim or (meta.hidden_dim // num_heads)
    last_layer = meta.num_layers - 1

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        raise ValueError("Could not access the language model head.")

    # One pass captures prev_hidden for every layer we need
    decomp = _run_decomposition_forward(
        model, config, input_ids, sorted(set(layers) | {last_layer})
    )
    mx.eval(*decomp["prev_hidden"].values(), *decomp["hidden_states"].values())

    full_logits = _norm_project(final_norm, lm_head, decomp["hidden_states"][last_layer][0, pos, :])
    mx.eval(full_logits)
    target_id, target_text = _resolve_target_token(tokenizer, full_logits, target_token)

    u = _get_unembed_vector(model, target_id)
    if u is None:
        raise ValueError("Could not extract unembedding vector.")
    mx.eval(u)

    seq_len = input_ids.shape[1]
    layer_entries: list[DlaLayerEntry] = []
    all_cells: list[tuple[float, int, int, str]] = []  # (|dla|, layer, head, top_token)

    for layer_idx in sorted(layers):
        target_layer_module = model_layers[layer_idx]
        prev_h = decomp["prev_hidden"].get(layer_idx)

        if prev_h is None or not _has_sublayers(target_layer_module):
            layer_entries.append(
                DlaLayerEntry(
                    layer=layer_idx,
                    is_decomposable=False,
                    heads=[
                        DlaHeadEntry(head=h, dla=0.0, fraction_of_layer=0.0, top_token="")
                        for h in range(num_heads)
                    ],
                    layer_total_dla=0.0,
                )
            )
            continue

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(prev_h.dtype)
        try:
            context, o_weight = _compute_per_head_attention(
                target_layer_module, prev_h, mask, pos, num_heads, num_kv_heads, head_dim
            )
            per_head = _per_head_dla(
                context, o_weight, u, pos, num_heads, head_dim, tokenizer, lm_head
            )
        except Exception as exc:
            logger.warning("Layer %d DLA failed: %s", layer_idx, exc)
            layer_entries.append(
                DlaLayerEntry(
                    layer=layer_idx,
                    is_decomposable=False,
                    heads=[
                        DlaHeadEntry(head=h, dla=0.0, fraction_of_layer=0.0, top_token="")
                        for h in range(num_heads)
                    ],
                    layer_total_dla=0.0,
                )
            )
            continue

        total_abs = sum(abs(v) for v, _ in per_head)
        layer_total = sum(v for v, _ in per_head)

        head_entries: list[DlaHeadEntry] = []
        for h_idx, (dla_val, top_tok) in enumerate(per_head):
            frac = abs(dla_val) / total_abs if total_abs > 1e-8 else 0.0
            head_entries.append(
                DlaHeadEntry(
                    head=h_idx,
                    dla=round(dla_val, 6),
                    fraction_of_layer=round(frac, 6),
                    top_token=top_tok,
                )
            )
            all_cells.append((abs(dla_val), layer_idx, h_idx, top_tok))

        layer_entries.append(
            DlaLayerEntry(
                layer=layer_idx,
                is_decomposable=True,
                heads=head_entries,
                layer_total_dla=round(layer_total, 6),
            )
        )

    all_cells.sort(key=lambda x: x[0], reverse=True)

    hot_cells: list[HotCell] = []
    for rank, (abs_dla, lay, hd, top_tok) in enumerate(all_cells[:top_k_cells]):
        signed_dla = 0.0
        frac = 0.0
        for entry in layer_entries:
            if entry.layer == lay:
                for he in entry.heads:
                    if he.head == hd:
                        signed_dla = he.dla
                        frac = he.fraction_of_layer
                        break
        hot_cells.append(
            HotCell(
                layer=lay,
                head=hd,
                dla=signed_dla,
                top_token=top_tok,
                fraction_of_layer=frac,
                abs_rank=rank + 1,
            )
        )

    top_dla = all_cells[0][0] if all_cells else 0.0
    summary: dict[str, Any] = {
        "target_token": target_text,
        "target_token_id": target_id,
        "total_cells_scanned": len(all_cells),
        "top_dla_value": round(top_dla, 4),
        "top_cell": (
            {"layer": hot_cells[0].layer, "head": hot_cells[0].head, "dla": hot_cells[0].dla}
            if hot_cells
            else None
        ),
        "copy_circuit": [{"layer": c.layer, "head": c.head, "dla": c.dla} for c in hot_cells[:3]],
    }

    return BatchDlaScanResult(
        prompt=prompt,
        target_token=target_text,
        target_token_id=target_id,
        position=position,
        num_layers_scanned=len(layer_entries),
        num_heads=num_heads,
        layers=layer_entries,
        hot_cells=hot_cells,
        summary=summary,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: extract_attention_output
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def extract_attention_output(
    prompt: str,
    layer: int,
    head: int,
    position: int = -1,
    top_k_tokens: int = 10,
) -> dict:
    """Extract the output vector of a specific attention head.

    Returns the head's contribution to the residual stream at the given
    position in the full hidden space.  Also projects onto the top-N
    token unembedding directions to reveal the content in vocabulary space.

    The dimensionality breakdown shows whether the content is effectively
    1-dimensional — one scalar in one direction — which supports the
    claim that factual content is compressible to 12 bytes (token_id +
    coefficient).

    Args:
        prompt:        Input text.
        layer:         Layer index.
        head:          Attention head index.
        position:      Token position to extract (-1 = last token).
        top_k_tokens:  Number of token directions to project onto.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "extract_attention_output"
        )

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "extract_attention_output",
        )
    if head < 0 or head >= meta.num_attention_heads:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Head {head} out of [0, {meta.num_attention_heads - 1}].",
            "extract_attention_output",
        )

    try:
        return await asyncio.to_thread(
            _extract_attention_output_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            head,
            position,
            max(1, min(top_k_tokens, 50)),
        )
    except Exception as exc:
        logger.exception("extract_attention_output failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "extract_attention_output")


def _extract_attention_output_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    head: int,
    position: int,
    top_k_tokens: int,
) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import _get_lm_projection, _run_decomposition_forward

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    num_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_heads
    head_dim = meta.head_dim or (meta.hidden_dim // num_heads)

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        raise ValueError("Could not access the language model head.")

    decomp = _run_decomposition_forward(model, config, input_ids, [layer])
    mx.eval(*decomp["prev_hidden"].values())

    prev_h = decomp["prev_hidden"][layer]
    target_layer_module = model_layers[layer]
    if not _has_sublayers(target_layer_module):
        raise ValueError(f"Layer {layer} does not have a decomposable attention sub-layer.")

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(prev_h.dtype)
    context, o_weight = _compute_per_head_attention(
        target_layer_module, prev_h, mask, pos, num_heads, num_kv_heads, head_dim
    )

    head_ctx = context[0, head, pos, :]
    w_slice = o_weight[:, head * head_dim : (head + 1) * head_dim]
    head_out = head_ctx @ w_slice.T
    mx.eval(head_out)

    head_out_np = np.array(head_out.tolist(), dtype=np.float32)
    vec_norm = float(np.linalg.norm(head_out_np))

    # Project through lm_head to get vocab-space coefficients
    raw_logits = lm_head(head_out.reshape(1, 1, -1))
    if hasattr(raw_logits, "logits"):
        raw_logits = raw_logits.logits[0, 0]
    elif isinstance(raw_logits, tuple):
        raw_logits = raw_logits[0][0, 0]
    else:
        raw_logits = raw_logits[0, 0]
    mx.eval(raw_logits)

    logits_np = np.array(raw_logits.tolist(), dtype=np.float32)
    order = np.argsort(-np.abs(logits_np))
    total_abs = float(np.sum(np.abs(logits_np)))

    top_projections: list[dict[str, Any]] = [
        {
            "token": tokenizer.decode([int(order[i])]),
            "token_id": int(order[i]),
            "coefficient": round(float(logits_np[order[i]]), 4),
            "fraction": round(abs(float(logits_np[order[i]])) / (total_abs + 1e-8), 6),
        }
        for i in range(min(top_k_tokens, len(logits_np)))
    ]

    # Dimensionality: cumulative |projection| mass
    sorted_abs = np.sort(np.abs(logits_np))[::-1]
    cum_frac = np.cumsum(sorted_abs) / (total_abs + 1e-8)

    def _dims_for(threshold: float) -> int:
        idxs = np.where(cum_frac >= threshold)[0]
        return int(idxs[0]) + 1 if len(idxs) > 0 else int(len(logits_np))

    d99 = _dims_for(0.99)
    dimensionality: dict[str, Any] = {
        "dims_for_95pct": _dims_for(0.95),
        "dims_for_99pct": d99,
        "dims_for_999pct": _dims_for(0.999),
        "top1_fraction": round(float(sorted_abs[0]) / (total_abs + 1e-8), 6),
        "top2_fraction": round(float(sorted_abs[0] + sorted_abs[1]) / (total_abs + 1e-8), 6)
        if len(sorted_abs) > 1
        else 1.0,
        "is_one_dimensional": d99 <= 2,
    }

    return AttentionOutputResult(
        prompt=prompt,
        layer=layer,
        head=head,
        position=position,
        vector=head_out_np.tolist(),
        vector_norm=round(vec_norm, 4),
        top_projections=top_projections,
        dimensionality=dimensionality,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: get_token_embedding
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def get_token_embedding(
    token: str,
) -> dict:
    """Get the input and unembedding vectors for a token.

    The unembedding vector is the direction used for DLA and content
    projection.  The input embedding is what the model places in the
    residual stream at positions where this token appears as input.

    For most modern models (Gemma, Llama, Mistral) these are tied —
    the same weight matrix is used for both embedding and projection.

    Args:
        token: Token string.  Both bare and space-prefixed forms are tried
               (e.g. "Paris" and " Paris"); the higher-logit form is used.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "get_token_embedding"
        )

    try:
        return await asyncio.to_thread(
            _get_token_embedding_impl,
            state.model,
            state.tokenizer,
            token,
        )
    except Exception as exc:
        logger.exception("get_token_embedding failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "get_token_embedding")


def _get_token_embedding_impl(model: Any, tokenizer: Any, token: str) -> dict:
    import mlx.core as mx

    from ..._residual_helpers import _get_embed_weight, _get_unembed_vector

    token_id = _resolve_token_to_id(tokenizer, token)
    if token_id is None:
        raise ValueError(f"Cannot encode {token!r} to a single token ID.")

    token_text = tokenizer.decode([token_id])

    u = _get_unembed_vector(model, token_id)
    if u is None:
        raise ValueError("Could not extract unembedding vector.")
    mx.eval(u)
    u_np = np.array(u.tolist(), dtype=np.float32)

    embed_w = _get_embed_weight(model)
    e_np: np.ndarray | None = None
    if embed_w is not None:
        e = embed_w[token_id]
        mx.eval(e)
        e_np = np.array(e.tolist(), dtype=np.float32)

    cos_sim = round(_cosine_sim(e_np, u_np), 6) if e_np is not None else None

    return EmbeddingResult(
        token=token_text,
        token_id=token_id,
        unembedding=u_np.tolist(),
        input_embedding=e_np.tolist() if e_np is not None else None,
        embeddings_tied=bool(getattr(model, "tie_word_embeddings", False)),
        unembedding_norm=round(float(np.linalg.norm(u_np)), 4),
        input_embedding_norm=round(float(np.linalg.norm(e_np)), 4) if e_np is not None else None,
        cosine_similarity=cos_sim,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: extract_k_vector
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def extract_k_vector(
    prompt: str,
    layer: int,
    kv_head: int,
    position: int = -1,
) -> dict:
    """Extract the Key vector for a specific KV head at a given position.

    Returns the K vector in head_dim space after k_norm and RoPE encoding.
    This is the vector the model uses for attention pattern matching
    (addressing), before the GQA expansion to query heads.

    Collect K vectors for the same KV head across multiple prompts with
    the same template (e.g. "X agreed to sell his Y") to analyse K-space
    crowding: how W_K maps distinct entities to overlapping K vectors.

    For GQA models, kv_head is in [0, num_kv_heads - 1].

    Args:
        prompt:   Input text.
        layer:    Layer index (use the copy layer from batch_dla_scan).
        kv_head:  KV head index.
        position: Token position to extract (-1 = last token).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "extract_k_vector"
        )

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "extract_k_vector",
        )
    num_kv_heads = meta.num_kv_heads or meta.num_attention_heads
    if kv_head < 0 or kv_head >= num_kv_heads:
        return make_error(
            ToolError.INVALID_INPUT,
            f"kv_head {kv_head} out of [0, {num_kv_heads - 1}].",
            "extract_k_vector",
        )

    try:
        return await asyncio.to_thread(
            _extract_k_vector_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            kv_head,
            position,
        )
    except Exception as exc:
        logger.exception("extract_k_vector failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "extract_k_vector")


def _extract_k_vector_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    kv_head: int,
    position: int,
) -> dict:
    import mlx.core as mx

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import _run_decomposition_forward

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    num_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_heads
    head_dim = meta.head_dim or (meta.hidden_dim // num_heads)

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()

    decomp = _run_decomposition_forward(model, config, input_ids, [layer])
    mx.eval(*decomp["prev_hidden"].values())

    prev_h = decomp["prev_hidden"][layer]
    target_layer_module = model_layers[layer]
    if not _has_sublayers(target_layer_module):
        raise ValueError(f"Layer {layer} does not have a decomposable attention sub-layer.")

    attn = target_layer_module.self_attn
    seq_len = input_ids.shape[1]
    batch_size = prev_h.shape[0]

    normed = target_layer_module.input_layernorm(prev_h)
    keys = attn.k_proj(normed)
    keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        keys = attn.k_norm(keys)

    if hasattr(attn, "rope") and attn.rope is not None:
        queries = (
            attn.q_proj(normed)
            .reshape(batch_size, seq_len, num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            queries = attn.q_norm(queries)
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    k_vec = keys[0, kv_head, pos, :]
    mx.eval(k_vec)
    k_np = np.array(k_vec.tolist(), dtype=np.float32)

    return KVectorResult(
        prompt=prompt,
        layer=layer,
        kv_head=kv_head,
        position=position,
        k_vector=k_np.tolist(),
        k_norm=round(float(np.linalg.norm(k_np)), 4),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: extract_q_vector
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def extract_q_vector(
    prompt: str,
    layer: int,
    head: int,
    position: int = -1,
) -> dict:
    """Extract the Query vector for a specific head at a given position.

    Returns the Q vector in head_dim space after q_norm and RoPE encoding.
    Use this alongside extract_k_vector to examine query-key compatibility
    and understand where in K-space the query is searching.

    Args:
        prompt:   Input text.
        layer:    Layer index.
        head:     Query head index (0-indexed, full head count).
        position: Token position to extract (-1 = last token).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "extract_q_vector"
        )

    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "extract_q_vector",
        )
    if head < 0 or head >= meta.num_attention_heads:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Head {head} out of [0, {meta.num_attention_heads - 1}].",
            "extract_q_vector",
        )

    try:
        return await asyncio.to_thread(
            _extract_q_vector_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            head,
            position,
        )
    except Exception as exc:
        logger.exception("extract_q_vector failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "extract_q_vector")


def _extract_q_vector_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    head: int,
    position: int,
) -> dict:
    import mlx.core as mx

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import _run_decomposition_forward

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = int(input_ids.shape[-1])
    pos = _resolve_position(num_tokens, position)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    num_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_heads
    head_dim = meta.head_dim or (meta.hidden_dim // num_heads)

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()

    decomp = _run_decomposition_forward(model, config, input_ids, [layer])
    mx.eval(*decomp["prev_hidden"].values())

    prev_h = decomp["prev_hidden"][layer]
    target_layer_module = model_layers[layer]
    if not _has_sublayers(target_layer_module):
        raise ValueError(f"Layer {layer} does not have a decomposable attention sub-layer.")

    attn = target_layer_module.self_attn
    seq_len = input_ids.shape[1]
    batch_size = prev_h.shape[0]

    normed = target_layer_module.input_layernorm(prev_h)
    queries = attn.q_proj(normed)
    queries = queries.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        queries = attn.q_norm(queries)

    if hasattr(attn, "rope") and attn.rope is not None:
        keys = (
            attn.k_proj(normed)
            .reshape(batch_size, seq_len, num_kv_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            keys = attn.k_norm(keys)
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    q_vec = queries[0, head, pos, :]
    mx.eval(q_vec)
    q_np = np.array(q_vec.tolist(), dtype=np.float32)

    return QVectorResult(
        prompt=prompt,
        layer=layer,
        head=head,
        position=position,
        q_vector=q_np.tolist(),
        q_norm=round(float(np.linalg.norm(q_np)), 4),
        num_heads=num_heads,
        head_dim=head_dim,
    ).model_dump()
