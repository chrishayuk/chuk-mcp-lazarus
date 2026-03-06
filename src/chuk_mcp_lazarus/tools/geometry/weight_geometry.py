"""
weight_geometry — map the supply side: what directions every component can push toward.

Extracts each attention head's output projection direction and the top-k
MLP neuron down_proj columns, projects them through the unembedding matrix,
and optionally runs PCA on all directions for the effective supply rank.
No forward pass required — purely from weights.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import dims_for_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class VocabEntry(BaseModel):
    """A token decoded from a push direction."""

    token: str
    score: float


class HeadDirection(BaseModel):
    """Push direction for one attention head."""

    head_idx: int
    norm: float
    top_tokens: list[VocabEntry]
    top_token: str


class NeuronDirection(BaseModel):
    """Push direction for one MLP neuron."""

    neuron_idx: int
    norm: float
    top_tokens: list[VocabEntry]
    top_token: str


class SupplySubspace(BaseModel):
    """PCA on all push directions — effective supply rank."""

    total_directions: int
    effective_rank_50: int
    effective_rank_80: int
    effective_rank_95: int
    top_singular_values: list[float] = Field(
        ..., description="Normalised singular values (top 20)."
    )


class WeightGeometryResult(BaseModel):
    """Result from weight_geometry."""

    layer: int
    num_heads: int
    num_neurons_analyzed: int
    intermediate_size: int
    hidden_dim: int
    head_directions: list[HeadDirection]
    neuron_directions: list[NeuronDirection]
    supply_subspace: SupplySubspace | None = None
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def weight_geometry(
    layer: int,
    top_k_neurons: int = 100,
    top_k_vocab: int = 5,
    include_pca: bool = True,
) -> dict:
    """Map what directions every component can push toward (supply side).

    Extracts each attention head's output-projection push direction and
    the top-k MLP neuron down_proj columns, projects them through
    the unembedding matrix, and optionally runs PCA on all directions.

    No forward pass required — purely from weights.

    Use this to answer: "What can layer N actually write into the
    residual stream? Which tokens can each head/neuron push toward?"

    Args:
        layer:          Layer index to analyse.
        top_k_neurons:  Number of MLP neurons to analyse (by L2 norm, 1-500).
        top_k_vocab:    Vocabulary tokens to decode per direction (1-20).
        include_pca:    Run PCA on all push directions for supply rank.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "weight_geometry")
    meta = state.metadata

    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "weight_geometry",
        )
    if top_k_neurons < 1 or top_k_neurons > 500:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k_neurons must be in [1, 500].",
            "weight_geometry",
        )
    if top_k_vocab < 1 or top_k_vocab > 20:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k_vocab must be in [1, 20].",
            "weight_geometry",
        )
    try:
        return await asyncio.to_thread(
            _weight_geometry_impl,
            state.model,
            state.config,
            meta,
            state.tokenizer,
            layer,
            top_k_neurons,
            top_k_vocab,
            include_pca,
        )
    except Exception as exc:
        logger.exception("weight_geometry failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "weight_geometry")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _weight_geometry_impl(
    model: Any,
    config: Any,
    meta: Any,
    tokenizer: Any,
    layer: int,
    top_k_neurons: int,
    top_k_vocab: int,
    include_pca: bool,
) -> dict:
    """Sync implementation of weight_geometry."""
    import mlx.core as mx

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..residual_tools import _get_lm_projection

    hooks = ModelHooks(model, model_config=config)
    model_layers = hooks._get_layers()
    target_layer = model_layers[layer]

    hidden_dim = meta.hidden_dim
    num_heads = meta.num_attention_heads
    head_dim = hidden_dim // num_heads
    intermediate_size = meta.intermediate_size

    lm_head = _get_lm_projection(model)

    # -- Phase 1: Head push directions --
    o_weight = target_layer.self_attn.o_proj.weight  # [hidden_dim, num_heads * head_dim]
    o_np = np.array(o_weight.tolist(), dtype=np.float32)

    head_dirs_np: list[np.ndarray] = []
    for head_i in range(num_heads):
        w_slice = o_np[:, head_i * head_dim : (head_i + 1) * head_dim]
        head_dir = w_slice.mean(axis=1)  # [hidden_dim]
        head_dirs_np.append(head_dir)

    # -- Phase 2: Neuron push directions (top-k by norm) --
    down_weight = target_layer.mlp.down_proj.weight  # [hidden_dim, intermediate_size]
    down_np = np.array(down_weight.tolist(), dtype=np.float32)

    neuron_norms = np.linalg.norm(down_np, axis=0)  # [intermediate_size]
    actual_neurons = min(top_k_neurons, down_np.shape[1])
    top_neuron_indices = np.argsort(-neuron_norms)[:actual_neurons]

    neuron_dirs_np: list[np.ndarray] = []
    neuron_indices: list[int] = []
    for nidx in top_neuron_indices:
        neuron_dirs_np.append(down_np[:, int(nidx)])
        neuron_indices.append(int(nidx))

    # -- Phase 3: Batch vocabulary projection --
    all_dirs = head_dirs_np + neuron_dirs_np
    n_dirs = len(all_dirs)

    if n_dirs > 0:
        dirs_stack = np.stack(all_dirs)  # [n_dirs, hidden_dim]
        dirs_mx = mx.array(dirs_stack.tolist())
        all_logits = lm_head(dirs_mx.reshape(1, n_dirs, -1))
        if hasattr(all_logits, "logits"):
            all_logits = all_logits.logits
        elif isinstance(all_logits, tuple):
            all_logits = all_logits[0]
        all_logits = all_logits[0]  # [n_dirs, vocab]
        mx.eval(all_logits)
        logits_np = np.array(all_logits.tolist(), dtype=np.float32)
    else:
        logits_np = np.zeros((0, 0), dtype=np.float32)

    # -- Phase 4: Build head results --
    head_results: list[HeadDirection] = []
    for head_i, hdir in enumerate(head_dirs_np):
        norm = float(np.linalg.norm(hdir))
        logits_row = logits_np[head_i]
        top_ids = np.argsort(-logits_row)[:top_k_vocab]

        top_tokens = [
            VocabEntry(
                token=tokenizer.decode([int(tid)]),
                score=round(float(logits_row[tid]), 4),
            )
            for tid in top_ids
        ]
        head_results.append(
            HeadDirection(
                head_idx=head_i,
                norm=round(norm, 6),
                top_tokens=top_tokens,
                top_token=top_tokens[0].token if top_tokens else "",
            )
        )

    # -- Phase 5: Build neuron results --
    neuron_results: list[NeuronDirection] = []
    for i, nidx in enumerate(neuron_indices):
        norm = float(neuron_norms[nidx])
        logits_row = logits_np[num_heads + i]
        top_ids = np.argsort(-logits_row)[:top_k_vocab]

        top_tokens = [
            VocabEntry(
                token=tokenizer.decode([int(tid)]),
                score=round(float(logits_row[tid]), 4),
            )
            for tid in top_ids
        ]
        neuron_results.append(
            NeuronDirection(
                neuron_idx=nidx,
                norm=round(norm, 6),
                top_tokens=top_tokens,
                top_token=top_tokens[0].token if top_tokens else "",
            )
        )

    # -- Phase 6: Supply PCA (optional) --
    supply_subspace: SupplySubspace | None = None
    if include_pca and n_dirs >= 2:
        dirs_matrix = np.stack(all_dirs)  # [n_dirs, hidden_dim]
        dirs_centered = dirs_matrix - dirs_matrix.mean(axis=0)
        _, s_vals, _ = np.linalg.svd(dirs_centered, full_matrices=False)
        total_var = float(np.sum(s_vals**2))
        if total_var < 1e-12:
            total_var = 1e-12

        cumulative_list: list[float] = []
        cumulative = 0.0
        for sv in s_vals:
            cumulative += float(sv**2) / total_var
            cumulative_list.append(cumulative)

        eff_50 = dims_for_threshold(cumulative_list, 0.50, n_dirs)
        eff_80 = dims_for_threshold(cumulative_list, 0.80, n_dirs)
        eff_95 = dims_for_threshold(cumulative_list, 0.95, n_dirs)

        # Normalise singular values
        max_sv = float(s_vals[0]) if len(s_vals) > 0 else 1.0
        if max_sv < 1e-12:
            max_sv = 1e-12
        top_svs = [round(float(sv) / max_sv, 6) for sv in s_vals[:20]]

        supply_subspace = SupplySubspace(
            total_directions=n_dirs,
            effective_rank_50=eff_50,
            effective_rank_80=eff_80,
            effective_rank_95=eff_95,
            top_singular_values=top_svs,
        )

    # -- Phase 7: Summary --
    summary: dict[str, Any] = {}
    if head_results:
        strongest_head = max(head_results, key=lambda h: h.norm)
        summary["strongest_head"] = {
            "head_idx": strongest_head.head_idx,
            "norm": strongest_head.norm,
            "top_token": strongest_head.top_token,
        }
    if neuron_results:
        strongest_neuron = neuron_results[0]  # already sorted by norm
        summary["strongest_neuron"] = {
            "neuron_idx": strongest_neuron.neuron_idx,
            "norm": strongest_neuron.norm,
            "top_token": strongest_neuron.top_token,
        }
    if supply_subspace:
        summary["supply_effective_rank_80"] = supply_subspace.effective_rank_80

    return WeightGeometryResult(
        layer=layer,
        num_heads=num_heads,
        num_neurons_analyzed=len(neuron_results),
        intermediate_size=intermediate_size,
        hidden_dim=hidden_dim,
        head_directions=head_results,
        neuron_directions=neuron_results,
        supply_subspace=supply_subspace,
        summary=summary,
    ).model_dump()
