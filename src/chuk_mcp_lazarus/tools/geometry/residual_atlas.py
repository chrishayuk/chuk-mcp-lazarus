"""
residual_atlas — map the residual stream via PCA on diverse prompt activations.

Collects hidden-state vectors from many prompts at specified layers,
runs PCA to find the dominant directions of variation, and projects
each principal component through the unembedding matrix to decode
what it represents in vocabulary space.
"""

import asyncio
import datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ...subspace_registry import SubspaceMetadata, SubspaceRegistry
from ._helpers import coerce_layers, collect_activations, effective_dimensionality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class VocabProjection(BaseModel):
    """A token decoded from a principal component direction."""

    token: str
    token_id: int
    projection_score: float = Field(
        ..., description="Logit from projecting PC through unembedding."
    )


class AtlasComponent(BaseModel):
    """One principal component of the residual stream at a layer."""

    component_idx: int = Field(..., description="1-indexed component number.")
    variance_explained: float
    cumulative_variance: float
    norm: float = Field(..., description="Singular value.")
    top_positive_tokens: list[VocabProjection]
    top_negative_tokens: list[VocabProjection]


class LayerAtlas(BaseModel):
    """Residual stream atlas at one layer."""

    layer: int
    num_prompts: int
    hidden_dim: int
    effective_dimensionality: dict[str, int] = Field(
        ...,
        description="PCs needed for variance thresholds: dims_for_50/80/90/95/99pct.",
    )
    total_variance_captured: float
    components: list[AtlasComponent]
    stored_subspace_name: str | None = None


class ResidualAtlasResult(BaseModel):
    """Result from residual_atlas."""

    layers: list[LayerAtlas]
    num_prompts: int
    max_components: int
    top_k_tokens: int
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True)
async def residual_atlas(
    prompts: list[str],
    layers: list[int] | int,
    token_position: int = -1,
    max_components: int = 50,
    top_k_tokens: int = 10,
    store_subspace: str | None = None,
) -> dict:
    """Map the residual stream structure via PCA on diverse prompt activations.

    Collects hidden-state vectors from many prompts at specified layers,
    runs PCA to find the dominant directions of variation, and projects
    each principal component through the unembedding matrix to decode
    what it represents in vocabulary space.

    Use this to answer: "What are the main axes of variation in the
    residual stream at layer N? What concepts do they correspond to?"

    Provide diverse prompts (different topics, languages, styles) to
    map the full structure. Use focused prompts to map a specific domain.

    Args:
        prompts:          Diverse prompts to extract activations from (10-2000).
        layers:           Layer index or list of layer indices.
        token_position:   Token position (-1 = last).
        max_components:   PCA components to analyse in detail (1-200).
        top_k_tokens:     Vocabulary tokens to decode per component (1-50).
        store_subspace:   If provided, store PCA basis in SubspaceRegistry.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "residual_atlas")
    meta = state.metadata

    layers_list = coerce_layers(layers) or []

    if not layers_list:
        return make_error(ToolError.INVALID_INPUT, "At least 1 layer required.", "residual_atlas")
    for lyr in layers_list:
        if lyr < 0 or lyr >= meta.num_layers:
            return make_error(
                ToolError.LAYER_OUT_OF_RANGE,
                f"Layer {lyr} out of [0, {meta.num_layers - 1}].",
                "residual_atlas",
            )
    if len(prompts) < 10:
        return make_error(
            ToolError.INVALID_INPUT, "At least 10 prompts required.", "residual_atlas"
        )
    if len(prompts) > 2000:
        return make_error(ToolError.INVALID_INPUT, "Maximum 2000 prompts.", "residual_atlas")
    if max_components < 1 or max_components > 200:
        return make_error(
            ToolError.INVALID_INPUT,
            "max_components must be in [1, 200].",
            "residual_atlas",
        )
    if top_k_tokens < 1 or top_k_tokens > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k_tokens must be in [1, 50].",
            "residual_atlas",
        )
    if max_components >= len(prompts):
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_components ({max_components}) must be < number of prompts ({len(prompts)}).",
            "residual_atlas",
        )
    try:
        return await asyncio.to_thread(
            _residual_atlas_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompts,
            layers_list,
            token_position,
            max_components,
            top_k_tokens,
            store_subspace,
        )
    except Exception as exc:
        logger.exception("residual_atlas failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "residual_atlas")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _residual_atlas_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompts: list[str],
    layers: list[int],
    token_position: int,
    max_components: int,
    top_k_tokens: int,
    store_subspace: str | None,
) -> dict:
    """Sync implementation of residual_atlas."""
    import mlx.core as mx

    from ..residual_tools import _get_lm_projection

    multi_layer = len(layers) > 1

    # -- Phase 1: Extract activations --
    per_layer_acts = collect_activations(model, config, tokenizer, prompts, layers, token_position)

    # -- Phase 2: Get lm_head --
    lm_head = _get_lm_projection(model)

    # -- Phase 3: PCA + vocabulary decode per layer --
    layer_results: list[LayerAtlas] = []

    for lyr in sorted(layers):
        acts = per_layer_acts[lyr]
        if len(acts) < 2:
            continue

        X = np.stack(acts)  # [n_prompts, hidden_dim]
        n, d = X.shape

        # Centre and SVD
        X_centered = X - X.mean(axis=0)

        n_components = min(max_components, n - 1, d)
        if n_components < 1:
            n_components = 1

        _, s_vals_full, Vt_full = np.linalg.svd(X_centered, full_matrices=False)

        total_var = float(np.sum(s_vals_full**2))
        if total_var < 1e-12:
            total_var = 1e-12

        s_vals = s_vals_full[:n_components]
        Vt = Vt_full[:n_components]

        # Effective dimensionality (shared helper)
        eff_dim, cumulative_list = effective_dimensionality(s_vals, n_components, total_var)
        cumulative = cumulative_list[-1] if cumulative_list else 0.0

        # -- Batch vocabulary projection: all PCs in one lm_head call --
        all_pcs_mx = mx.array(Vt.tolist())  # [n_components, hidden_dim]
        all_logits = lm_head(all_pcs_mx.reshape(1, n_components, -1))
        if hasattr(all_logits, "logits"):
            all_logits = all_logits.logits
        elif isinstance(all_logits, tuple):
            all_logits = all_logits[0]
        all_logits = all_logits[0]  # [n_components, vocab]
        mx.eval(all_logits)
        all_logits_np = np.array(all_logits.tolist(), dtype=np.float32)

        # Build components from batch results
        components: list[AtlasComponent] = []
        for i, sv in enumerate(s_vals):
            var_exp = float(sv**2) / total_var
            logits_row = all_logits_np[i]

            # Top positive tokens
            pos_order = np.argsort(-logits_row)
            top_pos = [
                VocabProjection(
                    token=tokenizer.decode([int(tid)]),
                    token_id=int(tid),
                    projection_score=round(float(logits_row[tid]), 4),
                )
                for tid in pos_order[:top_k_tokens]
            ]

            # Top negative tokens
            neg_order = np.argsort(logits_row)
            top_neg = [
                VocabProjection(
                    token=tokenizer.decode([int(tid)]),
                    token_id=int(tid),
                    projection_score=round(float(logits_row[tid]), 4),
                )
                for tid in neg_order[:top_k_tokens]
            ]

            components.append(
                AtlasComponent(
                    component_idx=i + 1,
                    variance_explained=round(var_exp, 6),
                    cumulative_variance=round(cumulative_list[i], 6),
                    norm=round(float(sv), 4),
                    top_positive_tokens=top_pos,
                    top_negative_tokens=top_neg,
                )
            )

        # -- Phase 4: Optional SubspaceRegistry storage --
        stored_name: str | None = None
        if store_subspace:
            stored_name = f"{store_subspace}_layer{lyr}" if multi_layer else store_subspace
            basis = Vt.astype(np.float32)
            sub_meta = SubspaceMetadata(
                name=stored_name,
                layer=lyr,
                rank=n_components,
                num_prompts=len(prompts),
                hidden_dim=d,
                variance_explained=[round(float(sv**2) / total_var, 6) for sv in s_vals],
                total_variance_explained=round(cumulative, 6),
                computed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            SubspaceRegistry.get().store(stored_name, basis, sub_meta)

        layer_results.append(
            LayerAtlas(
                layer=lyr,
                num_prompts=len(acts),
                hidden_dim=d,
                effective_dimensionality=eff_dim,
                total_variance_captured=round(cumulative, 6),
                components=components,
                stored_subspace_name=stored_name,
            )
        )

    # -- Phase 5: Summary --
    summary: dict[str, Any] = {"stored": store_subspace is not None}
    if layer_results:
        per_layer_rank = {
            lr.layer: lr.effective_dimensionality["dims_for_90pct"] for lr in layer_results
        }
        summary["per_layer_effective_rank_90pct"] = per_layer_rank
        most_conc = min(
            layer_results,
            key=lambda lr: lr.effective_dimensionality["dims_for_90pct"],
        )
        most_dist = max(
            layer_results,
            key=lambda lr: lr.effective_dimensionality["dims_for_90pct"],
        )
        summary["most_concentrated_layer"] = most_conc.layer
        summary["most_distributed_layer"] = most_dist.layer

    return ResidualAtlasResult(
        layers=layer_results,
        num_prompts=len(prompts),
        max_components=max_components,
        top_k_tokens=top_k_tokens,
        summary=summary,
    ).model_dump()
