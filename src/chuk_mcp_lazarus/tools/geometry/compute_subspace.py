"""
compute_subspace — extract PCA-based functional subspaces from activations.

Runs varied prompts through the model, collects hidden-state vectors at a
layer, and computes the top principal components via SVD.  The orthonormal
basis is stored in SubspaceRegistry for use with inject_residual.
"""

import asyncio
import datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel

from ..._extraction import extract_activation_at_layer
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ...subspace_registry import SubspaceMetadata, SubspaceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class PCAComponentEntry(BaseModel):
    """One principal component from the subspace PCA."""

    component: int
    variance_explained: float
    cumulative_variance: float


class ComputeSubspaceResult(BaseModel):
    """Result from compute_subspace."""

    subspace_name: str
    layer: int
    rank: int
    num_prompts: int
    hidden_dim: int
    components: list[PCAComponentEntry]
    total_variance_explained: float
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def compute_subspace(
    subspace_name: str,
    layer: int,
    prompts: list[str],
    rank: int = 10,
    token_position: int = -1,
) -> dict:
    """Compute a PCA subspace from model activations and store it in
    the SubspaceRegistry for use with inject_residual.

    Runs each prompt through the model at the given layer, collects
    hidden-state vectors at token_position, centres them, and computes
    the top-rank principal components via SVD.

    For capital-city experiments: pass prompts like
      ["The capital of Australia is", "The capital of France is", ...]
    PCA finds the directions of maximum variation — the signal that
    changes when you change the country.

    Args:
        subspace_name:  Name to store this subspace under.
        layer:          Layer to extract activations at.
        prompts:        Varied prompts (min 3, max 500).
        rank:           Number of PCA components to retain (1-100).
        token_position: Token position (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compute_subspace",
        )
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "compute_subspace",
        )
    if not subspace_name or not subspace_name.strip():
        return make_error(
            ToolError.INVALID_INPUT,
            "subspace_name must be non-empty.",
            "compute_subspace",
        )
    if len(prompts) < 3:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 3 prompts required.",
            "compute_subspace",
        )
    if len(prompts) > 500:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 500 prompts.",
            "compute_subspace",
        )
    if rank < 1 or rank > 100:
        return make_error(
            ToolError.INVALID_INPUT,
            f"rank must be in [1, 100], got {rank}.",
            "compute_subspace",
        )
    if rank >= len(prompts):
        return make_error(
            ToolError.INVALID_INPUT,
            f"rank ({rank}) must be less than number of prompts ({len(prompts)}).",
            "compute_subspace",
        )
    try:
        return await asyncio.to_thread(
            _compute_subspace_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            subspace_name,
            layer,
            prompts,
            rank,
            token_position,
        )
    except Exception as exc:
        logger.exception("compute_subspace failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "compute_subspace")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _dims_for_threshold(components: list[PCAComponentEntry], threshold: float) -> int:
    """Number of components needed to reach a cumulative variance threshold."""
    for c in components:
        if c.cumulative_variance >= threshold:
            return c.component
    return len(components) if components else 0


def _compute_subspace_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    subspace_name: str,
    layer: int,
    prompts: list[str],
    rank: int,
    token_position: int,
) -> dict:
    """Sync implementation of compute_subspace."""
    # Phase 1: Extract activations
    all_acts: list[np.ndarray] = []
    for p in prompts:
        act = extract_activation_at_layer(model, config, tokenizer, p, layer, token_position)
        all_acts.append(np.array(act, dtype=np.float32))

    X = np.stack(all_acts)  # [n_prompts, hidden_dim]

    # Phase 2: PCA via SVD
    global_mean = X.mean(axis=0)
    X_centered = X - global_mean

    n_components = min(rank, X_centered.shape[0] - 1, X_centered.shape[1])
    if n_components < 1:
        n_components = 1

    _, s_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)
    s_vals = s_vals[:n_components]
    Vt = Vt[:n_components]  # [n_components, hidden_dim]

    # Phase 3: Variance spectrum
    total_var = float(np.sum(s_vals**2))
    if total_var < 1e-12:
        total_var = 1e-12

    components: list[PCAComponentEntry] = []
    cumulative = 0.0
    var_explained_list: list[float] = []

    for i, sv in enumerate(s_vals):
        ve = float(sv**2) / total_var
        cumulative += ve
        var_explained_list.append(round(ve, 6))
        components.append(
            PCAComponentEntry(
                component=i + 1,
                variance_explained=round(ve, 6),
                cumulative_variance=round(cumulative, 6),
            )
        )

    # Phase 4: Store in SubspaceRegistry
    basis = Vt.astype(np.float32)
    sub_meta = SubspaceMetadata(
        name=subspace_name,
        layer=layer,
        rank=n_components,
        num_prompts=len(prompts),
        hidden_dim=int(X.shape[1]),
        variance_explained=var_explained_list,
        total_variance_explained=round(cumulative, 6),
        computed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    SubspaceRegistry.get().store(subspace_name, basis, sub_meta)

    # Phase 5: Result
    return ComputeSubspaceResult(
        subspace_name=subspace_name,
        layer=layer,
        rank=n_components,
        num_prompts=len(prompts),
        hidden_dim=int(X.shape[1]),
        components=components,
        total_variance_explained=round(cumulative, 6),
        summary={
            "stored": True,
            "effective_rank": n_components,
            "top_component_variance": var_explained_list[0] if var_explained_list else 0.0,
            "recommended_rank_for_80pct": _dims_for_threshold(components, 0.80),
        },
    ).model_dump()


# ---------------------------------------------------------------------------
# list_subspaces
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_subspaces() -> dict:
    """List all named subspaces stored in the SubspaceRegistry.

    Returns names, layers, ranks, and variance explained for each
    stored subspace.
    """
    reg = SubspaceRegistry.get()
    dump = reg.dump()
    return dump.model_dump()
