"""
residual_map — compact per-layer variance spectrum across the full model.

Like residual_atlas but lighter: runs PCA per layer and returns only the
effective dimensionality and variance spectrum (no vocabulary projections).
Designed for visualising how representation structure evolves across layers.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import collect_activations, effective_dimensionality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class LayerSummary(BaseModel):
    """Compact PCA summary at one layer."""

    layer: int
    effective_rank_50: int
    effective_rank_80: int
    effective_rank_90: int
    effective_rank_95: int
    effective_rank_99: int
    top_singular_values: list[float] = Field(
        ..., description="Normalised singular values (top 20)."
    )
    total_variance_captured: float


class ResidualMapResult(BaseModel):
    """Result from residual_map."""

    layers: list[LayerSummary]
    num_prompts: int
    hidden_dim: int
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_layers(num_layers: int, max_points: int = 12) -> list[int]:
    """Select evenly-spaced layers including first and last."""
    if num_layers <= max_points:
        return list(range(num_layers))
    step = max(1, (num_layers - 1) / (max_points - 1))
    layers = [round(i * step) for i in range(max_points)]
    # Ensure first and last
    if layers[0] != 0:
        layers[0] = 0
    if layers[-1] != num_layers - 1:
        layers[-1] = num_layers - 1
    return sorted(set(layers))


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True)
async def residual_map(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
    max_components: int = 100,
) -> dict:
    """Compact per-layer variance spectrum across the full model.

    Runs PCA at each layer and returns effective dimensionality and
    variance spectrum — no vocabulary projections. Use this to see
    how representation structure evolves across layers.

    Args:
        prompts:          Diverse prompts to extract activations from (10-2000).
        layers:           Layer indices (None = auto-select evenly-spaced layers).
        token_position:   Token position (-1 = last).
        max_components:   PCA components to compute (1-200).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "residual_map")
    meta = state.metadata

    if layers is None:
        layers_list = _auto_layers(meta.num_layers)
    else:
        layers_list = list(layers)

    if not layers_list:
        return make_error(ToolError.INVALID_INPUT, "At least 1 layer required.", "residual_map")
    for lyr in layers_list:
        if lyr < 0 or lyr >= meta.num_layers:
            return make_error(
                ToolError.LAYER_OUT_OF_RANGE,
                f"Layer {lyr} out of [0, {meta.num_layers - 1}].",
                "residual_map",
            )
    if len(prompts) < 10:
        return make_error(ToolError.INVALID_INPUT, "At least 10 prompts required.", "residual_map")
    if len(prompts) > 2000:
        return make_error(ToolError.INVALID_INPUT, "Maximum 2000 prompts.", "residual_map")
    if max_components < 1 or max_components > 200:
        return make_error(
            ToolError.INVALID_INPUT,
            "max_components must be in [1, 200].",
            "residual_map",
        )
    try:
        return await asyncio.to_thread(
            _residual_map_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompts,
            layers_list,
            token_position,
            max_components,
        )
    except Exception as exc:
        logger.exception("residual_map failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "residual_map")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _residual_map_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompts: list[str],
    layers: list[int],
    token_position: int,
    max_components: int,
) -> dict:
    """Sync implementation of residual_map."""
    # -- Phase 1: Extract activations --
    per_layer_acts = collect_activations(model, config, tokenizer, prompts, layers, token_position)

    # -- Phase 2: PCA per layer --
    layer_summaries: list[LayerSummary] = []
    hidden_dim = meta.hidden_dim

    for lyr in sorted(layers):
        acts = per_layer_acts[lyr]
        if len(acts) < 2:
            continue

        X = np.stack(acts)  # [n_prompts, hidden_dim]
        n, d = X.shape
        hidden_dim = d  # update from actual data

        X_centered = X - X.mean(axis=0)
        n_components = min(max_components, n - 1, d)
        if n_components < 1:
            n_components = 1

        _, s_vals, _ = np.linalg.svd(X_centered, full_matrices=False)

        total_var = float(np.sum(s_vals**2))
        if total_var < 1e-12:
            total_var = 1e-12

        # Effective ranks (shared helper)
        eff_dim, cumulative_list = effective_dimensionality(s_vals, n_components, total_var)
        captured = cumulative_list[-1] if cumulative_list else 0.0

        # Normalised singular values (top 20)
        max_sv = float(s_vals[0]) if len(s_vals) > 0 else 1.0
        if max_sv < 1e-12:
            max_sv = 1e-12
        top_svs = [round(float(sv) / max_sv, 6) for sv in s_vals[:20]]

        layer_summaries.append(
            LayerSummary(
                layer=lyr,
                effective_rank_50=eff_dim["dims_for_50pct"],
                effective_rank_80=eff_dim["dims_for_80pct"],
                effective_rank_90=eff_dim["dims_for_90pct"],
                effective_rank_95=eff_dim["dims_for_95pct"],
                effective_rank_99=eff_dim["dims_for_99pct"],
                top_singular_values=top_svs,
                total_variance_captured=round(captured, 6),
            )
        )

    # -- Phase 3: Summary --
    summary: dict[str, Any] = {}
    if layer_summaries:
        rank_progression = {ls.layer: ls.effective_rank_90 for ls in layer_summaries}
        summary["rank_progression_90pct"] = rank_progression

        peak = max(layer_summaries, key=lambda ls: ls.effective_rank_90)
        summary["peak_rank_layer"] = peak.layer
        summary["peak_rank_90"] = peak.effective_rank_90

        most_compact = min(layer_summaries, key=lambda ls: ls.effective_rank_90)
        summary["most_compact_layer"] = most_compact.layer
        summary["most_compact_rank_90"] = most_compact.effective_rank_90

    return ResidualMapResult(
        layers=layer_summaries,
        num_prompts=len(prompts),
        hidden_dim=hidden_dim,
        summary=summary,
    ).model_dump()
