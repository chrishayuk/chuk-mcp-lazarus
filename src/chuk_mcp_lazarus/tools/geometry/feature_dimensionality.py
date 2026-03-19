"""
feature_dimensionality — how many dimensions a feature occupies.

Estimates the effective dimensionality of a feature at a layer via PCA
spectrum analysis and classification-by-dimension. 1 dimension = clean
direction. 50 dimensions = holographically distributed.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel

from ..._extraction import extract_activation_at_layer
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class SpectrumEntry(BaseModel):
    """One principal component of the feature."""

    dimension: int
    variance_explained: float
    cumulative_variance: float
    top_token: str | None = None


class ClassificationByDim(BaseModel):
    """Classification accuracy using N PCA dimensions."""

    num_dimensions: int
    accuracy: float


class FeatureDimensionalityResult(BaseModel):
    """Result from feature_dimensionality — how many dimensions a feature occupies."""

    layer: int
    num_positive: int
    num_negative: int
    hidden_dim: int
    effective_dimensionality: dict[str, int]
    spectrum: list[SpectrumEntry]
    classification_by_dim: list[ClassificationByDim]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def feature_dimensionality(
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int = -1,
    max_dims: int = 50,
) -> dict:
    """Estimate the effective dimensionality of a feature at a layer.

    Extracts activations for positive/negative prompt groups and computes
    principal components of the difference. Reports how many dimensions
    are needed to capture the feature.

    1 dimension = clean direction. 50 dimensions = holographically distributed.

    Args:
        layer:             Layer to analyse.
        positive_prompts:  Prompts representing the positive class (min 2).
        negative_prompts:  Prompts representing the negative class (min 2).
        token_position:    Token position (-1 = last).
        max_dims:          Maximum dimensions to analyse (default: 50).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "feature_dimensionality"
        )
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "feature_dimensionality",
        )
    if len(positive_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT, "At least 2 positive prompts.", "feature_dimensionality"
        )
    if len(negative_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT, "At least 2 negative prompts.", "feature_dimensionality"
        )
    if max_dims < 1 or max_dims > 500:
        return make_error(
            ToolError.INVALID_INPUT, "max_dims must be in [1, 500].", "feature_dimensionality"
        )
    try:
        return await asyncio.to_thread(
            _feature_dimensionality_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            layer,
            positive_prompts,
            negative_prompts,
            token_position,
            max_dims,
        )
    except Exception as exc:
        logger.exception("feature_dimensionality failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "feature_dimensionality")


def _feature_dimensionality_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    layer: int,
    pos_prompts: list[str],
    neg_prompts: list[str],
    token_position: int,
    max_dims: int,
) -> dict:
    """Sync implementation of feature_dimensionality."""
    from ..._residual_helpers import _get_lm_projection, _project_to_logits

    # Extract activations for all prompts
    all_acts: list[np.ndarray] = []
    labels: list[int] = []

    for p in pos_prompts:
        act = extract_activation_at_layer(model, config, tokenizer, p, layer, token_position)
        all_acts.append(np.array(act, dtype=np.float32))
        labels.append(1)
    for p in neg_prompts:
        act = extract_activation_at_layer(model, config, tokenizer, p, layer, token_position)
        all_acts.append(np.array(act, dtype=np.float32))
        labels.append(0)

    X = np.stack(all_acts)
    y = np.array(labels)

    # PCA on centred data
    global_mean = X.mean(axis=0)
    X_centered = X - global_mean
    n_components = min(max_dims, X_centered.shape[0] - 1, X_centered.shape[1])
    if n_components < 1:
        n_components = 1

    _, s_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)
    s_vals = s_vals[:n_components]

    total_var = float(np.sum(s_vals**2))
    if total_var < 1e-12:
        total_var = 1e-12

    # Build spectrum
    cumulative = 0.0
    spectrum: list[SpectrumEntry] = []
    dims_50 = n_components
    dims_80 = n_components
    dims_95 = n_components
    dims_99 = n_components

    for i, sv in enumerate(s_vals):
        var_exp = float(sv**2) / total_var
        cumulative += var_exp

        # Top token for this PC direction
        top_tok: str | None = None
        if i < Vt.shape[0]:
            try:
                import mlx.core as mx

                lm_head = _get_lm_projection(model)
                logits = _project_to_logits(lm_head, mx.array(Vt[i]))
                tid = int(mx.argmax(logits).item())
                top_tok = tokenizer.decode([tid])
            except Exception:
                pass

        spectrum.append(
            SpectrumEntry(
                dimension=i + 1,
                variance_explained=round(var_exp, 6),
                cumulative_variance=round(cumulative, 6),
                top_token=top_tok,
            )
        )

        if cumulative >= 0.50 and dims_50 == n_components:
            dims_50 = i + 1
        if cumulative >= 0.80 and dims_80 == n_components:
            dims_80 = i + 1
        if cumulative >= 0.95 and dims_95 == n_components:
            dims_95 = i + 1
        if cumulative >= 0.99 and dims_99 == n_components:
            dims_99 = i + 1

    # Classification by dimension
    class_by_dim: list[ClassificationByDim] = []
    test_dims = [d for d in [1, 2, 3, 5, 10, 20, 50] if d <= n_components]

    for nd in test_dims:
        proj = X_centered @ Vt[:nd].T
        pos_proj = proj[y == 1].mean(axis=0)
        neg_proj = proj[y == 0].mean(axis=0)
        midpoint = (pos_proj + neg_proj) / 2
        direction = pos_proj - neg_proj
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm < 1e-12:
            acc = 0.5
        else:
            scores = (proj - midpoint) @ (direction / dir_norm)
            preds = (scores > 0).astype(int)
            acc = float(np.mean(preds == y))
        class_by_dim.append(ClassificationByDim(num_dimensions=nd, accuracy=round(acc, 4)))

    # Interpret
    if dims_80 <= 3:
        interpretation = "directional"
    elif dims_80 <= 20:
        interpretation = "subspace"
    else:
        interpretation = "distributed"

    summary = {
        "is_directional": dims_80 <= 3,
        "is_subspace": 3 < dims_80 <= 20,
        "is_distributed": dims_80 > 20,
        "interpretation": interpretation,
    }

    return FeatureDimensionalityResult(
        layer=layer,
        num_positive=len(pos_prompts),
        num_negative=len(neg_prompts),
        hidden_dim=meta.hidden_dim,
        effective_dimensionality={
            "dims_for_50pct": dims_50,
            "dims_for_80pct": dims_80,
            "dims_for_95pct": dims_95,
            "dims_for_99pct": dims_99,
        },
        spectrum=spectrum,
        classification_by_dim=class_by_dim,
        summary=summary,
    ).model_dump(exclude_none=True)
