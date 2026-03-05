"""
direction_angles — pairwise angles between any directions.

Computes pairwise angles between arbitrary directions in the model's
activation space at a specific layer. Supports tokens, neurons, residual
stream, FFN/attention outputs, head outputs, and steering vectors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import (
    DirectionSpec,
    DirectionType,
    _angle_deg,
    _cosine_sim,
    _extract_direction_vector,
    _parse_direction_spec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class DirectionEntry(BaseModel):
    """Metadata about a direction in the geometry analysis."""

    label: str
    type: DirectionType
    value: Any = None
    norm: float


class DirectionPairResult(BaseModel):
    """Pairwise relationship between two directions in the full space."""

    direction_a: str
    direction_b: str
    angle: float = Field(..., description="Angle in degrees in the full space.")
    cosine_similarity: float
    dot_product: float


class DirectionAnglesResult(BaseModel):
    """Result from direction_angles — pairwise angular relationships."""

    prompt: str
    layer: int
    hidden_dim: int
    directions: list[DirectionEntry]
    pairwise: list[DirectionPairResult]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def direction_angles(
    prompt: str,
    layer: int,
    directions: list[dict],
    token_position: int = -1,
) -> dict:
    """Compute pairwise angles between arbitrary directions in the model's
    activation space at a specific layer.

    Each direction is a dict with 'type' and optional 'value':
        {"type": "token", "value": "Sydney"}
        {"type": "neuron", "value": 9444}
        {"type": "residual"}
        {"type": "ffn_output"}
        {"type": "attention_output"}
        {"type": "head_output", "value": 1}
        {"type": "steering_vector", "value": "my_vector_name"}

    All angles are in degrees in the full native dimensionality.

    Args:
        prompt:         Input text.
        layer:          Layer to extract directions at.
        directions:     List of direction specifications.
        token_position: Token position (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "direction_angles"
        )
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "direction_angles",
        )
    if not directions or len(directions) < 2:
        return make_error(
            ToolError.INVALID_INPUT, "At least 2 directions required.", "direction_angles"
        )
    if len(directions) > 20:
        return make_error(ToolError.INVALID_INPUT, "Maximum 20 directions.", "direction_angles")
    # Validate and parse direction specs at the boundary
    specs: list[DirectionSpec] = []
    for raw in directions:
        spec, err = _parse_direction_spec(raw)
        if err is not None:
            return make_error(ToolError.INVALID_INPUT, err, "direction_angles")
        assert spec is not None
        specs.append(spec)
    try:
        return await asyncio.to_thread(
            _direction_angles_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            layer,
            specs,
            token_position,
        )
    except Exception as exc:
        logger.exception("direction_angles failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "direction_angles")


def _direction_angles_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    specs: list[DirectionSpec],
    token_position: int,
) -> dict:
    """Sync implementation of direction_angles."""
    import mlx.core as mx

    # Check if we need decomposition
    decomp_types = {
        DirectionType.RESIDUAL,
        DirectionType.FFN_OUTPUT,
        DirectionType.ATTENTION_OUTPUT,
        DirectionType.HEAD_OUTPUT,
    }
    needs_decomp = any(s.type in decomp_types for s in specs)

    decomp = None
    if needs_decomp:
        from ..residual_tools import _run_decomposition_forward

        tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = mx.array(tok_ids)
        decomp = _run_decomposition_forward(model, config, input_ids, [layer])

    # Extract all direction vectors
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    entries: list[DirectionEntry] = []

    for spec in specs:
        label, vec, err = _extract_direction_vector(
            model,
            config,
            tokenizer,
            meta,
            prompt,
            layer,
            spec,
            token_position,
            decomp,
        )
        if err is not None:
            return make_error(ToolError.INVALID_INPUT, err, "direction_angles")
        assert vec is not None
        labels.append(label)
        vectors.append(vec)
        entries.append(
            DirectionEntry(
                label=label,
                type=spec.type,
                value=spec.value,
                norm=round(float(np.linalg.norm(vec)), 4),
            )
        )

    # Pairwise angles
    pairwise: list[DirectionPairResult] = []
    most_aligned = ("", "", 180.0)
    most_opposed = ("", "", 0.0)
    most_orthogonal = ("", "", 180.0)

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cs = _cosine_sim(vectors[i], vectors[j])
            ang = _angle_deg(cs)
            dp = float(np.dot(vectors[i], vectors[j]))
            pairwise.append(
                DirectionPairResult(
                    direction_a=labels[i],
                    direction_b=labels[j],
                    angle=round(ang, 4),
                    cosine_similarity=round(cs, 6),
                    dot_product=round(dp, 4),
                )
            )
            if ang < most_aligned[2]:
                most_aligned = (labels[i], labels[j], ang)
            if ang > most_opposed[2]:
                most_opposed = (labels[i], labels[j], ang)
            if abs(ang - 90.0) < abs(most_orthogonal[2] - 90.0):
                most_orthogonal = (labels[i], labels[j], ang)

    summary = {
        "most_aligned_pair": {
            "a": most_aligned[0],
            "b": most_aligned[1],
            "angle": round(most_aligned[2], 4),
        },
        "most_opposed_pair": {
            "a": most_opposed[0],
            "b": most_opposed[1],
            "angle": round(most_opposed[2], 4),
        },
        "most_orthogonal_pair": {
            "a": most_orthogonal[0],
            "b": most_orthogonal[1],
            "angle": round(most_orthogonal[2], 4),
        },
    }

    return DirectionAnglesResult(
        prompt=prompt,
        layer=layer,
        hidden_dim=meta.hidden_dim,
        directions=entries,
        pairwise=pairwise,
        summary=summary,
    ).model_dump()
