"""
residual_trajectory — track residual rotation through layers.

Tracks how the residual stream moves through activation space across
layers, measured by angles to reference token directions. Reports
rotation angles, crossing layers, and orthogonal fractions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..._extraction import extract_activations_all_layers
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import (
    _angle_between,
    _auto_layers,
    _get_unembed_vec_np,
    _gram_schmidt,
    _resolve_token_to_id,
    _token_text_at,
    _unit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class TrajectoryAngle(BaseModel):
    """Angle and projection to a reference token at one layer."""

    token: str
    angle: float = Field(..., description="Angle in degrees in the full space.")
    projection: float
    fraction: float


class TrajectoryDelta(BaseModel):
    """Change in residual stream between consecutive layers."""

    rotation_angle: float = Field(..., description="Rotation in degrees from previous layer.")
    component_deltas: list[dict[str, Any]]


class TrajectoryPoint(BaseModel):
    """Residual stream position at one layer."""

    layer: int
    residual_norm: float
    angles: list[TrajectoryAngle]
    dominant_token: str
    dominant_angle: float
    orthogonal_fraction: float = Field(
        ...,
        description="Fraction of residual NOT in any reference token direction.",
    )
    delta_from_previous: TrajectoryDelta | None = None


class ResidualTrajectoryResult(BaseModel):
    """Result from residual_trajectory — the residual stream's path."""

    prompt: str
    token_position: int
    token_text: str
    hidden_dim: int
    reference_tokens: list[dict[str, Any]]
    num_layers: int
    trajectory: list[TrajectoryPoint]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def residual_trajectory(
    prompt: str,
    reference_tokens: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """Track how the residual stream moves through activation space across
    layers, measured by angles to reference token directions.

    At each layer, reports the angle to each reference token, the rotation
    from the previous layer, and what fraction of the residual is orthogonal
    to all reference tokens.

    Args:
        prompt:           Input text.
        reference_tokens: Tokens to measure against (e.g. ["Sydney", "Canberra"]).
        layers:           Layer indices (None = all layers).
        token_position:   Token position (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "residual_trajectory"
        )
    meta = state.metadata
    if not reference_tokens:
        return make_error(
            ToolError.INVALID_INPUT, "reference_tokens must be non-empty.", "residual_trajectory"
        )
    if len(reference_tokens) > 10:
        return make_error(
            ToolError.INVALID_INPUT, "Maximum 10 reference tokens.", "residual_trajectory"
        )
    if layers is None:
        layers = _auto_layers(meta.num_layers)
    for lyr in layers:
        if lyr < 0 or lyr >= meta.num_layers:
            return make_error(
                ToolError.LAYER_OUT_OF_RANGE,
                f"Layer {lyr} out of [0, {meta.num_layers - 1}].",
                "residual_trajectory",
            )
    try:
        return await asyncio.to_thread(
            _residual_trajectory_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            reference_tokens,
            layers,
            token_position,
        )
    except Exception as exc:
        logger.exception("residual_trajectory failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "residual_trajectory")


def _residual_trajectory_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    reference_tokens: list[str],
    layers: list[int],
    token_position: int,
) -> dict:
    """Sync implementation of residual_trajectory."""
    tok_text = _token_text_at(tokenizer, prompt, token_position)

    # Single forward pass for all layers
    layer_acts = extract_activations_all_layers(
        model,
        config,
        tokenizer,
        prompt,
        layers,
        token_position,
    )

    # Get unembed vectors for reference tokens
    ref_info: list[dict[str, Any]] = []
    ref_vecs: list[np.ndarray] = []
    ref_names: list[str] = []
    for tok in reference_tokens:
        tid = _resolve_token_to_id(tokenizer, tok)
        if tid is None:
            return make_error(
                ToolError.INVALID_INPUT, f"Cannot encode token '{tok}'.", "residual_trajectory"
            )
        vec = _get_unembed_vec_np(model, tid)
        if vec is None:
            return make_error(
                ToolError.INVALID_INPUT, f"No unembed vector for '{tok}'.", "residual_trajectory"
            )
        ref_vecs.append(vec)
        ref_names.append(tok)
        ref_info.append(
            {
                "token": tok,
                "token_id": tid,
                "unembed_norm": round(float(np.linalg.norm(vec)), 4),
            }
        )

    # Orthonormalize reference directions for subspace membership
    ref_ortho = _gram_schmidt(ref_vecs)

    # Build trajectory
    trajectory: list[TrajectoryPoint] = []
    prev_residual: np.ndarray | None = None
    total_rotation = 0.0
    max_rotation = 0.0
    max_rotation_layer = layers[0] if layers else 0
    crossings: list[int] = []
    prev_dominant = ""

    for lyr in sorted(layers):
        if lyr not in layer_acts:
            continue
        residual = np.array(layer_acts[lyr], dtype=np.float32)
        res_norm = float(np.linalg.norm(residual))

        # Angles to each reference token
        angles: list[TrajectoryAngle] = []
        for tok_name, tok_vec in zip(ref_names, ref_vecs):
            ang = _angle_between(residual, tok_vec)
            tok_unit = _unit(tok_vec)
            proj = float(np.dot(residual, tok_unit))
            frac = (proj**2) / (res_norm**2 + 1e-12)
            angles.append(
                TrajectoryAngle(
                    token=tok_name,
                    angle=round(ang, 4),
                    projection=round(proj, 4),
                    fraction=round(frac, 6),
                )
            )

        dom = min(angles, key=lambda a: a.angle)

        # Orthogonal fraction
        in_subspace = 0.0
        for bv in ref_ortho:
            p = float(np.dot(residual, bv))
            in_subspace += p**2
        ortho_frac = max(0.0, 1.0 - in_subspace / (res_norm**2 + 1e-12))

        # Delta from previous layer
        delta = None
        if prev_residual is not None:
            rot = _angle_between(residual, prev_residual)
            total_rotation += rot
            if rot > max_rotation:
                max_rotation = rot
                max_rotation_layer = lyr

            comp_deltas = []
            diff = residual - prev_residual
            for tok_name, tok_vec in zip(ref_names, ref_vecs):
                delta_proj = float(np.dot(diff, _unit(tok_vec)))
                comp_deltas.append({"token": tok_name, "delta_projection": round(delta_proj, 4)})

            delta = TrajectoryDelta(rotation_angle=round(rot, 4), component_deltas=comp_deltas)

        if prev_dominant and dom.token != prev_dominant:
            crossings.append(lyr)
        prev_dominant = dom.token

        trajectory.append(
            TrajectoryPoint(
                layer=lyr,
                residual_norm=round(res_norm, 4),
                angles=angles,
                dominant_token=dom.token,
                dominant_angle=round(dom.angle, 4),
                orthogonal_fraction=round(ortho_frac, 6),
                delta_from_previous=delta,
            )
        )
        prev_residual = residual

    summary = {
        "total_rotation": round(total_rotation, 4),
        "crossing_layers": crossings,
        "max_single_rotation": round(max_rotation, 4),
        "max_rotation_layer": max_rotation_layer,
        "final_dominant_token": trajectory[-1].dominant_token if trajectory else "",
        "final_dominant_angle": trajectory[-1].dominant_angle if trajectory else 0.0,
    }

    return ResidualTrajectoryResult(
        prompt=prompt,
        token_position=token_position,
        token_text=tok_text,
        hidden_dim=meta.hidden_dim,
        reference_tokens=ref_info,
        num_layers=len(trajectory),
        trajectory=trajectory,
        summary=summary,
    ).model_dump(exclude_none=True)
