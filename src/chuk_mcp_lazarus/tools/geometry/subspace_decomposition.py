"""
subspace_decomposition — decompose a target vector into basis components.

Decomposes a target vector into components along basis directions with
optional Gram-Schmidt orthogonalisation. Reports energy fractions and
the orthogonal residual. All in the full native dimensionality.
"""

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
    _angle_between,
    _extract_direction_vector,
    _gram_schmidt,
    _parse_direction_spec,
    _unit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class SubspaceComponent(BaseModel):
    """Projection of a target vector onto one basis direction."""

    direction_label: str
    projection: float = Field(
        ...,
        description="Signed scalar projection along this direction.",
    )
    fraction_of_target: float = Field(
        ...,
        description="|projection|^2 / |target|^2 (energy fraction).",
    )
    angle_to_target: float = Field(
        ...,
        description="Angle between direction and target in degrees.",
    )


class SubspaceSummary(BaseModel):
    """Summary of the subspace decomposition."""

    total_fraction_in_subspace: float
    residual_fraction: float
    residual_norm: float
    effective_subspace_dim: int


class SubspaceDecompositionResult(BaseModel):
    """Result from subspace_decomposition."""

    prompt: str
    layer: int
    hidden_dim: int
    target: dict[str, Any]
    components: list[SubspaceComponent]
    subspace_summary: SubspaceSummary
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def subspace_decomposition(
    prompt: str,
    layer: int,
    target: dict,
    basis_directions: list[dict],
    token_position: int = -1,
    orthogonalize: bool = True,
) -> dict:
    """Decompose a target vector into components along basis directions.

    Reports how much of the target lives in the subspace spanned by the
    basis versus orthogonal to it. All in the full native dimensionality.

    Uses Gram-Schmidt orthogonalisation by default so components are
    independent. Set orthogonalize=False for raw (possibly overlapping)
    projections.

    Args:
        prompt:           Input text.
        layer:            Layer to analyse.
        target:           Vector to decompose (same spec as direction_angles).
        basis_directions: Directions to project onto (same spec format).
        token_position:   Token position (-1 = last).
        orthogonalize:    Gram-Schmidt the basis first (default: True).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "subspace_decomposition"
        )
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "subspace_decomposition",
        )
    if not basis_directions:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 basis direction required.",
            "subspace_decomposition",
        )
    if len(basis_directions) > 20:
        return make_error(
            ToolError.INVALID_INPUT, "Maximum 20 basis directions.", "subspace_decomposition"
        )
    # Parse specs at boundary
    target_spec, t_err = _parse_direction_spec(target)
    if t_err is not None:
        return make_error(ToolError.INVALID_INPUT, f"Target: {t_err}", "subspace_decomposition")
    assert target_spec is not None
    basis_specs: list[DirectionSpec] = []
    for raw in basis_directions:
        spec, err = _parse_direction_spec(raw)
        if err is not None:
            return make_error(ToolError.INVALID_INPUT, f"Basis: {err}", "subspace_decomposition")
        assert spec is not None
        basis_specs.append(spec)
    try:
        return await asyncio.to_thread(
            _subspace_decomposition_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            layer,
            target_spec,
            basis_specs,
            token_position,
            orthogonalize,
        )
    except Exception as exc:
        logger.exception("subspace_decomposition failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "subspace_decomposition")


def _subspace_decomposition_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    target_spec: DirectionSpec,
    basis_specs: list[DirectionSpec],
    token_position: int,
    orthogonalize: bool,
) -> dict:
    """Sync implementation of subspace_decomposition."""
    import mlx.core as mx

    # Check if we need decomposition
    decomp_types = {
        DirectionType.RESIDUAL,
        DirectionType.FFN_OUTPUT,
        DirectionType.ATTENTION_OUTPUT,
        DirectionType.HEAD_OUTPUT,
    }
    all_specs = [target_spec] + basis_specs
    needs_decomp = any(s.type in decomp_types for s in all_specs)

    decomp = None
    if needs_decomp:
        from ..residual_tools import _run_decomposition_forward

        tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = mx.array(tok_ids)
        decomp = _run_decomposition_forward(model, config, input_ids, [layer])

    # Extract target
    t_label, t_vec, t_err = _extract_direction_vector(
        model,
        config,
        tokenizer,
        meta,
        prompt,
        layer,
        target_spec,
        token_position,
        decomp,
    )
    if t_err is not None:
        return make_error(ToolError.INVALID_INPUT, f"Target: {t_err}", "subspace_decomposition")
    assert t_vec is not None
    target_norm = float(np.linalg.norm(t_vec))

    # Extract basis directions
    basis_labels: list[str] = []
    basis_vecs: list[np.ndarray] = []
    for spec in basis_specs:
        b_label, b_vec, b_err = _extract_direction_vector(
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
        if b_err is not None:
            return make_error(ToolError.INVALID_INPUT, f"Basis: {b_err}", "subspace_decomposition")
        assert b_vec is not None
        basis_labels.append(b_label)
        basis_vecs.append(b_vec)

    # Optionally orthogonalise
    if orthogonalize:
        ortho_basis = _gram_schmidt(basis_vecs)
        effective_dim = len(ortho_basis)
        proj_basis = ortho_basis
        proj_labels = basis_labels[:effective_dim]
    else:
        proj_basis = [_unit(v) for v in basis_vecs]
        proj_labels = basis_labels
        effective_dim = len(proj_basis)

    # Project target onto each basis direction
    components: list[SubspaceComponent] = []
    total_energy = 0.0
    target_norm_sq = target_norm**2 if target_norm > 1e-12 else 1e-12

    for label, bv in zip(proj_labels, proj_basis):
        proj = float(np.dot(t_vec, bv))
        frac = (proj**2) / target_norm_sq
        ang = _angle_between(bv, t_vec)
        total_energy += frac
        components.append(
            SubspaceComponent(
                direction_label=label,
                projection=round(proj, 6),
                fraction_of_target=round(frac, 6),
                angle_to_target=round(ang, 4),
            )
        )

    # Residual (orthogonal complement)
    residual_vec = t_vec.copy().astype(np.float64)
    for bv in proj_basis:
        bv64 = bv.astype(np.float64)
        residual_vec -= float(np.dot(residual_vec, bv64)) * bv64
    residual_norm = float(np.linalg.norm(residual_vec))
    residual_frac = max(0.0, (residual_norm**2) / target_norm_sq)

    sub_summary = SubspaceSummary(
        total_fraction_in_subspace=round(
            min(total_energy, 1.0) if orthogonalize else total_energy,
            6,
        ),
        residual_fraction=round(residual_frac, 6),
        residual_norm=round(residual_norm, 4),
        effective_subspace_dim=effective_dim,
    )

    # Find dominant component
    dominant: dict[str, Any] = {}
    if components:
        dom = max(components, key=lambda c: abs(c.projection))
        dominant = {
            "direction": dom.direction_label,
            "projection": dom.projection,
            "fraction": dom.fraction_of_target,
        }

    summary = {
        "target_label": t_label,
        "target_norm": round(target_norm, 4),
        "dominant_component": dominant,
        "orthogonalized": orthogonalize,
    }

    return SubspaceDecompositionResult(
        prompt=prompt,
        layer=layer,
        hidden_dim=meta.hidden_dim,
        target={"label": t_label, "norm": round(target_norm, 4)},
        components=components,
        subspace_summary=sub_summary,
        summary=summary,
    ).model_dump()
