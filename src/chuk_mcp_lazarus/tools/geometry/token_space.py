"""
token_space — angles between token unembed vectors + residual stream.

Maps geometric relationships between token directions and the residual
stream at a specific layer. Works in the full native dimensionality.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..._extraction import extract_activation_at_layer
from ..._serialize import pca_2d
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import (
    _angle_between,
    _angle_deg,
    _cosine_sim,
    _get_unembed_vec_np,
    _resolve_token_to_id,
    _token_text_at,
    _unit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class TokenSpaceEntry(BaseModel):
    """A token's geometric position relative to the residual stream."""

    token: str
    token_id: int
    angle_to_residual: float = Field(
        ..., description="Angle in degrees between unembed vector and residual."
    )
    projection_on_residual: float = Field(
        ..., description="Scalar projection of residual onto token direction."
    )
    norm: float = Field(..., description="L2 norm of unembed vector.")


class TokenPairAngle(BaseModel):
    """Pairwise angular relationship between two tokens."""

    token_a: str
    token_b: str
    angle: float = Field(..., description="Angle in degrees in the full space.")
    cosine_similarity: float


class ResidualInfo(BaseModel):
    """Information about the residual stream at the analysis layer."""

    norm: float
    top_aligned_token: str
    top_aligned_angle: float


class TokenSpaceResult(BaseModel):
    """Result from token_space — geometric map of tokens vs residual."""

    prompt: str
    layer: int
    token_position: int
    token_text: str
    hidden_dim: int
    tokens: list[TokenSpaceEntry]
    pairwise_angles: list[TokenPairAngle]
    residual_info: ResidualInfo
    projection_2d: list[dict[str, Any]] | None = None
    projection_variance_explained: float | None = None
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def token_space(
    prompt: str,
    layer: int,
    tokens: list[str],
    token_position: int = -1,
    include_projection: bool = False,
) -> dict:
    """Map geometric relationships between token directions and the residual
    stream at a specific layer. Works in the full native dimensionality.

    Reports angles (degrees) and projections between each token's unembedding
    vector and the residual stream, plus pairwise angles between all tokens.

    Args:
        prompt:             Input text.
        layer:              Layer to analyse.
        tokens:             Tokens to map (e.g. ["Sydney", "Canberra"]).
        token_position:     Which token position to use (-1 = last).
        include_projection: Include lossy 2D PCA projection (default: False).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "token_space")
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "token_space",
        )
    if not tokens:
        return make_error(ToolError.INVALID_INPUT, "tokens list must be non-empty.", "token_space")
    if len(tokens) > 20:
        return make_error(ToolError.INVALID_INPUT, "Maximum 20 tokens.", "token_space")
    try:
        return await asyncio.to_thread(
            _token_space_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            layer,
            tokens,
            token_position,
            include_projection,
        )
    except Exception as exc:
        logger.exception("token_space failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "token_space")


def _token_space_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    tokens: list[str],
    token_position: int,
    include_projection: bool,
) -> dict:
    """Sync implementation of token_space."""
    tok_text = _token_text_at(tokenizer, prompt, token_position)

    # Get residual vector at layer (one forward pass)
    residual_list = extract_activation_at_layer(
        model,
        config,
        tokenizer,
        prompt,
        layer,
        token_position,
    )
    residual = np.array(residual_list, dtype=np.float32)
    residual_norm = float(np.linalg.norm(residual))

    # Get unembed vectors for each token (weight lookups, no forward pass)
    entries: list[TokenSpaceEntry] = []
    token_vecs: list[np.ndarray] = []
    token_names: list[str] = []

    for tok in tokens:
        tid = _resolve_token_to_id(tokenizer, tok)
        if tid is None:
            return make_error(
                ToolError.INVALID_INPUT, f"Cannot encode token '{tok}'.", "token_space"
            )
        vec = _get_unembed_vec_np(model, tid)
        if vec is None:
            return make_error(
                ToolError.INVALID_INPUT, f"No unembed vector for '{tok}'.", "token_space"
            )
        token_vecs.append(vec)
        token_names.append(tok)
        entries.append(
            TokenSpaceEntry(
                token=tok,
                token_id=tid,
                angle_to_residual=round(_angle_between(vec, residual), 4),
                projection_on_residual=round(float(np.dot(residual, _unit(vec))), 4),
                norm=round(float(np.linalg.norm(vec)), 4),
            )
        )

    # Pairwise angles between tokens
    pairwise: list[TokenPairAngle] = []
    for i in range(len(token_vecs)):
        for j in range(i + 1, len(token_vecs)):
            cs = _cosine_sim(token_vecs[i], token_vecs[j])
            pairwise.append(
                TokenPairAngle(
                    token_a=token_names[i],
                    token_b=token_names[j],
                    angle=round(_angle_deg(cs), 4),
                    cosine_similarity=round(cs, 6),
                )
            )

    # Residual info
    best_idx = int(np.argmin([e.angle_to_residual for e in entries]))
    res_info = ResidualInfo(
        norm=round(residual_norm, 4),
        top_aligned_token=entries[best_idx].token,
        top_aligned_angle=entries[best_idx].angle_to_residual,
    )

    # Optional PCA projection
    proj_2d = None
    var_explained = None
    if include_projection and len(token_vecs) >= 2:
        all_vecs = [v.tolist() for v in token_vecs] + [residual.tolist()]
        all_labels = token_names + ["residual"]
        pca = pca_2d(all_vecs)
        proj_2d = [
            {"label": lbl, "x": round(p[0], 6), "y": round(p[1], 6)}
            for lbl, p in zip(all_labels, pca)
        ]
        # Variance explained by first 2 PCs
        vecs_np = np.array(all_vecs, dtype=np.float32)
        centered = vecs_np - vecs_np.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = float(np.sum(s**2))
        if total_var > 0:
            var_explained = round(float(np.sum(s[:2] ** 2)) / total_var, 4)

    # Summary
    angles_to_res = [e.angle_to_residual for e in entries]
    farthest_idx = int(np.argmax(angles_to_res))
    inter_angles = [p.angle for p in pairwise] if pairwise else [0.0]
    summary = {
        "nearest_token": entries[best_idx].token,
        "nearest_angle": entries[best_idx].angle_to_residual,
        "farthest_token": entries[farthest_idx].token,
        "farthest_angle": entries[farthest_idx].angle_to_residual,
        "mean_inter_token_angle": round(float(np.mean(inter_angles)), 4),
    }

    return TokenSpaceResult(
        prompt=prompt,
        layer=layer,
        token_position=token_position,
        token_text=tok_text,
        hidden_dim=meta.hidden_dim,
        tokens=entries,
        pairwise_angles=pairwise,
        residual_info=res_info,
        projection_2d=proj_2d,
        projection_variance_explained=var_explained,
        summary=summary,
    ).model_dump(exclude_none=True)
