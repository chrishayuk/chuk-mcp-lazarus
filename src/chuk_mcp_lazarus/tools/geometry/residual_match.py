"""
residual_match — find prompts with the most similar residual streams.

Given a target prompt and candidates, computes cosine similarity and
angular distance between their residual streams at a specified layer.
Optionally measures similarity within a task-relevant subspace defined
by unembedding vectors, separately from the full space.
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
    _angle_between,
    _cosine_sim,
    _get_unembed_vec_np,
    _gram_schmidt,
    _resolve_token_to_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class SubspaceMatchInfo(BaseModel):
    """Subspace-specific similarity for one candidate."""

    subspace_cosine_similarity: float
    subspace_angle: float = Field(..., description="Angle in degrees within subspace.")
    orthogonal_cosine_similarity: float
    orthogonal_angle: float = Field(..., description="Angle in degrees in orthogonal complement.")


class MatchEntry(BaseModel):
    """Similarity of one candidate prompt to the target."""

    prompt: str
    full_cosine_similarity: float
    full_angle: float = Field(..., description="Angle in degrees.")
    subspace: SubspaceMatchInfo | None = None


class ResidualMatchResult(BaseModel):
    """Result from residual_match."""

    target_prompt: str
    layer: int
    token_position: int
    num_candidates: int
    matches: list[MatchEntry]
    best_full_match: str
    best_subspace_match: str | None = None
    subspace_dim: int | None = None
    subspace_tokens: list[str] | None = None
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def residual_match(
    target_prompt: str,
    candidate_prompts: list[str],
    layer: int,
    token_position: int = -1,
    subspace_tokens: list[str] | None = None,
) -> dict:
    """Find which candidate prompts produce the most similar residual
    stream to the target at a given layer.

    Useful for finding natural test cases for the Markov property:
    two prompts that arrive at similar states through different paths.
    If inject_residual shows identical downstream behaviour, the
    Markov property holds for that state.

    If subspace_tokens are provided, also reports similarity within
    the task-relevant subspace separately from the full space.

    Args:
        target_prompt:     Reference prompt.
        candidate_prompts: 1-20 prompts to compare against.
        layer:             Layer to compare at.
        token_position:    Token position (-1 = last).
        subspace_tokens:   Optional tokens defining a task-relevant
                           subspace for separate similarity reporting.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "residual_match",
        )
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "residual_match",
        )
    if not candidate_prompts:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 candidate required.",
            "residual_match",
        )
    if len(candidate_prompts) > 20:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 20 candidates.",
            "residual_match",
        )
    try:
        return await asyncio.to_thread(
            _residual_match_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            target_prompt,
            candidate_prompts,
            layer,
            token_position,
            subspace_tokens,
        )
    except Exception as exc:
        logger.exception("residual_match failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "residual_match")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _residual_match_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    target_prompt: str,
    candidate_prompts: list[str],
    layer: int,
    token_position: int,
    subspace_tokens: list[str] | None,
) -> dict:
    """Sync implementation of residual_match."""
    import mlx.core as mx

    from ..._residual_helpers import _extract_position, _run_decomposition_forward

    # -- Run target --
    target_ids = mx.array(tokenizer.encode(target_prompt, add_special_tokens=True))
    target_decomp = _run_decomposition_forward(model, config, target_ids, [layer])
    mx.eval(*target_decomp["hidden_states"].values())
    target_h = _extract_position(target_decomp["hidden_states"][layer], token_position)
    target_np = np.array(target_h.tolist(), dtype=np.float32)

    # -- Build subspace basis (if requested) --
    basis: list[np.ndarray] = []
    tokens_used: list[str] = []
    if subspace_tokens:
        unembed_vecs: list[np.ndarray] = []
        for tok in subspace_tokens:
            tid = _resolve_token_to_id(tokenizer, tok)
            if tid is None:
                continue
            uv = _get_unembed_vec_np(model, tid)
            if uv is not None:
                unembed_vecs.append(uv)
                tokens_used.append(tok)
        if unembed_vecs:
            basis = _gram_schmidt(unembed_vecs)

    # Project target onto subspace (if basis available)
    target_sub: np.ndarray | None = None
    target_orth: np.ndarray | None = None
    if basis:
        target_sub = np.zeros_like(target_np)
        for b in basis:
            target_sub = target_sub + float(np.dot(target_np, b)) * b
        target_orth = target_np - target_sub

    # -- Run each candidate --
    matches: list[MatchEntry] = []
    for cand_prompt in candidate_prompts:
        cand_ids = mx.array(tokenizer.encode(cand_prompt, add_special_tokens=True))
        cand_decomp = _run_decomposition_forward(model, config, cand_ids, [layer])
        mx.eval(*cand_decomp["hidden_states"].values())
        cand_h = _extract_position(cand_decomp["hidden_states"][layer], token_position)
        cand_np = np.array(cand_h.tolist(), dtype=np.float32)

        full_cos = _cosine_sim(target_np, cand_np)
        full_angle = _angle_between(target_np, cand_np)

        sub_info: SubspaceMatchInfo | None = None
        if basis and target_sub is not None and target_orth is not None:
            cand_sub = np.zeros_like(cand_np)
            for b in basis:
                cand_sub = cand_sub + float(np.dot(cand_np, b)) * b  # type: ignore[assignment]
            cand_orth = cand_np - cand_sub

            sub_info = SubspaceMatchInfo(
                subspace_cosine_similarity=round(_cosine_sim(target_sub, cand_sub), 6),
                subspace_angle=round(_angle_between(target_sub, cand_sub), 4),
                orthogonal_cosine_similarity=round(_cosine_sim(target_orth, cand_orth), 6),
                orthogonal_angle=round(_angle_between(target_orth, cand_orth), 4),
            )

        matches.append(
            MatchEntry(
                prompt=cand_prompt,
                full_cosine_similarity=round(full_cos, 6),
                full_angle=round(full_angle, 4),
                subspace=sub_info,
            )
        )

    # -- Sort by similarity descending --
    matches.sort(key=lambda m: m.full_cosine_similarity, reverse=True)

    # -- Find best matches --
    best_full = matches[0].prompt if matches else ""
    best_subspace: str | None = None
    if basis and matches:
        best_sub_entry = max(
            (m for m in matches if m.subspace is not None),
            key=lambda m: m.subspace.subspace_cosine_similarity,  # type: ignore[union-attr]
            default=None,
        )
        if best_sub_entry is not None:
            best_subspace = best_sub_entry.prompt

    # -- Summary --
    summary: dict[str, Any] = {
        "best_similarity": matches[0].full_cosine_similarity if matches else 0.0,
        "best_angle": matches[0].full_angle if matches else 180.0,
        "worst_similarity": matches[-1].full_cosine_similarity if matches else 0.0,
    }
    if basis:
        summary["subspace_dim"] = len(basis)
        if best_subspace and matches:
            best_sub_m = next((m for m in matches if m.prompt == best_subspace), None)
            if best_sub_m and best_sub_m.subspace:
                summary["best_subspace_similarity"] = best_sub_m.subspace.subspace_cosine_similarity

    return ResidualMatchResult(
        target_prompt=target_prompt,
        layer=layer,
        token_position=token_position,
        num_candidates=len(matches),
        matches=matches,
        best_full_match=best_full,
        best_subspace_match=best_subspace,
        subspace_dim=len(basis) if basis else None,
        subspace_tokens=tokens_used if tokens_used else None,
        summary=summary,
    ).model_dump()
