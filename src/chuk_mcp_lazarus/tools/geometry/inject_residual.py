"""
inject_residual — inject donor residual stream into recipient prompt.

Tests the Markov property: if downstream layers only care about the
current residual state, the output after injection should match the
donor's regardless of the recipient's history.  Supports full-space
and subspace-only injection.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..._generate import generate_text
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ...subspace_registry import SubspaceRegistry
from ._helpers import (
    _angle_between,
    _cosine_sim,
    _get_unembed_vec_np,
    _gram_schmidt,
    _resolve_token_to_id,
)

from ._injection_helpers import (
    TokenPrediction,
    _generate_from_hidden,
    _kl_divergence,
    _run_forward_with_injection,
    _top_k_from_logits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class DonorRecipientOutput(BaseModel):
    """Output from running donor or recipient through the model."""

    text: str
    top_prediction: str
    top_prediction_id: int
    probability: float
    top_k: list[TokenPrediction]


class InjectedOutput(BaseModel):
    """Output from the injected forward pass."""

    text: str
    top_prediction: str
    top_prediction_id: int
    probability: float
    top_k: list[TokenPrediction]


class ComparisonMetrics(BaseModel):
    """Comparison between donor, recipient, and injected outputs."""

    injected_matches_donor: bool
    injected_matches_recipient: bool
    donor_recipient_kl: float = Field(..., description="KL(donor || recipient).")
    donor_injected_kl: float = Field(..., description="KL(donor || injected). Low = Markov holds.")
    recipient_injected_kl: float = Field(..., description="KL(recipient || injected).")


class ResidualSimilarity(BaseModel):
    """Similarity between donor and recipient residuals at injection layer."""

    cosine_similarity: float
    angle: float = Field(..., description="Angle in degrees.")
    donor_norm: float
    recipient_norm: float


class SubspaceAnalysis(BaseModel):
    """Subspace projection analysis (when subspace_only=True)."""

    subspace_dim: int
    tokens_used: list[str]
    donor_subspace_fraction: float = Field(..., description="Energy fraction of donor in subspace.")
    recipient_subspace_fraction: float = Field(
        ..., description="Energy fraction of recipient in subspace."
    )
    subspace_cosine_similarity: float
    orthogonal_cosine_similarity: float
    subspace_name: str | None = None


class InjectResidualResult(BaseModel):
    """Result from inject_residual."""

    donor_prompt: str
    recipient_prompt: str
    layer: int
    donor_layer: int | None = None
    donor_position: int
    recipient_position: int
    subspace_only: bool
    patch_all_positions: bool
    donor_output: DonorRecipientOutput
    recipient_output: DonorRecipientOutput
    injected_output: InjectedOutput
    comparison: ComparisonMetrics
    residual_similarity: ResidualSimilarity
    subspace_analysis: SubspaceAnalysis | None = None
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def inject_residual(
    donor_prompt: str,
    recipient_prompt: str,
    layer: int,
    donor_layer: int | None = None,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    donor_position: int = -1,
    recipient_position: int = -1,
    subspace_only: bool = False,
    subspace_tokens: list[str] | None = None,
    subspace_name: str | None = None,
    patch_all_positions: bool = False,
) -> dict:
    """Inject the residual stream from one prompt into another at a
    specific layer and continue generation.

    Runs the donor prompt to capture its residual stream.  At the
    injection layer, replaces the recipient's residual with the
    donor's (full vector or subspace-only).  Then continues the
    recipient's forward pass to the final layer and generates text.

    This tests the Markov property: if downstream layers only care
    about the current residual state (not the history), the output
    should match the donor's.

    Args:
        donor_prompt:       Prompt whose residual is captured.
        recipient_prompt:   Prompt that receives the injected residual.
        layer:              Layer at which to inject into the recipient.
        donor_layer:        Layer at which to capture the donor's residual.
                            Defaults to layer.  Use this to capture at a
                            later layer (e.g. L33) and inject at an
                            earlier layer (e.g. L24) for recirculation.
        max_new_tokens:     Tokens to generate after injection.
        temperature:        Sampling temperature (0 = greedy).
        donor_position:     Token position in donor (-1 = last).
        recipient_position: Token position in recipient (-1 = last).
        subspace_only:      Only inject the subspace component.
        subspace_tokens:    Tokens defining the injection subspace
                            (required if subspace_only=True).
        subspace_name:      Name of a PCA subspace from SubspaceRegistry
                            (from compute_subspace).  Mutually exclusive
                            with subspace_tokens.
        patch_all_positions: If True, inject the donor's entire hidden
                            state tensor across all positions, replacing
                            the recipient's full residual stream at that
                            layer.  Incompatible with subspace_only.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "inject_residual",
        )
    meta = state.metadata
    layer = int(layer)
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "inject_residual",
        )
    donor_layer = int(donor_layer) if donor_layer is not None else None
    effective_donor_layer = donor_layer if donor_layer is not None else layer
    if effective_donor_layer < 0 or effective_donor_layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"donor_layer {effective_donor_layer} out of [0, {meta.num_layers - 1}].",
            "inject_residual",
        )
    if max_new_tokens < 1 or max_new_tokens > 500:
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_new_tokens must be 1-500, got {max_new_tokens}.",
            "inject_residual",
        )
    if temperature < 0:
        return make_error(
            ToolError.INVALID_INPUT,
            "temperature must be non-negative.",
            "inject_residual",
        )
    if subspace_tokens and subspace_name:
        return make_error(
            ToolError.INVALID_INPUT,
            "subspace_tokens and subspace_name are mutually exclusive.",
            "inject_residual",
        )
    if subspace_only and not subspace_tokens and subspace_name is None:
        return make_error(
            ToolError.INVALID_INPUT,
            "subspace_tokens or subspace_name required when subspace_only=True.",
            "inject_residual",
        )
    if patch_all_positions and (subspace_only or subspace_name is not None):
        return make_error(
            ToolError.INVALID_INPUT,
            "patch_all_positions=True is incompatible with subspace injection.",
            "inject_residual",
        )
    if subspace_name is not None and not SubspaceRegistry.get().exists(subspace_name):
        return make_error(
            ToolError.VECTOR_NOT_FOUND,
            f"Subspace '{subspace_name}' not found in SubspaceRegistry.",
            "inject_residual",
        )
    if subspace_tokens and len(subspace_tokens) > 20:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 20 subspace_tokens.",
            "inject_residual",
        )
    try:
        return await asyncio.to_thread(
            _inject_residual_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            donor_prompt,
            recipient_prompt,
            layer,
            donor_layer,
            max_new_tokens,
            temperature,
            donor_position,
            recipient_position,
            subspace_only,
            subspace_tokens,
            subspace_name,
            patch_all_positions,
        )
    except Exception as exc:
        logger.exception("inject_residual failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "inject_residual")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _inject_residual_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    donor_prompt: str,
    recipient_prompt: str,
    layer: int,
    donor_layer: int | None,
    max_new_tokens: int,
    temperature: float,
    donor_position: int,
    recipient_position: int,
    subspace_only: bool,
    subspace_tokens: list[str] | None,
    subspace_name: str | None,
    patch_all_positions: bool = False,
) -> dict:
    """Sync implementation of inject_residual."""
    import mlx.core as mx

    from ..._residual_helpers import (
        _extract_position,
        _get_lm_projection,
        _norm_project,
        _run_decomposition_forward,
    )

    from chuk_lazarus.introspection.hooks import ModelHooks

    # -- Phase 1: Setup --
    donor_ids_list = tokenizer.encode(donor_prompt, add_special_tokens=True)
    recip_ids_list = tokenizer.encode(recipient_prompt, add_special_tokens=True)
    donor_ids = mx.array(donor_ids_list)
    recip_ids = mx.array(recip_ids_list)

    num_donor = len(donor_ids_list)
    num_recip = len(recip_ids_list)
    d_pos = donor_position if donor_position >= 0 else num_donor + donor_position
    d_pos = max(0, min(d_pos, num_donor - 1))
    r_pos = recipient_position if recipient_position >= 0 else num_recip + recipient_position
    r_pos = max(0, min(r_pos, num_recip - 1))

    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not access lm_head.",
            "inject_residual",
        )

    last_layer = meta.num_layers - 1
    effective_donor_layer = donor_layer if donor_layer is not None else layer
    donor_capture_layers = sorted(set([effective_donor_layer, last_layer]))
    recip_capture_layers = sorted(set([layer, last_layer]))

    # -- Phase 2: Capture donor state --
    donor_decomp = _run_decomposition_forward(model, config, donor_ids, donor_capture_layers)
    mx.eval(*donor_decomp["hidden_states"].values())

    donor_vec_at_layer = _extract_position(
        donor_decomp["hidden_states"][effective_donor_layer], d_pos
    )
    donor_vec_final = _extract_position(donor_decomp["hidden_states"][last_layer], d_pos)
    donor_np = np.array(donor_vec_at_layer.tolist(), dtype=np.float32)

    # -- Phase 3: Capture recipient state --
    recip_decomp = _run_decomposition_forward(model, config, recip_ids, recip_capture_layers)
    mx.eval(*recip_decomp["hidden_states"].values())

    recip_vec_at_layer = _extract_position(recip_decomp["hidden_states"][layer], r_pos)
    recip_vec_final = _extract_position(recip_decomp["hidden_states"][last_layer], r_pos)
    recip_np = np.array(recip_vec_at_layer.tolist(), dtype=np.float32)

    # -- Phase 4: Compute injection vector --
    subspace_result: SubspaceAnalysis | None = None

    if subspace_only and subspace_tokens:
        unembed_vecs: list[np.ndarray] = []
        tokens_used: list[str] = []
        for tok in subspace_tokens:
            tid = _resolve_token_to_id(tokenizer, tok)
            if tid is None:
                continue
            uv = _get_unembed_vec_np(model, tid)
            if uv is not None:
                unembed_vecs.append(uv)
                tokens_used.append(tok)

        if not unembed_vecs:
            return make_error(
                ToolError.EXTRACTION_FAILED,
                "No valid unembed vectors for subspace_tokens.",
                "inject_residual",
            )

        basis = _gram_schmidt(unembed_vecs)
        if not basis:
            return make_error(
                ToolError.EXTRACTION_FAILED,
                "Subspace basis is empty after Gram-Schmidt.",
                "inject_residual",
            )

        # Project donor onto subspace
        donor_sub = np.zeros_like(donor_np)
        for b in basis:
            donor_sub = donor_sub + float(np.dot(donor_np, b)) * b  # type: ignore[assignment]

        # Get recipient's orthogonal component
        recip_sub = np.zeros_like(recip_np)
        for b in basis:
            recip_sub = recip_sub + float(np.dot(recip_np, b)) * b  # type: ignore[assignment]
        recip_orth = recip_np - recip_sub

        injection_np = donor_sub + recip_orth

        # Subspace analysis
        donor_norm_sq = float(np.dot(donor_np, donor_np)) + 1e-12
        recip_norm_sq = float(np.dot(recip_np, recip_np)) + 1e-12
        d_sub_frac = float(np.dot(donor_sub, donor_sub)) / donor_norm_sq
        r_sub_frac = float(np.dot(recip_sub, recip_sub)) / recip_norm_sq

        donor_orth = donor_np - donor_sub
        sub_cos = _cosine_sim(donor_sub, recip_sub)
        orth_cos = _cosine_sim(donor_orth, recip_orth)

        subspace_result = SubspaceAnalysis(
            subspace_dim=len(basis),
            tokens_used=tokens_used,
            donor_subspace_fraction=round(d_sub_frac, 6),
            recipient_subspace_fraction=round(r_sub_frac, 6),
            subspace_cosine_similarity=round(sub_cos, 6),
            orthogonal_cosine_similarity=round(orth_cos, 6),
        )
    elif subspace_name is not None:
        # Fetch PCA basis from SubspaceRegistry
        entry = SubspaceRegistry.get().fetch(subspace_name)
        if entry is None:
            return make_error(
                ToolError.VECTOR_NOT_FOUND,
                f"Subspace '{subspace_name}' not found.",
                "inject_residual",
            )
        basis_matrix, _sub_meta = entry  # [rank, hidden_dim]

        # Project donor onto PCA subspace
        donor_sub = basis_matrix.T @ (basis_matrix @ donor_np)
        # Get recipient's orthogonal component
        recip_sub = basis_matrix.T @ (basis_matrix @ recip_np)
        recip_orth = recip_np - recip_sub

        injection_np = donor_sub + recip_orth

        # Subspace analysis
        donor_norm_sq = float(np.dot(donor_np, donor_np)) + 1e-12
        recip_norm_sq = float(np.dot(recip_np, recip_np)) + 1e-12
        d_sub_frac = float(np.dot(donor_sub, donor_sub)) / donor_norm_sq
        r_sub_frac = float(np.dot(recip_sub, recip_sub)) / recip_norm_sq

        donor_orth = donor_np - donor_sub
        sub_cos = _cosine_sim(donor_sub, recip_sub)
        orth_cos = _cosine_sim(donor_orth, recip_orth)

        subspace_result = SubspaceAnalysis(
            subspace_dim=int(basis_matrix.shape[0]),
            tokens_used=[],
            donor_subspace_fraction=round(d_sub_frac, 6),
            recipient_subspace_fraction=round(r_sub_frac, 6),
            subspace_cosine_similarity=round(sub_cos, 6),
            orthogonal_cosine_similarity=round(orth_cos, 6),
            subspace_name=subspace_name,
        )
    else:
        injection_np = donor_np.copy()

    if patch_all_positions:
        # Full tensor [1, donor_seq, hidden_dim] — skip numpy round-trip
        injection_vec = donor_decomp["hidden_states"][effective_donor_layer]
    else:
        injection_vec = mx.array(injection_np.tolist())

    # -- Phase 5: Run injected forward pass --
    injected_hidden = _run_forward_with_injection(
        model, config, recip_ids, layer, r_pos, injection_vec, meta
    )
    mx.eval(injected_hidden)

    # -- Phase 6: Get logits for all three --
    donor_logits = _norm_project(final_norm, lm_head, donor_vec_final)
    recip_logits = _norm_project(final_norm, lm_head, recip_vec_final)
    inject_logit_pos = -1 if patch_all_positions else r_pos
    injected_vec_final = _extract_position(injected_hidden, inject_logit_pos)
    injected_logits = _norm_project(final_norm, lm_head, injected_vec_final)
    mx.eval(donor_logits, recip_logits, injected_logits)

    donor_logits_np = np.array(donor_logits.tolist(), dtype=np.float32)
    recip_logits_np = np.array(recip_logits.tolist(), dtype=np.float32)
    injected_logits_np = np.array(injected_logits.tolist(), dtype=np.float32)

    # -- Phase 7: Comparison metrics --
    d_top_k, d_top1_id, d_top1_tok, d_top1_prob = _top_k_from_logits(donor_logits_np, tokenizer)
    r_top_k, r_top1_id, r_top1_tok, r_top1_prob = _top_k_from_logits(recip_logits_np, tokenizer)
    i_top_k, i_top1_id, i_top1_tok, i_top1_prob = _top_k_from_logits(injected_logits_np, tokenizer)

    comparison = ComparisonMetrics(
        injected_matches_donor=(i_top1_id == d_top1_id),
        injected_matches_recipient=(i_top1_id == r_top1_id),
        donor_recipient_kl=round(_kl_divergence(donor_logits_np, recip_logits_np), 6),
        donor_injected_kl=round(_kl_divergence(donor_logits_np, injected_logits_np), 6),
        recipient_injected_kl=round(_kl_divergence(recip_logits_np, injected_logits_np), 6),
    )

    # -- Phase 8: Generate text --
    donor_text, _ = generate_text(model, tokenizer, donor_prompt, max_new_tokens, temperature)
    recip_text, _ = generate_text(model, tokenizer, recipient_prompt, max_new_tokens, temperature)
    if patch_all_positions:
        gen_ids = donor_ids
        gen_pos = -1
    else:
        gen_ids = recip_ids
        gen_pos = r_pos

    injected_text, _ = _generate_from_hidden(
        model,
        tokenizer,
        injected_hidden,
        final_norm,
        lm_head,
        gen_ids,
        gen_pos,
        max_new_tokens,
        temperature,
    )

    # -- Phase 9: Residual similarity --
    residual_sim = ResidualSimilarity(
        cosine_similarity=round(_cosine_sim(donor_np, recip_np), 6),
        angle=round(_angle_between(donor_np, recip_np), 4),
        donor_norm=round(float(np.linalg.norm(donor_np)), 4),
        recipient_norm=round(float(np.linalg.norm(recip_np)), 4),
    )

    # -- Phase 10: Summary --
    summary: dict[str, Any] = {
        "markov_holds": comparison.injected_matches_donor,
        "donor_injected_kl": comparison.donor_injected_kl,
        "donor_top1": d_top1_tok,
        "recipient_top1": r_top1_tok,
        "injected_top1": i_top1_tok,
        "residual_angle": residual_sim.angle,
    }
    if donor_layer is not None and donor_layer != layer:
        summary["donor_layer"] = effective_donor_layer
    if patch_all_positions:
        summary["donor_seq_len"] = num_donor
        summary["recipient_seq_len"] = num_recip
    if subspace_result:
        summary["subspace_dim"] = subspace_result.subspace_dim
        summary["donor_subspace_fraction"] = subspace_result.donor_subspace_fraction

    # -- Phase 11: Assemble result --
    return InjectResidualResult(
        donor_prompt=donor_prompt,
        recipient_prompt=recipient_prompt,
        layer=layer,
        donor_layer=donor_layer,
        donor_position=donor_position,
        recipient_position=recipient_position,
        subspace_only=subspace_only,
        patch_all_positions=patch_all_positions,
        donor_output=DonorRecipientOutput(
            text=donor_text,
            top_prediction=d_top1_tok,
            top_prediction_id=d_top1_id,
            probability=round(d_top1_prob, 6),
            top_k=d_top_k,
        ),
        recipient_output=DonorRecipientOutput(
            text=recip_text,
            top_prediction=r_top1_tok,
            top_prediction_id=r_top1_id,
            probability=round(r_top1_prob, 6),
            top_k=r_top_k,
        ),
        injected_output=InjectedOutput(
            text=injected_text,
            top_prediction=i_top1_tok,
            top_prediction_id=i_top1_id,
            probability=round(i_top1_prob, 6),
            top_k=i_top_k,
        ),
        comparison=comparison,
        residual_similarity=residual_sim,
        subspace_analysis=subspace_result,
        summary=summary,
    ).model_dump()
