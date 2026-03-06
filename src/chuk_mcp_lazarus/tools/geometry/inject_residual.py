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


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str
    token_id: int
    probability: float


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


class InjectResidualResult(BaseModel):
    """Result from inject_residual."""

    donor_prompt: str
    recipient_prompt: str
    layer: int
    donor_position: int
    recipient_position: int
    subspace_only: bool
    donor_output: DonorRecipientOutput
    recipient_output: DonorRecipientOutput
    injected_output: InjectedOutput
    comparison: ComparisonMetrics
    residual_similarity: ResidualSimilarity
    subspace_analysis: SubspaceAnalysis | None = None
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax in numpy."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) from logits, computed in numpy."""
    p = _softmax_np(p_logits)
    q = _softmax_np(q_logits)
    q = np.maximum(q, 1e-10)
    p = np.maximum(p, 1e-10)
    return float(np.sum(p * np.log(p / q)))


def _top_k_from_logits(
    logits_np: np.ndarray, tokenizer: Any, k: int = 10
) -> tuple[list[TokenPrediction], int, str, float]:
    """Extract top-k predictions from numpy logits.

    Returns (top_k_list, top1_id, top1_token, top1_prob).
    """
    probs = _softmax_np(logits_np)
    order = np.argsort(-probs)
    entries: list[TokenPrediction] = []
    for i in range(min(k, len(probs))):
        tid = int(order[i])
        entries.append(
            TokenPrediction(
                token=tokenizer.decode([tid]),
                token_id=tid,
                probability=round(float(probs[tid]), 6),
            )
        )
    top1_id = int(order[0])
    top1_token = tokenizer.decode([top1_id])
    top1_prob = float(probs[top1_id])
    return entries, top1_id, top1_token, top1_prob


# ---------------------------------------------------------------------------
# Forward pass with injection
# ---------------------------------------------------------------------------


def _run_forward_with_injection(
    model: Any,
    config: Any,
    input_ids: Any,
    target_layer: int,
    target_position: int,
    injection_vec: Any,
    metadata: Any,
) -> Any:
    """Run forward pass, replacing hidden state at target_layer/position.

    After layer target_layer completes normally, the hidden state at
    target_position is overwritten with injection_vec.  Remaining layers
    continue from this modified state.

    Returns the final hidden state [1, seq, hidden_dim].
    """
    import mlx.core as mx
    import mlx.nn as nn

    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    embed = helper._get_embed_tokens()
    embedding_scale = helper._get_embedding_scale()

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    h = embed(input_ids)
    if embedding_scale is not None:
        scale = mx.array(embedding_scale, dtype=h.dtype)
        h = h * scale

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    for layer_idx, layer_module in enumerate(model_layers):
        # Standard layer forward
        try:
            layer_out = layer_module(h, mask=mask, cache=None)
        except TypeError:
            try:
                layer_out = layer_module(h, cache=None)
            except TypeError:
                layer_out = layer_module(h)

        if hasattr(layer_out, "hidden_states"):
            h = layer_out.hidden_states
        elif isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out

        # After target layer, inject
        if layer_idx == target_layer:
            pos = target_position
            if pos < 0:
                pos = seq_len + pos
            pos = max(0, min(pos, seq_len - 1))
            before = h[:, :pos, :]
            after = h[:, pos + 1 :, :]
            injected = injection_vec.reshape(1, 1, -1)
            h = mx.concatenate([before, injected, after], axis=1)

    return h


# ---------------------------------------------------------------------------
# Generation from hidden state
# ---------------------------------------------------------------------------


def _generate_from_hidden(
    model: Any,
    tokenizer: Any,
    final_hidden: Any,
    final_norm: Any,
    lm_head: Any,
    input_ids: Any,
    position: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """Generate text starting from a modified hidden state.

    First token comes from the injected hidden state projected through
    norm + lm_head.  Subsequent tokens use normal model forward passes.
    """
    import mlx.core as mx

    from ..residual_tools import _extract_position, _norm_project

    # First token from injected hidden state
    vec = _extract_position(final_hidden, position)
    first_logits = _norm_project(final_norm, lm_head, vec)
    mx.eval(first_logits)

    if temperature <= 0:
        first_id = int(mx.argmax(first_logits).item())
    else:
        probs = mx.softmax(first_logits / temperature)
        first_id = int(mx.random.categorical(mx.log(probs)).item())

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if first_id == eos_token_id:
        return tokenizer.decode([first_id], skip_special_tokens=True), 1

    generated_ids = [first_id]

    # Subsequent tokens: normal model forward passes
    if input_ids.ndim == 1:
        flat_ids = input_ids
    else:
        flat_ids = input_ids.reshape(-1)
    current_ids = mx.concatenate([flat_ids, mx.array([first_id])])

    for _ in range(max_new_tokens - 1):
        logits = model(current_ids[None, :] if current_ids.ndim == 1 else current_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        if hasattr(logits, "logits"):
            logits = logits.logits
        next_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]

        if temperature <= 0:
            next_id = int(mx.argmax(next_logits).item())
        else:
            probs = mx.softmax(next_logits / temperature)
            next_id = int(mx.random.categorical(mx.log(probs)).item())

        if next_id == eos_token_id:
            break
        generated_ids.append(next_id)
        current_ids = mx.concatenate([current_ids.reshape(-1), mx.array([next_id])])

    return (
        tokenizer.decode(generated_ids, skip_special_tokens=True),
        len(generated_ids),
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def inject_residual(
    donor_prompt: str,
    recipient_prompt: str,
    layer: int,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    donor_position: int = -1,
    recipient_position: int = -1,
    subspace_only: bool = False,
    subspace_tokens: list[str] | None = None,
) -> dict:
    """Inject the residual stream from one prompt into another at a
    specific layer and continue generation.

    Runs the donor prompt through layers 0-L to capture its residual
    stream.  At layer L, replaces the recipient's residual with the
    donor's (full vector or subspace-only).  Then continues the
    recipient's forward pass from layer L+1 to the final layer and
    generates text.

    This tests the Markov property: if downstream layers only care
    about the current residual state (not the history), the output
    should match the donor's.

    Args:
        donor_prompt:       Prompt whose residual is captured.
        recipient_prompt:   Prompt that receives the injected residual.
        layer:              Layer at which to inject.
        max_new_tokens:     Tokens to generate after injection.
        temperature:        Sampling temperature (0 = greedy).
        donor_position:     Token position in donor (-1 = last).
        recipient_position: Token position in recipient (-1 = last).
        subspace_only:      Only inject the subspace component.
        subspace_tokens:    Tokens defining the injection subspace
                            (required if subspace_only=True).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "inject_residual",
        )
    meta = state.metadata
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
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
    if subspace_only and (not subspace_tokens or len(subspace_tokens) == 0):
        return make_error(
            ToolError.INVALID_INPUT,
            "subspace_tokens required when subspace_only=True.",
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
            max_new_tokens,
            temperature,
            donor_position,
            recipient_position,
            subspace_only,
            subspace_tokens,
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
    max_new_tokens: int,
    temperature: float,
    donor_position: int,
    recipient_position: int,
    subspace_only: bool,
    subspace_tokens: list[str] | None,
) -> dict:
    """Sync implementation of inject_residual."""
    import mlx.core as mx

    from ..residual_tools import (
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
    capture_layers = sorted(set([layer, last_layer]))

    # -- Phase 2: Capture donor state --
    donor_decomp = _run_decomposition_forward(model, config, donor_ids, capture_layers)
    mx.eval(*donor_decomp["hidden_states"].values())

    donor_vec_at_layer = _extract_position(donor_decomp["hidden_states"][layer], d_pos)
    donor_vec_final = _extract_position(donor_decomp["hidden_states"][last_layer], d_pos)
    donor_np = np.array(donor_vec_at_layer.tolist(), dtype=np.float32)

    # -- Phase 3: Capture recipient state --
    recip_decomp = _run_decomposition_forward(model, config, recip_ids, capture_layers)
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
    else:
        injection_np = donor_np.copy()

    injection_vec = mx.array(injection_np.tolist())

    # -- Phase 5: Run injected forward pass --
    injected_hidden = _run_forward_with_injection(
        model, config, recip_ids, layer, r_pos, injection_vec, meta
    )
    mx.eval(injected_hidden)

    # -- Phase 6: Get logits for all three --
    donor_logits = _norm_project(final_norm, lm_head, donor_vec_final)
    recip_logits = _norm_project(final_norm, lm_head, recip_vec_final)
    injected_vec_final = _extract_position(injected_hidden, r_pos)
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
    injected_text, _ = _generate_from_hidden(
        model,
        tokenizer,
        injected_hidden,
        final_norm,
        lm_head,
        recip_ids,
        r_pos,
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
    if subspace_result:
        summary["subspace_dim"] = subspace_result.subspace_dim
        summary["donor_subspace_fraction"] = subspace_result.donor_subspace_fraction

    # -- Phase 11: Assemble result --
    return InjectResidualResult(
        donor_prompt=donor_prompt,
        recipient_prompt=recipient_prompt,
        layer=layer,
        donor_position=donor_position,
        recipient_position=recipient_position,
        subspace_only=subspace_only,
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
