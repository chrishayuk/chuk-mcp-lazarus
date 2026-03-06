"""
computation_map — complete prediction flow in one tool call.

Runs a single decomposition forward pass and returns everything needed
to understand how the model processes a prompt: residual geometry,
component attribution, logit lens race, top heads, top neurons, and
candidate angles — structured for rendering as a network flow diagram.
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
    _auto_layers,
    _cosine_sim,
    _get_unembed_vec_np,
    _gram_schmidt,
    _resolve_token_to_id,
    _unit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class CandidateInfo(BaseModel):
    """Metadata about a candidate token."""

    token: str
    token_id: int
    unembed_norm: float


class CandidatePairAngle(BaseModel):
    """Pairwise angle between two candidate unembed vectors."""

    token_a: str
    token_b: str
    angle: float
    cosine_similarity: float


class CandidateAngle(BaseModel):
    """Angle from residual to one candidate direction at a layer."""

    token: str
    angle: float
    projection: float
    fraction: float


class LayerGeometry(BaseModel):
    """Residual geometry at one layer."""

    residual_norm: float
    angles: list[CandidateAngle]
    orthogonal_fraction: float
    rotation_from_previous: float | None = Field(
        None,
        description="Angle in degrees rotated from previous layer.",
    )


class CandidateAttribution(BaseModel):
    """Per-candidate logit delta from a component."""

    token: str
    attention_delta: float
    ffn_delta: float


class LayerAttribution(BaseModel):
    """Attribution of logit changes to attention vs FFN at a layer."""

    attention_logit: float = Field(..., description="Total logit change from attention.")
    ffn_logit: float = Field(..., description="Total logit change from FFN.")
    per_candidate: list[CandidateAttribution]


class CandidateRace(BaseModel):
    """Logit lens state for one candidate at one layer."""

    token: str
    logit: float
    probability: float
    rank: int


class TopHead(BaseModel):
    """A high-impact attention head."""

    head: int
    logit_contribution: float
    top_token: str


class TopNeuron(BaseModel):
    """A high-impact MLP neuron."""

    neuron_index: int
    activation: float
    logit_contribution: float
    top_token: str


class LayerTopComponents(BaseModel):
    """Top heads and neurons at a layer."""

    top_heads: list[TopHead]
    top_neurons: list[TopNeuron]


class ComputationMapLayer(BaseModel):
    """All data for one layer of the computation map."""

    layer: int
    geometry: LayerGeometry
    attribution: LayerAttribution
    race: list[CandidateRace]
    top_components: LayerTopComponents


class CrossingEvent(BaseModel):
    """A crossing where the leading candidate changed."""

    layer: int
    from_token: str
    to_token: str


class ComputationMapSummary(BaseModel):
    """Summary of the full computation map."""

    final_prediction: str
    final_probability: float
    total_rotation: float
    biggest_rotation_layer: int
    crossing_events: list[CrossingEvent]
    primary_target: str
    primary_target_id: int


class ComputationMapResult(BaseModel):
    """Result from computation_map."""

    prompt: str
    token_position: int
    token_text: str
    hidden_dim: int
    candidates: list[CandidateInfo]
    candidate_pairwise_angles: list[CandidatePairAngle]
    layers: list[ComputationMapLayer]
    summary: ComputationMapSummary


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True)
async def computation_map(
    prompt: str,
    candidates: list[str],
    layers: list[int] | None = None,
    top_k_heads: int = 3,
    top_k_neurons: int = 5,
    token_position: int = -1,
) -> dict:
    """Map the complete prediction flow for a prompt in one call.

    Runs a single decomposition forward pass and returns per-layer:
    residual geometry (angles to candidates), component attribution
    (attention vs FFN logit deltas), logit lens race (probabilities),
    top attention heads, and top MLP neurons.

    Structured for rendering as a network flow diagram.

    Args:
        prompt:         Input text.
        candidates:     Candidate tokens to track (e.g. ["Sydney", "Canberra"]).
        layers:         Layer indices to analyse (None = auto-select).
        top_k_heads:    Top attention heads per layer (default: 3).
        top_k_neurons:  Top MLP neurons per layer (default: 5).
        token_position: Token position to analyse (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "computation_map",
        )
    meta = state.metadata
    if not candidates:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 candidate token required.",
            "computation_map",
        )
    if len(candidates) > 10:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 10 candidate tokens.",
            "computation_map",
        )
    if layers is not None:
        for lyr in layers:
            if lyr < 0 or lyr >= meta.num_layers:
                return make_error(
                    ToolError.LAYER_OUT_OF_RANGE,
                    f"Layer {lyr} out of [0, {meta.num_layers - 1}].",
                    "computation_map",
                )
    try:
        return await asyncio.to_thread(
            _computation_map_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            candidates,
            layers,
            top_k_heads,
            top_k_neurons,
            token_position,
        )
    except Exception as exc:
        logger.exception("computation_map failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "computation_map")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_top_heads(
    target_layer: Any,
    prev_h: Any,
    pos: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: Any,
    unembed_mx: Any,
    lm_head: Any,
    tokenizer: Any,
    top_k: int,
) -> list[TopHead]:
    """Reconstruct per-head output and return top-k by |logit|."""
    import mlx.core as mx

    from .._serialize import to_pylist

    attn = target_layer.self_attn
    normed = target_layer.input_layernorm(prev_h)

    queries = attn.q_proj(normed)
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    batch_size = normed.shape[0]
    seq_len = normed.shape[1]

    queries = queries.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        queries = attn.q_norm(queries)
    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        keys = attn.k_norm(keys)

    if hasattr(attn, "rope") and attn.rope is not None:
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    attn_scale = getattr(attn, "scale", head_dim**-0.5)
    context = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn_scale,
        mask=mask,
    )
    mx.eval(context)

    o_weight = attn.o_proj.weight

    head_output_vecs = []
    for head_i in range(num_heads):
        head_ctx = context[0, head_i, pos, :]
        w_slice = o_weight[:, head_i * head_dim : (head_i + 1) * head_dim]
        head_output_vecs.append(head_ctx @ w_slice.T)

    mx.eval(*head_output_vecs)

    head_logits = [float((hv * unembed_mx).sum().item()) for hv in head_output_vecs]

    # Sort by absolute logit, take top_k
    ranked = sorted(range(num_heads), key=lambda i: abs(head_logits[i]), reverse=True)
    top_indices = ranked[:top_k]

    if top_indices:
        head_stack = mx.stack([head_output_vecs[i] for i in top_indices])
        all_logits = lm_head(head_stack.reshape(1, len(top_indices), -1))
        if hasattr(all_logits, "logits"):
            all_logits = all_logits.logits
        elif isinstance(all_logits, tuple):
            all_logits = all_logits[0]
        all_logits = all_logits[0]
        top_ids = mx.argmax(all_logits, axis=1)
        mx.eval(top_ids)
        top_ids_list = to_pylist(top_ids)
    else:
        top_ids_list = []

    results = []
    for rank, hi in enumerate(top_indices):
        top_tok = tokenizer.decode([top_ids_list[rank]])
        results.append(
            TopHead(
                head=hi,
                logit_contribution=round(head_logits[hi], 6),
                top_token=top_tok,
            )
        )
    return results


def _compute_top_neurons(
    target_layer: Any,
    prev_h: Any,
    attn_out: Any,
    pos: int,
    unembed_mx: Any,
    lm_head: Any,
    tokenizer: Any,
    top_k: int,
    four_norms: bool,
) -> list[TopNeuron]:
    """Reconstruct per-neuron FFN output and return top-k by |logit|."""
    import mlx.core as mx
    import mlx.nn as nn

    from .._serialize import to_pylist

    h_post_attn = prev_h + attn_out
    mlp = target_layer.mlp

    if four_norms:
        ffn_input = target_layer.pre_feedforward_layernorm(h_post_attn)
    else:
        ffn_input = target_layer.post_attention_layernorm(h_post_attn)

    has_gate = hasattr(mlp, "gate_proj")
    if has_gate:
        gate_out = mlp.gate_proj(ffn_input)
        up_out = mlp.up_proj(ffn_input)
        act_fn = getattr(mlp, "act", None) or getattr(mlp, "activation", None)
        if act_fn is not None:
            hidden_act = act_fn(gate_out) * up_out
        else:
            hidden_act = nn.silu(gate_out) * up_out
    else:
        up_out = mlp.up_proj(ffn_input)
        act_fn = getattr(mlp, "act", None) or getattr(mlp, "activation", None)
        if act_fn is not None:
            hidden_act = act_fn(up_out)
        else:
            hidden_act = nn.gelu(up_out)

    mx.eval(hidden_act)

    if hidden_act.ndim == 3:
        hidden_pos = hidden_act[0, pos, :]
    elif hidden_act.ndim == 2:
        hidden_pos = hidden_act[pos, :]
    else:
        hidden_pos = hidden_act

    down_weight = mlp.down_proj.weight
    neuron_projections = down_weight.T @ unembed_mx
    mx.eval(neuron_projections)

    neuron_logits = hidden_pos * neuron_projections
    mx.eval(neuron_logits)

    abs_logits = mx.abs(neuron_logits)
    sorted_indices = mx.argsort(abs_logits)[::-1]
    mx.eval(sorted_indices)
    sorted_list = to_pylist(sorted_indices)
    top_indices = sorted_list[:top_k]

    neuron_logits_list = to_pylist(neuron_logits)
    hidden_pos_list = to_pylist(hidden_pos)

    if top_indices:
        neuron_vecs = mx.stack([hidden_pos[i] * down_weight[:, i] for i in top_indices])
        batch_logits = lm_head(neuron_vecs.reshape(1, len(top_indices), -1))
        if hasattr(batch_logits, "logits"):
            batch_logits = batch_logits.logits
        elif isinstance(batch_logits, tuple):
            batch_logits = batch_logits[0]
        batch_logits = batch_logits[0]
        batch_top_ids = mx.argmax(batch_logits, axis=1)
        mx.eval(batch_top_ids)
        batch_top_ids_list = to_pylist(batch_top_ids)
    else:
        batch_top_ids_list = []

    results = []
    for rank, ni in enumerate(top_indices):
        top_tok = tokenizer.decode([batch_top_ids_list[rank]])
        results.append(
            TopNeuron(
                neuron_index=ni,
                activation=round(hidden_pos_list[ni], 6),
                logit_contribution=round(neuron_logits_list[ni], 6),
                top_token=top_tok,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _computation_map_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    candidate_tokens: list[str],
    layers: list[int] | None,
    top_k_heads: int,
    top_k_neurons: int,
    token_position: int,
) -> dict:
    """Sync implementation of computation_map."""
    import mlx.core as mx
    import mlx.nn as nn

    from ..residual_tools import (
        _extract_position,
        _get_lm_projection,
        _get_unembed_vector,
        _has_four_norms,
        _norm_project,
        _run_decomposition_forward,
    )

    # -- Phase 1: Setup --
    tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array(tok_ids)
    num_tokens = input_ids.shape[-1]
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    tok_text = tokenizer.decode([tok_ids[pos]])

    if layers is None:
        layers = _auto_layers(meta.num_layers)
    layers = sorted(layers)

    # Resolve candidates
    cand_ids: list[int] = []
    cand_vecs_np: list[np.ndarray] = []
    cand_vecs_mx: list[Any] = []
    cand_info: list[CandidateInfo] = []

    for tok in candidate_tokens:
        tid = _resolve_token_to_id(tokenizer, tok)
        if tid is None:
            return make_error(
                ToolError.INVALID_INPUT,
                f"Cannot encode candidate '{tok}'.",
                "computation_map",
            )
        vec_np = _get_unembed_vec_np(model, tid)
        if vec_np is None:
            return make_error(
                ToolError.INVALID_INPUT,
                f"No unembed vector for '{tok}'.",
                "computation_map",
            )
        vec_mx = _get_unembed_vector(model, tid)
        if vec_mx is None:
            return make_error(
                ToolError.INVALID_INPUT,
                f"No unembed vector (mx) for '{tok}'.",
                "computation_map",
            )
        cand_ids.append(tid)
        cand_vecs_np.append(vec_np)
        cand_vecs_mx.append(vec_mx)
        cand_info.append(
            CandidateInfo(
                token=tok,
                token_id=tid,
                unembed_norm=round(float(np.linalg.norm(vec_np)), 4),
            )
        )

    # Pairwise angles between candidates
    pairwise_angles: list[CandidatePairAngle] = []
    for i in range(len(cand_vecs_np)):
        for j in range(i + 1, len(cand_vecs_np)):
            cs = _cosine_sim(cand_vecs_np[i], cand_vecs_np[j])
            ang = _angle_between(cand_vecs_np[i], cand_vecs_np[j])
            pairwise_angles.append(
                CandidatePairAngle(
                    token_a=candidate_tokens[i],
                    token_b=candidate_tokens[j],
                    angle=round(ang, 4),
                    cosine_similarity=round(cs, 6),
                )
            )

    # Orthonormalize candidate directions for subspace fractions
    cand_ortho = _gram_schmidt(cand_vecs_np)

    # Model components
    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not access lm_head.",
            "computation_map",
        )

    num_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_heads
    head_dim = meta.head_dim or (meta.hidden_dim // num_heads)

    # -- Phase 2: Decomposition forward --
    decomp = _run_decomposition_forward(model, config, input_ids, layers)

    # Materialize everything
    to_eval = []
    for d in (decomp["hidden_states"], decomp["prev_hidden"]):
        to_eval.extend(d.values())
    for v in decomp["attn_outputs"].values():
        if v is not None:
            to_eval.append(v)
    for v in decomp["ffn_outputs"].values():
        if v is not None:
            to_eval.append(v)
    to_eval.append(decomp["embeddings"])
    mx.eval(*to_eval)

    # Causal mask for head reconstruction
    seq_len = len(tok_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(decomp["embeddings"].dtype)

    # -- Phase 3-7: Per-layer computation --
    map_layers: list[ComputationMapLayer] = []
    prev_residual_np: np.ndarray | None = None
    total_rotation = 0.0
    max_rotation = 0.0
    max_rotation_layer = layers[0] if layers else 0
    crossings: list[CrossingEvent] = []
    prev_leader = ""

    # Use first candidate as primary target for head/neuron attribution
    primary_unembed_mx = cand_vecs_mx[0]
    mx.eval(primary_unembed_mx)

    for lyr in layers:
        hidden = decomp["hidden_states"].get(lyr)
        if hidden is None:
            continue

        h_vec = _extract_position(hidden, pos)

        # -- Phase 3: Race state --
        logits = _norm_project(final_norm, lm_head, h_vec)
        mx.eval(logits)

        # Per-candidate race: extract logits then softmax via numpy
        race_entries: list[CandidateRace] = []
        cand_logit_vals: list[float] = []
        cand_logits_raw = [float(logits[tid].item()) for tid in cand_ids]
        # Softmax over candidates for probability estimates
        max_logit = max(cand_logits_raw) if cand_logits_raw else 0.0
        exp_vals = [np.exp(v - max_logit) for v in cand_logits_raw]
        sum_exp = sum(exp_vals)
        cand_probs = [e / sum_exp if sum_exp > 0 else 0.0 for e in exp_vals]

        for ci, (tok, tid) in enumerate(zip(candidate_tokens, cand_ids)):
            logit_val = cand_logits_raw[ci]
            prob_val = cand_probs[ci]
            cand_logit_vals.append(logit_val)
            race_entries.append(
                CandidateRace(
                    token=tok,
                    logit=round(logit_val, 4),
                    probability=round(prob_val, 6),
                    rank=0,  # filled below
                )
            )

        # Compute ranks
        sorted_cands = sorted(
            range(len(race_entries)), key=lambda i: cand_logit_vals[i], reverse=True
        )
        for rank, ci in enumerate(sorted_cands):
            race_entries[ci].rank = rank + 1

        # Detect crossing
        leader = race_entries[sorted_cands[0]].token
        if prev_leader and leader != prev_leader:
            crossings.append(CrossingEvent(layer=lyr, from_token=prev_leader, to_token=leader))
        prev_leader = leader

        # -- Phase 4: Geometry --
        h_np = np.array(h_vec.tolist(), dtype=np.float32)
        res_norm = float(np.linalg.norm(h_np))

        cand_angles: list[CandidateAngle] = []
        for tok, cvec in zip(candidate_tokens, cand_vecs_np):
            ang = _angle_between(h_np, cvec)
            c_unit = _unit(cvec)
            proj = float(np.dot(h_np, c_unit))
            frac = (proj**2) / (res_norm**2 + 1e-12)
            cand_angles.append(
                CandidateAngle(
                    token=tok,
                    angle=round(ang, 4),
                    projection=round(proj, 4),
                    fraction=round(frac, 6),
                )
            )

        # Orthogonal fraction
        in_subspace = 0.0
        for bv in cand_ortho:
            p = float(np.dot(h_np, bv))
            in_subspace += p**2
        ortho_frac = max(0.0, 1.0 - in_subspace / (res_norm**2 + 1e-12))

        # Rotation
        rotation: float | None = None
        if prev_residual_np is not None:
            rotation = _angle_between(h_np, prev_residual_np)
            total_rotation += rotation
            if rotation > max_rotation:
                max_rotation = rotation
                max_rotation_layer = lyr

        prev_residual_np = h_np

        geometry = LayerGeometry(
            residual_norm=round(res_norm, 4),
            angles=cand_angles,
            orthogonal_fraction=round(ortho_frac, 6),
            rotation_from_previous=round(rotation, 4) if rotation is not None else None,
        )

        # -- Phase 5: Attribution --
        prev_h = decomp["prev_hidden"].get(lyr)
        attn_out = decomp["attn_outputs"].get(lyr)
        ffn_out = decomp["ffn_outputs"].get(lyr)

        per_cand_attr: list[CandidateAttribution] = []
        total_attn_logit = 0.0
        total_ffn_logit = 0.0

        if prev_h is not None and attn_out is not None and ffn_out is not None:
            prev_vec = _extract_position(prev_h, pos)
            attn_vec = _extract_position(attn_out, pos)
            ffn_vec = _extract_position(ffn_out, pos)

            # Normalized attribution: logit at checkpoint
            pre_logits = _norm_project(final_norm, lm_head, prev_vec)
            post_attn_logits = _norm_project(final_norm, lm_head, prev_vec + attn_vec)
            post_ffn_logits = _norm_project(final_norm, lm_head, prev_vec + attn_vec + ffn_vec)
            mx.eval(pre_logits, post_attn_logits, post_ffn_logits)

            for tok, tid in zip(candidate_tokens, cand_ids):
                pre_l = float(pre_logits[tid].item())
                post_a_l = float(post_attn_logits[tid].item())
                post_f_l = float(post_ffn_logits[tid].item())
                a_delta = post_a_l - pre_l
                f_delta = post_f_l - post_a_l
                per_cand_attr.append(
                    CandidateAttribution(
                        token=tok,
                        attention_delta=round(a_delta, 4),
                        ffn_delta=round(f_delta, 4),
                    )
                )
                total_attn_logit += a_delta
                total_ffn_logit += f_delta

            if len(candidate_tokens) > 0:
                total_attn_logit /= len(candidate_tokens)
                total_ffn_logit /= len(candidate_tokens)
        else:
            for tok in candidate_tokens:
                per_cand_attr.append(
                    CandidateAttribution(
                        token=tok,
                        attention_delta=0.0,
                        ffn_delta=0.0,
                    )
                )

        attribution = LayerAttribution(
            attention_logit=round(total_attn_logit, 4),
            ffn_logit=round(total_ffn_logit, 4),
            per_candidate=per_cand_attr,
        )

        # -- Phase 6: Top heads --
        top_heads: list[TopHead] = []
        if prev_h is not None and lyr < len(model_layers):
            try:
                top_heads = _compute_top_heads(
                    model_layers[lyr],
                    prev_h,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    mask,
                    primary_unembed_mx,
                    lm_head,
                    tokenizer,
                    top_k_heads,
                )
            except Exception:
                logger.debug("Head reconstruction failed at layer %d", lyr)

        # -- Phase 7: Top neurons --
        top_neurons: list[TopNeuron] = []
        if prev_h is not None and attn_out is not None and lyr < len(model_layers):
            try:
                four_norms = _has_four_norms(model_layers[lyr])
                top_neurons = _compute_top_neurons(
                    model_layers[lyr],
                    prev_h,
                    attn_out,
                    pos,
                    primary_unembed_mx,
                    lm_head,
                    tokenizer,
                    top_k_neurons,
                    four_norms,
                )
            except Exception:
                logger.debug("Neuron reconstruction failed at layer %d", lyr)

        map_layers.append(
            ComputationMapLayer(
                layer=lyr,
                geometry=geometry,
                attribution=attribution,
                race=race_entries,
                top_components=LayerTopComponents(
                    top_heads=top_heads,
                    top_neurons=top_neurons,
                ),
            )
        )

    # -- Summary --
    final_race = map_layers[-1].race if map_layers else []
    if final_race:
        winner = min(final_race, key=lambda r: r.rank)
        final_prediction = winner.token
        final_probability = winner.probability
    else:
        final_prediction = ""
        final_probability = 0.0

    summary = ComputationMapSummary(
        final_prediction=final_prediction,
        final_probability=final_probability,
        total_rotation=round(total_rotation, 4),
        biggest_rotation_layer=max_rotation_layer,
        crossing_events=crossings,
        primary_target=candidate_tokens[0],
        primary_target_id=cand_ids[0],
    )

    return ComputationMapResult(
        prompt=prompt,
        token_position=token_position,
        token_text=tok_text,
        hidden_dim=meta.hidden_dim,
        candidates=cand_info,
        candidate_pairwise_angles=pairwise_angles,
        layers=map_layers,
        summary=summary,
    ).model_dump()
