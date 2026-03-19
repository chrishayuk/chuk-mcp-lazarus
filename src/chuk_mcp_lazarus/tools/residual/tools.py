"""
Residual stream tools: decomposition, clustering, logit attribution,
head attribution, neuron identification.

residual_decomposition decomposes each layer's contribution to the
residual stream into attention vs MLP components.  layer_clustering
computes representation similarity across prompts at each layer to
reveal where prompts converge or diverge.  logit_attribution uses
Direct Logit Attribution (DLA) to measure each component's
contribution to the predicted token's logit.  head_attribution
drills into a single layer's attention per head.  top_neurons
identifies the MLP neurons that contribute most.

All tools build directly on ModelHooks to capture intermediate
hidden states.
"""

import asyncio
import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, Field

from ..._serialize import cosine_similarity_matrix, hidden_state_to_list, to_pylist
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ..._residual_helpers import (  # noqa: F401
    _compute_clustering_scores,
    _extract_position,
    _get_embed_weight,
    _get_lm_projection,
    _get_unembed_vector,
    _has_four_norms,
    _has_sublayers,
    _l2_norm,
    _norm_project,
    _project_to_logits,
    _resolve_target_token,
    _run_decomposition_forward,
    _token_text,
    _tokenize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class LayerContribution(BaseModel):
    """Attention vs MLP contribution at a single layer."""

    layer: int
    total_norm: float = Field(..., description="L2 norm of the total residual delta at this layer.")
    attention_norm: float = Field(..., description="L2 norm of the attention sub-layer output.")
    ffn_norm: float = Field(..., description="L2 norm of the FFN/MLP sub-layer output.")
    attention_fraction: float = Field(
        ..., description="Attention's share of (attn_norm + ffn_norm). Range [0, 1]."
    )
    ffn_fraction: float = Field(
        ..., description="FFN's share of (attn_norm + ffn_norm). Range [0, 1]."
    )
    dominant_component: str = Field(
        ..., description="'attention' or 'ffn' -- whichever has larger norm."
    )


class ResidualDecompositionResult(BaseModel):
    """Result from residual_decomposition."""

    prompt: str
    token_position: int
    token_text: str
    num_tokens: int
    layers: list[LayerContribution]
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Summary statistics: attention_dominant_count, ffn_dominant_count, "
            "peak_layer (largest total_norm), peak_component."
        ),
    )


class LayerSimilarity(BaseModel):
    """Cosine similarity matrix and clustering scores at one layer."""

    layer: int
    similarity_matrix: list[list[float]] = Field(
        ..., description="Pairwise cosine similarity [n_prompts x n_prompts]."
    )
    mean_similarity: float = Field(..., description="Average off-diagonal similarity.")
    within_cluster_similarity: dict[str, float] | None = Field(
        None, description="Average similarity within each label group."
    )
    between_cluster_similarity: dict[str, float] | None = Field(
        None, description="Average similarity between label group pairs."
    )
    separation_score: float | None = Field(
        None,
        description=(
            "within_avg - between_avg. Positive = good separation. "
            "Only present when labels are provided."
        ),
    )


class LayerClusteringResult(BaseModel):
    """Result from layer_clustering."""

    prompts: list[str]
    labels: list[str] | None
    token_position: int
    num_layers_analyzed: int
    layers: list[LayerSimilarity]
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Summary: most_similar_layer, least_similar_layer, "
            "best_separation_layer (if labels), separation_trend."
        ),
    )


class LayerAttribution(BaseModel):
    """Per-layer logit attribution: how much each component pushes toward the target token."""

    layer: int
    attention_logit: float = Field(
        ..., description="Attention sub-layer's contribution to the target token logit."
    )
    ffn_logit: float = Field(
        ..., description="FFN/MLP sub-layer's contribution to the target token logit."
    )
    total_logit: float = Field(..., description="attention_logit + ffn_logit.")
    cumulative_logit: float = Field(
        ..., description="Running sum of all contributions up to and including this layer."
    )
    attention_top_token: str = Field(
        ...,
        description=(
            "Normalized mode: model's top prediction after attention (logit lens). "
            "Raw DLA mode: token the attention output vector points toward."
        ),
    )
    ffn_top_token: str = Field(
        ...,
        description=(
            "Normalized mode: model's top prediction after FFN (logit lens). "
            "Raw DLA mode: token the FFN output vector points toward."
        ),
    )


class LogitAttributionResult(BaseModel):
    """Result from logit_attribution."""

    prompt: str
    token_position: int
    token_text: str
    target_token: str
    target_token_id: int
    model_logit: float = Field(
        ..., description="Actual model logit for the target token (with final norm)."
    )
    model_probability: float = Field(..., description="Softmax probability of the target token.")
    embedding_logit: float = Field(
        ..., description="Embedding's direct contribution to the target token logit."
    )
    layers: list[LayerAttribution]
    attribution_sum: float = Field(
        ..., description="embedding_logit + sum of all layer total_logits."
    )
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Summary: top_positive_layer, top_negative_layer, "
            "top_positive_component, embedding_fraction."
        ),
    )


class HeadContribution(BaseModel):
    """Per-head logit contribution at a single layer."""

    head: int
    logit_contribution: float = Field(
        ..., description="This head's DLA contribution to the target token logit."
    )
    fraction_of_layer: float = Field(
        ..., description="This head's share of the layer total. Can be negative."
    )
    top_token: str = Field(..., description="Token this head's output vector pushes toward most.")


class HeadAttributionResult(BaseModel):
    """Result from head_attribution."""

    prompt: str
    layer: int
    token_position: int
    token_text: str
    target_token: str
    target_token_id: int
    num_heads: int
    heads: list[HeadContribution]
    layer_total_logit: float = Field(..., description="Sum of per-head logit contributions.")
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Summary: top_positive_head, top_negative_head, "
            "positive_head_count, negative_head_count, "
            "concentration (abs share of top-1 head)."
        ),
    )


class NeuronContribution(BaseModel):
    """A single neuron's contribution to the target token logit."""

    neuron_index: int
    activation: float = Field(
        ..., description="Post-gating activation value (SwiGLU: silu(gate)*up)."
    )
    logit_contribution: float = Field(
        ..., description="activation * dot(down_proj_column, unembed_vector)."
    )
    top_token: str = Field(..., description="Token this neuron pushes toward most strongly.")


class TopNeuronsResult(BaseModel):
    """Result from top_neurons."""

    prompt: str
    layer: int
    token_position: int
    token_text: str
    target_token: str
    target_token_id: int
    mlp_type: str = Field(..., description="Detected MLP type: 'swiglu' or 'standard'.")
    intermediate_size: int = Field(
        ..., description="MLP intermediate dimension (number of neurons)."
    )
    top_k: int
    top_positive: list[NeuronContribution] = Field(
        ..., description="Top-k neurons pushing TOWARD the target token."
    )
    top_negative: list[NeuronContribution] = Field(
        ..., description="Top-k neurons pushing AWAY from the target token."
    )
    total_neuron_logit: float = Field(..., description="Sum of all neuron contributions (raw DLA).")
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Summary: top_neuron_index, top_neuron_logit, "
            "positive_neuron_count, negative_neuron_count, "
            "concentration (abs share from top-10), "
            "sparsity (fraction with |contribution| < 1e-4)."
        ),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def residual_decomposition(
    prompt: str,
    layers: list[int] | None = None,
    position: int = -1,
) -> dict:
    """
    Decompose each layer's contribution to the residual stream into
    attention vs MLP components.

    For each requested layer, computes the L2 norm of the attention
    sub-layer output and the FFN/MLP sub-layer output.  Shows which
    component dominates at each layer depth.

    This answers: "Is this layer primarily routing information
    (attention) or retrieving knowledge (MLP)?"

    Args:
        prompt:   Input text.
        layers:   Layer indices to decompose.  None = all layers.
        position: Token position to analyze (-1 = last token).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "residual_decomposition",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        layers = list(range(num_layers))

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "residual_decomposition",
        )

    try:
        input_ids = _tokenize(state.tokenizer, prompt)
        num_tokens = input_ids.shape[-1]
        tok_text = _token_text(state.tokenizer, input_ids, position)

        captured = _run_decomposition_forward(
            state.model,
            state.config,
            input_ids,
            layers,
        )

        # Materialize lazy MLX arrays
        to_eval = [captured["embeddings"]]
        to_eval += list(captured["hidden_states"].values())
        to_eval += list(captured["prev_hidden"].values())
        to_eval += [v for v in captured["attn_outputs"].values() if v is not None]
        to_eval += [v for v in captured["ffn_outputs"].values() if v is not None]
        mx.eval(*to_eval)

        contributions: list[LayerContribution] = []

        for layer_idx in sorted(layers):
            curr_h = captured["hidden_states"].get(layer_idx)
            prev_h = captured["prev_hidden"].get(layer_idx)
            if curr_h is None or prev_h is None:
                continue

            curr_vec = _extract_position(curr_h, position)
            prev_vec = _extract_position(prev_h, position)

            delta = curr_vec - prev_vec
            total_norm = _l2_norm(delta)

            attn_out = captured["attn_outputs"].get(layer_idx)
            ffn_out = captured["ffn_outputs"].get(layer_idx)

            if attn_out is not None and ffn_out is not None:
                attn_vec = _extract_position(attn_out, position)
                ffn_vec = _extract_position(ffn_out, position)
                attn_norm = _l2_norm(attn_vec)
                ffn_norm = _l2_norm(ffn_vec)
            else:
                # Non-standard block: cannot decompose
                attn_norm = total_norm / 2.0
                ffn_norm = total_norm / 2.0

            denom = attn_norm + ffn_norm
            if denom > 0:
                attn_frac = attn_norm / denom
                ffn_frac = ffn_norm / denom
            else:
                attn_frac = 0.5
                ffn_frac = 0.5

            dominant = "attention" if attn_frac > ffn_frac else "ffn"

            contributions.append(
                LayerContribution(
                    layer=layer_idx,
                    total_norm=round(total_norm, 6),
                    attention_norm=round(attn_norm, 6),
                    ffn_norm=round(ffn_norm, 6),
                    attention_fraction=round(attn_frac, 6),
                    ffn_fraction=round(ffn_frac, 6),
                    dominant_component=dominant,
                )
            )

        # Summary
        attn_dom = sum(1 for c in contributions if c.dominant_component == "attention")
        ffn_dom = sum(1 for c in contributions if c.dominant_component == "ffn")
        peak = max(contributions, key=lambda c: c.total_norm) if contributions else None

        summary = {
            "attention_dominant_count": attn_dom,
            "ffn_dominant_count": ffn_dom,
            "peak_layer": peak.layer if peak else -1,
            "peak_total_norm": peak.total_norm if peak else 0.0,
            "peak_component": peak.dominant_component if peak else "none",
        }

        result = ResidualDecompositionResult(
            prompt=prompt,
            token_position=position,
            token_text=tok_text,
            num_tokens=int(num_tokens),
            layers=contributions,
            summary=summary,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("residual_decomposition failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "residual_decomposition")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def layer_clustering(
    prompts: list[str],
    layers: list[int] | None = None,
    labels: list[str] | None = None,
    position: int = -1,
) -> dict:
    """
    Compute representation similarity across prompts at each layer.
    Shows where prompts converge or diverge in representation space.

    For each layer, builds a cosine similarity matrix between all
    prompt pairs.  If labels are provided, computes within-cluster
    vs between-cluster similarity and a separation score.

    This answers: "At which layer do semantically similar prompts
    converge?  Where do different categories separate?"

    Args:
        prompts:  2-8 input strings to compare.
        layers:   Layers to analyze.  None = key layers
                  (0, n//4, n//2, 3n//4, n-1).
        labels:   Optional category label per prompt (same length as
                  prompts).  Enables cluster separation scoring.
        position: Token position to extract (-1 = last token).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "layer_clustering",
        )

    if not 2 <= len(prompts) <= 8:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need 2-8 prompts, got {len(prompts)}.",
            "layer_clustering",
        )

    if labels is not None and len(labels) != len(prompts):
        return make_error(
            ToolError.INVALID_INPUT,
            f"labels length ({len(labels)}) must match prompts length ({len(prompts)}).",
            "layer_clustering",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        n = num_layers
        layers = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "layer_clustering",
        )

    try:
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

        # Collect hidden states: layer -> list of vectors (one per prompt)
        prompt_vectors: dict[int, list[list[float]]] = {lay: [] for lay in layers}

        for prompt in prompts:
            input_ids = _tokenize(state.tokenizer, prompt)

            hooks = ModelHooks(state.model, model_config=state.config)
            hooks.configure(
                CaptureConfig(
                    layers=layers,
                    capture_hidden_states=True,
                )
            )
            hooks.forward(input_ids)
            mx.eval(hooks.state.hidden_states)

            for layer_idx in layers:
                vec = hidden_state_to_list(
                    hooks.state.hidden_states[layer_idx],
                    position=position,
                )
                prompt_vectors[layer_idx].append(vec)

        # Build per-layer results
        layer_results: list[LayerSimilarity] = []
        for layer_idx in sorted(layers):
            vectors = prompt_vectors[layer_idx]
            sim_matrix = cosine_similarity_matrix(vectors)

            # Mean off-diagonal similarity
            n = len(prompts)
            off_diag = []
            for i in range(n):
                for j in range(i + 1, n):
                    off_diag.append(sim_matrix[i][j])
            mean_sim = sum(off_diag) / len(off_diag) if off_diag else 0.0

            # Clustering scores (if labels)
            within_sim = None
            between_sim = None
            separation = None
            if labels is not None:
                within_sim, between_sim, separation = _compute_clustering_scores(
                    labels,
                    sim_matrix,
                )

            layer_results.append(
                LayerSimilarity(
                    layer=layer_idx,
                    similarity_matrix=[[round(v, 6) for v in row] for row in sim_matrix],
                    mean_similarity=round(mean_sim, 6),
                    within_cluster_similarity=within_sim,
                    between_cluster_similarity=between_sim,
                    separation_score=separation,
                )
            )

        # Summary
        most_similar = max(layer_results, key=lambda r: r.mean_similarity)
        least_similar = min(layer_results, key=lambda r: r.mean_similarity)

        summary: dict[str, Any] = {
            "most_similar_layer": most_similar.layer,
            "most_similar_value": most_similar.mean_similarity,
            "least_similar_layer": least_similar.layer,
            "least_similar_value": least_similar.mean_similarity,
        }

        if labels is not None:
            best_sep = max(
                (r for r in layer_results if r.separation_score is not None),
                key=lambda r: r.separation_score or 0.0,
                default=None,
            )
            if best_sep is not None:
                summary["best_separation_layer"] = best_sep.layer
                summary["best_separation_score"] = best_sep.separation_score

            summary["separation_trend"] = [
                {"layer": r.layer, "separation": r.separation_score}
                for r in layer_results
                if r.separation_score is not None
            ]

        result = LayerClusteringResult(
            prompts=prompts,
            labels=labels,
            token_position=position,
            num_layers_analyzed=len(layer_results),
            layers=layer_results,
            summary=summary,
        )
        return result.model_dump(exclude_none=True)

    except Exception as e:
        logger.exception("layer_clustering failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "layer_clustering")


def _logit_attribution_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    position: int,
    target_token: str | None,
    normalized: bool,
    num_layers: int,
) -> dict:
    """Sync implementation of logit_attribution.

    Runs a decomposition forward pass and computes per-layer attention/FFN
    contributions to the target token's logit.  Called via asyncio.to_thread
    from the async tool wrapper.  Also reused by attribution_sweep.

    Raises ValueError on invalid target_token or missing lm_head.
    """
    from chuk_lazarus.introspection.hooks import ModelHooks

    input_ids = _tokenize(tokenizer, prompt)
    tok_text = _token_text(tokenizer, input_ids, position)

    last_layer = num_layers - 1
    forward_layers = sorted(set(layers) | {last_layer})

    captured = _run_decomposition_forward(model, config, input_ids, forward_layers)

    lm_head = _get_lm_projection(model)
    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()

    if lm_head is None:
        raise ValueError("Could not access the language model head for this model.")

    # Materialize lazy MLX arrays
    to_eval = [captured["embeddings"]]
    to_eval += list(captured["hidden_states"].values())
    to_eval += list(captured["prev_hidden"].values())
    to_eval += [v for v in captured["attn_outputs"].values() if v is not None]
    to_eval += [v for v in captured["ffn_outputs"].values() if v is not None]
    mx.eval(*to_eval)

    # Compute actual model prediction (with final norm)
    final_hidden = captured["hidden_states"][last_layer]
    final_vec = _extract_position(final_hidden, position)

    full_logits = _norm_project(final_norm, lm_head, final_vec)
    mx.eval(full_logits)

    probs = mx.softmax(full_logits, axis=-1)
    mx.eval(probs)

    # Resolve target token (raises ValueError on bad token)
    target_id, target_text = _resolve_target_token(tokenizer, full_logits, target_token)

    model_logit = float(full_logits[target_id].item())
    model_prob = float(probs[target_id].item())

    # Embedding contribution
    embed_vec = _extract_position(captured["embeddings"], position)

    if normalized:
        embed_logits = _norm_project(final_norm, lm_head, embed_vec)
    else:
        embed_logits = _project_to_logits(lm_head, embed_vec)
    mx.eval(embed_logits)
    embedding_logit = float(embed_logits[target_id].item())

    # Per-layer attribution
    cumulative = embedding_logit
    attributions: list[LayerAttribution] = []

    for layer_idx in sorted(layers):
        attn_out = captured["attn_outputs"].get(layer_idx)
        ffn_out = captured["ffn_outputs"].get(layer_idx)
        prev_h = captured["prev_hidden"].get(layer_idx)
        post_h = captured["hidden_states"].get(layer_idx)

        if normalized and prev_h is not None and attn_out is not None and ffn_out is not None:
            # ── Normalized mode: logit deltas through norm ──
            prev_vec = _extract_position(prev_h, position)
            attn_vec = _extract_position(attn_out, position)
            post_vec = _extract_position(post_h, position)
            post_attn_vec = prev_vec + attn_vec  # residual after attention

            logits_pre = _norm_project(final_norm, lm_head, prev_vec)
            logits_post_attn = _norm_project(final_norm, lm_head, post_attn_vec)
            logits_post = _norm_project(final_norm, lm_head, post_vec)
            mx.eval(logits_pre, logits_post_attn, logits_post)

            attn_contribution = float(logits_post_attn[target_id].item()) - float(
                logits_pre[target_id].item()
            )
            ffn_contribution = float(logits_post[target_id].item()) - float(
                logits_post_attn[target_id].item()
            )

            # Top tokens: logit lens prediction at each checkpoint
            attn_top_id = int(mx.argmax(logits_post_attn).item())
            ffn_top_id = int(mx.argmax(logits_post).item())
            attn_top_token = tokenizer.decode([attn_top_id])
            ffn_top_token = tokenizer.decode([ffn_top_id])

            # Cumulative = actual logit lens at this layer
            cumul = float(logits_post[target_id].item())

        elif attn_out is not None and ffn_out is not None:
            # ── Raw DLA mode ──
            attn_vec = _extract_position(attn_out, position)
            attn_logits = _project_to_logits(lm_head, attn_vec)
            mx.eval(attn_logits)
            attn_contribution = float(attn_logits[target_id].item())
            attn_top_id = int(mx.argmax(attn_logits).item())
            attn_top_token = tokenizer.decode([attn_top_id])

            ffn_vec = _extract_position(ffn_out, position)
            ffn_logits = _project_to_logits(lm_head, ffn_vec)
            mx.eval(ffn_logits)
            ffn_contribution = float(ffn_logits[target_id].item())
            ffn_top_id = int(mx.argmax(ffn_logits).item())
            ffn_top_token = tokenizer.decode([ffn_top_id])

            cumulative += attn_contribution + ffn_contribution
            cumul = cumulative

        else:
            # Non-decomposable block
            attn_contribution = 0.0
            ffn_contribution = 0.0
            attn_top_token = "?"
            ffn_top_token = "?"
            cumul = cumulative

        total = attn_contribution + ffn_contribution

        attributions.append(
            LayerAttribution(
                layer=layer_idx,
                attention_logit=round(attn_contribution, 6),
                ffn_logit=round(ffn_contribution, 6),
                total_logit=round(total, 6),
                cumulative_logit=round(cumul, 6),
                attention_top_token=attn_top_token,
                ffn_top_token=ffn_top_token,
            )
        )

    # attribution_sum: last cumulative value
    attribution_sum = (
        round(attributions[-1].cumulative_logit, 6) if attributions else round(embedding_logit, 6)
    )

    # Summary
    top_pos: LayerAttribution | None = None
    top_neg: LayerAttribution | None = None
    total_attn = 0.0
    total_ffn = 0.0
    if attributions:
        top_pos = max(attributions, key=lambda a: a.total_logit)
        top_neg = min(attributions, key=lambda a: a.total_logit)
        total_attn = sum(a.attention_logit for a in attributions)
        total_ffn = sum(a.ffn_logit for a in attributions)

    summary: dict[str, Any] = {
        "mode": "normalized" if normalized else "raw_dla",
        "top_positive_layer": top_pos.layer if top_pos else -1,
        "top_positive_logit": top_pos.total_logit if top_pos else 0.0,
        "top_negative_layer": top_neg.layer if top_neg else -1,
        "top_negative_logit": top_neg.total_logit if top_neg else 0.0,
        "total_attention_logit": round(total_attn, 6),
        "total_ffn_logit": round(total_ffn, 6),
        "dominant_component": "attention" if total_attn > total_ffn else "ffn",
        "embedding_logit": round(embedding_logit, 6),
    }

    result = LogitAttributionResult(
        prompt=prompt,
        token_position=position,
        token_text=tok_text,
        target_token=target_text,
        target_token_id=target_id,
        model_logit=round(model_logit, 6),
        model_probability=round(model_prob, 6),
        embedding_logit=round(embedding_logit, 6),
        layers=attributions,
        attribution_sum=attribution_sum,
        summary=summary,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def logit_attribution(
    prompt: str,
    layers: list[int] | None = None,
    position: int = -1,
    target_token: str | None = None,
    normalized: bool = True,
) -> dict:
    """
    Logit attribution: measure each layer's attention and FFN
    contribution to the predicted token's logit.

    Two modes:

    normalized=True (default): At each layer, compute the model's
    logit for the target token before and after each sub-component.
    The delta is the component's contribution.  Values are on the
    same scale as actual model logits.  Best for large models
    (Gemma, Llama) where embedding scaling makes raw DLA noisy.

    normalized=False: Raw Direct Logit Attribution (DLA). Projects
    each component's output directly through the unembedding matrix
    without final norm.  Contributions are additive by construction.
    Best for smaller models (GPT-2, SmolLM2) with moderate scales.

    In normalized mode, also shows the logit lens prediction after
    each sub-component (what the model would predict at that point).
    In raw DLA mode, shows what token each component's output vector
    points toward in the unembedding space.

    This answers: "Which layers and components are responsible for
    predicting this specific token?"

    Args:
        prompt:       Input text.
        layers:       Layer indices to analyze.  None = sample ~12 layers.
        position:     Token position to analyze (-1 = last token).
        target_token: Token to attribute (string). None = model's top-1
                      prediction.
        normalized:   True = normalized deltas (default). False = raw DLA.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "logit_attribution",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        if num_layers <= 12:
            layers = list(range(num_layers))
        else:
            step = max(1, num_layers // 12)
            layers = list(range(0, num_layers, step))
            if (num_layers - 1) not in layers:
                layers.append(num_layers - 1)

    if layers is not None and not layers:
        return make_error(
            ToolError.INVALID_INPUT,
            "layers must not be empty when specified.",
            "logit_attribution",
        )

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "logit_attribution",
        )

    try:
        result = await asyncio.to_thread(
            _logit_attribution_impl,
            state.model,
            state.config,
            state.tokenizer,
            prompt,
            layers,
            position,
            target_token,
            normalized,
            num_layers,
        )
        return result
    except ValueError as exc:
        return make_error(ToolError.INVALID_INPUT, str(exc), "logit_attribution")
    except Exception as e:
        logger.exception("logit_attribution failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "logit_attribution")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def head_attribution(
    prompt: str,
    layer: int,
    target_token: str | None = None,
    position: int = -1,
) -> dict:
    """Per-head logit attribution: decompose a layer's attention into
    individual head contributions to the target token's logit.

    Uses Direct Logit Attribution (DLA) without final norm.  Each head's
    contribution is exact because ``o_proj`` is linear -- the sum of
    per-head contributions equals the total raw-DLA attention contribution.

    Typical workflow: run ``logit_attribution`` first to find the peak
    attention layer, then drill into that layer with ``head_attribution``.

    Args:
        prompt:        Input text.
        layer:         Single layer index to decompose.
        target_token:  Token to attribute (None = model's top-1 prediction).
        position:      Token position to analyse (-1 = last token).

    Returns:
        HeadAttributionResult dict with per-head logit contributions.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "No model loaded. Call load_model first.",
            "head_attribution",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "head_attribution",
        )

    try:
        result = await asyncio.to_thread(
            _head_attribution_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            target_token,
            position,
        )
        return result

    except Exception as e:
        logger.exception("head_attribution failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "head_attribution")


def _head_attribution_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    metadata: Any,
    prompt: str,
    layer: int,
    target_token: str | None,
    position: int,
) -> dict:
    """Synchronous implementation of head_attribution."""
    from chuk_lazarus.introspection.hooks import ModelHooks

    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = input_ids.shape[-1]
    pos = position if position >= 0 else num_tokens + position
    pos = max(0, min(pos, num_tokens - 1))
    tok_text = tokenizer.decode([to_pylist(input_ids)[pos]])

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    # Get model components
    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    embed = helper._get_embed_tokens()
    embedding_scale = helper._get_embedding_scale()
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not access the language model head for this model.",
            "head_attribution",
        )

    num_heads = metadata.num_attention_heads
    num_kv_heads = metadata.num_kv_heads or num_heads
    head_dim = metadata.head_dim or (metadata.hidden_dim // num_heads)

    # Forward through embedding
    h = embed(input_ids)
    if embedding_scale is not None:
        scale = mx.array(embedding_scale, dtype=h.dtype)
        h = h * scale

    # Forward through layers 0..layer-1
    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    for idx in range(layer):
        ly = model_layers[idx]
        try:
            out = ly(h, mask=mask, cache=None)
        except TypeError:
            try:
                out = ly(h, cache=None)
            except TypeError:
                out = ly(h)
        if hasattr(out, "hidden_states"):
            h = out.hidden_states
        elif isinstance(out, tuple):
            h = out[0]
        else:
            h = out

    # Also get full model output for target token resolution
    h_for_full = h
    for idx in range(layer, len(model_layers)):
        ly = model_layers[idx]
        try:
            out = ly(h_for_full, mask=mask, cache=None)
        except TypeError:
            try:
                out = ly(h_for_full, cache=None)
            except TypeError:
                out = ly(h_for_full)
        if hasattr(out, "hidden_states"):
            h_for_full = out.hidden_states
        elif isinstance(out, tuple):
            h_for_full = out[0]
        else:
            h_for_full = out

    final_vec = h_for_full[0, pos, :]
    if final_norm is not None:
        final_vec_normed = final_norm(final_vec.reshape(1, -1))[0]
    else:
        final_vec_normed = final_vec
    full_logits = _project_to_logits(lm_head, final_vec_normed)
    mx.eval(full_logits)

    # Resolve target token
    try:
        target_id, target_text = _resolve_target_token(
            tokenizer,
            full_logits,
            target_token,
        )
    except ValueError as exc:
        return make_error(
            ToolError.INVALID_INPUT,
            str(exc),
            "head_attribution",
        )

    # Now decompose the target layer's attention
    target_layer = model_layers[layer]
    attn = target_layer.self_attn

    # Apply input layernorm
    normed = target_layer.input_layernorm(h)

    # Q, K, V projections
    queries = attn.q_proj(normed)
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    batch_size = normed.shape[0]

    # Reshape to [batch, heads, seq, head_dim]
    queries = queries.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # Q/K norms (Gemma-style)
    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        queries = attn.q_norm(queries)
    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        keys = attn.k_norm(keys)

    # RoPE
    if hasattr(attn, "rope") and attn.rope is not None:
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    # GQA repeat
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    # SDPA → [batch, heads, seq, head_dim]
    attn_scale = getattr(attn, "scale", head_dim**-0.5)
    context = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn_scale,
        mask=mask,
    )
    mx.eval(context)

    # Per-head output through o_proj slices
    o_weight = attn.o_proj.weight  # [hidden_dim, num_heads * head_dim]

    # Get unembed vector for efficient per-head target logit
    u = _get_unembed_vector(model, target_id)
    if u is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not extract unembedding vector.",
            "head_attribution",
        )
    mx.eval(u)

    # Extract per-head outputs at target position
    head_results: list[HeadContribution] = []
    head_output_vecs: list[mx.array] = []

    for head_i in range(num_heads):
        head_ctx = context[0, head_i, pos, :]  # [head_dim]
        w_slice = o_weight[:, head_i * head_dim : (head_i + 1) * head_dim]
        head_out = head_ctx @ w_slice.T  # [hidden_dim]
        head_output_vecs.append(head_out)

    # Batch evaluate
    mx.eval(*head_output_vecs)

    # Per-head target logits via dot product with unembed vector
    head_logits = [float((hv * u).sum().item()) for hv in head_output_vecs]
    layer_total = sum(head_logits)

    # Batch project all heads through lm_head for top tokens
    head_stack = mx.stack(head_output_vecs)  # [num_heads, hidden_dim]
    all_logits = lm_head(head_stack.reshape(1, num_heads, -1))
    if hasattr(all_logits, "logits"):
        all_logits = all_logits.logits
    elif isinstance(all_logits, tuple):
        all_logits = all_logits[0]
    all_logits = all_logits[0]  # [num_heads, vocab]
    top_ids = mx.argmax(all_logits, axis=1)  # [num_heads]
    mx.eval(top_ids)
    top_ids_list = to_pylist(top_ids)

    for head_i in range(num_heads):
        frac = head_logits[head_i] / layer_total if abs(layer_total) > 1e-8 else 0.0
        top_tok = tokenizer.decode([top_ids_list[head_i]])
        head_results.append(
            HeadContribution(
                head=head_i,
                logit_contribution=round(head_logits[head_i], 6),
                fraction_of_layer=round(frac, 6),
                top_token=top_tok,
            )
        )

    # Summary
    pos_heads = [h for h in head_results if h.logit_contribution > 0]
    neg_heads = [h for h in head_results if h.logit_contribution < 0]
    top_pos = max(head_results, key=lambda h: h.logit_contribution)
    top_neg = min(head_results, key=lambda h: h.logit_contribution)
    concentration = (
        abs(top_pos.logit_contribution) / abs(layer_total) if abs(layer_total) > 1e-8 else 0.0
    )

    summary: dict[str, Any] = {
        "top_positive_head": top_pos.head,
        "top_positive_logit": top_pos.logit_contribution,
        "top_negative_head": top_neg.head,
        "top_negative_logit": top_neg.logit_contribution,
        "positive_head_count": len(pos_heads),
        "negative_head_count": len(neg_heads),
        "concentration": round(concentration, 6),
    }

    result = HeadAttributionResult(
        prompt=prompt,
        layer=layer,
        token_position=position,
        token_text=tok_text,
        target_token=target_text,
        target_token_id=target_id,
        num_heads=num_heads,
        heads=head_results,
        layer_total_logit=round(layer_total, 6),
        summary=summary,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def top_neurons(
    prompt: str,
    layer: int,
    target_token: str | None = None,
    position: int = -1,
    top_k: int = 20,
) -> dict:
    """Identify which MLP neurons contribute most to the predicted token.

    Decomposes the MLP at a single layer into per-neuron logit
    contributions using DLA.  For SwiGLU (Gemma, Llama):
    ``hidden = silu(gate_proj(x)) * up_proj(x)``.  Each neuron's
    contribution is ``activation[i] * dot(down_proj[:, i], unembed_vector)``.

    The sum of all neuron contributions equals the total raw-DLA FFN
    contribution for this layer.

    Args:
        prompt:       Input text.
        layer:        Layer index to analyse.
        target_token: Token to attribute (None = model's top-1 prediction).
        position:     Token position (-1 = last token).
        top_k:        Number of top neurons to return (positive + negative).

    Returns:
        TopNeuronsResult dict with top promoting and suppressing neurons.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "No model loaded. Call load_model first.",
            "top_neurons",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "top_neurons",
        )

    top_k = max(1, min(top_k, 200))

    try:
        result = await asyncio.to_thread(
            _top_neurons_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            target_token,
            position,
            top_k,
        )
        return result

    except Exception as e:
        logger.exception("top_neurons failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "top_neurons")


def _top_neurons_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    metadata: Any,
    prompt: str,
    layer: int,
    target_token: str | None,
    position: int,
    top_k: int,
) -> dict:
    """Synchronous implementation of top_neurons."""
    from chuk_lazarus.introspection.hooks import ModelHooks

    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = input_ids.shape[-1]
    pos = position if position >= 0 else num_tokens + position
    pos = max(0, min(pos, num_tokens - 1))
    tok_text = tokenizer.decode([to_pylist(input_ids)[pos]])

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    num_layers = metadata.num_layers
    last_layer = num_layers - 1
    layers_to_capture = sorted(set([layer, last_layer]))

    # Run decomposition forward to get prev_hidden and attn_outputs
    captured = _run_decomposition_forward(
        model,
        config,
        input_ids,
        layers_to_capture,
    )

    # Materialize
    to_eval = list(captured["hidden_states"].values())
    to_eval += list(captured["prev_hidden"].values())
    to_eval += [v for v in captured["attn_outputs"].values() if v is not None]
    mx.eval(*to_eval)

    # Get model logits for target token resolution
    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not access the language model head for this model.",
            "top_neurons",
        )

    final_hidden = captured["hidden_states"][last_layer]
    final_vec = _extract_position(final_hidden, pos)

    if final_norm is not None:
        final_vec_normed = final_norm(final_vec.reshape(1, -1))[0]
    else:
        final_vec_normed = final_vec
    full_logits = _project_to_logits(lm_head, final_vec_normed)
    mx.eval(full_logits)

    try:
        target_id, target_text = _resolve_target_token(
            tokenizer,
            full_logits,
            target_token,
        )
    except ValueError as exc:
        return make_error(
            ToolError.INVALID_INPUT,
            str(exc),
            "top_neurons",
        )

    # Reconstruct FFN input for the target layer
    prev_h = captured["prev_hidden"][layer]
    attn_out = captured["attn_outputs"].get(layer)

    if attn_out is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            f"Layer {layer} does not have decomposable attention output.",
            "top_neurons",
        )

    # h_post_attn = prev_hidden + attn_output (attn_output is already
    # post-normed for Gemma 3 due to _run_decomposition_forward)
    h_post_attn = prev_h + attn_out

    target_layer = helper._get_layers()[layer]
    mlp = target_layer.mlp

    # Compute FFN input (apply correct norm)
    if _has_four_norms(target_layer):
        ffn_input = target_layer.pre_feedforward_layernorm(h_post_attn)
    else:
        ffn_input = target_layer.post_attention_layernorm(h_post_attn)

    # Detect MLP type and compute post-gate activations
    has_gate = hasattr(mlp, "gate_proj")
    if has_gate:
        mlp_type = "swiglu"
        gate_out = mlp.gate_proj(ffn_input)
        up_out = mlp.up_proj(ffn_input)
        # Detect activation function
        act_fn = getattr(mlp, "act", None) or getattr(mlp, "activation", None)
        if act_fn is not None:
            hidden_act = act_fn(gate_out) * up_out
        else:
            hidden_act = nn.silu(gate_out) * up_out
    else:
        mlp_type = "standard"
        up_out = mlp.up_proj(ffn_input)
        act_fn = getattr(mlp, "act", None) or getattr(mlp, "activation", None)
        if act_fn is not None:
            hidden_act = act_fn(up_out)
        else:
            hidden_act = nn.gelu(up_out)

    mx.eval(hidden_act)

    # Extract at target position: [intermediate_size]
    if hidden_act.ndim == 3:
        hidden_pos = hidden_act[0, pos, :]
    elif hidden_act.ndim == 2:
        hidden_pos = hidden_act[pos, :]
    else:
        hidden_pos = hidden_act

    intermediate_size = hidden_pos.shape[0]

    # Get unembed vector and compute efficient per-neuron projections
    u = _get_unembed_vector(model, target_id)
    if u is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not extract unembedding vector.",
            "top_neurons",
        )
    mx.eval(u)

    # down_proj.weight: [hidden_dim, intermediate_size]
    down_weight = mlp.down_proj.weight
    # neuron_projections = down_weight.T @ u → [intermediate_size]
    neuron_projections = down_weight.T @ u
    mx.eval(neuron_projections)

    # Per-neuron logit contributions
    neuron_logits = hidden_pos * neuron_projections  # [intermediate_size]
    mx.eval(neuron_logits)

    total_neuron_logit = float(neuron_logits.sum().item())

    # Sort by absolute value to find top-k
    abs_logits = mx.abs(neuron_logits)
    sorted_indices = mx.argsort(abs_logits)[::-1]
    mx.eval(sorted_indices)
    sorted_list = to_pylist(sorted_indices)

    # Split into positive and negative
    neuron_logits_list = to_pylist(neuron_logits)
    hidden_pos_list = to_pylist(hidden_pos)

    pos_neurons = [(i, neuron_logits_list[i]) for i in sorted_list if neuron_logits_list[i] > 0]
    neg_neurons = [(i, neuron_logits_list[i]) for i in sorted_list if neuron_logits_list[i] < 0]

    top_pos_indices = [n[0] for n in pos_neurons[:top_k]]
    top_neg_indices = [n[0] for n in neg_neurons[:top_k]]
    all_top_indices = top_pos_indices + top_neg_indices

    # Batch project top neurons through lm_head for top tokens
    if all_top_indices:
        neuron_vecs = mx.stack(
            [hidden_pos[i] * down_weight[:, i] for i in all_top_indices]
        )  # [n, hidden_dim]
        batch_logits = lm_head(neuron_vecs.reshape(1, len(all_top_indices), -1))
        if hasattr(batch_logits, "logits"):
            batch_logits = batch_logits.logits
        elif isinstance(batch_logits, tuple):
            batch_logits = batch_logits[0]
        batch_logits = batch_logits[0]  # [n, vocab]
        batch_top_ids = mx.argmax(batch_logits, axis=1)
        mx.eval(batch_top_ids)
        batch_top_ids_list = to_pylist(batch_top_ids)
    else:
        batch_top_ids_list = []

    # Build result lists
    def make_contribution(idx_in_all: int, neuron_idx: int) -> NeuronContribution:
        top_tok = tokenizer.decode([batch_top_ids_list[idx_in_all]])
        return NeuronContribution(
            neuron_index=neuron_idx,
            activation=round(hidden_pos_list[neuron_idx], 6),
            logit_contribution=round(neuron_logits_list[neuron_idx], 6),
            top_token=top_tok,
        )

    top_positive = [make_contribution(i, idx) for i, idx in enumerate(top_pos_indices)]
    offset = len(top_pos_indices)
    top_negative = [make_contribution(offset + i, idx) for i, idx in enumerate(top_neg_indices)]

    # Summary
    pos_count = sum(1 for v in neuron_logits_list if v > 0)
    neg_count = sum(1 for v in neuron_logits_list if v < 0)
    sparse_count = sum(1 for v in neuron_logits_list if abs(v) < 1e-4)
    sparsity = sparse_count / intermediate_size if intermediate_size > 0 else 0.0

    top_10_abs = sorted(neuron_logits_list, key=lambda x: abs(x), reverse=True)[:10]
    concentration = sum(abs(v) for v in top_10_abs) / (abs(total_neuron_logit) + 1e-8)

    summary: dict[str, Any] = {
        "top_neuron_index": sorted_list[0] if sorted_list else -1,
        "top_neuron_logit": round(neuron_logits_list[sorted_list[0]], 6) if sorted_list else 0.0,
        "positive_neuron_count": pos_count,
        "negative_neuron_count": neg_count,
        "concentration_top10": round(concentration, 6),
        "sparsity": round(sparsity, 6),
    }

    result = TopNeuronsResult(
        prompt=prompt,
        layer=layer,
        token_position=position,
        token_text=tok_text,
        target_token=target_text,
        target_token_id=target_id,
        mlp_type=mlp_type,
        intermediate_size=intermediate_size,
        top_k=top_k,
        top_positive=top_positive,
        top_negative=top_negative,
        total_neuron_logit=round(total_neuron_logit, 6),
        summary=summary,
    )
    return result.model_dump()
