"""
Neuron analysis tools: discover_neurons, analyze_neuron.

Finds discriminative MLP neurons between prompt groups (discover_neurons)
and reports activation statistics for specific neurons (analyze_neuron).
Uses the shared _extraction helpers — no chuk-lazarus neuron service import.
"""

import asyncio
import logging
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..._extraction import extract_activation_at_layer
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class DiscoveredNeuronEntry(BaseModel):
    """A single discovered neuron."""

    neuron_idx: int
    separation: float = Field(..., description="Cohen's d separation score.")
    best_pair: list[str] | None = Field(None, description="Label pair with best separation.")
    overall_std: float
    mean_range: float
    group_means: dict[str, float]


class DiscoverNeuronsResult(BaseModel):
    """Result from discover_neurons."""

    layer: int
    top_k: int
    num_prompts: int
    num_labels: int
    labels: list[str]
    neurons: list[DiscoveredNeuronEntry]


class NeuronActivationEntry(BaseModel):
    """Activation statistics for a single neuron."""

    neuron_idx: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float


class AnalyzeNeuronResult(BaseModel):
    """Result from analyze_neuron."""

    layer: int
    num_prompts: int
    neurons: list[NeuronActivationEntry]
    prompts: list[str]
    per_prompt_activations: list[dict[str, Any]] | None = Field(
        None,
        description="Per-prompt activation values for each neuron (when detailed=True).",
    )


# ---------------------------------------------------------------------------
# Tool: discover_neurons
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def discover_neurons(
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    positive_label: str = "positive",
    negative_label: str = "negative",
    top_k: int = 20,
    token_position: int = -1,
) -> dict:
    """
    Discover the most discriminative MLP neurons between two groups
    of prompts at a given layer.

    Ranks all neurons by Cohen's d separation score between the
    positive and negative prompt groups. Returns the top-k neurons
    with the highest separation.

    Useful for identifying which specific neurons encode a concept
    (e.g. language identity, factual knowledge, sentiment).

    Args:
        layer:             Layer to analyze.
        positive_prompts:  Prompts representing one class.
        negative_prompts:  Prompts representing the other class.
        positive_label:    Label for positive class (default: "positive").
        negative_label:    Label for negative class (default: "negative").
        top_k:             Number of top neurons to return (default: 20).
        token_position:    Token position to extract (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "discover_neurons",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "discover_neurons",
        )

    if not positive_prompts:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 positive prompt is required.",
            "discover_neurons",
        )

    if not negative_prompts:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 negative prompt is required.",
            "discover_neurons",
        )

    if top_k < 1 or top_k > 1000:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k must be between 1 and 1000.",
            "discover_neurons",
        )

    try:
        result = await asyncio.to_thread(
            _discover_neurons_impl,
            state.model,
            state.config,
            state.tokenizer,
            layer,
            positive_prompts,
            negative_prompts,
            positive_label,
            negative_label,
            top_k,
            token_position,
        )
        return result
    except Exception as e:
        logger.exception("discover_neurons failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "discover_neurons")


def _discover_neurons_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    positive_label: str,
    negative_label: str,
    top_k: int,
    token_position: int,
) -> dict:
    """Sync implementation of discover_neurons."""
    # Step 1: Extract activations for all prompts
    all_prompts = positive_prompts + negative_prompts
    labels = [positive_label] * len(positive_prompts) + [negative_label] * len(negative_prompts)

    all_activations: list[list[float]] = []
    for prompt in all_prompts:
        vec = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        all_activations.append(vec)

    full_activations = np.array(all_activations, dtype=np.float32)
    num_neurons = full_activations.shape[1]

    # Step 2: Group by label
    unique_labels = sorted(set(labels))
    label_groups: dict[str, np.ndarray] = {}
    for lbl in unique_labels:
        indices = [i for i, la in enumerate(labels) if la == lbl]
        label_groups[lbl] = full_activations[indices]

    # Step 3: Compute per-neuron separation
    single_sample = all(len(label_groups[lbl]) == 1 for lbl in unique_labels)

    neuron_scores: list[DiscoveredNeuronEntry] = []
    for neuron_idx in range(num_neurons):
        group_means: dict[str, float] = {}
        group_stds: list[float] = []
        for lbl in unique_labels:
            vals = label_groups[lbl][:, neuron_idx]
            group_means[lbl] = float(np.mean(vals))
            group_stds.append(float(np.std(vals)))

        overall_std = float(np.std(full_activations[:, neuron_idx]))

        # Find max pairwise separation (Cohen's d)
        max_separation = 0.0
        best_pair: list[str] | None = None
        for i, lbl1 in enumerate(unique_labels):
            for j, lbl2 in enumerate(unique_labels):
                if i >= j:
                    continue
                mean_diff = abs(group_means[lbl1] - group_means[lbl2])

                if single_sample:
                    separation = mean_diff / overall_std if overall_std > 1e-6 else 0.0
                else:
                    pooled_std = math.sqrt((group_stds[i] ** 2 + group_stds[j] ** 2) / 2)
                    separation = mean_diff / pooled_std if pooled_std > 1e-6 else 0.0

                if separation > max_separation:
                    max_separation = separation
                    best_pair = [lbl1, lbl2]

        mean_vals = list(group_means.values())
        mean_range = max(mean_vals) - min(mean_vals) if mean_vals else 0.0

        neuron_scores.append(
            DiscoveredNeuronEntry(
                neuron_idx=neuron_idx,
                separation=round(max_separation, 4),
                best_pair=best_pair,
                overall_std=round(overall_std, 6),
                mean_range=round(mean_range, 6),
                group_means={k: round(v, 6) for k, v in group_means.items()},
            )
        )

    # Step 4: Sort and take top-k
    neuron_scores.sort(key=lambda x: -x.separation)
    top_neurons = neuron_scores[:top_k]

    result = DiscoverNeuronsResult(
        layer=layer,
        top_k=top_k,
        num_prompts=len(all_prompts),
        num_labels=len(unique_labels),
        labels=unique_labels,
        neurons=top_neurons,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Tool: analyze_neuron
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def analyze_neuron(
    layer: int,
    neuron_indices: list[int],
    prompts: list[str],
    token_position: int = -1,
    detailed: bool = False,
) -> dict:
    """
    Analyze specific neurons' activations across a set of prompts.

    Returns min, max, mean, and standard deviation for each neuron.
    Use discover_neurons first to find interesting neuron indices,
    then use this tool to drill into their behavior on specific prompts.

    Args:
        layer:           Layer to analyze.
        neuron_indices:  Neuron indices to analyze (from discover_neurons).
        prompts:         Prompts to analyze.
        token_position:  Token position to extract (-1 = last).
        detailed:        If True, include per-prompt activation values.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "analyze_neuron",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "analyze_neuron",
        )

    if not prompts:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 prompt is required.",
            "analyze_neuron",
        )

    if not neuron_indices:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 neuron index is required.",
            "analyze_neuron",
        )

    if len(neuron_indices) > 100:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 100 neuron indices allowed.",
            "analyze_neuron",
        )

    if len(prompts) > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 50 prompts allowed.",
            "analyze_neuron",
        )

    hidden_dim = state.metadata.hidden_dim
    out_of_range = [i for i in neuron_indices if i < 0 or i >= hidden_dim]
    if out_of_range:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Neuron indices {out_of_range} out of range [0, {hidden_dim - 1}].",
            "analyze_neuron",
        )

    try:
        result = await asyncio.to_thread(
            _analyze_neuron_impl,
            state.model,
            state.config,
            state.tokenizer,
            layer,
            neuron_indices,
            prompts,
            token_position,
            detailed,
        )
        return result
    except Exception as e:
        logger.exception("analyze_neuron failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "analyze_neuron")


def _analyze_neuron_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    layer: int,
    neuron_indices: list[int],
    prompts: list[str],
    token_position: int,
    detailed: bool,
) -> dict:
    """Sync implementation of analyze_neuron."""
    # Extract activations for all prompts
    all_activations: list[list[float]] = []
    for prompt in prompts:
        vec = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        all_activations.append(vec)

    activations = np.array(all_activations, dtype=np.float32)

    # Compute statistics per neuron
    neuron_results: list[NeuronActivationEntry] = []
    for idx in neuron_indices:
        vals = activations[:, idx]
        neuron_results.append(
            NeuronActivationEntry(
                neuron_idx=idx,
                min_val=round(float(vals.min()), 6),
                max_val=round(float(vals.max()), 6),
                mean_val=round(float(vals.mean()), 6),
                std_val=round(float(vals.std()), 6),
            )
        )

    # Optionally include per-prompt activations
    per_prompt: list[dict[str, Any]] | None = None
    if detailed:
        per_prompt = []
        for i, prompt in enumerate(prompts):
            entry: dict[str, Any] = {"prompt": prompt}
            for idx in neuron_indices:
                entry[f"neuron_{idx}"] = round(float(activations[i, idx]), 6)
            per_prompt.append(entry)

    result = AnalyzeNeuronResult(
        layer=layer,
        num_prompts=len(prompts),
        neurons=neuron_results,
        prompts=prompts,
        per_prompt_activations=per_prompt,
    )
    return result.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Result models for neuron_trace
# ---------------------------------------------------------------------------


class NeuronInfo(BaseModel):
    """Source neuron information."""

    layer: int
    neuron_index: int
    activation: float
    output_direction_norm: float
    top_token: str


class AlignedHead(BaseModel):
    """Attention head aligned with the neuron direction."""

    head: int
    alignment: float


class TraceLayerEntry(BaseModel):
    """Alignment of neuron direction with a downstream layer's components."""

    layer: int
    residual_alignment: float = Field(
        ..., description="Cosine sim of neuron direction with residual stream."
    )
    attention_alignment: float | None = Field(
        None, description="Cosine sim with attention output (None if unavailable)."
    )
    ffn_alignment: float | None = Field(
        None, description="Cosine sim with FFN output (None if unavailable)."
    )
    residual_projection: float = Field(
        ..., description="Dot product of neuron direction with residual stream."
    )
    top_aligned_heads: list[AlignedHead] | None = None


class NeuronTraceResult(BaseModel):
    """Result from neuron_trace."""

    prompt: str
    token_position: int
    token_text: str
    neuron: NeuronInfo
    num_trace_layers: int
    trace: list[TraceLayerEntry]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool: neuron_trace
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def neuron_trace(
    prompt: str,
    layer: int,
    neuron_index: int,
    target_layers: list[int] | None = None,
    token_position: int = -1,
    top_k_heads: int = 5,
) -> dict:
    """
    Trace a neuron's influence through downstream layers.

    Computes the output direction of a specific FFN neuron at the source
    layer, then measures how aligned that direction is with the residual
    stream, attention output, and FFN output at subsequent layers.

    Useful for understanding how a neuron's signal propagates: which
    attention heads at layer N read from neuron X's output at layer M?

    Args:
        prompt:         Input text.
        layer:          Source layer (where the neuron lives).
        neuron_index:   Index of the FFN neuron.
        target_layers:  Downstream layers to trace through. None = auto.
        token_position: Token position (-1 = last).
        top_k_heads:    Number of top-aligned heads to return (1–64).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "neuron_trace",
        )

    num_layers = state.metadata.num_layers
    intermediate_size = state.metadata.intermediate_size

    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "neuron_trace",
        )

    if neuron_index < 0 or neuron_index >= intermediate_size:
        return make_error(
            ToolError.INVALID_INPUT,
            f"neuron_index {neuron_index} out of range [0, {intermediate_size - 1}].",
            "neuron_trace",
        )

    if target_layers is None:
        target_layers = list(range(layer + 1, min(layer + 10, num_layers)))

    if not target_layers:
        return make_error(
            ToolError.INVALID_INPUT,
            "No target layers to trace (source layer may be the last layer).",
            "neuron_trace",
        )

    invalid = [t for t in target_layers if t <= layer]
    if invalid:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Target layers {invalid} must be > source layer {layer}.",
            "neuron_trace",
        )

    out_of_range = [t for t in target_layers if t >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Target layers {out_of_range} out of range [0, {num_layers - 1}].",
            "neuron_trace",
        )

    if top_k_heads < 1 or top_k_heads > 64:
        return make_error(
            ToolError.INVALID_INPUT,
            f"top_k_heads must be 1–64, got {top_k_heads}.",
            "neuron_trace",
        )

    try:
        result = await asyncio.to_thread(
            _neuron_trace_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            layer,
            neuron_index,
            sorted(target_layers),
            token_position,
            top_k_heads,
        )
        return result
    except ValueError as exc:
        return make_error(ToolError.INVALID_INPUT, str(exc), "neuron_trace")
    except Exception as e:
        logger.exception("neuron_trace failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "neuron_trace")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clamped to [-1, 1]."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))


def _neuron_trace_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    metadata: Any,
    prompt: str,
    layer: int,
    neuron_index: int,
    target_layers: list[int],
    token_position: int,
    top_k_heads: int,
) -> dict:
    """Sync implementation of neuron_trace."""
    import mlx.core as mx

    from ..._residual_helpers import (
        _extract_position,
        _get_lm_projection,
        _project_to_logits,
        _run_decomposition_forward,
    )

    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = input_ids.shape[-1]
    ids_list = input_ids.tolist() if hasattr(input_ids, "tolist") else list(range(num_tokens))
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    if isinstance(ids_list, list) and ids_list:
        tok_text = tokenizer.decode([ids_list[pos] if isinstance(ids_list[pos], int) else 0])
    else:
        tok_text = ""

    # Run decomposition forward to capture all needed layers
    all_layers = sorted(set([layer] + target_layers))
    decomp = _run_decomposition_forward(model, config, input_ids, all_layers)

    # Get the MLP at source layer to extract neuron direction
    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    model_layers = helper._get_layers()
    source_layer = model_layers[layer]
    mlp = source_layer.mlp

    # Get hidden state entering the source layer's MLP
    # For standard arch: prev_hidden → input_layernorm → self_attn → post_attention_layernorm → MLP
    # We need the FFN input. We have attn_outputs and prev_hidden
    prev_h = decomp["prev_hidden"].get(layer)
    attn_out = decomp["attn_outputs"].get(layer)

    if prev_h is not None and attn_out is not None:
        # FFN input = prev_h + attn_out (then post_attention_layernorm applied)
        ffn_input_pre = _extract_position(prev_h, token_position) + _extract_position(
            attn_out, token_position
        )
    elif prev_h is not None:
        ffn_input_pre = _extract_position(prev_h, token_position)
    else:
        # Fallback: use hidden state at previous layer or embeddings
        h_prev = decomp["hidden_states"].get(layer - 1) if layer > 0 else decomp["embeddings"]
        ffn_input_pre = _extract_position(h_prev, token_position)

    # Apply post_attention_layernorm to get actual FFN input
    ffn_input_normed = source_layer.post_attention_layernorm(ffn_input_pre.reshape(1, -1))[0]

    # Compute neuron activation via SwiGLU: silu(gate * input) * up * input
    gate_out = mlp.gate_proj(ffn_input_normed.reshape(1, -1))[0]
    up_out = mlp.up_proj(ffn_input_normed.reshape(1, -1))[0]

    # SwiGLU activation at neuron_index
    gate_val = gate_out[neuron_index]
    up_val = up_out[neuron_index]
    silu_val = gate_val / (1.0 + mx.array(math.exp(-float(gate_val))))  # silu approximation
    activation = float((silu_val * up_val).item())

    # Neuron output direction = down_proj weight column
    down_weight = mlp.down_proj.weight  # [hidden_dim, intermediate_size]
    direction = down_weight[:, neuron_index]
    mx.eval(direction)

    # Convert to numpy for cosine computations
    dir_np = np.array(direction.tolist(), dtype=np.float32)
    dir_norm = float(np.linalg.norm(dir_np))

    # Top token from neuron direction
    lm_head = _get_lm_projection(model)
    dir_logits = _project_to_logits(lm_head, direction)
    mx.eval(dir_logits)
    top_token_id = int(mx.argmax(dir_logits).item())
    top_token = tokenizer.decode([top_token_id])

    neuron_info = NeuronInfo(
        layer=layer,
        neuron_index=neuron_index,
        activation=round(activation, 6),
        output_direction_norm=round(dir_norm, 6),
        top_token=top_token,
    )

    # Trace through target layers
    trace: list[TraceLayerEntry] = []
    max_residual_alignment = 0.0
    max_alignment_layer = target_layers[0]

    for tl in target_layers:
        h_state = decomp["hidden_states"].get(tl)
        attn_state = decomp["attn_outputs"].get(tl)
        ffn_state = decomp["ffn_outputs"].get(tl)

        if h_state is None:
            continue

        h_vec = _extract_position(h_state, token_position)
        h_np = np.array(h_vec.tolist(), dtype=np.float32)

        res_align = _cosine_sim(dir_np, h_np)
        res_proj = float(np.dot(dir_np / (dir_norm + 1e-8), h_np))

        attn_align = None
        if attn_state is not None:
            attn_vec = _extract_position(attn_state, token_position)
            attn_np = np.array(attn_vec.tolist(), dtype=np.float32)
            attn_align = _cosine_sim(dir_np, attn_np)

        ffn_align = None
        if ffn_state is not None:
            ffn_vec = _extract_position(ffn_state, token_position)
            ffn_np = np.array(ffn_vec.tolist(), dtype=np.float32)
            ffn_align = _cosine_sim(dir_np, ffn_np)

        if abs(res_align) > abs(max_residual_alignment):
            max_residual_alignment = res_align
            max_alignment_layer = tl

        trace.append(
            TraceLayerEntry(
                layer=tl,
                residual_alignment=round(res_align, 6),
                attention_alignment=round(attn_align, 6) if attn_align is not None else None,
                ffn_alignment=round(ffn_align, 6) if ffn_align is not None else None,
                residual_projection=round(res_proj, 6),
                top_aligned_heads=None,  # Head-level tracing requires per-head decomposition
            )
        )

    summary = {
        "max_residual_alignment": round(max_residual_alignment, 6),
        "max_alignment_layer": max_alignment_layer,
        "num_trace_layers": len(trace),
        "neuron_activation": round(activation, 6),
        "top_token": top_token,
    }

    result = NeuronTraceResult(
        prompt=prompt,
        token_position=token_position,
        token_text=tok_text,
        neuron=neuron_info,
        num_trace_layers=len(trace),
        trace=trace,
        summary=summary,
    )
    return result.model_dump(exclude_none=True)
