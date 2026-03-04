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

from .._extraction import extract_activation_at_layer
from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

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
