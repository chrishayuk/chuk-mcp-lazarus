"""
Model comparison tools: load_comparison_model, compare_weights,
compare_representations, compare_attention, compare_generations,
unload_comparison_model.

Compare a primary model (in ModelState) with a second model (in
ComparisonState) to find where fine-tuning changed weights,
activations, attention patterns, and generated outputs.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from .._compare import (
    activation_divergence,
    attention_divergence,
    weight_divergence,
)
from ..comparison_state import ComparisonState
from ..errors import ToolError, make_error
from ..model_state import ModelState, WeightDType
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class LoadComparisonResult(BaseModel):
    """Result from load_comparison_model."""

    model_id: str
    family: str
    architecture: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    parameter_count: int
    status: str = "loaded"


class WeightDivergenceResult(BaseModel):
    """Result from compare_weights."""

    primary_model: str
    comparison_model: str
    num_layers_compared: int
    divergences: list[dict] = Field(
        ..., description="Per-layer, per-component weight divergence."
    )
    summary: dict = Field(
        ..., description="Top divergent layers and overall statistics."
    )


class RepresentationDivergenceResult(BaseModel):
    """Result from compare_representations."""

    primary_model: str
    comparison_model: str
    num_prompts: int
    num_layers_compared: int
    divergences: list[dict] = Field(
        ..., description="Per-layer, per-prompt activation divergence."
    )
    layer_averages: list[dict] = Field(
        ..., description="Average divergence per layer across prompts."
    )


class AttentionDivergenceResult(BaseModel):
    """Result from compare_attention."""

    primary_model: str
    comparison_model: str
    prompt: str
    num_layers_compared: int
    divergences: list[dict] = Field(
        ..., description="Per-layer, per-head attention divergence."
    )
    top_divergent_heads: list[dict] = Field(
        ..., description="Heads with highest JS divergence."
    )


class CompareGenerationsResult(BaseModel):
    """Result from compare_generations."""

    prompt: str
    primary_model: str
    comparison_model: str
    primary_output: str
    comparison_output: str
    primary_tokens: int
    comparison_tokens: int
    outputs_match: bool


class UnloadComparisonResult(BaseModel):
    """Result from unload_comparison_model."""

    model_id: str
    status: str = "unloaded"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_layers(layers: list[int] | None, num_layers: int) -> list[int]:
    """Validate and normalize layer list."""
    if layers is None:
        return list(range(num_layers))

    out_of_range = [l for l in layers if l < 0 or l >= num_layers]
    if out_of_range:
        raise ValueError(f"Layers {out_of_range} out of range [0, {num_layers - 1}].")
    return sorted(set(layers))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def load_comparison_model(
    model_id: str,
    dtype: str = "bfloat16",
) -> dict:
    """
    Load a second model for two-model comparison analysis.

    The comparison model is held separately from the primary model and
    is used by compare_weights, compare_representations, and
    compare_attention. Models must share the same architecture
    (num_layers, hidden_dim) for meaningful comparison.

    Args:
        model_id: HuggingFace model ID (e.g. "google/translategemma-4b-it").
        dtype:    Weight dtype: "bfloat16", "float16", or "float32".
    """
    try:
        weight_dtype = WeightDType(dtype)
    except ValueError:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid dtype '{dtype}'. Use: bfloat16, float16, float32.",
            "load_comparison_model",
        )

    try:
        comp = ComparisonState.get()
        metadata = await asyncio.to_thread(comp.load, model_id, weight_dtype)

        result = LoadComparisonResult(
            model_id=metadata.model_id,
            family=metadata.family,
            architecture=metadata.architecture,
            num_layers=metadata.num_layers,
            hidden_dim=metadata.hidden_dim,
            num_attention_heads=metadata.num_attention_heads,
            parameter_count=metadata.parameter_count,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("load_comparison_model failed")
        return make_error(ToolError.LOAD_FAILED, str(e), "load_comparison_model")


@mcp.tool(read_only_hint=True)
async def compare_weights(
    layers: list[int] | None = None,
) -> dict:
    """
    Compare weight matrices between primary and comparison models.

    Returns per-layer, per-component Frobenius norm difference and
    cosine similarity. No inference needed -- operates directly on
    weight tensors. Cheapest comparison operation.

    Args:
        layers: Layer indices to compare. None = all layers.
    """
    primary = ModelState.get()
    if not primary.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compare_weights",
        )

    comp = ComparisonState.get()
    if not comp.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_comparison_model() first.",
            "compare_weights",
        )

    try:
        comp.require_compatible(primary.metadata)
    except ValueError as e:
        return make_error(ToolError.COMPARISON_INCOMPATIBLE, str(e), "compare_weights")

    try:
        validated = _validate_layers(layers, primary.metadata.num_layers)
    except ValueError as e:
        return make_error(ToolError.LAYER_OUT_OF_RANGE, str(e), "compare_weights")

    try:
        divergences = await asyncio.to_thread(
            weight_divergence, primary.model, comp.model, validated
        )

        # Summarize: average divergence per layer
        layer_totals: dict[int, list[float]] = {}
        for d in divergences:
            layer_totals.setdefault(d["layer"], []).append(d["frobenius_norm_diff"])

        layer_avgs = sorted(
            [
                {"layer": l, "avg_frobenius": round(sum(v) / len(v), 6)}
                for l, v in layer_totals.items()
            ],
            key=lambda x: x["avg_frobenius"],
            reverse=True,
        )

        result = WeightDivergenceResult(
            primary_model=primary.metadata.model_id,
            comparison_model=comp.metadata.model_id,
            num_layers_compared=len(validated),
            divergences=divergences,
            summary={
                "top_divergent_layers": layer_avgs[:5],
                "total_components": len(divergences),
            },
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("compare_weights failed")
        return make_error(ToolError.COMPARISON_FAILED, str(e), "compare_weights")


@mcp.tool(read_only_hint=True)
async def compare_representations(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Compare hidden-state activations between primary and comparison
    models for the same prompts. Shows where representations diverge.

    Best used with translation prompts in different languages to find
    where fine-tuning changed the model's internal representations.

    Args:
        prompts:        1-8 input strings.
        layers:         Layer indices to compare. None = all layers.
        token_position: Token position (-1 = last).
    """
    primary = ModelState.get()
    if not primary.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compare_representations",
        )

    comp = ComparisonState.get()
    if not comp.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_comparison_model() first.",
            "compare_representations",
        )

    try:
        comp.require_compatible(primary.metadata)
    except ValueError as e:
        return make_error(ToolError.COMPARISON_INCOMPATIBLE, str(e), "compare_representations")

    if not 1 <= len(prompts) <= 8:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need 1-8 prompts, got {len(prompts)}.",
            "compare_representations",
        )

    try:
        validated = _validate_layers(layers, primary.metadata.num_layers)
    except ValueError as e:
        return make_error(ToolError.LAYER_OUT_OF_RANGE, str(e), "compare_representations")

    try:
        divergences = await asyncio.to_thread(
            activation_divergence,
            primary.model,
            primary.config,
            comp.model,
            comp.config,
            primary.tokenizer,
            prompts,
            validated,
            token_position,
        )

        # Average divergence per layer across prompts
        layer_sums: dict[int, list[float]] = {}
        for d in divergences:
            layer_sums.setdefault(d["layer"], []).append(d["cosine_similarity"])

        layer_avgs = [
            {
                "layer": l,
                "avg_cosine_similarity": round(sum(v) / len(v), 6),
                "avg_divergence": round(1.0 - sum(v) / len(v), 6),
            }
            for l, v in sorted(layer_sums.items())
        ]

        result = RepresentationDivergenceResult(
            primary_model=primary.metadata.model_id,
            comparison_model=comp.metadata.model_id,
            num_prompts=len(prompts),
            num_layers_compared=len(validated),
            divergences=divergences,
            layer_averages=layer_avgs,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("compare_representations failed")
        return make_error(ToolError.COMPARISON_FAILED, str(e), "compare_representations")


@mcp.tool(read_only_hint=True)
async def compare_attention(
    prompt: str,
    layers: list[int] | None = None,
) -> dict:
    """
    Compare attention patterns between primary and comparison models.

    Returns per-head Jensen-Shannon divergence and cosine similarity
    for the last token's attention distribution at each layer.

    Args:
        prompt: Input text.
        layers: Layer indices to compare. None = all layers.
    """
    primary = ModelState.get()
    if not primary.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compare_attention",
        )

    comp = ComparisonState.get()
    if not comp.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_comparison_model() first.",
            "compare_attention",
        )

    try:
        comp.require_compatible(primary.metadata)
    except ValueError as e:
        return make_error(ToolError.COMPARISON_INCOMPATIBLE, str(e), "compare_attention")

    try:
        validated = _validate_layers(layers, primary.metadata.num_layers)
    except ValueError as e:
        return make_error(ToolError.LAYER_OUT_OF_RANGE, str(e), "compare_attention")

    try:
        divergences = await asyncio.to_thread(
            attention_divergence,
            primary.model,
            primary.config,
            comp.model,
            comp.config,
            primary.tokenizer,
            prompt,
            validated,
        )

        # Top divergent heads
        top_heads = sorted(divergences, key=lambda d: d["js_divergence"], reverse=True)[:10]

        result = AttentionDivergenceResult(
            primary_model=primary.metadata.model_id,
            comparison_model=comp.metadata.model_id,
            prompt=prompt,
            num_layers_compared=len(validated),
            divergences=divergences,
            top_divergent_heads=top_heads,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("compare_attention failed")
        return make_error(ToolError.COMPARISON_FAILED, str(e), "compare_attention")


@mcp.tool(read_only_hint=True)
async def compare_generations(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> dict:
    """
    Generate text from both models using the same prompt and compare
    the outputs side-by-side.

    Essential for seeing *what actually changes* in model output after
    fine-tuning, before diving into internal representations.

    Args:
        prompt:         Input text.
        max_new_tokens: Maximum tokens to generate (default: 100).
        temperature:    Sampling temperature. 0 = greedy (default).
    """
    primary = ModelState.get()
    if not primary.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compare_generations",
        )

    comp = ComparisonState.get()
    if not comp.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_comparison_model() first.",
            "compare_generations",
        )

    try:
        from .._generate import generate_text as _gen

        primary_out, primary_tokens = await asyncio.to_thread(
            _gen, primary.model, primary.tokenizer, prompt,
            max_new_tokens, temperature,
        )
        comp_out, comp_tokens = await asyncio.to_thread(
            _gen, comp.model, comp.tokenizer, prompt,
            max_new_tokens, temperature,
        )

        result = CompareGenerationsResult(
            prompt=prompt,
            primary_model=primary.metadata.model_id,
            comparison_model=comp.metadata.model_id,
            primary_output=primary_out,
            comparison_output=comp_out,
            primary_tokens=primary_tokens,
            comparison_tokens=comp_tokens,
            outputs_match=primary_out.strip() == comp_out.strip(),
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("compare_generations failed")
        return make_error(ToolError.COMPARISON_FAILED, str(e), "compare_generations")


@mcp.tool()
async def unload_comparison_model() -> dict:
    """
    Unload the comparison model and free VRAM.

    After unloading, the primary model in ModelState remains available.
    """
    comp = ComparisonState.get()
    if not comp.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "No comparison model loaded.",
            "unload_comparison_model",
        )

    try:
        model_id = comp.metadata.model_id
        await asyncio.to_thread(comp.unload)
        result = UnloadComparisonResult(model_id=model_id)
        return result.model_dump()

    except Exception as e:
        logger.exception("unload_comparison_model failed")
        return make_error(ToolError.COMPARISON_FAILED, str(e), "unload_comparison_model")
