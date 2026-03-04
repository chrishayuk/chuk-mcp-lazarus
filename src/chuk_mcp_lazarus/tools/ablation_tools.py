"""
Ablation tools: ablate_layers, patch_activations.

Ablation zeroes out layer weights to test causal necessity.
Activation patching swaps hidden states from one prompt into another
to measure information flow. Both are key tools for causal
mechanistic interpretability.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from .._generate import generate_text
from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class AblateLayersResult(BaseModel):
    """Result from ablate_layers."""

    prompt: str
    ablated_layers: list[int]
    ablation_type: str
    component: str
    ablated_output: str
    baseline_output: str
    output_similarity: float = Field(
        ..., description="Word-level Jaccard similarity between ablated and baseline outputs."
    )
    disruption_score: float = Field(
        ..., description="1 - output_similarity. Higher = more disruption."
    )


class PatchActivationsResult(BaseModel):
    """Result from patch_activations."""

    source_prompt: str
    target_prompt: str
    patched_layer: int
    patched_output: str
    baseline_output: str
    source_output: str
    recovery_rate: float = Field(
        ..., description="0-1: how much source behaviour was recovered in the patched run."
    )
    effect_size: float = Field(
        ..., description="recovery_rate - 0.5. Positive = patching moved output toward source."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def ablate_layers(
    prompt: str,
    layers: list[int],
    max_new_tokens: int = 50,
    ablation_type: str = "zero",
    component: str = "mlp",
) -> dict:
    """
    Generate text with specific layers ablated (zeroed out).

    Compares ablated output with a baseline to measure how much each
    layer contributes. High disruption = the layer is causally important.

    Args:
        prompt:         Input text.
        layers:         Layer indices to ablate.
        max_new_tokens: Tokens to generate.
        ablation_type:  "zero" (only supported type currently).
        component:      "mlp", "attention", or "both".
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "ablate_layers",
        )

    if not layers:
        return make_error(
            ToolError.INVALID_INPUT,
            "layers must not be empty.",
            "ablate_layers",
        )

    if max_new_tokens < 1 or max_new_tokens > 1000:
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_new_tokens must be 1-1000, got {max_new_tokens}.",
            "ablate_layers",
        )

    num_layers = state.metadata.num_layers
    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "ablate_layers",
        )

    if ablation_type not in ("zero",):
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid ablation_type '{ablation_type}'. Use 'zero'.",
            "ablate_layers",
        )

    valid_components = ("mlp", "attention", "both")
    if component not in valid_components:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid component '{component}'. Use one of: {', '.join(valid_components)}.",
            "ablate_layers",
        )

    try:
        result = await asyncio.to_thread(
            _ablate_layers_impl,
            state.model,
            state.tokenizer,
            state.config,
            prompt,
            layers,
            max_new_tokens,
            ablation_type,
            component,
        )
        return result

    except Exception as e:
        logger.exception("ablate_layers failed")
        return make_error(ToolError.ABLATION_FAILED, str(e), "ablate_layers")


def _ablate_layers_impl(
    model: Any,
    tokenizer: Any,
    config: Any,
    prompt: str,
    layers: list[int],
    max_new_tokens: int,
    ablation_type: str,
    component: str,
) -> dict:
    """Sync implementation of ablate_layers."""
    from chuk_lazarus.introspection.ablation.adapter import ModelAdapter
    from chuk_lazarus.introspection.ablation.study import AblationStudy
    from chuk_lazarus.introspection.ablation.config import (
        AblationConfig,
        ComponentType,
    )

    component_map = {
        "mlp": ComponentType.MLP,
        "attention": ComponentType.ATTENTION,
        "both": ComponentType.BOTH,
    }
    comp = component_map[component]

    adapter = ModelAdapter(model, tokenizer, config)
    study = AblationStudy(adapter)
    abl_config = AblationConfig(max_new_tokens=max_new_tokens, component=comp)

    baseline_output, _ = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
    )

    ablated_output = study.ablate_and_generate(
        prompt=prompt,
        layers=layers,
        component=comp,
        config=abl_config,
    )

    similarity = _word_overlap_similarity(baseline_output, ablated_output)
    disruption = 1.0 - similarity

    result = AblateLayersResult(
        prompt=prompt,
        ablated_layers=layers,
        ablation_type=ablation_type,
        component=component,
        ablated_output=ablated_output,
        baseline_output=baseline_output,
        output_similarity=round(similarity, 4),
        disruption_score=round(disruption, 4),
    )
    return result.model_dump()


@mcp.tool()
async def patch_activations(
    source_prompt: str,
    target_prompt: str,
    layer: int,
    max_new_tokens: int = 50,
) -> dict:
    """
    Activation patching: run target_prompt but replace the hidden state
    at a layer with the hidden state from source_prompt.

    If the patched output shifts toward what source_prompt would
    produce, it proves that layer carries the information that
    distinguishes the two prompts.

    Args:
        source_prompt:  Prompt whose activations we borrow.
        target_prompt:  Prompt we generate for.
        layer:          Layer at which to patch.
        max_new_tokens: Tokens to generate.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "patch_activations",
        )

    if max_new_tokens < 1 or max_new_tokens > 1000:
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_new_tokens must be 1-1000, got {max_new_tokens}.",
            "patch_activations",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "patch_activations",
        )

    try:
        result = await asyncio.to_thread(
            _patch_activations_impl,
            state.model,
            state.tokenizer,
            source_prompt,
            target_prompt,
            layer,
        )
        return result

    except Exception as e:
        logger.exception("patch_activations failed")
        return make_error(ToolError.ABLATION_FAILED, str(e), "patch_activations")


def _patch_activations_impl(
    model: Any,
    tokenizer: Any,
    source_prompt: str,
    target_prompt: str,
    layer: int,
) -> dict:
    """Sync implementation of patch_activations."""
    from chuk_lazarus.introspection.interventions import (
        CounterfactualIntervention,
    )

    ci = CounterfactualIntervention(model=model, tokenizer=tokenizer)

    patch_result = ci.patch_run(
        clean_prompt=source_prompt,
        corrupt_prompt=target_prompt,
        patch_layers=[layer],
        patch_positions=[-1],
    )

    result = PatchActivationsResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        patched_layer=layer,
        patched_output=patch_result.patched_output,
        baseline_output=patch_result.corrupt_output,
        source_output=patch_result.clean_output,
        recovery_rate=round(float(patch_result.recovery_rate), 4),
        effect_size=round(float(patch_result.effect_size), 4),
    )
    return result.model_dump()
