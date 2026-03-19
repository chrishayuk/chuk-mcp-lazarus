"""
Causal tracing tools: trace_token, full_causal_trace.

Causal tracing identifies which components are causally responsible for
a specific prediction.  trace_token ablates each layer and measures the
effect on target-token probability.  full_causal_trace builds a
position x layer patching grid (Meng et al. 2022 style).  Both wrap
the CounterfactualIntervention backend from chuk-lazarus.

Ablation tools: ablate_layers, patch_activations.

Ablation zeroes out layer weights to test causal necessity.
Activation patching swaps hidden states from one prompt into another
to measure information flow. Both are key tools for causal
mechanistic interpretability.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..._generate import generate_text
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models (causal)
# ---------------------------------------------------------------------------


class LayerEffect(BaseModel):
    """Effect of ablating a single layer on target token probability."""

    layer: int
    effect: float = Field(
        ...,
        description=(
            "Change in target token probability when this layer is ablated. "
            "Positive means the layer was helping the prediction."
        ),
    )


class TraceTokenResult(BaseModel):
    """Result from trace_token (causal tracing)."""

    prompt: str
    target_token: str
    target_token_id: int
    baseline_prob: float = Field(..., description="Probability of target token with no ablation.")
    layer_effects: list[LayerEffect] = Field(
        ..., description="Per-layer causal effect on target token."
    )
    critical_layers: list[int] = Field(
        ...,
        description="Layers above effect_threshold, sorted by effect magnitude.",
    )
    peak_layer: int = Field(..., description="Layer with the largest causal effect.")
    peak_effect: float = Field(..., description="Effect magnitude at the peak layer.")
    effect_threshold: float = Field(..., description="Threshold used to determine critical layers.")


class FullCausalTraceResult(BaseModel):
    """Result from full_causal_trace (position x layer heatmap)."""

    prompt: str
    target_token: str
    tokens: list[str] = Field(..., description="All tokens in the prompt.")
    effects: list[list[float]] = Field(
        ...,
        description=(
            "2D effect grid [position][layer].  Each value is the recovery "
            "rate when patching the clean activation at that (position, layer) "
            "into a corrupt forward pass."
        ),
    )
    critical_positions: list[int] = Field(
        ..., description="Token positions with the highest causal effects."
    )
    critical_layers: list[int] = Field(..., description="Layers with the highest causal effects.")
    num_positions: int = Field(..., description="Number of token positions (rows in the grid).")
    num_layers_tested: int = Field(
        ..., description="Number of layers tested (columns in the grid)."
    )


# ---------------------------------------------------------------------------
# Tools (causal)
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def trace_token(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
    effect_threshold: float = 0.1,
) -> dict:
    """
    Causal tracing: find which layers are causally responsible for
    predicting a specific token.

    Ablates each layer one at a time and measures how much the target
    token's probability drops.  Layers where ablation causes the largest
    drop are causally important for the prediction.

    This is the interventional complement to track_token (which is
    observational).  track_token shows where a prediction emerges;
    trace_token proves which layers are *necessary* for it.

    Args:
        prompt:           Input text.
        token:            Target token to trace (e.g. "Paris").
        layers:           Layer indices to test.  None = all layers.
        effect_threshold: Minimum effect to be considered "critical" (0-1).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "trace_token",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        layers = list(range(num_layers))

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "trace_token",
        )

    if effect_threshold < 0 or effect_threshold > 1:
        return make_error(
            ToolError.INVALID_INPUT,
            f"effect_threshold must be 0-1, got {effect_threshold}.",
            "trace_token",
        )

    try:
        from chuk_lazarus.introspection.interventions import (
            CounterfactualIntervention,
        )

        ci = CounterfactualIntervention(
            model=state.model,
            tokenizer=state.tokenizer,
        )

        trace_result = await asyncio.to_thread(
            ci.trace_token,
            prompt=prompt,
            target_token=token,
            layers=layers,
            effect_threshold=effect_threshold,
        )

        layer_effects = [
            LayerEffect(layer=int(layer), effect=round(float(effect), 6))
            for layer, effect in trace_result.layer_effects
        ]

        result = TraceTokenResult(
            prompt=trace_result.prompt,
            target_token=trace_result.target_token,
            target_token_id=int(trace_result.target_token_id),
            baseline_prob=round(float(trace_result.baseline_prob), 6),
            layer_effects=layer_effects,
            critical_layers=list(trace_result.critical_layers),
            peak_layer=int(trace_result.peak_layer),
            peak_effect=round(float(trace_result.peak_effect), 6),
            effect_threshold=effect_threshold,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("trace_token failed")
        return make_error(ToolError.ABLATION_FAILED, str(e), "trace_token")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def full_causal_trace(
    prompt: str,
    token: str,
    corrupt_prompt: str | None = None,
    layers: list[int] | None = None,
) -> dict:
    """
    Full causal tracing: build a position x layer heatmap of causal
    importance (Meng et al. 2022 style).

    For each (position, layer) pair, patches the clean activation into
    a corrupt forward pass and measures how much of the target token's
    probability is recovered.  High recovery at a cell means that
    position's representation at that layer carries information needed
    for the prediction.

    The result is a 2D grid suitable for heatmap visualization.
    Dimensions: [num_positions x num_layers_tested].

    Args:
        prompt:         Clean input text.
        token:          Target token to trace (e.g. "Paris").
        corrupt_prompt: Corrupted version of the prompt.  If None the
                        backend auto-corrupts by adding noise.
        layers:         Layer indices to test.  None = all layers.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "full_causal_trace",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        layers = list(range(num_layers))

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "full_causal_trace",
        )

    try:
        from chuk_lazarus.introspection.interventions import (
            CounterfactualIntervention,
        )

        ci = CounterfactualIntervention(
            model=state.model,
            tokenizer=state.tokenizer,
        )

        trace_result = await asyncio.to_thread(
            ci.full_causal_trace,
            prompt=prompt,
            target_token=token,
            corrupt_prompt=corrupt_prompt,
            layers=layers,
        )

        effects = [
            [round(float(e), 6) for e in pos_effects] for pos_effects in trace_result.effects
        ]

        result = FullCausalTraceResult(
            prompt=trace_result.prompt,
            target_token=trace_result.target_token,
            tokens=list(trace_result.tokens),
            effects=effects,
            critical_positions=list(trace_result.critical_positions),
            critical_layers=list(trace_result.critical_layers),
            num_positions=len(effects),
            num_layers_tested=len(layers),
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("full_causal_trace failed")
        return make_error(ToolError.ABLATION_FAILED, str(e), "full_causal_trace")


# ---------------------------------------------------------------------------
# Result models (ablation)
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
# Internal helpers (ablation)
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
# Tools (ablation)
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
