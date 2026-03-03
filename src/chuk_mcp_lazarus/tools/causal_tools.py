"""
Causal tracing tools: trace_token, full_causal_trace.

Causal tracing identifies which components are causally responsible for
a specific prediction.  trace_token ablates each layer and measures the
effect on target-token probability.  full_causal_trace builds a
position × layer patching grid (Meng et al. 2022 style).  Both wrap
the CounterfactualIntervention backend from chuk-lazarus.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
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
    baseline_prob: float = Field(
        ..., description="Probability of target token with no ablation."
    )
    layer_effects: list[LayerEffect] = Field(
        ..., description="Per-layer causal effect on target token."
    )
    critical_layers: list[int] = Field(
        ...,
        description="Layers above effect_threshold, sorted by effect magnitude.",
    )
    peak_layer: int = Field(
        ..., description="Layer with the largest causal effect."
    )
    peak_effect: float = Field(
        ..., description="Effect magnitude at the peak layer."
    )
    effect_threshold: float = Field(
        ..., description="Threshold used to determine critical layers."
    )


class FullCausalTraceResult(BaseModel):
    """Result from full_causal_trace (position × layer heatmap)."""

    prompt: str
    target_token: str
    tokens: list[str] = Field(
        ..., description="All tokens in the prompt."
    )
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
    critical_layers: list[int] = Field(
        ..., description="Layers with the highest causal effects."
    )
    num_positions: int = Field(
        ..., description="Number of token positions (rows in the grid)."
    )
    num_layers_tested: int = Field(
        ..., description="Number of layers tested (columns in the grid)."
    )


# ---------------------------------------------------------------------------
# Tools
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

    out_of_range = [l for l in layers if l < 0 or l >= num_layers]
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
    Full causal tracing: build a position × layer heatmap of causal
    importance (Meng et al. 2022 style).

    For each (position, layer) pair, patches the clean activation into
    a corrupt forward pass and measures how much of the target token's
    probability is recovered.  High recovery at a cell means that
    position's representation at that layer carries information needed
    for the prediction.

    The result is a 2D grid suitable for heatmap visualization.
    Dimensions: [num_positions × num_layers_tested].

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

    out_of_range = [l for l in layers if l < 0 or l >= num_layers]
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
            [round(float(e), 6) for e in pos_effects]
            for pos_effects in trace_result.effects
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
        return make_error(
            ToolError.ABLATION_FAILED, str(e), "full_causal_trace"
        )
