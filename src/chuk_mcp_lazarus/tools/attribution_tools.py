"""
Attribution sweep tool: batch logit attribution across multiple prompts.

Runs logit_attribution for each prompt and aggregates per-layer statistics
(mean/std of attention, FFN, and total contributions). Useful for comparing
how different prompts route through the model.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class LayerSummaryEntry(BaseModel):
    """Aggregated attribution statistics for a single layer."""

    layer: int
    mean_attention_logit: float
    mean_ffn_logit: float
    mean_total_logit: float
    std_attention_logit: float
    std_ffn_logit: float
    std_total_logit: float


class PromptSummaryEntry(BaseModel):
    """Per-prompt summary row extracted from logit attribution results."""

    prompt: str
    label: str | None = None
    embedding_logit: float
    net_attention: float = Field(
        ..., description="Sum of attention contributions across all analyzed layers."
    )
    net_ffn: float = Field(..., description="Sum of FFN contributions across all analyzed layers.")
    final_logit: float = Field(..., description="Model's logit for the target token.")
    top_prediction: str = Field(..., description="Target token (model's prediction or specified).")
    probability: float = Field(..., description="Model's probability for the target token.")
    dominant_component: str = Field(
        ..., description="'attention' or 'ffn' based on net contribution."
    )
    top_positive_layer: int
    top_negative_layer: int


class AttributionSweepResult(BaseModel):
    """Result from attribution_sweep."""

    num_prompts: int
    num_layers: int
    target_token: str | None
    normalized: bool
    labels: list[str] | None = None
    per_prompt: list[dict[str, Any]] = Field(
        ..., description="Individual logit_attribution results."
    )
    prompt_summary: list[PromptSummaryEntry] = Field(
        ..., description="Clean per-prompt summary with net attention/FFN/prediction."
    )
    layer_summary: list[LayerSummaryEntry] = Field(
        ..., description="Per-layer mean/std across all prompts."
    )
    dominant_layer: int = Field(..., description="Layer with highest mean |total_logit|.")
    dominant_component: str = Field(
        ..., description="'attention' or 'ffn' based on total contribution."
    )


# ---------------------------------------------------------------------------
# Tool: attribution_sweep
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def attribution_sweep(
    prompts: list[str],
    layers: list[int] | None = None,
    position: int = -1,
    target_token: str | None = None,
    normalized: bool = True,
    labels: list[str] | None = None,
) -> dict:
    """
    Run logit attribution across multiple prompts and aggregate results.

    For each prompt, computes per-layer attention and FFN contributions
    to the target token's logit (same as logit_attribution). Then
    aggregates across prompts: mean and std of each component per layer.

    Useful for finding which layers consistently matter across a class
    of prompts, or for comparing attribution patterns between groups.

    Args:
        prompts:      List of prompts to analyze.
        layers:       Layer indices. None = sample ~12 layers.
        position:     Token position to analyze (-1 = last token).
        target_token: Token to attribute. None = model's top-1 per prompt.
        normalized:   True = normalized deltas (default). False = raw DLA.
        labels:       Optional labels for each prompt (must match length).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "attribution_sweep",
        )

    if not prompts:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 prompt is required.",
            "attribution_sweep",
        )

    if len(prompts) > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 50 prompts allowed.",
            "attribution_sweep",
        )

    if labels is not None and len(labels) != len(prompts):
        return make_error(
            ToolError.INVALID_INPUT,
            f"labels length ({len(labels)}) must match prompts length ({len(prompts)}).",
            "attribution_sweep",
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

    if not layers:
        return make_error(
            ToolError.INVALID_INPUT,
            "layers must not be empty when specified.",
            "attribution_sweep",
        )

    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "attribution_sweep",
        )

    try:
        result = await asyncio.to_thread(
            _attribution_sweep_impl,
            state.model,
            state.config,
            state.tokenizer,
            prompts,
            sorted(layers),
            position,
            target_token,
            normalized,
            num_layers,
            labels,
        )
        return result
    except ValueError as exc:
        return make_error(ToolError.INVALID_INPUT, str(exc), "attribution_sweep")
    except Exception as e:
        logger.exception("attribution_sweep failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "attribution_sweep")


def _attribution_sweep_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    position: int,
    target_token: str | None,
    normalized: bool,
    num_layers: int,
    labels: list[str] | None,
) -> dict:
    """Sync implementation of attribution_sweep."""
    from .residual_tools import _logit_attribution_impl

    # Run attribution for each prompt
    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        attribution = _logit_attribution_impl(
            model,
            config,
            tokenizer,
            prompt,
            layers,
            position,
            target_token,
            normalized,
            num_layers,
        )
        per_prompt.append(attribution)

    # Aggregate per-layer statistics
    layer_data: dict[int, dict[str, list[float]]] = {}
    for layer_idx in layers:
        layer_data[layer_idx] = {
            "attention": [],
            "ffn": [],
            "total": [],
        }

    for prompt_result in per_prompt:
        result_layers = prompt_result.get("layers", [])
        for layer_entry in result_layers:
            lid = layer_entry["layer"]
            if lid in layer_data:
                layer_data[lid]["attention"].append(layer_entry["attention_logit"])
                layer_data[lid]["ffn"].append(layer_entry["ffn_logit"])
                layer_data[lid]["total"].append(layer_entry["total_logit"])

    import numpy as np

    layer_summary: list[LayerSummaryEntry] = []
    for layer_idx in layers:
        d = layer_data[layer_idx]
        attn_arr = np.array(d["attention"], dtype=np.float64) if d["attention"] else np.zeros(1)
        ffn_arr = np.array(d["ffn"], dtype=np.float64) if d["ffn"] else np.zeros(1)
        total_arr = np.array(d["total"], dtype=np.float64) if d["total"] else np.zeros(1)

        layer_summary.append(
            LayerSummaryEntry(
                layer=layer_idx,
                mean_attention_logit=round(float(attn_arr.mean()), 6),
                mean_ffn_logit=round(float(ffn_arr.mean()), 6),
                mean_total_logit=round(float(total_arr.mean()), 6),
                std_attention_logit=round(float(attn_arr.std()), 6),
                std_ffn_logit=round(float(ffn_arr.std()), 6),
                std_total_logit=round(float(total_arr.std()), 6),
            )
        )

    # Build per-prompt summaries
    prompt_summaries: list[PromptSummaryEntry] = []
    for i, prompt_result in enumerate(per_prompt):
        result_layers = prompt_result.get("layers", [])
        net_attn = sum(lay["attention_logit"] for lay in result_layers)
        net_ffn = sum(lay["ffn_logit"] for lay in result_layers)
        summary_data = prompt_result.get("summary", {})

        prompt_summaries.append(
            PromptSummaryEntry(
                prompt=prompts[i],
                label=labels[i] if labels else None,
                embedding_logit=prompt_result.get("embedding_logit", 0.0),
                net_attention=round(net_attn, 6),
                net_ffn=round(net_ffn, 6),
                final_logit=prompt_result.get("model_logit", 0.0),
                top_prediction=prompt_result.get("target_token", ""),
                probability=prompt_result.get("model_probability", 0.0),
                dominant_component="attention" if net_attn > net_ffn else "ffn",
                top_positive_layer=summary_data.get("top_positive_layer", -1),
                top_negative_layer=summary_data.get("top_negative_layer", -1),
            )
        )

    # Dominant layer = highest mean |total_logit|
    dominant_layer = layers[0]
    max_abs_total = 0.0
    for entry in layer_summary:
        abs_total = abs(entry.mean_total_logit)
        if abs_total > max_abs_total:
            max_abs_total = abs_total
            dominant_layer = entry.layer

    # Dominant component
    total_attn = sum(e.mean_attention_logit for e in layer_summary)
    total_ffn = sum(e.mean_ffn_logit for e in layer_summary)
    dominant_component = "attention" if total_attn > total_ffn else "ffn"

    result = AttributionSweepResult(
        num_prompts=len(prompts),
        num_layers=len(layers),
        target_token=target_token,
        normalized=normalized,
        labels=labels,
        per_prompt=per_prompt,
        prompt_summary=prompt_summaries,
        layer_summary=layer_summary,
        dominant_layer=dominant_layer,
        dominant_component=dominant_component,
    )
    return result.model_dump(exclude_none=True)
