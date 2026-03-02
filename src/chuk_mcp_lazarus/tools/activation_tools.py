"""
Activation extraction tools: extract_activations, compare_activations.

Wraps chuk-lazarus ModelHooks to capture hidden states at specific
layers and token positions. All heavy lifting is delegated to the
hooks infrastructure; this module translates MCP parameters to
CaptureConfig and serialises the results.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
from pydantic import BaseModel, Field

from .._serialize import (
    cosine_similarity_matrix,
    hidden_state_to_list,
    pca_2d,
)
from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """Result from extract_activations."""

    prompt: str
    token_position: int
    token_text: str
    num_tokens: int
    activations: dict[str, list[float]] = Field(
        ..., description="Layer index (str) -> activation vector."
    )
    attention_shapes: dict[str, list[int]] | None = Field(
        None, description="Layer index (str) -> attention tensor shape, if captured."
    )


class ComparisonResult(BaseModel):
    """Result from compare_activations."""

    layer: int
    prompts: list[str]
    cosine_similarity_matrix: list[list[float]]
    pca_2d: list[list[float]]
    centroid_distance: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(tokenizer: Any, prompt: str) -> mx.array:
    """Encode a prompt and return MLX token IDs."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    return mx.array(ids)


def _token_text(tokenizer: Any, token_ids: mx.array, position: int) -> str:
    """Decode a single token at the given position."""
    ids_list = token_ids.tolist()
    if isinstance(ids_list[0], list):
        ids_list = ids_list[0]
    idx = position if position >= 0 else len(ids_list) + position
    idx = max(0, min(idx, len(ids_list) - 1))
    return tokenizer.decode([ids_list[idx]])


def _run_hooks(
    model: Any,
    config: Any,
    input_ids: mx.array,
    layers: list[int],
    capture_attention: bool = False,
) -> Any:
    """Create ModelHooks, run forward pass, return CapturedState."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    hooks = ModelHooks(model, model_config=config)
    hooks.configure(
        CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
            capture_attention_weights=capture_attention,
        )
    )
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)
    return hooks.state


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def extract_activations(
    prompt: str,
    layers: list[int],
    token_position: int = -1,
    capture_attention: bool = False,
) -> dict:
    """
    Run a forward pass and return hidden-state activations at the
    specified layers for one token position.

    Args:
        prompt:            Input text.
        layers:            Layer indices (0-indexed).
        token_position:    Which token's activation to return.
                           -1 = last token (default).
        capture_attention: Also return attention weights (memory-heavy).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "extract_activations",
        )

    num_layers = state.metadata.num_layers
    out_of_range = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "extract_activations",
        )

    try:
        input_ids = _tokenize(state.tokenizer, prompt)
        num_tokens = input_ids.shape[-1]
        tok_text = _token_text(state.tokenizer, input_ids, token_position)

        captured = _run_hooks(
            state.model, state.config, input_ids, layers,
            capture_attention=capture_attention,
        )

        activations: dict[str, list[float]] = {}
        for layer_idx in layers:
            if layer_idx in captured.hidden_states:
                activations[str(layer_idx)] = hidden_state_to_list(
                    captured.hidden_states[layer_idx], position=token_position
                )

        attention_shapes: dict[str, list[int]] | None = None
        if capture_attention and captured.attention_weights:
            attention_shapes = {
                str(k): list(v.shape) for k, v in captured.attention_weights.items()
            }

        result = ExtractionResult(
            prompt=prompt,
            token_position=token_position,
            token_text=tok_text,
            num_tokens=int(num_tokens),
            activations=activations,
            attention_shapes=attention_shapes,
        )
        return result.model_dump(exclude_none=True)

    except Exception as e:
        logger.exception("extract_activations failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "extract_activations")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def compare_activations(
    prompts: list[str],
    layer: int,
    token_position: int = -1,
) -> dict:
    """
    Extract activations at a single layer for multiple prompts.
    Returns pairwise cosine similarities and a 2-D PCA projection.

    Useful for checking whether prompts in different languages converge
    to the same representation at a given layer.

    Args:
        prompts:        2-8 input strings.
        layer:          Layer index.
        token_position: Token position (default: last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compare_activations",
        )

    if not 2 <= len(prompts) <= 8:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need 2-8 prompts, got {len(prompts)}.",
            "compare_activations",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "compare_activations",
        )

    try:
        import numpy as np

        vectors: list[list[float]] = []
        for prompt in prompts:
            input_ids = _tokenize(state.tokenizer, prompt)
            captured = _run_hooks(state.model, state.config, input_ids, [layer])
            vec = hidden_state_to_list(
                captured.hidden_states[layer], position=token_position
            )
            vectors.append(vec)

        sim_matrix = cosine_similarity_matrix(vectors)
        projection = pca_2d(vectors)

        n = len(vectors)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(1.0 - sim_matrix[i][j])
        centroid_dist = float(np.mean(distances)) if distances else 0.0

        result = ComparisonResult(
            layer=layer,
            prompts=prompts,
            cosine_similarity_matrix=sim_matrix,
            pca_2d=projection,
            centroid_distance=centroid_dist,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("compare_activations failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "compare_activations")
