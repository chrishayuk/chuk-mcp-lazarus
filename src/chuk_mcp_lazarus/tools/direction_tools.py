"""
Direction extraction tools: extract_direction.

Extracts linear directions in activation space that separate two classes
of prompts. Supports multiple methods: difference of means, LDA, PCA, and
logistic regression probe weights. Extracted directions are stored in the
SteeringVectorRegistry for immediate use with steer_and_generate().
"""

import asyncio
import datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .._extraction import extract_activation_at_layer
from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp
from ..steering_store import SteeringVectorRegistry, VectorMetadata

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = {"diff_means", "lda", "probe_weights", "pca"}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class DirectionResult(BaseModel):
    """Result from extract_direction."""

    direction_name: str
    layer: int
    method: str
    separation_score: float = Field(
        ..., description="Cohen's d between positive and negative projections."
    )
    accuracy: float = Field(..., description="Classification accuracy using midpoint threshold.")
    mean_projection_positive: float
    mean_projection_negative: float
    positive_label: str
    negative_label: str
    vector_norm: float
    num_positive: int
    num_negative: int
    stored_as_steering_vector: bool = Field(
        True,
        description="Whether the direction was stored in the steering vector registry.",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def extract_direction(
    direction_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    method: str = "diff_means",
    positive_label: str = "positive",
    negative_label: str = "negative",
    token_position: int = -1,
) -> dict:
    """
    Extract a linear direction in activation space that separates
    two classes of prompts.

    The direction can be used for activation steering or for
    understanding what concepts are represented at a given layer.

    Supports multiple extraction methods:
    - "diff_means": Difference of means (default, fastest)
    - "lda": Linear Discriminant Analysis (maximizes separation)
    - "probe_weights": Logistic regression probe weights
    - "pca": First principal component of centered activations

    The extracted direction is automatically stored as a steering
    vector and can be used with steer_and_generate().

    Args:
        direction_name:    Name to store this direction under.
        layer:             Layer to extract the direction at.
        positive_prompts:  Prompts representing the positive class (min 2).
        negative_prompts:  Prompts representing the negative class (min 2).
        method:            Extraction method (default: "diff_means").
        positive_label:    Human-readable label for positive class.
        negative_label:    Human-readable label for negative class.
        token_position:    Token position to extract (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "extract_direction",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "extract_direction",
        )

    if method not in _ALLOWED_METHODS:
        return make_error(
            ToolError.INVALID_INPUT,
            f"method must be one of {sorted(_ALLOWED_METHODS)}, got '{method}'.",
            "extract_direction",
        )

    if len(positive_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 2 positive_prompts are required.",
            "extract_direction",
        )

    if len(negative_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 2 negative_prompts are required.",
            "extract_direction",
        )

    try:
        result = await asyncio.to_thread(
            _extract_direction_impl,
            state.model,
            state.config,
            state.tokenizer,
            direction_name,
            layer,
            positive_prompts,
            negative_prompts,
            method,
            positive_label,
            negative_label,
            token_position,
        )
        return result
    except Exception as e:
        logger.exception("extract_direction failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "extract_direction")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _extract_direction_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    direction_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    method: str,
    positive_label: str,
    negative_label: str,
    token_position: int,
) -> dict:
    """Sync implementation of extract_direction."""
    import mlx.core as mx
    from chuk_lazarus.introspection.circuit.collector import CollectedActivations
    from chuk_lazarus.introspection.circuit.directions import (
        DirectionExtractor,
        DirectionMethod,
    )

    # Step 1: Extract activations for all prompts
    pos_vecs: list[list[float]] = []
    for prompt in positive_prompts:
        vec = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        pos_vecs.append(vec)

    neg_vecs: list[list[float]] = []
    for prompt in negative_prompts:
        vec = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        neg_vecs.append(vec)

    # Step 2: Build CollectedActivations
    all_vecs = pos_vecs + neg_vecs
    labels = [1] * len(positive_prompts) + [0] * len(negative_prompts)
    hidden_array = mx.array(np.array(all_vecs, dtype=np.float32))

    collected = CollectedActivations(
        hidden_states={layer: hidden_array},
        labels=labels,
        prompts=positive_prompts + negative_prompts,
        dataset_label_names={1: positive_label, 0: negative_label},
        dataset_name=direction_name,
        hidden_size=len(all_vecs[0]),
    )

    # Step 3: Extract direction
    method_enum = DirectionMethod(method)
    extractor = DirectionExtractor(collected)
    extracted = extractor.extract_direction(
        layer=layer,
        method=method_enum,
        positive_label=1,
        negative_label=0,
    )

    # Step 4: Store in SteeringVectorRegistry
    direction_np = np.array(extracted.direction, dtype=np.float32)
    vector_norm = float(np.linalg.norm(direction_np))

    # Compute cosine separability for VectorMetadata
    pos_mean = np.mean(np.array(pos_vecs, dtype=np.float32), axis=0)
    neg_mean = np.mean(np.array(neg_vecs, dtype=np.float32), axis=0)
    cos_sim = float(
        np.dot(pos_mean, neg_mean) / (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean) + 1e-8)
    )
    separability = 1.0 - cos_sim

    metadata = VectorMetadata(
        name=direction_name,
        layer=layer,
        vector_norm=vector_norm,
        separability_score=separability,
        num_positive=len(positive_prompts),
        num_negative=len(negative_prompts),
        computed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    SteeringVectorRegistry.get().store(direction_name, direction_np, metadata)

    # Step 5: Build result
    result = DirectionResult(
        direction_name=direction_name,
        layer=layer,
        method=method,
        separation_score=round(extracted.separation_score, 4),
        accuracy=round(extracted.accuracy, 4),
        mean_projection_positive=round(extracted.mean_projection_positive, 4),
        mean_projection_negative=round(extracted.mean_projection_negative, 4),
        positive_label=positive_label,
        negative_label=negative_label,
        vector_norm=round(vector_norm, 4),
        num_positive=len(positive_prompts),
        num_negative=len(negative_prompts),
        stored_as_steering_vector=True,
    )
    return result.model_dump()
