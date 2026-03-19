"""
Steering tools: compute_steering_vector, steer_and_generate, list_steering_vectors.

Computes contrastive activation vectors (mean-difference between two sets
of activations) and applies them to the model's residual stream during
generation. This allows redirecting model behaviour -- e.g. steering a
translation prompt from French output to German output.

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

from ..._extraction import extract_activation_at_layer
from ..._generate import generate_text
from ..._serialize import cosine_similarity_matrix
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ...steering_store import SteeringVectorRegistry, VectorMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models (steering)
# ---------------------------------------------------------------------------


class ComputeVectorResult(BaseModel):
    """Result from compute_steering_vector."""

    vector_name: str
    layer: int
    vector_norm: float
    cosine_similarity_within_positive: float
    cosine_similarity_within_negative: float
    separability_score: float
    num_positive: int
    num_negative: int


class SteerAndGenerateResult(BaseModel):
    """Result from steer_and_generate."""

    prompt: str
    vector_name: str
    alpha: float
    layer: int
    steered_output: str
    baseline_output: str
    steered_tokens: int
    baseline_tokens: int


class ListVectorsResult(BaseModel):
    """Result from list_steering_vectors."""

    vectors: list[dict[str, Any]]
    count: int


# ---------------------------------------------------------------------------
# Internal helpers (steering)
# ---------------------------------------------------------------------------


def _mean_vector(vectors: list[list[float]]) -> np.ndarray:
    """Compute the mean of a list of activation vectors."""
    return np.mean(np.array(vectors, dtype=np.float32), axis=0)


def _mean_pairwise_similarity(vectors: list[list[float]]) -> float:
    """Average pairwise cosine similarity within a group."""
    if len(vectors) < 2:
        return 1.0
    sim = cosine_similarity_matrix(vectors)
    n = len(vectors)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim[i][j]
            count += 1
    return total / count if count > 0 else 1.0


def _generate_steered(
    model: Any,
    tokenizer: Any,
    config: Any,
    prompt: str,
    direction: np.ndarray,
    layer: int,
    alpha: float = 20.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> tuple[str, int]:
    """Generate text with a steering vector applied via ActivationSteering."""
    from chuk_lazarus.introspection.steering.core import ActivationSteering
    from chuk_lazarus.introspection.steering.config import SteeringConfig

    steerer = ActivationSteering(model=model, tokenizer=tokenizer)
    steerer.add_direction(layer=layer, direction=direction, name="mcp_steer")

    steering_config = SteeringConfig(
        layers=[layer],
        coefficient=alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    output = steerer.generate(prompt, config=steering_config)
    # Count tokens in generated portion
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    output_ids = tokenizer.encode(prompt + output, add_special_tokens=True)
    num_new = len(output_ids) - len(prompt_ids)

    return output, max(num_new, 0)


# ---------------------------------------------------------------------------
# Tools (steering)
# ---------------------------------------------------------------------------


@mcp.tool()
async def compute_steering_vector(
    vector_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int = -1,
) -> dict:
    """
    Compute a steering vector as the mean difference between two sets
    of activations (contrastive activation addition).

    The vector points from the negative centroid to the positive
    centroid. Adding it to the residual stream during generation
    pushes outputs toward the positive class.

    Args:
        vector_name:       Name to store this vector under.
        layer:             Layer to compute the vector at.
        positive_prompts:  Target direction (e.g. German sentences).
        negative_prompts:  Source direction (e.g. English sentences).
        token_position:    Token position (default: last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "compute_steering_vector",
        )

    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "compute_steering_vector",
        )

    if len(positive_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need at least 2 positive prompts, got {len(positive_prompts)}.",
            "compute_steering_vector",
        )

    if len(negative_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need at least 2 negative prompts, got {len(negative_prompts)}.",
            "compute_steering_vector",
        )

    try:
        result = await asyncio.to_thread(
            _compute_steering_vector_impl,
            state.model,
            state.config,
            state.tokenizer,
            vector_name,
            layer,
            positive_prompts,
            negative_prompts,
            token_position,
        )
        return result

    except Exception as e:
        logger.exception("compute_steering_vector failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "compute_steering_vector")


def _compute_steering_vector_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    vector_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int,
) -> dict:
    """Sync implementation of compute_steering_vector."""
    pos_vecs: list[list[float]] = []
    for prompt in positive_prompts:
        vec = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        pos_vecs.append(vec)

    neg_vecs: list[list[float]] = []
    for prompt in negative_prompts:
        vec = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        neg_vecs.append(vec)

    # Compute steering vector: mean_positive - mean_negative
    pos_mean = _mean_vector(pos_vecs)
    neg_mean = _mean_vector(neg_vecs)
    direction = pos_mean - neg_mean

    vector_norm = float(np.linalg.norm(direction))
    sim_pos = _mean_pairwise_similarity(pos_vecs)
    sim_neg = _mean_pairwise_similarity(neg_vecs)

    # Separability: cosine distance between centroids
    cos_sim = float(
        np.dot(pos_mean, neg_mean) / (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean) + 1e-8)
    )
    separability = 1.0 - cos_sim

    # Store
    metadata = VectorMetadata(
        name=vector_name,
        layer=layer,
        vector_norm=vector_norm,
        separability_score=separability,
        num_positive=len(positive_prompts),
        num_negative=len(negative_prompts),
        computed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    SteeringVectorRegistry.get().store(vector_name, direction, metadata)

    result = ComputeVectorResult(
        vector_name=vector_name,
        layer=layer,
        vector_norm=round(vector_norm, 4),
        cosine_similarity_within_positive=round(sim_pos, 4),
        cosine_similarity_within_negative=round(sim_neg, 4),
        separability_score=round(separability, 4),
        num_positive=len(positive_prompts),
        num_negative=len(negative_prompts),
    )
    return result.model_dump()


@mcp.tool()
async def steer_and_generate(
    prompt: str,
    vector_name: str,
    alpha: float = 20.0,
    max_new_tokens: int = 100,
) -> dict:
    """
    Generate text with a steering vector applied to the residual stream.

    Runs both a steered and baseline generation for comparison. The
    alpha parameter controls steering strength -- start at 10-20 and
    adjust based on results.

    Args:
        prompt:         Input prompt.
        vector_name:    Name of a stored steering vector.
        alpha:          Steering strength (start at 10-20).
        max_new_tokens: Maximum tokens to generate.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "steer_and_generate",
        )

    registry = SteeringVectorRegistry.get()
    entry = registry.fetch(vector_name)
    if entry is None:
        return make_error(
            ToolError.VECTOR_NOT_FOUND,
            f"Vector '{vector_name}' not found. Use list_steering_vectors() to see available vectors.",
            "steer_and_generate",
        )

    direction, meta = entry

    if max_new_tokens < 1 or max_new_tokens > 1000:
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_new_tokens must be 1-1000, got {max_new_tokens}.",
            "steer_and_generate",
        )

    try:
        result = await asyncio.to_thread(
            _steer_and_generate_impl,
            state.model,
            state.tokenizer,
            state.config,
            prompt,
            vector_name,
            direction,
            meta.layer,
            alpha,
            max_new_tokens,
        )
        return result

    except Exception as e:
        logger.exception("steer_and_generate failed")
        return make_error(ToolError.GENERATION_FAILED, str(e), "steer_and_generate")


def _steer_and_generate_impl(
    model: Any,
    tokenizer: Any,
    config: Any,
    prompt: str,
    vector_name: str,
    direction: np.ndarray,
    layer: int,
    alpha: float,
    max_new_tokens: int,
) -> dict:
    """Sync implementation of steer_and_generate."""
    baseline_text, baseline_tokens = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
    )

    steered_text, steered_tokens = _generate_steered(
        model,
        tokenizer,
        config,
        prompt,
        direction,
        layer,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
    )

    result = SteerAndGenerateResult(
        prompt=prompt,
        vector_name=vector_name,
        alpha=alpha,
        layer=layer,
        steered_output=steered_text,
        baseline_output=baseline_text,
        steered_tokens=steered_tokens,
        baseline_tokens=baseline_tokens,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_steering_vectors() -> dict:
    """
    List all steering vectors in memory.

    Returns summary metadata for every computed vector, including
    name, layer, vector_norm, separability_score, and computed_at.
    """
    try:
        registry = SteeringVectorRegistry.get()
        dump = registry.dump()
        return dump.model_dump()
    except Exception as e:
        logger.exception("list_steering_vectors failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "list_steering_vectors")


# ---------------------------------------------------------------------------
# Result models (direction)
# ---------------------------------------------------------------------------

_ALLOWED_METHODS = {"diff_means", "lda", "probe_weights", "pca"}


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
# Tool (direction)
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
# Implementation (direction)
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
