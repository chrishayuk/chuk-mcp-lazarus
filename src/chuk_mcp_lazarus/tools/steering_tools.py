"""
Steering tools: compute_steering_vector, steer_and_generate, list_steering_vectors.

Computes contrastive activation vectors (mean-difference between two sets
of activations) and applies them to the model's residual stream during
generation. This allows redirecting model behaviour -- e.g. steering a
translation prompt from French output to German output.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, Field

from .._generate import generate_text
from .._serialize import cosine_similarity_matrix, hidden_state_to_list
from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp
from ..steering_store import SteeringVectorRegistry, VectorMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
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
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_activation(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    token_position: int = -1,
) -> list[float]:
    """Extract activation vector at one layer for one prompt."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    hooks = ModelHooks(model, model_config=config)
    hooks.configure(CaptureConfig(layers=[layer], capture_hidden_states=True))
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)
    return hidden_state_to_list(hooks.state.hidden_states[layer], position=token_position)


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
# Tools
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
        # Extract activations for both sets
        pos_vecs: list[list[float]] = []
        for prompt in positive_prompts:
            vec = _extract_activation(
                state.model, state.config, state.tokenizer,
                prompt, layer, token_position,
            )
            pos_vecs.append(vec)

        neg_vecs: list[list[float]] = []
        for prompt in negative_prompts:
            vec = _extract_activation(
                state.model, state.config, state.tokenizer,
                prompt, layer, token_position,
            )
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
            np.dot(pos_mean, neg_mean)
            / (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean) + 1e-8)
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

    except Exception as e:
        logger.exception("compute_steering_vector failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "compute_steering_vector")


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

    try:
        # Baseline generation (no steering)
        baseline_text, baseline_tokens = generate_text(
            state.model, state.tokenizer,
            prompt, max_new_tokens=max_new_tokens,
        )

        # Steered generation
        steered_text, steered_tokens = _generate_steered(
            state.model, state.tokenizer, state.config,
            prompt, direction, meta.layer,
            alpha=alpha, max_new_tokens=max_new_tokens,
        )

        result = SteerAndGenerateResult(
            prompt=prompt,
            vector_name=vector_name,
            alpha=alpha,
            layer=meta.layer,
            steered_output=steered_text,
            baseline_output=baseline_text,
            steered_tokens=steered_tokens,
            baseline_tokens=baseline_tokens,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("steer_and_generate failed")
        return make_error(ToolError.GENERATION_FAILED, str(e), "steer_and_generate")


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
