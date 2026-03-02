"""
Generation and prediction tools: generate_text, predict_next_token,
tokenize, logit_lens.

These tools expose the model's basic I/O capabilities that are essential
for an AI agent to understand *what* a model produces, complementing
the interpretability tools that explain *why* it produces it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import mlx.core as mx
from pydantic import BaseModel, Field

from ..errors import ToolError, make_error
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class GenerateResult(BaseModel):
    """Result from generate_text."""

    prompt: str
    output: str
    num_tokens: int
    temperature: float
    max_new_tokens: int


class PredictionEntry(BaseModel):
    """A single next-token prediction."""

    token: str
    token_id: int
    probability: float
    log_probability: float


class PredictResult(BaseModel):
    """Result from predict_next_token."""

    prompt: str
    num_input_tokens: int
    predictions: list[PredictionEntry]


class TokenInfo(BaseModel):
    """Info for a single token."""

    position: int
    token_id: int
    token_text: str


class TokenizeResult(BaseModel):
    """Result from tokenize."""

    text: str
    num_tokens: int
    tokens: list[TokenInfo]
    token_ids: list[int]


class LayerPredictionEntry(BaseModel):
    """Top-k predictions at a single layer."""

    layer: int
    top_tokens: list[str]
    top_probabilities: list[float]
    top_token_ids: list[int]


class LogitLensResult(BaseModel):
    """Result from logit_lens."""

    prompt: str
    token_position: int
    token_text: str
    num_layers_analyzed: int
    predictions: list[LayerPredictionEntry]
    summary: dict = Field(
        ..., description="Where the final prediction first emerges."
    )


class TokenLayerEntry(BaseModel):
    """Probability and rank of a tracked token at one layer."""

    layer: int
    probability: float
    rank: int
    is_top1: bool


class TrackTokenResult(BaseModel):
    """Result from track_token."""

    prompt: str
    target_token: str
    target_token_id: int
    token_position: int
    token_text: str
    layers: list[TokenLayerEntry]
    emergence_layer: int | None = Field(
        None, description="First layer where target is top-1 prediction."
    )
    peak_layer: int
    peak_probability: float


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> dict:
    """
    Generate text from the loaded model.

    Uses greedy decoding when temperature is 0, otherwise samples.
    This is the basic "what does the model say?" tool -- essential for
    seeing actual model outputs before and after interpretability work.

    Args:
        prompt:         Input text.
        max_new_tokens: Maximum tokens to generate (default: 100).
        temperature:    Sampling temperature. 0 = greedy (default).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "generate_text",
        )

    if max_new_tokens < 1 or max_new_tokens > 1000:
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_new_tokens must be 1-1000, got {max_new_tokens}.",
            "generate_text",
        )

    try:
        from .._generate import generate_text as _gen

        output, num_tokens = await asyncio.to_thread(
            _gen, state.model, state.tokenizer, prompt,
            max_new_tokens, temperature,
        )

        result = GenerateResult(
            prompt=prompt,
            output=output,
            num_tokens=num_tokens,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("generate_text failed")
        return make_error(ToolError.GENERATION_FAILED, str(e), "generate_text")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def predict_next_token(
    prompt: str,
    top_k: int = 10,
) -> dict:
    """
    Get the model's top-k next-token predictions with probabilities.

    Shows what the model thinks comes next without generating text.
    Useful for understanding model confidence and comparing how
    different prompts affect the prediction distribution.

    Args:
        prompt: Input text.
        top_k:  Number of top predictions to return (default: 10).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "predict_next_token",
        )

    if top_k < 1 or top_k > 100:
        return make_error(
            ToolError.INVALID_INPUT,
            f"top_k must be 1-100, got {top_k}.",
            "predict_next_token",
        )

    try:
        predictions = await asyncio.to_thread(
            _predict_next, state.model, state.tokenizer, prompt, top_k,
        )
        return predictions

    except Exception as e:
        logger.exception("predict_next_token failed")
        return make_error(ToolError.GENERATION_FAILED, str(e), "predict_next_token")


def _predict_next(model: Any, tokenizer: Any, prompt: str, top_k: int) -> dict:
    """Compute top-k next-token predictions."""
    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_input = input_ids.shape[-1]

    logits = model(input_ids[None, :] if input_ids.ndim == 1 else input_ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    if hasattr(logits, "logits"):
        logits = logits.logits

    next_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]

    probs = mx.softmax(next_logits, axis=-1)
    log_probs = mx.log(probs + 1e-10)

    sorted_indices = mx.argsort(probs)[::-1][:top_k]
    mx.eval(sorted_indices, probs, log_probs)

    top_ids = sorted_indices.tolist()
    top_probs = probs[sorted_indices].tolist()
    top_log_probs = log_probs[sorted_indices].tolist()

    entries = []
    for tid, p, lp in zip(top_ids, top_probs, top_log_probs):
        token_text = tokenizer.decode([tid])
        entries.append(PredictionEntry(
            token=token_text,
            token_id=tid,
            probability=round(p, 6),
            log_probability=round(lp, 4),
        ))

    result = PredictResult(
        prompt=prompt,
        num_input_tokens=int(num_input),
        predictions=entries,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def tokenize(
    text: str,
) -> dict:
    """
    Show how text is tokenized by the loaded model's tokenizer.

    Essential for understanding attention patterns (which token is being
    attended to) and debugging multi-token words. Returns token IDs,
    decoded text for each token, and positions.

    Args:
        text: Input text to tokenize.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "tokenize",
        )

    try:
        token_ids = state.tokenizer.encode(text, add_special_tokens=True)

        tokens = []
        for i, tid in enumerate(token_ids):
            decoded = state.tokenizer.decode([tid])
            tokens.append(TokenInfo(
                position=i,
                token_id=tid,
                token_text=decoded,
            ))

        result = TokenizeResult(
            text=text,
            num_tokens=len(token_ids),
            tokens=tokens,
            token_ids=token_ids,
        )
        return result.model_dump()

    except Exception as e:
        logger.exception("tokenize failed")
        return make_error(ToolError.GENERATION_FAILED, str(e), "tokenize")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def logit_lens(
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 5,
    token_position: int = -1,
) -> dict:
    """
    Apply the "logit lens" technique: project hidden states from each
    layer back to vocabulary space to see how the model's prediction
    evolves through its depth.

    Shows what the model would predict if computation stopped at each
    layer. Critical for understanding when specific knowledge emerges
    (e.g., "at which layer does the model start predicting the correct
    translation token?").

    Args:
        prompt:         Input text.
        layers:         Layer indices to analyze. None = sample ~10 layers.
        top_k:          Number of top predictions per layer (default: 5).
        token_position: Which token to analyze (-1 = last, default).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "logit_lens",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        # Sample ~10 layers across the model
        if num_layers <= 10:
            layers = list(range(num_layers))
        else:
            step = max(1, num_layers // 10)
            layers = list(range(0, num_layers, step))
            if (num_layers - 1) not in layers:
                layers.append(num_layers - 1)

    out_of_range = [l for l in layers if l < 0 or l >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "logit_lens",
        )

    try:
        result = await asyncio.to_thread(
            _logit_lens_impl, state.model, state.config,
            state.tokenizer, prompt, sorted(layers), top_k, token_position,
        )
        return result

    except Exception as e:
        logger.exception("logit_lens failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "logit_lens")


def _logit_lens_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    top_k: int,
    token_position: int,
) -> dict:
    """Run logit lens analysis using ModelHooks.get_layer_logits."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = input_ids.shape[-1]

    # Resolve token text at the position
    ids_list = input_ids.tolist()
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    token_text = tokenizer.decode([ids_list[pos]])

    hooks = ModelHooks(model, model_config=config)
    hooks.configure(
        CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
        )
    )
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)

    predictions = []
    for layer_idx in layers:
        logits = hooks.get_layer_logits(layer_idx, normalize=True)
        if logits is None:
            continue

        # Get logits at the target position
        if logits.ndim == 3:
            pos_logits = logits[0, token_position, :]
        elif logits.ndim == 2:
            pos_logits = logits[token_position, :]
        else:
            pos_logits = logits

        probs = mx.softmax(pos_logits, axis=-1)
        sorted_indices = mx.argsort(probs)[::-1][:top_k]
        mx.eval(sorted_indices, probs)

        top_ids = sorted_indices.tolist()
        top_probs = probs[sorted_indices].tolist()
        top_tokens = [tokenizer.decode([tid]) for tid in top_ids]

        predictions.append(LayerPredictionEntry(
            layer=layer_idx,
            top_tokens=top_tokens,
            top_probabilities=[round(p, 6) for p in top_probs],
            top_token_ids=top_ids,
        ))

    # Summary: when does the final prediction first emerge?
    final_token = predictions[-1].top_tokens[0] if predictions else None
    emergence_layer = None
    if final_token is not None:
        for pred in predictions:
            if pred.top_tokens[0] == final_token:
                emergence_layer = pred.layer
                break

    summary = {
        "final_prediction": final_token,
        "emergence_layer": emergence_layer,
        "total_layers": len(predictions),
    }

    result = LogitLensResult(
        prompt=prompt,
        token_position=token_position,
        token_text=token_text,
        num_layers_analyzed=len(predictions),
        predictions=predictions,
        summary=summary,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def track_token(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Track a specific token's probability and rank across layers.

    Uses the logit lens technique to project hidden states at each layer
    back to vocabulary space, then finds the target token's probability
    and rank. Answers "at which layer does the model start predicting
    this specific token?"

    Args:
        prompt:         Input text.
        token:          Target token to track (string, e.g. "Paris").
        layers:         Layer indices to analyze. None = sample ~10 layers.
        token_position: Which input token to analyze (-1 = last, default).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "track_token",
        )

    num_layers = state.metadata.num_layers

    if layers is None:
        if num_layers <= 10:
            layers = list(range(num_layers))
        else:
            step = max(1, num_layers // 10)
            layers = list(range(0, num_layers, step))
            if (num_layers - 1) not in layers:
                layers.append(num_layers - 1)

    out_of_range = [l for l in layers if l < 0 or l >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "track_token",
        )

    # Resolve target token ID
    token_ids = state.tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Could not encode token {token!r}.",
            "track_token",
        )
    target_token_id = token_ids[0]

    try:
        result = await asyncio.to_thread(
            _track_token_impl, state.model, state.config,
            state.tokenizer, prompt, sorted(layers), target_token_id,
            token, token_position,
        )
        return result

    except Exception as e:
        logger.exception("track_token failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "track_token")


def _track_token_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    target_token_id: int,
    target_token: str,
    token_position: int,
) -> dict:
    """Track a specific token's probability across layers."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    num_tokens = input_ids.shape[-1]

    # Resolve token text at position
    ids_list = input_ids.tolist()
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    token_text = tokenizer.decode([ids_list[pos]])

    hooks = ModelHooks(model, model_config=config)
    hooks.configure(
        CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
        )
    )
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)

    entries = []
    emergence_layer = None
    peak_layer = layers[0]
    peak_probability = 0.0

    for layer_idx in layers:
        logits = hooks.get_layer_logits(layer_idx, normalize=True)
        if logits is None:
            continue

        # Get logits at the target position
        if logits.ndim == 3:
            pos_logits = logits[0, token_position, :]
        elif logits.ndim == 2:
            pos_logits = logits[token_position, :]
        else:
            pos_logits = logits

        probs = mx.softmax(pos_logits, axis=-1)

        # Get target token's probability
        target_prob = float(probs[target_token_id])

        # Get rank (how many tokens have higher probability)
        rank = int(mx.sum(probs > probs[target_token_id]))

        is_top1 = rank == 0
        if is_top1 and emergence_layer is None:
            emergence_layer = layer_idx

        if target_prob > peak_probability:
            peak_probability = target_prob
            peak_layer = layer_idx

        entries.append(TokenLayerEntry(
            layer=layer_idx,
            probability=round(target_prob, 6),
            rank=rank,
            is_top1=is_top1,
        ))

    result = TrackTokenResult(
        prompt=prompt,
        target_token=target_token,
        target_token_id=target_token_id,
        token_position=token_position,
        token_text=token_text,
        layers=entries,
        emergence_layer=emergence_layer,
        peak_layer=peak_layer,
        peak_probability=round(peak_probability, 6),
    )
    return result.model_dump()
