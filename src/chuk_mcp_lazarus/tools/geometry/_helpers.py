"""
Shared helpers for geometry tools.

Enums, typed specs, math utilities, token resolution, and direction
vector extraction used across all geometry tools.
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel

from ...steering_store import SteeringVectorRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and typed specs
# ---------------------------------------------------------------------------


class DirectionType(str, Enum):
    """Valid direction source types for geometry tools."""

    TOKEN = "token"
    NEURON = "neuron"
    RESIDUAL = "residual"
    FFN_OUTPUT = "ffn_output"
    ATTENTION_OUTPUT = "attention_output"
    HEAD_OUTPUT = "head_output"
    STEERING_VECTOR = "steering_vector"


class DirectionSpec(BaseModel):
    """Typed specification for a direction vector."""

    type: DirectionType
    value: str | int | None = None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clamped to [-1, 1]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _angle_deg(cos_sim: float) -> float:
    """Cosine similarity -> angle in degrees."""
    return float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two vectors in the full space."""
    return _angle_deg(_cosine_sim(a, b))


def _unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalise a vector. Returns zero vector if norm is near-zero."""
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _gram_schmidt(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Gram-Schmidt orthogonalisation. Returns orthonormal basis."""
    basis: list[np.ndarray] = []
    for v in vectors:
        w = v.copy().astype(np.float64)
        for b in basis:
            w -= float(np.dot(w, b)) * b
        n = float(np.linalg.norm(w))
        if n > 1e-10:
            basis.append((w / n).astype(np.float32))
    return basis


def coerce_layers(layers: Any) -> list[int] | None:
    """Coerce a layers argument from MCP into a list[int].

    MCP may deliver ``layers: "28"`` (str) instead of ``28`` (int),
    or ``["0", "16", "33"]`` instead of ``[0, 16, 33]``.
    Returns None if *layers* is None (for optional-layers tools).
    """
    if layers is None:
        return None
    if isinstance(layers, (int, float)):
        return [int(layers)]
    if isinstance(layers, str):
        return [int(layers)]
    # list / tuple — coerce each element
    return [int(x) for x in layers]


def _auto_layers(num_layers: int, max_points: int = 20) -> list[int]:
    """Auto-select evenly spaced layers, always including 0 and last."""
    if num_layers <= max_points:
        return list(range(num_layers))
    step = max(1, num_layers // max_points)
    layers = list(range(0, num_layers, step))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return layers


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _resolve_token_to_id(tokenizer: Any, token: str) -> int | None:
    """Encode a token string to a single token ID (tries bare + space-prefixed)."""
    for variant in (token, " " + token):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            return ids[0]
    return None


def _get_unembed_vec_np(model: Any, token_id: int) -> np.ndarray | None:
    """Get unembedding vector for a token as numpy float32."""
    from ..residual_tools import _get_unembed_vector

    vec: Any = _get_unembed_vector(model, token_id)
    if vec is None:
        return None
    raw: list[float] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return np.array(raw, dtype=np.float32)


def _token_text_at(tokenizer: Any, prompt: str, position: int) -> str:
    """Decode a single token from a prompt at the given position."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    idx = position if position >= 0 else len(ids) + position
    idx = max(0, min(idx, len(ids) - 1))
    return tokenizer.decode([ids[idx]])


# ---------------------------------------------------------------------------
# Direction vector extraction
# ---------------------------------------------------------------------------


def _parse_direction_spec(raw: dict) -> tuple[DirectionSpec | None, str | None]:
    """Parse a raw dict into a typed DirectionSpec. Returns (spec, error)."""
    if not isinstance(raw, dict) or "type" not in raw:
        return None, "Each direction must be a dict with 'type'."
    try:
        return DirectionSpec(**raw), None
    except Exception as exc:
        return None, f"Invalid direction spec: {exc}"


def _extract_direction_vector(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layer: int,
    spec: DirectionSpec,
    token_position: int,
    decomp: dict | None,
) -> tuple[str, np.ndarray | None, str | None]:
    """Extract a direction vector from a typed spec.

    Returns (label, vector_np, error_message).
    decomp is the result of _run_decomposition_forward if already computed.
    """
    from ..residual_tools import _extract_position

    if spec.type == DirectionType.TOKEN:
        if spec.value is None:
            return ("", None, "Token direction requires 'value'.")
        tid = _resolve_token_to_id(tokenizer, str(spec.value))
        if tid is None:
            return ("", None, f"Cannot encode token '{spec.value}'.")
        tvec = _get_unembed_vec_np(model, tid)
        if tvec is None:
            return ("", None, f"No unembed vector for token '{spec.value}'.")
        return (f"token:{spec.value}", tvec, None)

    if spec.type == DirectionType.STEERING_VECTOR:
        if spec.value is None:
            return ("", None, "Steering vector requires 'value' (name).")
        reg = SteeringVectorRegistry.get()
        sv_result = reg.fetch(str(spec.value))
        if sv_result is None:
            return ("", None, f"Steering vector '{spec.value}' not found.")
        sv_vec, _ = sv_result
        return (f"sv:{spec.value}", sv_vec.astype(np.float32), None)

    if spec.type == DirectionType.NEURON:
        if spec.value is None:
            return ("", None, "Neuron direction requires 'value' (neuron index).")
        import mlx.core as mx
        from chuk_lazarus.introspection.hooks import ModelHooks

        helper = ModelHooks(model, model_config=config)
        model_layers = helper._get_layers()
        if layer >= len(model_layers):
            return ("", None, f"Layer {layer} out of range.")
        target = model_layers[layer]
        down_weight = target.mlp.down_proj.weight
        if isinstance(down_weight, mx.array):
            col: Any = down_weight[:, int(spec.value)]
            nraw: list[float] = col.tolist() if hasattr(col, "tolist") else list(col)
            return (f"neuron:{spec.value}", np.array(nraw, dtype=np.float32), None)
        return ("", None, f"Cannot extract neuron {spec.value} direction.")

    # Remaining types need decomposition data
    if decomp is None:
        return ("", None, f"Decomposition data not available for type '{spec.type.value}'.")

    if spec.type == DirectionType.RESIDUAL:
        h = decomp["hidden_states"].get(layer)
        if h is None:
            return ("", None, f"No hidden state at layer {layer}.")
        rvec: Any = _extract_position(h, token_position)
        rraw: list[float] = rvec.tolist() if hasattr(rvec, "tolist") else list(rvec)
        return ("residual", np.array(rraw, dtype=np.float32), None)

    if spec.type == DirectionType.FFN_OUTPUT:
        ffn = decomp["ffn_outputs"].get(layer)
        if ffn is None:
            return ("", None, f"No FFN output at layer {layer}.")
        fvec: Any = _extract_position(ffn, token_position)
        fraw: list[float] = fvec.tolist() if hasattr(fvec, "tolist") else list(fvec)
        return ("ffn_output", np.array(fraw, dtype=np.float32), None)

    if spec.type == DirectionType.ATTENTION_OUTPUT:
        attn = decomp["attn_outputs"].get(layer)
        if attn is None:
            return ("", None, f"No attention output at layer {layer}.")
        avec: Any = _extract_position(attn, token_position)
        araw: list[float] = avec.tolist() if hasattr(avec, "tolist") else list(avec)
        return ("attn_output", np.array(araw, dtype=np.float32), None)

    if spec.type == DirectionType.HEAD_OUTPUT:
        if spec.value is None:
            return ("", None, "Head output requires 'value' (head index).")
        import mlx.core as mx
        from chuk_lazarus.introspection.hooks import ModelHooks

        attn_out = decomp["attn_outputs"].get(layer)
        if attn_out is None:
            return ("", None, f"No attention output at layer {layer}.")

        helper = ModelHooks(model, model_config=config)
        model_layers = helper._get_layers()
        target = model_layers[layer]
        hidden_dim = meta.hidden_dim
        num_heads = meta.num_attention_heads
        head_dim = hidden_dim // num_heads
        head_idx = int(spec.value)
        if head_idx < 0 or head_idx >= num_heads:
            return ("", None, f"Head {head_idx} out of [0, {num_heads - 1}].")

        # Use o_proj weight column slice as the head's output direction
        o_proj = target.self_attn.o_proj
        w_slice = o_proj.weight[:, head_idx * head_dim : (head_idx + 1) * head_dim]
        head_dir: Any = mx.mean(w_slice, axis=1)
        hraw: list[float] = head_dir.tolist() if hasattr(head_dir, "tolist") else list(head_dir)
        return (f"head:{spec.value}", np.array(hraw, dtype=np.float32), None)

    return ("", None, f"Unhandled direction type '{spec.type.value}'.")


# ---------------------------------------------------------------------------
# Shared PCA helpers
# ---------------------------------------------------------------------------


def collect_activations(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    token_position: int = -1,
) -> dict[int, list[np.ndarray]]:
    """Collect per-layer activations for many prompts.

    Single-layer uses ``extract_activation_at_layer`` (lighter).
    Multi-layer uses ``extract_activations_all_layers`` (one forward pass).

    Returns ``{layer: [np.ndarray(hidden_dim,), ...]}``.
    """
    from ..._extraction import extract_activation_at_layer, extract_activations_all_layers

    per_layer: dict[int, list[np.ndarray]] = {lyr: [] for lyr in layers}

    if len(layers) > 1:
        for p in prompts:
            layer_acts = extract_activations_all_layers(
                model, config, tokenizer, p, layers, token_position
            )
            for lyr in layers:
                if lyr in layer_acts:
                    per_layer[lyr].append(np.array(layer_acts[lyr], dtype=np.float32))
    else:
        lyr = layers[0]
        for p in prompts:
            act = extract_activation_at_layer(model, config, tokenizer, p, lyr, token_position)
            per_layer[lyr].append(np.array(act, dtype=np.float32))

    return per_layer


def dims_for_threshold(cumulative_vars: list[float], threshold: float, total: int) -> int:
    """Number of components needed to reach a cumulative variance threshold."""
    for i, cv in enumerate(cumulative_vars):
        if cv >= threshold:
            return i + 1
    return total


def effective_dimensionality(
    s_vals: np.ndarray,
    n_components: int,
    total_var: float,
) -> tuple[dict[str, int], list[float]]:
    """Compute effective dimensionality at standard thresholds.

    Returns ``(eff_dim_dict, cumulative_list)`` where eff_dim_dict has keys
    ``dims_for_50pct`` through ``dims_for_99pct``.
    """
    cumulative = 0.0
    cumulative_list: list[float] = []

    for sv in s_vals[:n_components]:
        cumulative += float(sv**2) / total_var
        cumulative_list.append(cumulative)

    thresholds = [
        ("dims_for_50pct", 0.50),
        ("dims_for_80pct", 0.80),
        ("dims_for_90pct", 0.90),
        ("dims_for_95pct", 0.95),
        ("dims_for_99pct", 0.99),
    ]
    eff_dim = {
        name: dims_for_threshold(cumulative_list, thresh, n_components)
        for name, thresh in thresholds
    }
    return eff_dim, cumulative_list
