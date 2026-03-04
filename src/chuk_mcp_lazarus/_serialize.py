"""
Serialization helpers for the JSON-safe boundary.

Converts MLX arrays, NumPy arrays, and other non-JSON types to plain
Python objects at the tool return boundary.
"""

from __future__ import annotations

from typing import Any


def mx_to_list(arr: Any) -> Any:
    """Convert an MLX array to a plain Python list (nested shape preserved)."""
    import mlx.core as mx

    if isinstance(arr, mx.array):
        return arr.tolist()
    return list(arr)


def np_to_python(val: Any) -> int | float | str | bool | None:
    """Convert a NumPy scalar to a plain Python scalar."""
    import numpy as np

    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)) and val.ndim == 0:
        return val.item()
    return val


def serialize_config(config: Any) -> dict[str, Any]:
    """Extract JSON-safe metadata from a model config object."""
    result: dict[str, Any] = {}
    for field in (
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
        "head_dim",
    ):
        val = getattr(config, field, None)
        if val is not None:
            result[field] = np_to_python(val)
    return result


def hidden_state_to_list(arr: Any, position: int = -1) -> Any:
    """Extract a single position's activation from a hidden state tensor.

    Handles shapes: [hidden], [seq, hidden], [batch, seq, hidden].
    Returns a flat list of floats.
    """
    import mlx.core as mx

    if isinstance(arr, mx.array):
        if arr.ndim == 1:
            return arr.tolist()
        if arr.ndim == 2:
            return arr[position].tolist()
        if arr.ndim == 3:
            return arr[0, position].tolist()
    return list(arr)


def cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Compute pairwise cosine similarity for a list of vectors."""
    import numpy as np

    vecs = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = vecs / norms
    sim = (normed @ normed.T).tolist()
    return sim


def pca_2d(vectors: list[list[float]]) -> list[list[float]]:
    """Project vectors to 2D via PCA. Returns list of [x, y] pairs."""
    import numpy as np

    vecs = np.array(vectors, dtype=np.float32)
    centered = vecs - vecs.mean(axis=0)
    if centered.shape[0] < 2:
        return [[0.0, 0.0]] * centered.shape[0]
    # SVD-based PCA (no sklearn needed for 2 components)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = (centered @ vt[:2].T).tolist()
    return projected
