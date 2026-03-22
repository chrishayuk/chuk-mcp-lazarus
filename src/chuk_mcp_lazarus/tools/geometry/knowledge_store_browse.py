"""
knowledge_store_browse — browse knowledge store windows for Context Map.

Tools:
    knowledge_store_info   — load store metadata (n_windows, has_boundaries, etc.)
    knowledge_store_window — load a specific window's text and token IDs.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class StoreInfoResult(BaseModel):
    """Metadata from a knowledge store."""

    store_path: str
    model_id: str = ""
    n_windows: int = 0
    window_size: int = 512
    entries_per_window: int = 8
    total_entries: int = 0
    has_boundary_residual: bool = False
    has_window_tokens: bool = False
    version: int = 0


class WindowResult(BaseModel):
    """A single window's text and tokens."""

    store_path: str
    window_id: int
    n_windows: int
    text: str
    token_ids: list[int]
    num_tokens: int


class BoundaryResult(BaseModel):
    """A boundary residual vector."""

    store_path: str
    window_id: int
    hidden_dim: int
    residual: list[float]


# ---------------------------------------------------------------------------
# Tool: knowledge_store_info
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def knowledge_store_info(
    store_path: str,
) -> dict:
    """Load metadata from a knowledge store directory.

    Returns the number of windows, window size, whether boundary residuals
    are available, and other store metadata.  Use this to populate the
    Context Map window selector.

    Args:
        store_path: Path to the knowledge store directory.
    """
    try:
        return await asyncio.to_thread(_store_info_impl, store_path)
    except Exception as exc:
        logger.exception("knowledge_store_info failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "knowledge_store_info")


def _store_info_impl(store_path: str) -> dict:
    path = Path(store_path)
    if not path.is_dir():
        return make_error(
            ToolError.INVALID_INPUT,
            f"Store path does not exist: {store_path}",
            "knowledge_store_info",
        )

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return make_error(
            ToolError.INVALID_INPUT,
            f"No manifest.json found in {store_path}",
            "knowledge_store_info",
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Count windows from window_token_lists.npz
    n_windows = 0
    has_window_tokens = False
    wt_path = path / "window_token_lists.npz"
    if wt_path.exists():
        has_window_tokens = True
        wt = np.load(str(wt_path))
        n_windows = len(wt.files)
    elif "n_windows" in manifest:
        n_windows = manifest["n_windows"]

    return StoreInfoResult(
        store_path=store_path,
        model_id=manifest.get("model_id", ""),
        n_windows=n_windows,
        window_size=manifest.get("window_size", 512),
        entries_per_window=manifest.get("entries_per_window", 8),
        total_entries=manifest.get("total_entries", 0),
        has_boundary_residual=(path / "boundary_residual.npy").exists(),
        has_window_tokens=has_window_tokens,
        version=manifest.get("version", 0),
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: knowledge_store_window
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def knowledge_store_window(
    store_path: str,
    window_id: int,
) -> dict:
    """Load a specific window's text from a knowledge store.

    Decodes the window's token IDs back into text using the currently
    loaded model's tokenizer.  Use this to populate the Context Map
    prompt when browsing a knowledge store.

    Args:
        store_path: Path to the knowledge store directory.
        window_id:  Window index (0-based).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(ToolError.MODEL_NOT_LOADED, "Call load_model() first.", "knowledge_store_window")

    try:
        return await asyncio.to_thread(
            _store_window_impl, store_path, window_id, state.tokenizer,
        )
    except Exception as exc:
        logger.exception("knowledge_store_window failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "knowledge_store_window")


def _store_window_impl(store_path: str, window_id: int, tokenizer: Any) -> dict:
    path = Path(store_path)
    wt_path = path / "window_token_lists.npz"
    if not wt_path.exists():
        return make_error(
            ToolError.INVALID_INPUT,
            f"No window_token_lists.npz in {store_path}",
            "knowledge_store_window",
        )

    wt = np.load(str(wt_path))
    n_windows = len(wt.files)

    if window_id < 0 or window_id >= n_windows:
        return make_error(
            ToolError.INVALID_INPUT,
            f"window_id {window_id} out of range [0, {n_windows - 1}].",
            "knowledge_store_window",
        )

    key = str(window_id)
    token_ids = wt[key].tolist()
    text = tokenizer.decode(token_ids)

    return WindowResult(
        store_path=store_path,
        window_id=window_id,
        n_windows=n_windows,
        text=text,
        token_ids=token_ids,
        num_tokens=len(token_ids),
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool: load_boundary_residual
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def load_boundary_residual(
    store_path: str,
    window_id: int = -1,
) -> dict:
    """Load a boundary residual from a knowledge store.

    Currently only the final boundary residual is saved (after all windows).
    Use window_id=-1 (default) to load it.  Per-window boundaries will be
    supported in a future version.

    The returned residual can be passed as initial_residual to context_map.

    Args:
        store_path: Path to the knowledge store directory.
        window_id:  Window index (-1 = final boundary).  Currently only -1 is supported.
    """
    try:
        return await asyncio.to_thread(_load_boundary_impl, store_path, window_id)
    except Exception as exc:
        logger.exception("load_boundary_residual failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "load_boundary_residual")


def _load_boundary_impl(store_path: str, window_id: int) -> dict:
    path = Path(store_path)

    if window_id != -1:
        # Future: per-window boundaries from boundaries/ directory
        boundary_path = path / "boundaries" / f"boundary_{window_id:03d}.npy"
        if not boundary_path.exists():
            return make_error(
                ToolError.INVALID_INPUT,
                f"Per-window boundary not found: {boundary_path}. "
                f"Only the final boundary is currently saved. Use window_id=-1.",
                "load_boundary_residual",
            )
    else:
        # Final boundary
        boundary_path = path / "boundary_residual.npy"
        if not boundary_path.exists():
            # Try final_residual.npy (vec_inject output)
            boundary_path = path / "final_residual.npy"

    if not boundary_path.exists():
        return make_error(
            ToolError.INVALID_INPUT,
            f"No boundary residual found in {store_path}.",
            "load_boundary_residual",
        )

    residual_np = np.load(str(boundary_path))

    # Flatten to 1D [hidden_dim]
    residual_flat = residual_np.flatten().astype(np.float32)
    hidden_dim = len(residual_flat)

    return BoundaryResult(
        store_path=store_path,
        window_id=window_id,
        hidden_dim=hidden_dim,
        residual=residual_flat.tolist(),
    ).model_dump()
