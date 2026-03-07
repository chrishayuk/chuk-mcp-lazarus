"""
build_dark_table / list_dark_tables — precompute subspace coordinate lookup tables.

For each reference prompt, extracts the hidden state at a layer, projects onto
a named PCA subspace, and stores the [rank]-dimensional coordinate vector.
These coordinates can later be injected via subspace_surgery in "lookup" mode
without an extra forward pass.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...dark_table_registry import (
    DarkTableMetadata,
    DarkTableRegistry,
)
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ...subspace_registry import SubspaceRegistry


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class DarkTableEntry(BaseModel):
    """One entry in the built table."""

    key: str = Field(..., description="Lookup key for this entry.")
    prompt: str = Field(..., description="Prompt used to extract coordinates.")
    coordinate_norm: float = Field(..., description="L2 norm of the coordinate vector.")


class BuildDarkTableResult(BaseModel):
    """Result from build_dark_table."""

    table_name: str
    subspace_name: str
    layer: int
    rank: int
    num_entries: int
    token_position: int
    entries: list[DarkTableEntry]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Sync implementation
# ---------------------------------------------------------------------------


def _build_dark_table_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    table_name: str,
    subspace_name: str,
    layer: int,
    entries: dict[str, str],
    token_position: int,
) -> dict:
    from ..._extraction import extract_activation_at_layer

    # Fetch subspace basis
    entry = SubspaceRegistry.get().fetch(subspace_name)
    if entry is None:
        return make_error(
            ToolError.VECTOR_NOT_FOUND,
            f"Subspace '{subspace_name}' not found.",
            "build_dark_table",
        )
    basis, sub_meta = entry  # basis: [rank, hidden_dim]

    coordinates: dict[str, np.ndarray] = {}
    result_entries: list[DarkTableEntry] = []

    for key, prompt in entries.items():
        act = extract_activation_at_layer(model, config, tokenizer, prompt, layer, token_position)
        act_np = np.array(act, dtype=np.float32)  # [hidden_dim]

        # Project onto subspace: dot each basis vector with activation
        coords = basis @ act_np  # [rank]
        coordinates[key] = coords

        result_entries.append(
            DarkTableEntry(
                key=key,
                prompt=prompt,
                coordinate_norm=round(float(np.linalg.norm(coords)), 6),
            )
        )

    # Store in registry
    dt_meta = DarkTableMetadata(
        table_name=table_name,
        subspace_name=subspace_name,
        layer=layer,
        rank=int(basis.shape[0]),
        num_entries=len(coordinates),
        token_position=token_position,
        computed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    DarkTableRegistry.get().store(table_name, coordinates, dt_meta)

    # Summary
    norms = [float(np.linalg.norm(c)) for c in coordinates.values()]
    summary: dict[str, Any] = {
        "stored": True,
        "mean_coordinate_norm": round(sum(norms) / max(len(norms), 1), 6),
        "min_coordinate_norm": round(min(norms) if norms else 0.0, 6),
        "max_coordinate_norm": round(max(norms) if norms else 0.0, 6),
    }

    return BuildDarkTableResult(
        table_name=table_name,
        subspace_name=subspace_name,
        layer=layer,
        rank=int(basis.shape[0]),
        num_entries=len(coordinates),
        token_position=token_position,
        entries=result_entries,
        summary=summary,
    ).model_dump()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def build_dark_table(
    table_name: str,
    subspace_name: str,
    layer: int,
    entries: dict[str, str],
    token_position: int = -1,
) -> dict:
    """Precompute dark coordinate lookup table.

    For each entry (key → prompt), extracts the hidden state at the given
    layer, projects onto the named PCA subspace, and stores the resulting
    [rank]-dimensional coordinate vector.  These coordinates can be injected
    via subspace_surgery(mode="lookup") with zero extra forward passes.

    Call compute_subspace first to create the PCA subspace.
    Call subspace_surgery(mode="lookup") after to inject stored coordinates.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "build_dark_table",
        )

    num_layers = state.metadata.num_layers
    layer = int(layer)

    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "build_dark_table",
        )

    if not table_name or not table_name.strip():
        return make_error(
            ToolError.INVALID_INPUT,
            "table_name must be non-empty.",
            "build_dark_table",
        )

    if not SubspaceRegistry.get().exists(subspace_name):
        return make_error(
            ToolError.VECTOR_NOT_FOUND,
            f"Subspace '{subspace_name}' not found. Call compute_subspace first.",
            "build_dark_table",
        )

    if not entries or len(entries) < 1:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 entry required.",
            "build_dark_table",
        )

    if len(entries) > 200:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 200 entries.",
            "build_dark_table",
        )

    try:
        return await asyncio.to_thread(
            _build_dark_table_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            table_name,
            subspace_name,
            layer,
            entries,
            token_position,
        )
    except Exception as e:
        return make_error(
            ToolError.GEOMETRY_FAILED,
            f"build_dark_table failed: {e}",
            "build_dark_table",
        )


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_dark_tables() -> dict:
    """List all dark tables in the DarkTableRegistry.

    Returns table names, subspace associations, entry counts, and timestamps.
    """
    reg = DarkTableRegistry.get()
    dump = reg.dump()
    return dump.model_dump()
