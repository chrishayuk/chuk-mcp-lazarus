"""Tests for compute_subspace tool and SubspaceRegistry.

Tests cover:
  - SubspaceRegistry: singleton, store/fetch, exists, list, clear, dump (10)
  - compute_subspace: async validation (8)
  - _compute_subspace_impl: output keys, rank, registry, basis, variance (9)
  - list_subspaces: empty, with stored entry (2)
  - SubspaceMetadata model: frozen, dump (2)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.subspace_registry import (
    SubspaceMetadata,
    SubspaceRegistry,
    SubspaceRegistryDump,
)

# ---------------------------------------------------------------------------
# Mock data: 6 prompts × 8-dim activations
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
MOCK_ACTIVATIONS = [RNG.standard_normal(8).astype(np.float32) for _ in range(6)]


def _make_meta(name: str = "test_sub", layer: int = 2, rank: int = 3) -> SubspaceMetadata:
    return SubspaceMetadata(
        name=name,
        layer=layer,
        rank=rank,
        num_prompts=6,
        hidden_dim=8,
        variance_explained=[0.5, 0.3, 0.2],
        total_variance_explained=1.0,
        computed_at="2024-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# SubspaceRegistry
# ---------------------------------------------------------------------------


class TestSubspaceRegistry:
    def test_singleton(self) -> None:
        r1 = SubspaceRegistry.get()
        r2 = SubspaceRegistry.get()
        assert r1 is r2

    def test_store_and_fetch(self) -> None:
        reg = SubspaceRegistry.get()
        basis = np.eye(3, 8, dtype=np.float32)
        meta = _make_meta()
        reg.store("test_sub", basis, meta)
        result = reg.fetch("test_sub")
        assert result is not None
        b, m = result
        assert b.shape == (3, 8)
        assert m.name == "test_sub"

    def test_fetch_missing(self) -> None:
        reg = SubspaceRegistry.get()
        assert reg.fetch("nonexistent") is None

    def test_exists(self) -> None:
        reg = SubspaceRegistry.get()
        assert not reg.exists("test_sub")
        reg.store("test_sub", np.eye(3, 8, dtype=np.float32), _make_meta())
        assert reg.exists("test_sub")

    def test_list_all(self) -> None:
        reg = SubspaceRegistry.get()
        reg.store("alpha", np.eye(2, 8, dtype=np.float32), _make_meta("alpha", rank=2))
        reg.store("beta", np.eye(4, 8, dtype=np.float32), _make_meta("beta", rank=4))
        entries = reg.list_all()
        assert len(entries) == 2
        assert entries[0].name == "alpha"
        assert entries[1].name == "beta"

    def test_clear(self) -> None:
        reg = SubspaceRegistry.get()
        reg.store("test_sub", np.eye(3, 8, dtype=np.float32), _make_meta())
        assert reg.count > 0
        reg.clear()
        assert reg.count == 0

    def test_count(self) -> None:
        reg = SubspaceRegistry.get()
        assert reg.count == 0
        reg.store("a", np.eye(1, 8, dtype=np.float32), _make_meta("a", rank=1))
        assert reg.count == 1

    def test_dump(self) -> None:
        reg = SubspaceRegistry.get()
        reg.store("test_sub", np.eye(3, 8, dtype=np.float32), _make_meta())
        dump = reg.dump()
        assert isinstance(dump, SubspaceRegistryDump)
        assert dump.count == 1
        assert dump.subspaces[0].name == "test_sub"

    def test_overwrite(self) -> None:
        reg = SubspaceRegistry.get()
        reg.store("test_sub", np.eye(3, 8, dtype=np.float32), _make_meta(rank=3))
        reg.store("test_sub", np.eye(5, 8, dtype=np.float32), _make_meta(rank=5))
        result = reg.fetch("test_sub")
        assert result is not None
        assert result[0].shape == (5, 8)

    def test_shape_preserved(self) -> None:
        reg = SubspaceRegistry.get()
        basis = RNG.standard_normal((4, 8)).astype(np.float32)
        reg.store("shaped", basis, _make_meta("shaped", rank=4))
        result = reg.fetch("shaped")
        assert result is not None
        np.testing.assert_array_equal(result[0], basis)


# ---------------------------------------------------------------------------
# compute_subspace — async validation
# ---------------------------------------------------------------------------


class TestComputeSubspace:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 0, ["a", "b", "c"])
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 99, ["a", "b", "c"])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 0, ["a", "b"])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 0, [f"p{i}" for i in range(501)])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_rank_too_large(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 0, ["a", "b", "c"], rank=101)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_rank_zero(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 0, ["a", "b", "c"], rank=0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_rank_exceeds_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("sub", 0, ["a", "b", "c"], rank=3)
        assert result["error_type"] == "InvalidInput"
        assert "less than" in result["message"]

    @pytest.mark.asyncio
    async def test_empty_name(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import compute_subspace

        result = await compute_subspace("  ", 0, ["a", "b", "c"])
        assert result["error_type"] == "InvalidInput"


# ---------------------------------------------------------------------------
# _compute_subspace_impl tests
# ---------------------------------------------------------------------------


def _run(
    prompts: list[str] | None = None,
    rank: int = 2,
    layer: int = 1,
    token_position: int = -1,
) -> dict:
    """Run _compute_subspace_impl with mock activations."""
    from chuk_mcp_lazarus.tools.geometry.compute_subspace import (
        _compute_subspace_impl,
    )

    if prompts is None:
        prompts = [f"prompt_{i}" for i in range(6)]

    call_idx = [0]

    def fake_extract(
        model: Any, config: Any, tokenizer: Any, prompt: str, ly: int, pos: int
    ) -> list:
        idx = call_idx[0]
        call_idx[0] += 1
        return MOCK_ACTIVATIONS[idx % len(MOCK_ACTIVATIONS)].tolist()

    meta = MagicMock()
    meta.num_layers = 4
    meta.hidden_dim = 8

    with patch(
        "chuk_mcp_lazarus.tools.geometry.compute_subspace.extract_activation_at_layer",
        side_effect=fake_extract,
    ):
        return _compute_subspace_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            meta,
            "test_subspace",
            layer,
            prompts,
            rank,
            token_position,
        )


class TestComputeSubspaceImpl:
    def test_output_keys(self) -> None:
        result = _run()
        assert "subspace_name" in result
        assert "layer" in result
        assert "rank" in result
        assert "num_prompts" in result
        assert "hidden_dim" in result
        assert "components" in result
        assert "total_variance_explained" in result
        assert "summary" in result

    def test_rank_stored(self) -> None:
        result = _run(rank=2)
        assert result["rank"] == 2

    def test_stored_in_registry(self) -> None:
        _run()
        reg = SubspaceRegistry.get()
        assert reg.exists("test_subspace")

    def test_basis_shape(self) -> None:
        _run(rank=3)
        reg = SubspaceRegistry.get()
        entry = reg.fetch("test_subspace")
        assert entry is not None
        basis, meta = entry
        assert basis.shape[0] == 3
        assert basis.shape[1] == 8

    def test_variance_sums_to_at_most_one(self) -> None:
        result = _run()
        assert result["total_variance_explained"] <= 1.0 + 1e-6

    def test_cumulative_monotone(self) -> None:
        result = _run(rank=3)
        components = result["components"]
        for i in range(1, len(components)):
            assert components[i]["cumulative_variance"] >= components[i - 1]["cumulative_variance"]

    def test_components_count(self) -> None:
        result = _run(rank=4)
        assert len(result["components"]) == 4

    def test_summary_keys(self) -> None:
        result = _run()
        s = result["summary"]
        assert "stored" in s
        assert "effective_rank" in s
        assert "top_component_variance" in s
        assert "recommended_rank_for_80pct" in s

    def test_overwrite(self) -> None:
        _run(rank=2)
        _run(rank=3)
        reg = SubspaceRegistry.get()
        entry = reg.fetch("test_subspace")
        assert entry is not None
        assert entry[0].shape[0] == 3


# ---------------------------------------------------------------------------
# list_subspaces
# ---------------------------------------------------------------------------


class TestListSubspaces:
    @pytest.mark.asyncio
    async def test_empty_registry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import list_subspaces

        result = await list_subspaces()
        assert result["count"] == 0
        assert result["subspaces"] == []

    @pytest.mark.asyncio
    async def test_lists_stored_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.compute_subspace import list_subspaces

        reg = SubspaceRegistry.get()
        reg.store("my_sub", np.eye(3, 8, dtype=np.float32), _make_meta("my_sub"))
        result = await list_subspaces()
        assert result["count"] == 1
        assert result["subspaces"][0]["name"] == "my_sub"


# ---------------------------------------------------------------------------
# SubspaceMetadata model
# ---------------------------------------------------------------------------


class TestSubspaceMetadataModel:
    def test_frozen(self) -> None:
        meta = _make_meta()
        with pytest.raises(Exception):
            meta.name = "changed"  # type: ignore[misc]

    def test_model_dump(self) -> None:
        meta = _make_meta()
        d = meta.model_dump()
        assert d["name"] == "test_sub"
        assert d["rank"] == 3
        assert "variance_explained" in d
