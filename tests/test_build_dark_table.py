"""Tests for build_dark_table and list_dark_tables tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.dark_table_registry import (
    DarkTableMetadata,
    DarkTableRegistry,
)
from chuk_mcp_lazarus.subspace_registry import SubspaceMetadata, SubspaceRegistry
from chuk_mcp_lazarus.tools.geometry.build_dark_table import (
    BuildDarkTableResult,
    DarkTableEntry,
    _build_dark_table_impl,
    build_dark_table,
    list_dark_tables,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 8
RANK = 3
RNG = np.random.default_rng(42)
MOCK_ACTIVATIONS = [RNG.standard_normal(DIM).astype(np.float32) for _ in range(5)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_subspace(name: str = "test_sub") -> np.ndarray:
    """Pre-store a subspace and return the basis."""
    basis = np.eye(RANK, DIM, dtype=np.float32)
    meta = SubspaceMetadata(
        name=name,
        layer=5,
        rank=RANK,
        num_prompts=10,
        hidden_dim=DIM,
        variance_explained=[0.4, 0.3, 0.2],
        total_variance_explained=0.9,
        computed_at="2026-01-01T00:00:00+00:00",
    )
    SubspaceRegistry.get().store(name, basis, meta)
    return basis


# ---------------------------------------------------------------------------
# Async validation tests
# ---------------------------------------------------------------------------


class TestBuildDarkTable:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        result = await build_dark_table(
            table_name="t", subspace_name="s", layer=0, entries={"a": "prompt"}
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await build_dark_table(
            table_name="t", subspace_name="test_sub", layer=99, entries={"a": "p"}
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_table_name(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await build_dark_table(
            table_name="", subspace_name="test_sub", layer=0, entries={"a": "p"}
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_subspace_not_found(self, loaded_model_state: Any) -> None:
        result = await build_dark_table(
            table_name="t", subspace_name="missing", layer=0, entries={"a": "p"}
        )
        assert result["error"] is True
        assert result["error_type"] == "VectorNotFound"

    @pytest.mark.asyncio
    async def test_empty_entries(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await build_dark_table(
            table_name="t", subspace_name="test_sub", layer=0, entries={}
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_entries(self, loaded_model_state: Any) -> None:
        _store_subspace()
        entries = {str(i): f"prompt {i}" for i in range(201)}
        result = await build_dark_table(
            table_name="t", subspace_name="test_sub", layer=0, entries=entries
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success_returns_dict(self, loaded_model_state: Any) -> None:
        _store_subspace()
        fake_result = {"table_name": "t", "num_entries": 1}
        with patch(
            "chuk_mcp_lazarus.tools.geometry.build_dark_table._build_dark_table_impl",
            return_value=fake_result,
        ):
            result = await build_dark_table(
                table_name="t",
                subspace_name="test_sub",
                layer=0,
                entries={"a": "p"},
            )
        assert result["table_name"] == "t"

    @pytest.mark.asyncio
    async def test_exception_handling(self, loaded_model_state: Any) -> None:
        _store_subspace()
        with patch(
            "chuk_mcp_lazarus.tools.geometry.build_dark_table._build_dark_table_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await build_dark_table(
                table_name="t",
                subspace_name="test_sub",
                layer=0,
                entries={"a": "p"},
            )
        assert result["error"] is True
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# Impl tests
# ---------------------------------------------------------------------------


class TestBuildDarkTableImpl:
    def _run(
        self,
        entries: dict[str, str] | None = None,
        subspace_name: str = "test_sub",
        layer: int = 1,
        token_position: int = -1,
    ) -> dict:
        _store_subspace(subspace_name)

        if entries is None:
            entries = {"a": "prompt a", "b": "prompt b"}

        call_idx = [0]

        def fake_extract(
            model: Any,
            config: Any,
            tokenizer: Any,
            prompt: str,
            ly: int,
            pos: int,
        ) -> list:
            idx = call_idx[0]
            call_idx[0] += 1
            return MOCK_ACTIVATIONS[idx % len(MOCK_ACTIVATIONS)].tolist()

        with patch(
            "chuk_mcp_lazarus._extraction.extract_activation_at_layer",
            side_effect=fake_extract,
        ):
            return _build_dark_table_impl(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                "test_table",
                subspace_name,
                layer,
                entries,
                token_position,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for key in [
            "table_name",
            "subspace_name",
            "layer",
            "rank",
            "num_entries",
            "token_position",
            "entries",
            "summary",
        ]:
            assert key in r, f"Missing key: {key}"

    def test_stored_in_registry(self) -> None:
        self._run()
        reg = DarkTableRegistry.get()
        assert reg.exists("test_table")

    def test_num_entries_match(self) -> None:
        r = self._run(entries={"x": "p1", "y": "p2", "z": "p3"})
        assert r["num_entries"] == 3
        assert len(r["entries"]) == 3

    def test_coordinate_shape(self) -> None:
        self._run()
        reg = DarkTableRegistry.get()
        result = reg.fetch("test_table")
        assert result is not None
        coords, _ = result
        for key, vec in coords.items():
            assert vec.shape == (RANK,), f"Key {key} has shape {vec.shape}"

    def test_coordinate_norm_positive(self) -> None:
        r = self._run()
        for entry in r["entries"]:
            assert entry["coordinate_norm"] > 0

    def test_summary_keys(self) -> None:
        r = self._run()
        s = r["summary"]
        assert "stored" in s
        assert s["stored"] is True
        assert "mean_coordinate_norm" in s
        assert "min_coordinate_norm" in s
        assert "max_coordinate_norm" in s

    def test_overwrite_table(self) -> None:
        self._run(entries={"a": "p1", "b": "p2"})
        self._run(entries={"x": "p3"})
        reg = DarkTableRegistry.get()
        result = reg.fetch("test_table")
        assert result is not None
        coords, meta = result
        assert "x" in coords
        assert meta.num_entries == 1

    def test_rank_matches_subspace(self) -> None:
        r = self._run()
        assert r["rank"] == RANK

    def test_single_entry(self) -> None:
        r = self._run(entries={"only": "single prompt"})
        assert r["num_entries"] == 1
        assert len(r["entries"]) == 1
        assert r["entries"][0]["key"] == "only"


# ---------------------------------------------------------------------------
# list_dark_tables tests
# ---------------------------------------------------------------------------


class TestListDarkTables:
    @pytest.mark.asyncio
    async def test_empty_registry(self) -> None:
        DarkTableRegistry.get().clear()
        result = await list_dark_tables()
        assert result["count"] == 0
        assert result["tables"] == []

    @pytest.mark.asyncio
    async def test_lists_stored_entry(self) -> None:
        DarkTableRegistry.get().clear()
        meta = DarkTableMetadata(
            table_name="listed",
            subspace_name="sub",
            layer=1,
            rank=3,
            num_entries=2,
            token_position=-1,
            computed_at="2026-01-01T00:00:00+00:00",
        )
        DarkTableRegistry.get().store(
            "listed",
            {"a": np.zeros(3, dtype=np.float32)},
            meta,
        )
        result = await list_dark_tables()
        assert result["count"] == 1
        assert result["tables"][0]["table_name"] == "listed"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestBuildDarkTableModels:
    def test_dark_table_entry(self) -> None:
        e = DarkTableEntry(key="k", prompt="p", coordinate_norm=1.5)
        d = e.model_dump()
        assert d["key"] == "k"
        assert d["coordinate_norm"] == 1.5

    def test_build_dark_table_result(self) -> None:
        r = BuildDarkTableResult(
            table_name="t",
            subspace_name="s",
            layer=5,
            rank=3,
            num_entries=1,
            token_position=-1,
            entries=[DarkTableEntry(key="k", prompt="p", coordinate_norm=1.0)],
            summary={"stored": True},
        )
        d = r.model_dump()
        assert d["table_name"] == "t"
        assert d["rank"] == 3
        assert len(d["entries"]) == 1
