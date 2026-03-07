"""Tests for DarkTableRegistry singleton."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from chuk_mcp_lazarus.dark_table_registry import (
    DarkTableListEntry,
    DarkTableMetadata,
    DarkTableRegistry,
    DarkTableRegistryDump,
)


def _make_meta(
    table_name: str = "test_table",
    subspace_name: str = "test_subspace",
    layer: int = 5,
    rank: int = 3,
    num_entries: int = 2,
    token_position: int = -1,
) -> DarkTableMetadata:
    return DarkTableMetadata(
        table_name=table_name,
        subspace_name=subspace_name,
        layer=layer,
        rank=rank,
        num_entries=num_entries,
        token_position=token_position,
        computed_at="2026-01-01T00:00:00+00:00",
    )


def _make_coords(rank: int = 3) -> dict[str, np.ndarray]:
    return {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32)[:rank],
        "b": np.array([4.0, 5.0, 6.0], dtype=np.float32)[:rank],
    }


class TestDarkTableRegistry:
    def test_singleton(self) -> None:
        r1 = DarkTableRegistry.get()
        r2 = DarkTableRegistry.get()
        assert r1 is r2

    def test_store_and_fetch(self) -> None:
        reg = DarkTableRegistry.get()
        coords = _make_coords()
        meta = _make_meta(table_name="t1")
        reg.store("t1", coords, meta)
        result = reg.fetch("t1")
        assert result is not None
        fetched_coords, fetched_meta = result
        assert "a" in fetched_coords
        assert "b" in fetched_coords
        np.testing.assert_array_equal(fetched_coords["a"], coords["a"])
        assert fetched_meta.table_name == "t1"

    def test_fetch_missing(self) -> None:
        reg = DarkTableRegistry.get()
        assert reg.fetch("nonexistent") is None

    def test_lookup(self) -> None:
        reg = DarkTableRegistry.get()
        coords = _make_coords()
        reg.store("t2", coords, _make_meta(table_name="t2"))
        vec = reg.lookup("t2", "a")
        assert vec is not None
        np.testing.assert_array_equal(vec, coords["a"])

    def test_lookup_missing_table(self) -> None:
        reg = DarkTableRegistry.get()
        assert reg.lookup("no_table", "a") is None

    def test_lookup_missing_key(self) -> None:
        reg = DarkTableRegistry.get()
        reg.store("t3", _make_coords(), _make_meta(table_name="t3"))
        assert reg.lookup("t3", "no_key") is None

    def test_exists(self) -> None:
        reg = DarkTableRegistry.get()
        reg.store("t4", _make_coords(), _make_meta(table_name="t4"))
        assert reg.exists("t4") is True
        assert reg.exists("missing") is False

    def test_list_all(self) -> None:
        reg = DarkTableRegistry.get()
        reg.store("t5", _make_coords(), _make_meta(table_name="t5"))
        entries = reg.list_all()
        names = [e.table_name for e in entries]
        assert "t5" in names

    def test_clear(self) -> None:
        reg = DarkTableRegistry.get()
        reg.store("t6", _make_coords(), _make_meta(table_name="t6"))
        assert reg.count > 0
        reg.clear()
        assert reg.count == 0

    def test_count(self) -> None:
        reg = DarkTableRegistry.get()
        reg.clear()
        assert reg.count == 0
        reg.store("t7", _make_coords(), _make_meta(table_name="t7"))
        assert reg.count == 1

    def test_dump(self) -> None:
        reg = DarkTableRegistry.get()
        reg.clear()
        reg.store("t8", _make_coords(), _make_meta(table_name="t8"))
        d = reg.dump()
        assert isinstance(d, DarkTableRegistryDump)
        assert d.count == 1
        assert len(d.tables) == 1
        assert d.tables[0].table_name == "t8"

    def test_overwrite(self) -> None:
        reg = DarkTableRegistry.get()
        reg.store("t9", _make_coords(), _make_meta(table_name="t9", num_entries=2))
        new_coords = {"x": np.array([9.0, 9.0, 9.0], dtype=np.float32)}
        reg.store("t9", new_coords, _make_meta(table_name="t9", num_entries=1))
        result = reg.fetch("t9")
        assert result is not None
        fetched_coords, fetched_meta = result
        assert "x" in fetched_coords
        assert "a" not in fetched_coords
        assert fetched_meta.num_entries == 1

    def test_metadata_frozen(self) -> None:
        meta = _make_meta()
        with pytest.raises(ValidationError):
            meta.table_name = "changed"  # type: ignore[misc]

    def test_metadata_model_dump(self) -> None:
        meta = _make_meta()
        d = meta.model_dump()
        assert d["table_name"] == "test_table"
        assert d["subspace_name"] == "test_subspace"
        assert d["layer"] == 5
        assert d["rank"] == 3
        assert d["num_entries"] == 2
        assert d["token_position"] == -1

    def test_list_entry_model(self) -> None:
        entry = DarkTableListEntry(
            table_name="t",
            subspace_name="s",
            layer=1,
            rank=2,
            num_entries=3,
            computed_at="2026-01-01T00:00:00+00:00",
        )
        d = entry.model_dump()
        assert d["table_name"] == "t"
        assert d["rank"] == 2
