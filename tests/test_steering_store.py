"""Tests for steering_store.py — SteeringVectorRegistry singleton."""

import numpy as np

from chuk_mcp_lazarus.steering_store import (
    SteeringVectorRegistry,
    VectorListEntry,
    VectorMetadata,
    VectorRegistryDump,
)


def _make_metadata(name: str = "test_vec", layer: int = 0) -> VectorMetadata:
    return VectorMetadata(
        name=name,
        layer=layer,
        vector_norm=1.5,
        separability_score=0.8,
        num_positive=10,
        num_negative=10,
        computed_at="2024-01-01T00:00:00",
    )


class TestSteeringVectorRegistry:
    def test_singleton(self) -> None:
        a = SteeringVectorRegistry.get()
        b = SteeringVectorRegistry.get()
        assert a is b

    def test_store_and_fetch(self) -> None:
        reg = SteeringVectorRegistry.get()
        vec = np.array([1.0, 2.0, 3.0])
        meta = _make_metadata()
        reg.store("test", vec, meta)

        result = reg.fetch("test")
        assert result is not None
        np.testing.assert_array_equal(result[0], vec)
        assert result[1] is meta

    def test_fetch_missing(self) -> None:
        reg = SteeringVectorRegistry.get()
        assert reg.fetch("nonexistent") is None

    def test_exists(self) -> None:
        reg = SteeringVectorRegistry.get()
        reg.store("exists_test", np.zeros(3), _make_metadata("exists_test"))
        assert reg.exists("exists_test") is True
        assert reg.exists("nope") is False

    def test_list_all(self) -> None:
        reg = SteeringVectorRegistry.get()
        reg.store("v1", np.zeros(3), _make_metadata("v1", 0))
        reg.store("v2", np.zeros(3), _make_metadata("v2", 1))

        entries = reg.list_all()
        assert len(entries) >= 2
        assert all(isinstance(e, VectorListEntry) for e in entries)

    def test_count(self) -> None:
        reg = SteeringVectorRegistry.get()
        reg.clear()
        assert reg.count == 0
        reg.store("c1", np.zeros(3), _make_metadata("c1"))
        assert reg.count == 1

    def test_clear(self) -> None:
        reg = SteeringVectorRegistry.get()
        reg.store("clear_test", np.zeros(3), _make_metadata("clear_test"))
        reg.clear()
        assert reg.count == 0

    def test_dump(self) -> None:
        reg = SteeringVectorRegistry.get()
        reg.clear()
        reg.store("d1", np.zeros(3), _make_metadata("d1"))
        dump = reg.dump()
        assert isinstance(dump, VectorRegistryDump)
        assert dump.count == 1
        assert len(dump.vectors) == 1

    def test_overwrite(self) -> None:
        reg = SteeringVectorRegistry.get()
        meta1 = _make_metadata("ow", layer=0)
        meta2 = _make_metadata("ow", layer=5)
        reg.store("ow", np.zeros(3), meta1)
        reg.store("ow", np.ones(3), meta2)
        result = reg.fetch("ow")
        assert result is not None
        assert result[1].layer == 5
        np.testing.assert_array_equal(result[0], np.ones(3))
