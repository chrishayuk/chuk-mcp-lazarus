"""Tests for probe_store.py — ProbeRegistry singleton."""

from unittest.mock import MagicMock

from chuk_mcp_lazarus.probe_store import (
    ProbeListEntry,
    ProbeMetadata,
    ProbeRegistry,
    ProbeRegistryDump,
    ProbeType,
)


def _make_metadata(name: str = "test_probe", layer: int = 0) -> ProbeMetadata:
    return ProbeMetadata(
        name=name,
        layer=layer,
        probe_type=ProbeType.LINEAR,
        classes=["a", "b"],
        num_examples=100,
        train_accuracy=0.95,
        val_accuracy=0.90,
        trained_at="2024-01-01T00:00:00",
    )


class TestProbeType:
    def test_values(self) -> None:
        assert ProbeType.LINEAR == "linear"
        assert ProbeType.MLP == "mlp"


class TestProbeRegistry:
    def test_singleton(self) -> None:
        a = ProbeRegistry.get()
        b = ProbeRegistry.get()
        assert a is b

    def test_store_and_fetch(self) -> None:
        reg = ProbeRegistry.get()
        meta = _make_metadata()
        model = MagicMock()
        reg.store("test", model, meta)

        result = reg.fetch("test")
        assert result is not None
        assert result[0] is model
        assert result[1] is meta

    def test_fetch_missing(self) -> None:
        reg = ProbeRegistry.get()
        assert reg.fetch("nonexistent") is None

    def test_exists(self) -> None:
        reg = ProbeRegistry.get()
        reg.store("exists_test", MagicMock(), _make_metadata("exists_test"))
        assert reg.exists("exists_test") is True
        assert reg.exists("nope") is False

    def test_list_all(self) -> None:
        reg = ProbeRegistry.get()
        reg.store("p1", MagicMock(), _make_metadata("p1", 0))
        reg.store("p2", MagicMock(), _make_metadata("p2", 1))

        entries = reg.list_all()
        assert len(entries) >= 2
        assert all(isinstance(e, ProbeListEntry) for e in entries)

    def test_count(self) -> None:
        reg = ProbeRegistry.get()
        reg.clear()
        assert reg.count == 0
        reg.store("c1", MagicMock(), _make_metadata("c1"))
        assert reg.count == 1

    def test_clear(self) -> None:
        reg = ProbeRegistry.get()
        reg.store("clear_test", MagicMock(), _make_metadata("clear_test"))
        reg.clear()
        assert reg.count == 0
        assert reg.fetch("clear_test") is None

    def test_dump(self) -> None:
        reg = ProbeRegistry.get()
        reg.clear()
        reg.store("d1", MagicMock(), _make_metadata("d1"))
        dump = reg.dump()
        assert isinstance(dump, ProbeRegistryDump)
        assert dump.count == 1
        assert len(dump.probes) == 1

    def test_overwrite(self) -> None:
        reg = ProbeRegistry.get()
        meta1 = _make_metadata("ow", layer=0)
        meta2 = _make_metadata("ow", layer=5)
        reg.store("ow", MagicMock(), meta1)
        reg.store("ow", MagicMock(), meta2)
        result = reg.fetch("ow")
        assert result is not None
        assert result[1].layer == 5
