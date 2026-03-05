"""Tests for experiment_store.py — ExperimentStore singleton."""

import threading
from pathlib import Path

import pytest

from chuk_mcp_lazarus.experiment_store import (
    ExperimentStore,
)


class TestExperimentStoreSingleton:
    """Singleton behaviour and reset."""

    def test_get_returns_same_instance(self) -> None:
        a = ExperimentStore.get()
        b = ExperimentStore.get()
        assert a is b

    def test_reset_clears_instance(self) -> None:
        a = ExperimentStore.get()
        ExperimentStore.reset()
        b = ExperimentStore.get()
        assert a is not b

    def test_thread_safety(self) -> None:
        """Multiple threads get the same singleton."""
        instances: list[ExperimentStore] = []

        def grab() -> None:
            instances.append(ExperimentStore.get())

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(inst is instances[0] for inst in instances)


class TestExperimentStoreCreate:
    """create() method."""

    def test_create_returns_uuid(self) -> None:
        store = ExperimentStore.get()
        eid = store.create("test_exp", "model/test")
        assert len(eid) == 36  # UUID format

    def test_create_with_description_and_tags(self) -> None:
        store = ExperimentStore.get()
        eid = store.create(
            "test_exp",
            "model/test",
            description="An experiment",
            tags=["tag1", "tag2"],
        )
        exp = store.get_experiment(eid)
        assert exp is not None
        assert exp.metadata.description == "An experiment"
        assert exp.metadata.tags == ["tag1", "tag2"]

    def test_create_captures_model_id(self) -> None:
        store = ExperimentStore.get()
        eid = store.create("test_exp", "SmolLM2-135M")
        exp = store.get_experiment(eid)
        assert exp is not None
        assert exp.metadata.model_id == "SmolLM2-135M"


class TestExperimentStoreAddResult:
    """add_result() method."""

    def test_add_result_increments_steps(self) -> None:
        store = ExperimentStore.get()
        eid = store.create("test_exp", "model/test")
        store.add_result(eid, "step_1", {"accuracy": 0.95})
        store.add_result(eid, "step_2", {"loss": 0.05})
        exp = store.get_experiment(eid)
        assert exp is not None
        assert len(exp.steps) == 2
        assert exp.steps[0].step_name == "step_1"
        assert exp.steps[1].step_name == "step_2"

    def test_add_result_stores_data(self) -> None:
        store = ExperimentStore.get()
        eid = store.create("test_exp", "model/test")
        store.add_result(eid, "step_1", {"key": "value", "num": 42})
        exp = store.get_experiment(eid)
        assert exp is not None
        assert exp.steps[0].data["key"] == "value"
        assert exp.steps[0].data["num"] == 42

    def test_add_result_not_found_raises(self) -> None:
        store = ExperimentStore.get()
        with pytest.raises(KeyError):
            store.add_result("nonexistent-id", "step_1", {})


class TestExperimentStoreGetAndList:
    """get_experiment() and list_experiments()."""

    def test_get_nonexistent_returns_none(self) -> None:
        store = ExperimentStore.get()
        assert store.get_experiment("nonexistent") is None

    def test_list_empty(self) -> None:
        store = ExperimentStore.get()
        result = store.list_experiments()
        assert result["count"] == 0
        assert result["experiments"] == []

    def test_list_populated(self) -> None:
        store = ExperimentStore.get()
        store.create("exp1", "model/a")
        store.create("exp2", "model/b")
        result = store.list_experiments()
        assert result["count"] == 2


class TestExperimentStoreDelete:
    """delete_experiment()."""

    def test_delete_removes_experiment(self) -> None:
        store = ExperimentStore.get()
        eid = store.create("test_exp", "model/test")
        assert store.delete_experiment(eid)
        assert store.get_experiment(eid) is None

    def test_delete_nonexistent_returns_false(self) -> None:
        store = ExperimentStore.get()
        assert not store.delete_experiment("nonexistent")


class TestExperimentStoreDisk:
    """Disk persistence: save and load."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        store = ExperimentStore.get()
        store._base_dir = tmp_path

        eid = store.create("test_exp", "model/test", description="round trip")
        store.add_result(eid, "step_1", {"value": 42})

        # Verify file exists
        fpath = tmp_path / f"{eid}.json"
        assert fpath.exists()

        # Clear in-memory and reload
        with store._access_lock:
            store._experiments.clear()
        assert store.get_experiment(eid) is None

        assert store.load_from_disk(eid)
        exp = store.get_experiment(eid)
        assert exp is not None
        assert exp.metadata.name == "test_exp"
        assert len(exp.steps) == 1
        assert exp.steps[0].data["value"] == 42

    def test_load_nonexistent_returns_false(self, tmp_path: Path) -> None:
        store = ExperimentStore.get()
        store._base_dir = tmp_path
        assert not store.load_from_disk("nonexistent")

    def test_create_auto_saves(self, tmp_path: Path) -> None:
        store = ExperimentStore.get()
        store._base_dir = tmp_path
        eid = store.create("auto_save", "model/test")
        assert (tmp_path / f"{eid}.json").exists()

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        store = ExperimentStore.get()
        store._base_dir = tmp_path
        eid = store.create("test_exp", "model/test")
        fpath = tmp_path / f"{eid}.json"
        assert fpath.exists()
        store.delete_experiment(eid)
        assert not fpath.exists()
