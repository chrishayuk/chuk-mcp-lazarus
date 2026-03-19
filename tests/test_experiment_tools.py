"""Tests for tools/experiment_tools.py — experiment persistence tools."""

from unittest.mock import MagicMock

import pytest

from chuk_mcp_lazarus.tools.experiment_tools import (
    add_experiment_result,
    create_experiment,
    get_experiment,
    list_experiments,
)


# ---------------------------------------------------------------------------
# TestCreateExperiment
# ---------------------------------------------------------------------------


class TestCreateExperiment:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await create_experiment(name="test")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        result = await create_experiment(name="my_exp", description="test desc", tags=["a", "b"])
        assert "error" not in result
        assert "experiment_id" in result
        assert result["name"] == "my_exp"
        assert result["model_id"] == "test/model"
        assert result["tags"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_default_tags(self, loaded_model_state: MagicMock) -> None:
        result = await create_experiment(name="my_exp")
        assert "error" not in result
        assert result["tags"] == []


# ---------------------------------------------------------------------------
# TestAddExperimentResult
# ---------------------------------------------------------------------------


class TestAddExperimentResult:
    @pytest.mark.asyncio
    async def test_not_found(self, loaded_model_state: MagicMock) -> None:
        result = await add_experiment_result(
            experiment_id="nonexistent",
            step_name="step1",
            result={"key": "val"},
        )
        assert result["error"] is True
        assert result["error_type"] == "ExperimentNotFound"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        create_result = await create_experiment(name="test_exp")
        eid = create_result["experiment_id"]

        result = await add_experiment_result(
            experiment_id=eid,
            step_name="logit_attribution",
            result={"accuracy": 0.95},
        )
        assert "error" not in result
        assert result["experiment_id"] == eid
        assert result["step_name"] == "logit_attribution"
        assert result["total_steps"] == 1

    @pytest.mark.asyncio
    async def test_multiple_steps(self, loaded_model_state: MagicMock) -> None:
        create_result = await create_experiment(name="multi")
        eid = create_result["experiment_id"]

        await add_experiment_result(eid, "step1", {"a": 1})
        result = await add_experiment_result(eid, "step2", {"b": 2})
        assert result["total_steps"] == 2


# ---------------------------------------------------------------------------
# TestGetExperiment
# ---------------------------------------------------------------------------


class TestGetExperiment:
    @pytest.mark.asyncio
    async def test_not_found(self, loaded_model_state: MagicMock) -> None:
        result = await get_experiment(experiment_id="nonexistent")
        assert result["error"] is True
        assert result["error_type"] == "ExperimentNotFound"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        create_result = await create_experiment(name="get_test")
        eid = create_result["experiment_id"]
        await add_experiment_result(eid, "step1", {"val": 42})

        result = await get_experiment(experiment_id=eid)
        assert "error" not in result
        assert result["metadata"]["name"] == "get_test"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["data"]["val"] == 42


# ---------------------------------------------------------------------------
# TestListExperiments
# ---------------------------------------------------------------------------


class TestListExperiments:
    @pytest.mark.asyncio
    async def test_empty(self, loaded_model_state: MagicMock) -> None:
        result = await list_experiments()
        assert result["count"] == 0
        assert result["experiments"] == []

    @pytest.mark.asyncio
    async def test_populated(self, loaded_model_state: MagicMock) -> None:
        await create_experiment(name="exp1")
        await create_experiment(name="exp2")
        result = await list_experiments()
        assert result["count"] == 2


# ---------------------------------------------------------------------------
# Exception-path tests
# ---------------------------------------------------------------------------


class TestCreateExperimentError:
    """Test exception paths in create_experiment (lines 60-62)."""

    @pytest.mark.asyncio
    async def test_store_exception(self, loaded_model_state: MagicMock) -> None:
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        mock_store.create.side_effect = RuntimeError("disk full")
        with patch("chuk_mcp_lazarus.tools.experiment_tools.ExperimentStore") as mock_es:
            mock_es.get.return_value = mock_store
            result = await create_experiment(name="bad_exp")
        assert result["error"] is True
        assert result["error_type"] == "ExperimentStoreError"


class TestAddExperimentResultError:
    """Test general exception path in add_experiment_result (lines 91-93)."""

    @pytest.mark.asyncio
    async def test_store_general_exception(self, loaded_model_state: MagicMock) -> None:
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        mock_store.add_result.side_effect = IOError("write failed")
        with patch("chuk_mcp_lazarus.tools.experiment_tools.ExperimentStore") as mock_es:
            mock_es.get.return_value = mock_store
            result = await add_experiment_result(
                experiment_id="any_id",
                step_name="step",
                result={"val": 1},
            )
        assert result["error"] is True
        assert result["error_type"] == "ExperimentStoreError"


class TestGetExperimentDiskFallback:
    """Test disk fallback path in get_experiment (line 118)."""

    @pytest.mark.asyncio
    async def test_load_from_disk_success(self, loaded_model_state: MagicMock) -> None:
        from unittest.mock import MagicMock, patch

        import datetime

        from chuk_mcp_lazarus.experiment_store import ExperimentDetail, ExperimentMetadata

        mock_store = MagicMock()

        # Build a minimal ExperimentDetail for model_dump
        meta = ExperimentMetadata(
            experiment_id="some_disk_id",
            name="disk_exp",
            model_id="test/model",
            created_at=datetime.datetime.now().isoformat(),
        )
        real_exp = ExperimentDetail(metadata=meta, steps=[])

        call_count = [0]

        def fake_get(eid):
            if call_count[0] == 0:
                call_count[0] += 1
                return None  # not in memory on first call
            return real_exp

        mock_store.get_experiment.side_effect = fake_get
        mock_store.load_from_disk.return_value = True

        with patch("chuk_mcp_lazarus.tools.experiment_tools.ExperimentStore") as mock_es:
            mock_es.get.return_value = mock_store
            result = await get_experiment(experiment_id="some_disk_id")

        # load_from_disk should have been called once
        mock_store.load_from_disk.assert_called_once_with("some_disk_id")
        assert "error" not in result
        assert result["metadata"]["name"] == "disk_exp"
