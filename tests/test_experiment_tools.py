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
