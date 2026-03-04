"""Tests for resources.py — MCP resource functions."""

from unittest.mock import MagicMock, patch

from chuk_mcp_lazarus.model_state import ModelMetadata
from chuk_mcp_lazarus.resources import (
    comparison_state_resource,
    model_info_resource,
    probes_registry_resource,
    vectors_registry_resource,
)


class TestModelInfoResource:
    def test_not_loaded(self) -> None:
        state = MagicMock()
        state.is_loaded = False
        with patch("chuk_mcp_lazarus.resources.ModelState.get", return_value=state):
            result = model_info_resource()
        assert result == {"loaded": False}

    def test_loaded(self) -> None:
        state = MagicMock()
        state.is_loaded = True
        state.metadata.model_dump.return_value = {"model_id": "test", "num_layers": 4}
        with patch("chuk_mcp_lazarus.resources.ModelState.get", return_value=state):
            result = model_info_resource()
        assert result["loaded"] is True
        assert result["model_id"] == "test"


class TestProbesRegistryResource:
    def test_empty(self) -> None:
        reg = MagicMock()
        reg.dump.return_value.model_dump.return_value = {"probes": [], "count": 0}
        with patch("chuk_mcp_lazarus.resources.ProbeRegistry.get", return_value=reg):
            result = probes_registry_resource()
        assert result["count"] == 0
        assert result["probes"] == []


class TestVectorsRegistryResource:
    def test_empty(self) -> None:
        reg = MagicMock()
        reg.dump.return_value.model_dump.return_value = {"vectors": [], "count": 0}
        with patch("chuk_mcp_lazarus.resources.SteeringVectorRegistry.get", return_value=reg):
            result = vectors_registry_resource()
        assert result["count"] == 0


class TestComparisonStateResource:
    def test_not_loaded(self) -> None:
        comp = MagicMock()
        comp.is_loaded = False
        with patch("chuk_mcp_lazarus.resources.ComparisonState.get", return_value=comp):
            result = comparison_state_resource()
        assert result == {"loaded": False}

    def test_loaded(self) -> None:
        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata.model_dump.return_value = {"model_id": "comp", "num_layers": 4}
        with patch("chuk_mcp_lazarus.resources.ComparisonState.get", return_value=comp):
            result = comparison_state_resource()
        assert result["loaded"] is True
        assert result["model_id"] == "comp"
