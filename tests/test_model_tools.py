"""Tests for tools/model_tools.py — load_model, get_model_info."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.model_tools import get_model_info, load_model


class TestLoadModel:
    @pytest.mark.asyncio
    async def test_invalid_dtype(self) -> None:
        result = await load_model(model_id="test", dtype="invalid")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from chuk_mcp_lazarus.model_state import ModelMetadata

        meta = ModelMetadata(
            model_id="test/model",
            family="test",
            architecture="test_arch",
            num_layers=4,
            hidden_dim=64,
            num_attention_heads=4,
            parameter_count=1000,
        )
        state = MagicMock()
        state.load.return_value = meta
        with patch("chuk_mcp_lazarus.tools.model_tools.ModelState.get", return_value=state):
            result = await load_model(model_id="test/model", dtype="bfloat16")
        assert "error" not in result
        assert result["model_id"] == "test/model"
        assert result["status"] == "loaded"

    @pytest.mark.asyncio
    async def test_load_failure(self) -> None:
        state = MagicMock()
        state.load.side_effect = RuntimeError("boom")
        with patch("chuk_mcp_lazarus.tools.model_tools.ModelState.get", return_value=state):
            result = await load_model(model_id="bad/model")
        assert result["error"] is True
        assert result["error_type"] == "LoadFailed"


class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await get_model_info()
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_loaded(self, loaded_model_state: MagicMock) -> None:
        result = await get_model_info()
        assert "error" not in result
        assert result["model_id"] == "test/model"
        assert result["num_layers"] == 4
