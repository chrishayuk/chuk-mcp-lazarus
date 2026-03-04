"""Tests for tools/causal_tools.py — trace_token, full_causal_trace."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.causal_tools import full_causal_trace, trace_token


class TestTraceToken:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await trace_token(prompt="hello", token="world")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await trace_token(prompt="hello", token="world", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_effect_threshold(self, loaded_model_state: MagicMock) -> None:
        result = await trace_token(prompt="hello", token="world", effect_threshold=-1.0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_effect_threshold_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await trace_token(prompt="hello", token="world", effect_threshold=1.5)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_ci = MagicMock()
        mock_trace = MagicMock()
        mock_trace.prompt = "hello"
        mock_trace.target_token = "world"
        mock_trace.target_token_id = 42
        mock_trace.baseline_prob = 0.8
        mock_trace.layer_effects = [(0, 0.5), (1, 0.1)]
        mock_trace.critical_layers = [0]
        mock_trace.peak_layer = 0
        mock_trace.peak_effect = 0.5
        mock_ci.trace_token.return_value = mock_trace

        with patch(
            "chuk_mcp_lazarus.tools.causal_tools.CounterfactualIntervention",
            return_value=mock_ci,
        ):
            result = await trace_token(prompt="hello", token="world", layers=[0, 1])
        assert "error" not in result
        assert result["target_token"] == "world"

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: MagicMock) -> None:
        """When layers=None, should use all layers."""
        mock_ci = MagicMock()
        mock_trace = MagicMock()
        mock_trace.prompt = "hello"
        mock_trace.target_token = "world"
        mock_trace.target_token_id = 42
        mock_trace.baseline_prob = 0.8
        mock_trace.layer_effects = [(i, 0.1) for i in range(4)]
        mock_trace.critical_layers = [0]
        mock_trace.peak_layer = 0
        mock_trace.peak_effect = 0.1
        mock_ci.trace_token.return_value = mock_trace

        with patch(
            "chuk_mcp_lazarus.tools.causal_tools.CounterfactualIntervention",
            return_value=mock_ci,
        ):
            result = await trace_token(prompt="hello", token="world")
        assert "error" not in result


class TestFullCausalTrace:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await full_causal_trace(prompt="hello", token="world")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await full_causal_trace(prompt="hello", token="world", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_ci = MagicMock()
        mock_trace = MagicMock()
        mock_trace.prompt = "hello"
        mock_trace.target_token = "world"
        mock_trace.tokens = ["hello"]
        mock_trace.effects = [[0.5, 0.2]]
        mock_trace.critical_positions = [0]
        mock_trace.critical_layers = [0]
        mock_ci.full_causal_trace.return_value = mock_trace

        with patch(
            "chuk_mcp_lazarus.tools.causal_tools.CounterfactualIntervention",
            return_value=mock_ci,
        ):
            result = await full_causal_trace(prompt="hello", token="world", layers=[0, 1])
        assert "error" not in result
        assert result["target_token"] == "world"
