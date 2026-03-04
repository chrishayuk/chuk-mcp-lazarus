"""Tests for tools/residual_tools.py — residual stream decomposition tools."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.residual_tools import (
    head_attribution,
    layer_clustering,
    logit_attribution,
    residual_decomposition,
    top_neurons,
)


class TestResidualDecomposition:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await residual_decomposition(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await residual_decomposition(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "decomposition": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._residual_decomposition_impl",
            return_value=mock_result,
        ):
            result = await residual_decomposition(prompt="hello", layers=[0])
        assert "error" not in result


class TestLayerClustering:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await layer_clustering(prompts=["a", "b"])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_too_few_prompts(self, loaded_model_state: MagicMock) -> None:
        result = await layer_clustering(prompts=["only_one"])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompts": ["a", "b"],
            "num_prompts": 2,
            "num_layers_analyzed": 1,
            "layers": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._layer_clustering_impl",
            return_value=mock_result,
        ):
            result = await layer_clustering(prompts=["a", "b"])
        assert "error" not in result


class TestLogitAttribution:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await logit_attribution(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "target_token": "world",
            "target_token_id": 42,
            "model_logit": 5.0,
            "attribution_sum": 5.0,
            "attributions": [],
            "top_positive_layers": [],
            "top_negative_layers": [],
            "normalized": True,
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._logit_attribution_impl",
            return_value=mock_result,
        ):
            result = await logit_attribution(prompt="hello")
        assert "error" not in result


class TestHeadAttribution:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await head_attribution(prompt="hello", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await head_attribution(prompt="hello", layer=99)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "target_token": "world",
            "target_token_id": 42,
            "layer_total_logit": 1.0,
            "heads": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._head_attribution_impl",
            return_value=mock_result,
        ):
            result = await head_attribution(prompt="hello", layer=0)
        assert "error" not in result


class TestTopNeurons:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await top_neurons(prompt="hello", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await top_neurons(prompt="hello", layer=99)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await top_neurons(prompt="hello", layer=0, top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "target_token": "world",
            "target_token_id": 42,
            "top_k": 10,
            "neurons": [],
            "total_neuron_logit": 0.0,
            "top_neuron_logit": 0.0,
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._top_neurons_impl",
            return_value=mock_result,
        ):
            result = await top_neurons(prompt="hello", layer=0)
        assert "error" not in result
