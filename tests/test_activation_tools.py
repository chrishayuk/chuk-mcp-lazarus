"""Tests for tools/activation_tools.py — extract_activations, compare_activations."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.activation_tools import (
    _token_text,
    compare_activations,
    extract_activations,
)


class TestTokenText:
    def test_positive_position(self) -> None:
        import mlx.core as mx

        tok = MagicMock()
        tok.decode.return_value = "hello"
        ids = mx.array([10, 20, 30])
        result = _token_text(tok, ids, 0)
        assert isinstance(result, str)

    def test_negative_position(self) -> None:
        import mlx.core as mx

        tok = MagicMock()
        tok.decode.return_value = "last"
        ids = mx.array([10, 20, 30])
        result = _token_text(tok, ids, -1)
        assert isinstance(result, str)


class TestExtractActivations:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await extract_activations(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await extract_activations(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "num_tokens": 5,
            "activations": {"0": [0.1, 0.2, 0.3]},
        }
        with patch(
            "chuk_mcp_lazarus.tools.activation_tools._extract_activations_impl",
            return_value=mock_result,
        ):
            result = await extract_activations(prompt="hello", layers=[0])
        assert "error" not in result
        assert result["prompt"] == "hello"


class TestCompareActivations:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await compare_activations(prompts=["a", "b"], layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_too_few_prompts(self, loaded_model_state: MagicMock) -> None:
        result = await compare_activations(prompts=["only_one"], layer=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_prompts(self, loaded_model_state: MagicMock) -> None:
        result = await compare_activations(prompts=["a"] * 9, layer=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await compare_activations(prompts=["a", "b"], layer=99)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "layer": 0,
            "prompts": ["a", "b"],
            "cosine_similarity_matrix": [[1.0, 0.9], [0.9, 1.0]],
            "pca_2d": [[0.0, 0.0], [1.0, 1.0]],
            "centroid_distance": 0.1,
        }
        with patch(
            "chuk_mcp_lazarus.tools.activation_tools._compare_activations_impl",
            return_value=mock_result,
        ):
            result = await compare_activations(prompts=["a", "b"], layer=0)
        assert "error" not in result
        assert result["layer"] == 0
