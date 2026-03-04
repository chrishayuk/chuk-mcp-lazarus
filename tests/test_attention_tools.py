"""Tests for tools/attention_tools.py — attention_pattern, attention_heads."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.attention_tools import attention_heads, attention_pattern


class TestAttentionPattern:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_top_k_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello", top_k=101)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "tokens": ["hello"],
            "num_layers_analyzed": 1,
            "patterns": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_pattern_impl",
            return_value=mock_result,
        ):
            result = await attention_pattern(prompt="hello", layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "tokens": ["hello"],
            "num_layers_analyzed": 3,
            "patterns": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_pattern_impl",
            return_value=mock_result,
        ):
            result = await attention_pattern(prompt="hello")
        assert "error" not in result


class TestAttentionHeads:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await attention_heads(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await attention_heads(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await attention_heads(prompt="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "tokens": ["hello"],
            "num_heads_analyzed": 4,
            "heads": [],
            "summary": {"most_focused_heads": [], "most_diffuse_heads": []},
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_heads_impl",
            return_value=mock_result,
        ):
            result = await attention_heads(prompt="hello", layers=[0])
        assert "error" not in result
