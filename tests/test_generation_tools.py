"""Tests for tools/generation_tools.py — generate_text, predict_next_token, etc."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.tools.generation_tools import (
    generate_text,
    predict_next_token,
    tokenize,
)


class TestGenerateText:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await generate_text(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        result = await generate_text(prompt="hello", max_new_tokens=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "generated_text": "world",
            "num_tokens_generated": 1,
            "finish_reason": "max_tokens",
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._generate_text_impl",
            return_value=mock_result,
        ):
            result = await generate_text(prompt="hello", max_new_tokens=10)
        assert "error" not in result
        assert result["prompt"] == "hello"


class TestPredictNextToken:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await predict_next_token(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await predict_next_token(prompt="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_top_k_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await predict_next_token(prompt="hello", top_k=1001)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "top_predictions": [{"token": "world", "probability": 0.5, "log_probability": -0.7}],
            "num_candidates": 1,
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._predict_next_token_impl",
            return_value=mock_result,
        ):
            result = await predict_next_token(prompt="hello", top_k=5)
        assert "error" not in result


class TestTokenize:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await tokenize(text="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        result = await tokenize(text="hello world")
        assert "error" not in result
        assert "tokens" in result or "token_ids" in result
