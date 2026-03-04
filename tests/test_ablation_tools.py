"""Tests for tools/ablation_tools.py — ablate_layers, patch_activations."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.ablation_tools import (
    _word_overlap_similarity,
    ablate_layers,
    patch_activations,
)


class TestWordOverlapSimilarity:
    """Pure function — no mocking needed."""

    def test_identical(self) -> None:
        assert _word_overlap_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert _word_overlap_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _word_overlap_similarity("hello world foo", "hello bar baz")
        assert 0.0 < sim < 1.0

    def test_both_empty(self) -> None:
        assert _word_overlap_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert _word_overlap_similarity("hello", "") == 0.0
        assert _word_overlap_similarity("", "hello") == 0.0

    def test_case_insensitive(self) -> None:
        assert _word_overlap_similarity("Hello World", "hello world") == 1.0


class TestAblateLayers:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], max_new_tokens=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_tokens_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], max_new_tokens=1001)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_ablation_type(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], ablation_type="random")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_component(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], component="invalid")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "ablated_layers": [0],
            "ablation_type": "zero",
            "component": "mlp",
            "ablated_output": "output",
            "baseline_output": "output",
            "output_similarity": 1.0,
            "disruption_score": 0.0,
        }
        with patch(
            "chuk_mcp_lazarus.tools.ablation_tools._ablate_layers_impl",
            return_value=mock_result,
        ):
            result = await ablate_layers(prompt="hello", layers=[0])
        assert "error" not in result
        assert result["prompt"] == "hello"


class TestPatchActivations:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await patch_activations(source_prompt="a", target_prompt="b", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        result = await patch_activations(
            source_prompt="a", target_prompt="b", layer=0, max_new_tokens=0
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await patch_activations(source_prompt="a", target_prompt="b", layer=99)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "source_prompt": "a",
            "target_prompt": "b",
            "patched_layer": 0,
            "patched_output": "out",
            "baseline_output": "base",
            "source_output": "src",
            "recovery_rate": 0.7,
            "effect_size": 0.2,
        }
        with patch(
            "chuk_mcp_lazarus.tools.ablation_tools._patch_activations_impl",
            return_value=mock_result,
        ):
            result = await patch_activations(source_prompt="a", target_prompt="b", layer=0)
        assert "error" not in result
        assert result["patched_layer"] == 0
