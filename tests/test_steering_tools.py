"""Tests for tools/steering_tools.py — steering vector tools."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.steering_store import SteeringVectorRegistry, VectorMetadata
from chuk_mcp_lazarus.tools.steering_tools import (
    _mean_pairwise_similarity,
    _mean_vector,
    compute_steering_vector,
    list_steering_vectors,
    steer_and_generate,
)


class TestMeanVector:
    def test_basic(self) -> None:
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        result = _mean_vector(vecs)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_single(self) -> None:
        result = _mean_vector([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


class TestMeanPairwiseSimilarity:
    def test_single_vector(self) -> None:
        assert _mean_pairwise_similarity([[1.0, 0.0]]) == 1.0

    def test_identical_vectors(self) -> None:
        sim = _mean_pairwise_similarity([[1.0, 0.0], [1.0, 0.0]])
        assert sim == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_vectors(self) -> None:
        sim = _mean_pairwise_similarity([[1.0, 0.0], [0.0, 1.0]])
        assert sim == pytest.approx(0.0, abs=1e-4)


class TestComputeSteeringVector:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await compute_steering_vector(
            vector_name="test",
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await compute_steering_vector(
            vector_name="test",
            layer=99,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_positive(self, loaded_model_state: MagicMock) -> None:
        result = await compute_steering_vector(
            vector_name="test",
            layer=0,
            positive_prompts=["a"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_few_negative(self, loaded_model_state: MagicMock) -> None:
        result = await compute_steering_vector(
            vector_name="test",
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["c"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "vector_name": "test",
            "layer": 0,
            "vector_norm": 1.5,
            "cosine_similarity_within_positive": 0.9,
            "cosine_similarity_within_negative": 0.85,
            "separability_score": 0.7,
            "num_positive": 2,
            "num_negative": 2,
        }
        with patch(
            "chuk_mcp_lazarus.tools.steering_tools._compute_steering_vector_impl",
            return_value=mock_result,
        ):
            result = await compute_steering_vector(
                vector_name="test",
                layer=0,
                positive_prompts=["a", "b"],
                negative_prompts=["c", "d"],
            )
        assert "error" not in result
        assert result["vector_name"] == "test"


class TestSteerAndGenerate:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await steer_and_generate(prompt="hello", vector_name="test")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_vector_not_found(self, loaded_model_state: MagicMock) -> None:
        result = await steer_and_generate(prompt="hello", vector_name="nonexistent")
        assert result["error"] is True
        assert result["error_type"] == "VectorNotFound"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        # Store a vector first
        reg = SteeringVectorRegistry.get()
        meta = VectorMetadata(
            name="test_vec",
            layer=0,
            vector_norm=1.0,
            separability_score=0.5,
            num_positive=2,
            num_negative=2,
            computed_at="2024-01-01",
        )
        reg.store("test_vec", np.zeros(64), meta)

        result = await steer_and_generate(
            prompt="hello", vector_name="test_vec", max_new_tokens=0
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        reg = SteeringVectorRegistry.get()
        meta = VectorMetadata(
            name="test_vec2",
            layer=0,
            vector_norm=1.0,
            separability_score=0.5,
            num_positive=2,
            num_negative=2,
            computed_at="2024-01-01",
        )
        reg.store("test_vec2", np.zeros(64), meta)

        mock_result = {
            "prompt": "hello",
            "vector_name": "test_vec2",
            "alpha": 20.0,
            "layer": 0,
            "steered_output": "steered",
            "baseline_output": "baseline",
            "steered_tokens": 5,
            "baseline_tokens": 5,
        }
        with patch(
            "chuk_mcp_lazarus.tools.steering_tools._steer_and_generate_impl",
            return_value=mock_result,
        ):
            result = await steer_and_generate(prompt="hello", vector_name="test_vec2")
        assert "error" not in result


class TestListSteeringVectors:
    @pytest.mark.asyncio
    async def test_empty(self) -> None:
        result = await list_steering_vectors()
        assert "error" not in result
        assert "count" in result
