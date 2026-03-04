"""Tests for tools/steering_tools.py — steering vector tools."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.steering_store import SteeringVectorRegistry, VectorMetadata
from chuk_mcp_lazarus.tools.steering_tools import (
    _compute_steering_vector_impl,
    _mean_pairwise_similarity,
    _mean_vector,
    compute_steering_vector,
    list_steering_vectors,
    steer_and_generate,
)


# ---------------------------------------------------------------------------
# _mean_vector
# ---------------------------------------------------------------------------


class TestMeanVector:
    def test_basic(self) -> None:
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        result = _mean_vector(vecs)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_single(self) -> None:
        result = _mean_vector([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# _mean_pairwise_similarity
# ---------------------------------------------------------------------------


class TestMeanPairwiseSimilarity:
    def test_single_vector(self) -> None:
        assert _mean_pairwise_similarity([[1.0, 0.0]]) == 1.0

    def test_identical_vectors(self) -> None:
        sim = _mean_pairwise_similarity([[1.0, 0.0], [1.0, 0.0]])
        assert sim == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_vectors(self) -> None:
        sim = _mean_pairwise_similarity([[1.0, 0.0], [0.0, 1.0]])
        assert sim == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_steering_vector
# ---------------------------------------------------------------------------


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

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        """When _compute_steering_vector_impl raises, returns ExtractionFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.steering_tools._compute_steering_vector_impl",
            side_effect=RuntimeError("extract boom"),
        ):
            result = await compute_steering_vector(
                vector_name="test",
                layer=0,
                positive_prompts=["a", "b"],
                negative_prompts=["c", "d"],
            )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "extract boom" in result["message"]


# ---------------------------------------------------------------------------
# _compute_steering_vector_impl
# ---------------------------------------------------------------------------


class TestComputeSteeringVectorImpl:
    def test_impl_with_mocked_extraction(self) -> None:
        """_compute_steering_vector_impl computes direction and stores vector."""
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_tokenizer = MagicMock()

        # Return deterministic vectors so we can verify the math
        # Positive centroid = [1, 0, 0, 0], Negative centroid = [0, 1, 0, 0]
        call_count = {"n": 0}
        pos_vecs = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        neg_vecs = [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        all_vecs = pos_vecs + neg_vecs

        def mock_extract(model, config, tokenizer, prompt, layer, token_position):
            idx = call_count["n"]
            call_count["n"] += 1
            return all_vecs[idx]

        with patch(
            "chuk_mcp_lazarus.tools.steering_tools.extract_activation_at_layer",
            side_effect=mock_extract,
        ):
            result = _compute_steering_vector_impl(
                model=mock_model,
                config=mock_config,
                tokenizer=mock_tokenizer,
                vector_name="impl_test",
                layer=2,
                positive_prompts=["pos1", "pos2"],
                negative_prompts=["neg1", "neg2"],
                token_position=-1,
            )

        assert "error" not in result
        assert result["vector_name"] == "impl_test"
        assert result["layer"] == 2
        assert result["num_positive"] == 2
        assert result["num_negative"] == 2
        # direction = [1,0,0,0] - [0,1,0,0] = [1,-1,0,0], norm = sqrt(2)
        assert result["vector_norm"] == pytest.approx(np.sqrt(2), abs=1e-3)
        # Within-group similarity should be 1.0 (identical vectors)
        assert result["cosine_similarity_within_positive"] == pytest.approx(1.0, abs=1e-3)
        assert result["cosine_similarity_within_negative"] == pytest.approx(1.0, abs=1e-3)
        # Separability: cos([1,0,0,0], [0,1,0,0]) = 0 => separability = 1.0
        assert result["separability_score"] == pytest.approx(1.0, abs=1e-3)

        # Verify stored in registry
        reg = SteeringVectorRegistry.get()
        entry = reg.fetch("impl_test")
        assert entry is not None
        direction, meta = entry
        np.testing.assert_allclose(direction, [1.0, -1.0, 0.0, 0.0], atol=1e-5)


# ---------------------------------------------------------------------------
# steer_and_generate
# ---------------------------------------------------------------------------


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

        result = await steer_and_generate(prompt="hello", vector_name="test_vec", max_new_tokens=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_tokens_too_high(self, loaded_model_state: MagicMock) -> None:
        """max_new_tokens=1001 exceeds the 1-1000 range."""
        reg = SteeringVectorRegistry.get()
        meta = VectorMetadata(
            name="test_vec_high",
            layer=0,
            vector_norm=1.0,
            separability_score=0.5,
            num_positive=2,
            num_negative=2,
            computed_at="2024-01-01",
        )
        reg.store("test_vec_high", np.zeros(64), meta)

        result = await steer_and_generate(
            prompt="hello", vector_name="test_vec_high", max_new_tokens=1001
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "1-1000" in result["message"]

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

    @pytest.mark.asyncio
    async def test_exception_returns_generation_failed(self, loaded_model_state: MagicMock) -> None:
        """When _steer_and_generate_impl raises, returns GenerationFailed."""
        reg = SteeringVectorRegistry.get()
        meta = VectorMetadata(
            name="test_vec_exc",
            layer=0,
            vector_norm=1.0,
            separability_score=0.5,
            num_positive=2,
            num_negative=2,
            computed_at="2024-01-01",
        )
        reg.store("test_vec_exc", np.zeros(64), meta)

        with patch(
            "chuk_mcp_lazarus.tools.steering_tools._steer_and_generate_impl",
            side_effect=RuntimeError("gen boom"),
        ):
            result = await steer_and_generate(prompt="hello", vector_name="test_vec_exc")
        assert result["error"] is True
        assert result["error_type"] == "GenerationFailed"
        assert "gen boom" in result["message"]


# ---------------------------------------------------------------------------
# _steer_and_generate_impl
# ---------------------------------------------------------------------------


class TestSteerAndGenerateImpl:
    def test_impl_with_mocked_generators(self) -> None:
        """_steer_and_generate_impl calls generate_text and _generate_steered."""
        from chuk_mcp_lazarus.tools.steering_tools import _steer_and_generate_impl

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_config = MagicMock()
        direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        with (
            patch(
                "chuk_mcp_lazarus.tools.steering_tools.generate_text",
                return_value=("baseline output text", 3),
            ),
            patch(
                "chuk_mcp_lazarus.tools.steering_tools._generate_steered",
                return_value=("steered output text", 5),
            ),
        ):
            result = _steer_and_generate_impl(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=mock_config,
                prompt="test prompt",
                vector_name="vec1",
                direction=direction,
                layer=2,
                alpha=15.0,
                max_new_tokens=50,
            )

        assert "error" not in result
        assert result["prompt"] == "test prompt"
        assert result["vector_name"] == "vec1"
        assert result["alpha"] == 15.0
        assert result["layer"] == 2
        assert result["steered_output"] == "steered output text"
        assert result["baseline_output"] == "baseline output text"
        assert result["steered_tokens"] == 5
        assert result["baseline_tokens"] == 3


# ---------------------------------------------------------------------------
# list_steering_vectors
# ---------------------------------------------------------------------------


class TestListSteeringVectors:
    @pytest.mark.asyncio
    async def test_empty(self) -> None:
        result = await list_steering_vectors()
        assert "error" not in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_with_stored_vectors(self) -> None:
        """list_steering_vectors should return metadata for stored vectors."""
        reg = SteeringVectorRegistry.get()
        for name, layer in [("vec_a", 0), ("vec_b", 3)]:
            meta = VectorMetadata(
                name=name,
                layer=layer,
                vector_norm=1.5,
                separability_score=0.8,
                num_positive=3,
                num_negative=3,
                computed_at="2024-06-01",
            )
            reg.store(name, np.zeros(64), meta)

        result = await list_steering_vectors()
        assert "error" not in result
        assert result["count"] == 2
        assert len(result["vectors"]) == 2
        names = {v["name"] for v in result["vectors"]}
        assert names == {"vec_a", "vec_b"}

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self) -> None:
        """When registry.dump() raises, list_steering_vectors returns ExtractionFailed."""
        with patch.object(SteeringVectorRegistry, "get", side_effect=RuntimeError("registry boom")):
            result = await list_steering_vectors()
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "registry boom" in result["message"]
