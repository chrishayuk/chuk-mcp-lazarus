"""Tests for tools/generation_tools.py — generate_text, predict_next_token, etc."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.tools.generation_tools import (
    _embedding_neighbors_impl,
    _logit_lens_impl,
    _predict_next,
    _track_token_impl,
    embedding_neighbors,
    generate_text,
    logit_lens,
    predict_next_token,
    tokenize,
    track_token,
)


# ---------------------------------------------------------------------------
# TestGenerateText
# ---------------------------------------------------------------------------


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
    async def test_max_tokens_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await generate_text(prompt="hello", max_new_tokens=1001)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus._generate.generate_text",
            return_value=("world", 1),
        ):
            result = await generate_text(prompt="hello", max_new_tokens=10)
        assert "error" not in result
        assert result["prompt"] == "hello"
        assert result["output"] == "world"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus._generate.generate_text",
            side_effect=RuntimeError("boom"),
        ):
            result = await generate_text(prompt="hello", max_new_tokens=10)
        assert result["error"] is True
        assert result["error_type"] == "GenerationFailed"


# ---------------------------------------------------------------------------
# TestPredictNextToken
# ---------------------------------------------------------------------------


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
        import mlx.core as mx

        logits = np.zeros((1, 5, 100), dtype=np.float32)
        logits[0, -1, 42] = 10.0
        loaded_model_state.model.return_value = mx.array(logits)

        result = await predict_next_token(prompt="hello", top_k=5)
        assert "error" not in result
        assert "predictions" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        """predict_next_token should catch exceptions and return error envelope."""
        loaded_model_state.model.side_effect = RuntimeError("model exploded")

        result = await predict_next_token(prompt="hello", top_k=5)
        assert result["error"] is True
        assert result["error_type"] == "GenerationFailed"


# ---------------------------------------------------------------------------
# TestPredictNext (sync helper)
# ---------------------------------------------------------------------------


class TestPredictNextHelper:
    """Test the _predict_next synchronous helper directly."""

    def test_basic_output_structure(self) -> None:
        import mlx.core as mx

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        logits = np.zeros((1, 5, 100), dtype=np.float32)
        logits[0, -1, 42] = 10.0
        model.return_value = mx.array(logits)

        result = _predict_next(model, tokenizer, "hello", top_k=5)

        assert "prompt" in result
        assert result["prompt"] == "hello"
        assert "predictions" in result
        assert len(result["predictions"]) == 5
        assert result["num_input_tokens"] == 5

    def test_top_prediction_is_highest_logit(self) -> None:
        import mlx.core as mx

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        logits = np.zeros((1, 3, 100), dtype=np.float32)
        logits[0, -1, 77] = 20.0  # token 77 has the highest logit
        model.return_value = mx.array(logits)

        result = _predict_next(model, tokenizer, "test", top_k=3)

        assert result["predictions"][0]["token_id"] == 77
        assert result["predictions"][0]["probability"] > 0.5

    def test_handles_tuple_logits(self) -> None:
        """Model may return (logits, ...) tuple."""
        import mlx.core as mx

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        logits = mx.array(np.zeros((1, 3, 100), dtype=np.float32))
        model.return_value = (logits, "extra_output")

        result = _predict_next(model, tokenizer, "test", top_k=3)
        assert "predictions" in result


# ---------------------------------------------------------------------------
# TestTokenize
# ---------------------------------------------------------------------------


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
        assert "token_ids" in result
        assert "tokens" in result
        assert "num_tokens" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        """Tokenize exception path: encode raises -> GenerationFailed error."""
        loaded_model_state.tokenizer.encode.side_effect = RuntimeError("encode failed")

        result = await tokenize(text="hello")
        assert result["error"] is True
        assert result["error_type"] == "GenerationFailed"
        assert "encode failed" in result["message"]


# ---------------------------------------------------------------------------
# TestLogitLens
# ---------------------------------------------------------------------------


class TestLogitLens:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await logit_lens(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await logit_lens(prompt="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await logit_lens(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "num_layers_analyzed": 1,
            "layers": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._logit_lens_impl",
            return_value=mock_result,
        ):
            result = await logit_lens(prompt="hello", layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_layers_none(self, loaded_model_state: MagicMock) -> None:
        """When layers=None, logit_lens should auto-select layers and succeed."""
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "num_layers_analyzed": 4,
            "layers": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._logit_lens_impl",
            return_value=mock_result,
        ):
            result = await logit_lens(prompt="hello", layers=None)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._logit_lens_impl",
            side_effect=RuntimeError("hooks failed"),
        ):
            result = await logit_lens(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"


# ---------------------------------------------------------------------------
# TestLogitLensImpl (sync helper)
# ---------------------------------------------------------------------------


class TestLogitLensImpl:
    """Test _logit_lens_impl directly using the stubbed ModelHooks."""

    def test_basic_output_structure(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _logit_lens_impl(
            model,
            config,
            tokenizer,
            prompt="hello",
            layers=[0, 1],
            top_k=5,
            token_position=-1,
        )

        assert isinstance(result, dict)
        assert result["prompt"] == "hello"
        assert "predictions" in result
        assert "summary" in result
        assert result["num_layers_analyzed"] >= 0
        assert result["token_position"] == -1
        assert "token_text" in result

    def test_predictions_have_expected_fields(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _logit_lens_impl(
            model,
            config,
            tokenizer,
            prompt="hello",
            layers=[0],
            top_k=3,
            token_position=-1,
        )

        for pred in result["predictions"]:
            assert "layer" in pred
            assert "top_tokens" in pred
            assert "top_probabilities" in pred
            assert "top_token_ids" in pred

    def test_summary_fields(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _logit_lens_impl(
            model,
            config,
            tokenizer,
            prompt="test",
            layers=[0],
            top_k=3,
            token_position=-1,
        )

        summary = result["summary"]
        assert "final_prediction" in summary
        assert "emergence_layer" in summary
        assert "total_layers" in summary

    def test_token_position_zero(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _logit_lens_impl(
            model,
            config,
            tokenizer,
            prompt="hello",
            layers=[0],
            top_k=3,
            token_position=0,
        )

        assert result["token_position"] == 0


# ---------------------------------------------------------------------------
# TestTrackToken
# ---------------------------------------------------------------------------


class TestTrackToken:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await track_token(prompt="hello", token="world")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await track_token(prompt="hello", token="world", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_token_encoding(self, loaded_model_state: MagicMock) -> None:
        loaded_model_state.tokenizer.encode.return_value = []
        result = await track_token(prompt="hello", token="")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "target_token": "world",
            "target_token_id": 42,
            "layers": [],
            "emergence_layer": 0,
            "peak_layer": 0,
            "peak_probability": 0.5,
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._track_token_impl",
            return_value=mock_result,
        ):
            result = await track_token(prompt="hello", token="world", layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_layers_none(self, loaded_model_state: MagicMock) -> None:
        """When layers=None, track_token should auto-select layers and succeed."""
        mock_result = {
            "prompt": "hello",
            "target_token": "world",
            "target_token_id": 42,
            "layers": [],
            "emergence_layer": 0,
            "peak_layer": 0,
            "peak_probability": 0.5,
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._track_token_impl",
            return_value=mock_result,
        ):
            result = await track_token(prompt="hello", token="world", layers=None)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._track_token_impl",
            side_effect=RuntimeError("track failed"),
        ):
            result = await track_token(prompt="hello", token="world", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"


# ---------------------------------------------------------------------------
# TestTrackTokenImpl (sync helper)
# ---------------------------------------------------------------------------


class TestTrackTokenImpl:
    """Test _track_token_impl directly using stubbed ModelHooks."""

    def test_basic_output_structure(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _track_token_impl(
            model,
            config,
            tokenizer,
            prompt="hello",
            layers=[0, 1],
            target_token_id=42,
            target_token="world",
            token_position=-1,
        )

        assert isinstance(result, dict)
        assert result["prompt"] == "hello"
        assert result["target_token"] == "world"
        assert result["target_token_id"] == 42
        assert "layers" in result
        assert "emergence_layer" in result
        assert "peak_layer" in result
        assert "peak_probability" in result
        assert result["token_position"] == -1

    def test_layer_entries_have_expected_fields(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _track_token_impl(
            model,
            config,
            tokenizer,
            prompt="hello",
            layers=[0],
            target_token_id=42,
            target_token="world",
            token_position=-1,
        )

        for entry in result["layers"]:
            assert "layer" in entry
            assert "probability" in entry
            assert "rank" in entry
            assert "is_top1" in entry

    def test_peak_probability_non_negative(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _track_token_impl(
            model,
            config,
            tokenizer,
            prompt="test",
            layers=[0, 1, 2],
            target_token_id=10,
            target_token="foo",
            token_position=-1,
        )

        assert result["peak_probability"] >= 0.0

    def test_token_position_zero(self) -> None:

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        result = _track_token_impl(
            model,
            config,
            tokenizer,
            prompt="hello",
            layers=[0],
            target_token_id=42,
            target_token="world",
            token_position=0,
        )

        assert result["token_position"] == 0


# ---------------------------------------------------------------------------
# TestEmbeddingNeighbors
# ---------------------------------------------------------------------------


class TestEmbeddingNeighbors:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await embedding_neighbors(token="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await embedding_neighbors(token="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "token": "hello",
            "token_id": 42,
            "resolved_form": "hello",
            "neighbors": [],
            "num_neighbors": 0,
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._embedding_neighbors_impl",
            return_value=mock_result,
        ):
            result = await embedding_neighbors(token="hello", top_k=5)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_top_k_clamped_to_100(self, loaded_model_state: MagicMock) -> None:
        """top_k > 100 should be clamped to 100 (no error), then call impl."""
        mock_result = {
            "token": "hello",
            "token_id": 42,
            "resolved_form": "hello",
            "neighbors": [],
            "num_neighbors": 0,
        }
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._embedding_neighbors_impl",
            return_value=mock_result,
        ):
            result = await embedding_neighbors(token="hello", top_k=200)
        # Should succeed (top_k clamped, not rejected)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.generation_tools._embedding_neighbors_impl",
            side_effect=RuntimeError("embed failed"),
        ):
            result = await embedding_neighbors(token="hello", top_k=5)
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"


# ---------------------------------------------------------------------------
# TestEmbeddingNeighborsImpl (sync helper)
# ---------------------------------------------------------------------------


class TestEmbeddingNeighborsImpl:
    """Test _embedding_neighbors_impl directly."""

    def _make_model_with_embeddings(self, vocab_size: int = 100, hidden_dim: int = 64):
        """Build a model with an embedding matrix suitable for neighbor search."""
        import mlx.core as mx

        model = MagicMock()
        # Deterministic embeddings for reproducibility
        np.random.seed(42)
        embed_weight = mx.array(np.random.randn(vocab_size, hidden_dim).astype(np.float32))
        model.model.embed_tokens.weight = embed_weight
        return model

    def _make_tokenizer(self):
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        # For bare token: return [42]; for " " + token: return [42]
        tok.encode.side_effect = lambda text, add_special_tokens=True: [42] if text.strip() else []
        tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        return tok

    def test_basic_output_structure(self) -> None:

        model = self._make_model_with_embeddings()
        tokenizer = self._make_tokenizer()

        result = _embedding_neighbors_impl(model, tokenizer, "hello", top_k=5)

        assert isinstance(result, dict)
        assert "query_token" in result
        assert "query_token_id" in result
        assert "resolved_form" in result
        assert "embedding_dim" in result
        assert "vocab_size" in result
        assert "top_k" in result
        assert "neighbors" in result
        assert "self_similarity" in result
        assert result["query_token"] == "hello"
        assert result["query_token_id"] == 42
        assert result["embedding_dim"] == 64
        assert result["vocab_size"] == 100

    def test_self_similarity_close_to_one(self) -> None:

        model = self._make_model_with_embeddings()
        tokenizer = self._make_tokenizer()

        result = _embedding_neighbors_impl(model, tokenizer, "hello", top_k=5)

        assert abs(result["self_similarity"] - 1.0) < 0.01

    def test_neighbors_do_not_include_self(self) -> None:

        model = self._make_model_with_embeddings()
        tokenizer = self._make_tokenizer()

        result = _embedding_neighbors_impl(model, tokenizer, "hello", top_k=10)

        neighbor_ids = [n["token_id"] for n in result["neighbors"]]
        assert 42 not in neighbor_ids

    def test_top_k_limits_neighbors(self) -> None:

        model = self._make_model_with_embeddings()
        tokenizer = self._make_tokenizer()

        result = _embedding_neighbors_impl(model, tokenizer, "hello", top_k=3)
        assert len(result["neighbors"]) == 3

    def test_unencodable_token_returns_error(self) -> None:

        model = self._make_model_with_embeddings()
        tokenizer = MagicMock()
        # Both bare and space-prefixed encoding return empty
        tokenizer.encode.return_value = []

        result = _embedding_neighbors_impl(model, tokenizer, "UNENCODABLE", top_k=5)

        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    def test_no_embedding_matrix_returns_error(self) -> None:

        model = MagicMock()
        # Remove embed_tokens entirely so the code finds no embedding matrix
        model.model.embed_tokens = None

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [42]
        tokenizer.decode.side_effect = lambda ids, **kw: "tok42"

        result = _embedding_neighbors_impl(model, tokenizer, "hello", top_k=5)

        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    def test_cosine_similarity_values_in_range(self) -> None:

        model = self._make_model_with_embeddings()
        tokenizer = self._make_tokenizer()

        result = _embedding_neighbors_impl(model, tokenizer, "hello", top_k=10)

        for neighbor in result["neighbors"]:
            assert -1.0 <= neighbor["cosine_similarity"] <= 1.0
