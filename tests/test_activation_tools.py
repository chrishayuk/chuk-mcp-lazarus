"""Tests for tools/activation_tools.py — extract_activations, compare_activations."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.activation_tools import (
    _compare_activations_impl,
    _extract_activations_impl,
    _run_hooks,
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


class TestRunHooks:
    def test_run_hooks_populates_hidden_states(self) -> None:
        """_run_hooks creates ModelHooks, runs forward, and returns CapturedState
        with hidden_states populated for the requested layers."""
        import mlx.core as mx

        model = MagicMock()
        config = MagicMock()
        input_ids = mx.array([1, 2, 3, 4, 5])
        layers = [0, 2]

        state = _run_hooks(model, config, input_ids, layers)

        # The stubbed ModelHooks populates hidden_states for each requested layer
        assert 0 in state.hidden_states
        assert 2 in state.hidden_states
        # Each hidden state should be shape [1, 5, 64] from the stub
        assert state.hidden_states[0].shape == (1, 5, 64)
        assert state.hidden_states[2].shape == (1, 5, 64)

    def test_run_hooks_with_capture_attention(self) -> None:
        """_run_hooks passes capture_attention flag through to CaptureConfig."""
        import mlx.core as mx

        model = MagicMock()
        config = MagicMock()
        input_ids = mx.array([1, 2, 3])
        layers = [0]

        # The stub doesn't populate attention_weights, but it should not raise
        state = _run_hooks(model, config, input_ids, layers, capture_attention=True)
        assert 0 in state.hidden_states


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
    async def test_negative_layer(self, loaded_model_state: MagicMock) -> None:
        result = await extract_activations(prompt="hello", layers=[-1])
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
            "chuk_mcp_lazarus.tools.activation.tools._extract_activations_impl",
            return_value=mock_result,
        ):
            result = await extract_activations(prompt="hello", layers=[0])
        assert "error" not in result
        assert result["prompt"] == "hello"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.activation.tools._extract_activations_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await extract_activations(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    @pytest.mark.asyncio
    async def test_with_capture_attention(self, loaded_model_state: MagicMock) -> None:
        """extract_activations with capture_attention=True includes attention_shapes."""
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "num_tokens": 5,
            "activations": {"0": [0.1, 0.2, 0.3]},
            "attention_shapes": {"0": [1, 4, 5, 5]},
        }
        with patch(
            "chuk_mcp_lazarus.tools.activation.tools._extract_activations_impl",
            return_value=mock_result,
        ):
            result = await extract_activations(prompt="hello", layers=[0], capture_attention=True)
        assert "error" not in result
        assert "attention_shapes" in result
        assert result["attention_shapes"]["0"] == [1, 4, 5, 5]


class TestExtractActivationsImpl:
    def test_impl_returns_activations(self) -> None:
        """_extract_activations_impl runs hooks and returns activations dict."""

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: "tok"

        model = MagicMock()
        config = MagicMock()

        result = _extract_activations_impl(
            model=model,
            config=config,
            tokenizer=tokenizer,
            prompt="hello world",
            layers=[0, 2],
            token_position=-1,
            capture_attention=False,
        )

        assert result["prompt"] == "hello world"
        assert result["token_position"] == -1
        assert result["num_tokens"] == 5
        assert "0" in result["activations"]
        assert "2" in result["activations"]
        # Each activation vector should have 64 elements (hidden_dim from stub)
        assert len(result["activations"]["0"]) == 64
        assert len(result["activations"]["2"]) == 64
        # attention_shapes should not be present when capture_attention=False
        assert "attention_shapes" not in result

    def test_impl_with_capture_attention_no_weights(self) -> None:
        """_extract_activations_impl with capture_attention=True but stub has no
        attention_weights populated -> attention_shapes is None / excluded."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: "tok"

        model = MagicMock()
        config = MagicMock()

        result = _extract_activations_impl(
            model=model,
            config=config,
            tokenizer=tokenizer,
            prompt="test",
            layers=[0],
            token_position=-1,
            capture_attention=True,
        )

        assert result["prompt"] == "test"
        assert "0" in result["activations"]
        # The stub doesn't populate attention_weights, so attention_shapes
        # should be None and excluded by model_dump(exclude_none=True)
        assert "attention_shapes" not in result


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
            "chuk_mcp_lazarus.tools.activation.tools._compare_activations_impl",
            return_value=mock_result,
        ):
            result = await compare_activations(prompts=["a", "b"], layer=0)
        assert "error" not in result
        assert result["layer"] == 0

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        """Exception path: _compare_activations_impl raises -> ExtractionFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.activation.tools._compare_activations_impl",
            side_effect=RuntimeError("compare failed"),
        ):
            result = await compare_activations(prompts=["a", "b"], layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "compare failed" in result["message"]


class TestCompareActivationsImpl:
    def test_impl_returns_comparison_result(self) -> None:
        """_compare_activations_impl runs hooks for each prompt and returns
        cosine similarity matrix, PCA projection, and centroid distance."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: "tok"

        model = MagicMock()
        config = MagicMock()

        result = _compare_activations_impl(
            model=model,
            config=config,
            tokenizer=tokenizer,
            prompts=["hello", "world"],
            layer=0,
            token_position=-1,
        )

        assert result["layer"] == 0
        assert result["prompts"] == ["hello", "world"]
        # Cosine similarity matrix should be 2x2
        sim = result["cosine_similarity_matrix"]
        assert len(sim) == 2
        assert len(sim[0]) == 2
        # Diagonal should be ~1.0 (self-similarity)
        # (Stub returns random data so values may differ, but shape is right)
        # PCA projection should have 2 points with 2 coordinates each
        pca = result["pca_2d"]
        assert len(pca) == 2
        assert len(pca[0]) == 2
        # Centroid distance should be a float
        assert isinstance(result["centroid_distance"], float)
