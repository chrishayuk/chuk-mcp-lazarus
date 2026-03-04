"""Tests for tools/residual_tools.py — residual stream decomposition tools.

Covers pure helper functions, tool error paths, tool success paths,
and exception handling for all five residual tools:
  residual_decomposition, layer_clustering, logit_attribution,
  head_attribution, top_neurons.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from chuk_mcp_lazarus.tools.residual_tools import (
    _compute_clustering_scores,
    _extract_position,
    _get_embed_weight,
    _get_lm_projection,
    _get_unembed_vector,
    _has_four_norms,
    _has_sublayers,
    _head_attribution_impl,
    _l2_norm,
    _norm_project,
    _project_to_logits,
    _resolve_target_token,
    _run_decomposition_forward,
    _top_neurons_impl,
    head_attribution,
    layer_clustering,
    logit_attribution,
    residual_decomposition,
    top_neurons,
)


# ---------------------------------------------------------------------------
# Patch missing MLX stub functions needed for _run_decomposition_forward
# The conftest installs a minimal MLX stub; some functions are absent.
# ---------------------------------------------------------------------------


def _ensure_mlx_stub_functions() -> None:
    """Add missing MLX functions to the stub if they are absent."""
    if not hasattr(mx, "stop_gradient"):
        mx.stop_gradient = lambda x: x  # type: ignore[attr-defined]

    if not hasattr(mx, "stack"):

        def _stack(arrays: list, axis: int = 0) -> mx.array:
            arrs = [a._data if hasattr(a, "_data") else np.array(a) for a in arrays]
            return mx.array(np.stack(arrs, axis=axis))

        mx.stack = _stack  # type: ignore[attr-defined]

    if not hasattr(nn, "silu"):

        def _silu(x: Any) -> mx.array:
            d = x._data if hasattr(x, "_data") else np.array(x)
            return mx.array(d * (1.0 / (1.0 + np.exp(-d))))

        nn.silu = _silu  # type: ignore[attr-defined]

    if not hasattr(nn, "gelu"):

        def _gelu(x: Any) -> mx.array:
            d = x._data if hasattr(x, "_data") else np.array(x)
            return mx.array(0.5 * d * (1 + np.tanh(np.sqrt(2 / np.pi) * (d + 0.044715 * d**3))))

        nn.gelu = _gelu  # type: ignore[attr-defined]

    if not hasattr(nn, "MultiHeadAttention"):

        class _MockMHA:
            @staticmethod
            def create_additive_causal_mask(seq_len: int) -> mx.array:
                mask = np.zeros((seq_len, seq_len), dtype=np.float32)
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        mask[i, j] = -1e9
                return mx.array(mask)

        nn.MultiHeadAttention = _MockMHA  # type: ignore[attr-defined]


_ensure_mlx_stub_functions()


# ============================================================================
# 1. Pure helper tests
# ============================================================================


class TestHasSublayers:
    """Tests for _has_sublayers(layer)."""

    def test_all_attrs_present(self) -> None:
        layer = MagicMock(spec=["self_attn", "input_layernorm", "post_attention_layernorm", "mlp"])
        assert _has_sublayers(layer) is True

    def test_missing_self_attn(self) -> None:
        layer = MagicMock(spec=["input_layernorm", "post_attention_layernorm", "mlp"])
        assert _has_sublayers(layer) is False

    def test_missing_input_layernorm(self) -> None:
        layer = MagicMock(spec=["self_attn", "post_attention_layernorm", "mlp"])
        assert _has_sublayers(layer) is False

    def test_missing_post_attention_layernorm(self) -> None:
        layer = MagicMock(spec=["self_attn", "input_layernorm", "mlp"])
        assert _has_sublayers(layer) is False

    def test_missing_mlp(self) -> None:
        layer = MagicMock(spec=["self_attn", "input_layernorm", "post_attention_layernorm"])
        assert _has_sublayers(layer) is False

    def test_empty_object(self) -> None:
        layer = object()
        assert _has_sublayers(layer) is False

    def test_with_extra_attrs(self) -> None:
        """Extra attrs beyond the four required should still return True."""
        layer = MagicMock(
            spec=[
                "self_attn",
                "input_layernorm",
                "post_attention_layernorm",
                "mlp",
                "dropout",
                "extra",
            ]
        )
        assert _has_sublayers(layer) is True


class TestHasFourNorms:
    """Tests for _has_four_norms(layer)."""

    def test_both_norms_present(self) -> None:
        layer = MagicMock(spec=["pre_feedforward_layernorm", "post_feedforward_layernorm"])
        assert _has_four_norms(layer) is True

    def test_missing_pre(self) -> None:
        layer = MagicMock(spec=["post_feedforward_layernorm"])
        assert _has_four_norms(layer) is False

    def test_missing_post(self) -> None:
        layer = MagicMock(spec=["pre_feedforward_layernorm"])
        assert _has_four_norms(layer) is False

    def test_neither_present(self) -> None:
        layer = MagicMock(spec=["self_attn", "mlp"])
        assert _has_four_norms(layer) is False

    def test_plain_object(self) -> None:
        assert _has_four_norms(object()) is False


class TestL2Norm:
    """Tests for _l2_norm(vec)."""

    def test_known_3_4_5_triangle(self) -> None:
        vec = mx.array([3.0, 4.0])
        assert abs(_l2_norm(vec) - 5.0) < 1e-5

    def test_unit_vector(self) -> None:
        vec = mx.array([1.0, 0.0, 0.0])
        assert abs(_l2_norm(vec) - 1.0) < 1e-5

    def test_zero_vector(self) -> None:
        vec = mx.array([0.0, 0.0])
        assert abs(_l2_norm(vec)) < 1e-7

    def test_negative_values(self) -> None:
        vec = mx.array([-3.0, -4.0])
        assert abs(_l2_norm(vec) - 5.0) < 1e-5

    def test_single_element(self) -> None:
        vec = mx.array([7.0])
        assert abs(_l2_norm(vec) - 7.0) < 1e-5

    def test_larger_vector(self) -> None:
        data = np.ones(64, dtype=np.float32)
        vec = mx.array(data)
        expected = float(np.sqrt(64.0))
        assert abs(_l2_norm(vec) - expected) < 1e-4


class TestExtractPosition:
    """Tests for _extract_position(tensor, position)."""

    def test_3d_tensor(self) -> None:
        # [batch=1, seq=3, hidden=4]
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        tensor = mx.array(data)
        result = _extract_position(tensor, 1)
        expected = data[0, 1, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_3d_tensor_first_position(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        tensor = mx.array(data)
        result = _extract_position(tensor, 0)
        expected = data[0, 0, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_3d_tensor_last_position(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        tensor = mx.array(data)
        result = _extract_position(tensor, 2)
        expected = data[0, 2, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_2d_tensor(self) -> None:
        # [seq=3, hidden=4]
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = mx.array(data)
        result = _extract_position(tensor, 2)
        expected = data[2, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_1d_tensor_returned_as_is(self) -> None:
        data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        tensor = mx.array(data)
        result = _extract_position(tensor, 0)
        np.testing.assert_array_almost_equal(np.array(result.tolist()), data)

    def test_negative_position_3d(self) -> None:
        """Negative position should work via MLX indexing."""
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        tensor = mx.array(data)
        result = _extract_position(tensor, -1)
        expected = data[0, -1, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)


class TestComputeClusteringScores:
    """Tests for _compute_clustering_scores(labels, sim_matrix)."""

    def test_two_groups_perfect_separation(self) -> None:
        # Group A (indices 0,1) = very similar to each other (0.95)
        # Group B (indices 2,3) = very similar to each other (0.90)
        # A vs B = low similarity (0.1)
        sim = [
            [1.0, 0.95, 0.1, 0.1],
            [0.95, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.90],
            [0.1, 0.1, 0.90, 1.0],
        ]
        labels = ["A", "A", "B", "B"]
        within, between, separation = _compute_clustering_scores(labels, sim)

        assert "A" in within
        assert "B" in within
        assert abs(within["A"] - 0.95) < 1e-5
        assert abs(within["B"] - 0.90) < 1e-5

        assert "A vs B" in between
        assert abs(between["A vs B"] - 0.1) < 1e-5

        # Separation should be positive (within > between)
        assert separation > 0

    def test_single_item_groups(self) -> None:
        """When a group has a single element, within-cluster defaults to 1.0."""
        sim = [
            [1.0, 0.5],
            [0.5, 1.0],
        ]
        labels = ["A", "B"]
        within, between, separation = _compute_clustering_scores(labels, sim)

        assert within["A"] == 1.0
        assert within["B"] == 1.0
        assert abs(between["A vs B"] - 0.5) < 1e-5
        # separation = avg_within - avg_between = 1.0 - 0.5 = 0.5
        assert abs(separation - 0.5) < 1e-5

    def test_negative_separation(self) -> None:
        """Between-cluster similarity higher than within-cluster."""
        sim = [
            [1.0, 0.1, 0.9, 0.9],
            [0.1, 1.0, 0.9, 0.9],
            [0.9, 0.9, 1.0, 0.1],
            [0.9, 0.9, 0.1, 1.0],
        ]
        labels = ["A", "A", "B", "B"]
        within, between, separation = _compute_clustering_scores(labels, sim)

        assert within["A"] == 0.1
        assert within["B"] == 0.1
        # between A vs B: (0.9+0.9+0.9+0.9)/4 = 0.9
        assert abs(between["A vs B"] - 0.9) < 1e-5
        assert separation < 0

    def test_three_groups(self) -> None:
        sim = [
            [1.0, 0.8, 0.2, 0.3],
            [0.8, 1.0, 0.2, 0.3],
            [0.2, 0.2, 1.0, 0.4],
            [0.3, 0.3, 0.4, 1.0],
        ]
        labels = ["X", "X", "Y", "Z"]
        within, between, separation = _compute_clustering_scores(labels, sim)

        assert "X" in within
        assert "Y" in within
        assert "Z" in within
        assert abs(within["X"] - 0.8) < 1e-5
        # Y and Z are singletons
        assert within["Y"] == 1.0
        assert within["Z"] == 1.0
        # Between keys: "X vs Y", "X vs Z", "Y vs Z"
        assert len(between) == 3

    def test_all_same_label(self) -> None:
        """All prompts in one group: within is average off-diagonal, no between pairs."""
        sim = [
            [1.0, 0.5, 0.6],
            [0.5, 1.0, 0.7],
            [0.6, 0.7, 1.0],
        ]
        labels = ["A", "A", "A"]
        within, between, separation = _compute_clustering_scores(labels, sim)

        # Within should be mean of (0.5, 0.6, 0.7) = 0.6
        assert abs(within["A"] - 0.6) < 1e-5
        assert len(between) == 0
        # avg_between is 0 because there are no between pairs
        # separation = avg_within - 0 = within["A"]
        expected_sep = round(within["A"] - 0.0, 6)
        assert abs(separation - expected_sep) < 1e-5


class TestGetEmbedWeight:
    """Tests for _get_embed_weight(model)."""

    def test_plain_mx_array(self) -> None:
        data = np.random.randn(100, 64).astype(np.float32)
        embed_weight = mx.array(data)

        model = MagicMock()
        model.model.embed_tokens.weight = embed_weight

        result = _get_embed_weight(model)
        assert result is embed_weight

    def test_nested_wrapper(self) -> None:
        """embed_tokens.weight is an nn.Embedding wrapper -- actual array at .weight.weight."""
        inner_array = mx.array(np.random.randn(100, 64).astype(np.float32))

        wrapper = MagicMock()
        wrapper.weight = inner_array

        model = MagicMock()
        model.model.embed_tokens.weight = wrapper

        result = _get_embed_weight(model)
        assert result is inner_array

    def test_no_embed_tokens(self) -> None:
        model = MagicMock(spec=["model"])
        model.model = MagicMock(spec=[])  # No embed_tokens
        result = _get_embed_weight(model)
        assert result is None

    def test_no_weight_attr(self) -> None:
        model = MagicMock()
        model.model.embed_tokens = MagicMock(spec=[])  # No weight
        result = _get_embed_weight(model)
        assert result is None

    def test_weight_is_none(self) -> None:
        model = MagicMock()
        model.model.embed_tokens.weight = None
        result = _get_embed_weight(model)
        assert result is None

    def test_no_model_inner(self) -> None:
        """Model without .model attribute: use model itself as inner."""
        data = np.random.randn(100, 64).astype(np.float32)
        embed_weight = mx.array(data)

        model = MagicMock(spec=["embed_tokens"])
        model.embed_tokens.weight = embed_weight

        result = _get_embed_weight(model)
        assert result is embed_weight

    def test_wrapper_weight_not_mx_array(self) -> None:
        """weight is not mx.array and weight.weight is also not mx.array -> None."""
        model = MagicMock()
        model.model.embed_tokens.weight = "not_an_array"

        result = _get_embed_weight(model)
        assert result is None


class TestGetLmProjection:
    """Tests for _get_lm_projection(model)."""

    def test_tied_embeddings_path(self) -> None:
        mock_as_linear = MagicMock()
        model = MagicMock()
        model.tie_word_embeddings = True
        model.model.embed_tokens.as_linear = mock_as_linear

        result = _get_lm_projection(model)
        assert result is mock_as_linear

    def test_lm_head_fallback(self) -> None:
        model = MagicMock()
        model.tie_word_embeddings = False
        mock_lm_head = MagicMock()
        model.lm_head = mock_lm_head

        result = _get_lm_projection(model)
        assert result is mock_lm_head

    def test_no_tie_no_as_linear_uses_lm_head(self) -> None:
        """tie_word_embeddings=True but no as_linear attr on embed -> fall to lm_head."""
        model = MagicMock()
        model.tie_word_embeddings = True
        # Remove as_linear
        model.model.embed_tokens = MagicMock(spec=["weight"])
        mock_lm_head = MagicMock()
        model.lm_head = mock_lm_head

        result = _get_lm_projection(model)
        assert result is mock_lm_head

    def test_model_hooks_fallback(self) -> None:
        """No lm_head either -> falls back to ModelHooks._get_lm_head()."""
        model = MagicMock(spec=[])  # No attrs

        mock_hooks_lm_head = MagicMock()
        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as MockHooks:
            MockHooks.return_value._get_lm_head.return_value = mock_hooks_lm_head
            result = _get_lm_projection(model)
        assert result is mock_hooks_lm_head

    def test_tie_true_embed_none(self) -> None:
        """tie=True but embed_tokens is None -> falls to lm_head."""
        model = MagicMock()
        model.tie_word_embeddings = True
        model.model.embed_tokens = None
        mock_lm_head = MagicMock()
        model.lm_head = mock_lm_head

        result = _get_lm_projection(model)
        assert result is mock_lm_head


class TestProjectToLogits:
    """Tests for _project_to_logits(lm_head, vec)."""

    def test_standard_output_shape(self) -> None:
        """lm_head returns [batch, seq, vocab] -> extracts [vocab]."""
        vocab_size = 100
        logits_data = np.random.randn(1, 1, vocab_size).astype(np.float32)

        def mock_lm_head(x: Any) -> mx.array:
            return mx.array(logits_data)

        vec = mx.array(np.random.randn(64).astype(np.float32))
        result = _project_to_logits(mock_lm_head, vec)
        assert result.shape == (vocab_size,)

    def test_output_with_logits_attr(self) -> None:
        """lm_head returns object with .logits attribute."""
        vocab_size = 50
        logits_data = mx.array(np.random.randn(1, 1, vocab_size).astype(np.float32))

        class HeadOutput:
            logits = logits_data

        def mock_lm_head(x: Any) -> HeadOutput:
            return HeadOutput()

        vec = mx.array(np.random.randn(64).astype(np.float32))
        result = _project_to_logits(mock_lm_head, vec)
        assert result.shape == (vocab_size,)

    def test_output_tuple(self) -> None:
        """lm_head returns tuple -> extracts first element."""
        vocab_size = 30
        logits_data = mx.array(np.random.randn(1, 1, vocab_size).astype(np.float32))

        def mock_lm_head(x: Any) -> tuple:
            return (logits_data, None)

        vec = mx.array(np.random.randn(64).astype(np.float32))
        result = _project_to_logits(mock_lm_head, vec)
        assert result.shape == (vocab_size,)

    def test_input_vec_reshaped_to_1_1_hidden(self) -> None:
        """Verify the input vec gets reshaped to [1, 1, hidden_dim]."""
        calls: list[mx.array] = []

        def tracking_head(x: Any) -> mx.array:
            calls.append(x)
            return mx.array(np.zeros((1, 1, 10), dtype=np.float32))

        vec = mx.array(np.random.randn(64).astype(np.float32))
        _project_to_logits(tracking_head, vec)
        assert len(calls) == 1
        assert calls[0].shape == (1, 1, 64)


class TestGetUnembedVector:
    """Tests for _get_unembed_vector(model, token_id)."""

    def test_tied_embeddings(self) -> None:
        embed_weight = mx.array(np.random.randn(100, 64).astype(np.float32))
        model = MagicMock()
        model.tie_word_embeddings = True
        model.model.embed_tokens.weight = embed_weight

        result = _get_unembed_vector(model, 5)
        assert result is not None
        expected = np.array(embed_weight.tolist())[5]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_lm_head_weight(self) -> None:
        """Non-tied model: uses lm_head.weight."""
        lm_weight = mx.array(np.random.randn(100, 64).astype(np.float32))
        model = MagicMock()
        model.tie_word_embeddings = False
        model.lm_head.weight = lm_weight

        result = _get_unembed_vector(model, 10)
        assert result is not None
        expected = np.array(lm_weight.tolist())[10]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_embed_weight_fallback(self) -> None:
        """Non-tied, no direct lm_head weight -> falls back to embed_weight."""
        embed_weight = mx.array(np.random.randn(100, 64).astype(np.float32))
        model = MagicMock()
        model.tie_word_embeddings = False
        model.lm_head.weight = "not_an_mx_array"
        model.model.embed_tokens.weight = embed_weight

        result = _get_unembed_vector(model, 7)
        assert result is not None

    def test_no_weights_returns_none(self) -> None:
        model = MagicMock(spec=[])
        result = _get_unembed_vector(model, 0)
        assert result is None

    def test_no_lm_head_no_embed_returns_none(self) -> None:
        model = MagicMock()
        model.tie_word_embeddings = False
        model.lm_head = None
        model.model.embed_tokens = None
        result = _get_unembed_vector(model, 0)
        assert result is None


class TestResolveTargetToken:
    """Tests for _resolve_target_token(tokenizer, logits, target)."""

    def test_none_target_returns_argmax(self) -> None:
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "top_prediction"

        logits = mx.array(np.zeros(100, dtype=np.float32))
        # Make token 42 the highest logit
        logits_data = np.zeros(100, dtype=np.float32)
        logits_data[42] = 10.0
        logits = mx.array(logits_data)

        tid, text = _resolve_target_token(tokenizer, logits, None)
        assert tid == 42
        tokenizer.decode.assert_called_with([42])

    def test_string_target_picks_best_variant(self) -> None:
        tokenizer = MagicMock()
        # "hello" -> [10], " hello" -> [20]
        tokenizer.encode.side_effect = lambda s, add_special_tokens=False: {
            "hello": [10],
            " hello": [20],
        }[s]
        tokenizer.decode.return_value = "hello"

        logits_data = np.zeros(100, dtype=np.float32)
        logits_data[10] = 5.0
        logits_data[20] = 8.0  # space-prefixed variant has higher logit
        logits = mx.array(logits_data)

        tid, text = _resolve_target_token(tokenizer, logits, "hello")
        assert tid == 20  # Picks the higher-logit variant

    def test_string_target_first_variant_wins(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda s, add_special_tokens=False: {
            "cat": [15],
            " cat": [25],
        }[s]
        tokenizer.decode.return_value = "cat"

        logits_data = np.zeros(100, dtype=np.float32)
        logits_data[15] = 9.0  # bare variant is higher
        logits_data[25] = 3.0
        logits = mx.array(logits_data)

        tid, text = _resolve_target_token(tokenizer, logits, "cat")
        assert tid == 15

    def test_unencodable_target_raises(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []

        logits = mx.array(np.zeros(100, dtype=np.float32))

        with pytest.raises(ValueError, match="Could not encode"):
            _resolve_target_token(tokenizer, logits, "unknown_token_xyz")

    def test_single_variant_used(self) -> None:
        """If encode returns same ID for both variants, no duplication."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda s, add_special_tokens=False: [42]
        tokenizer.decode.return_value = "tok"

        logits_data = np.zeros(100, dtype=np.float32)
        logits_data[42] = 1.0
        logits = mx.array(logits_data)

        tid, text = _resolve_target_token(tokenizer, logits, "something")
        assert tid == 42


class TestNormProject:
    """Tests for _norm_project(final_norm, lm_head, vec)."""

    def test_with_norm(self) -> None:
        """Applies norm then projects."""
        call_log: list[str] = []

        def mock_norm(x: Any) -> mx.array:
            call_log.append("norm")
            return x

        def mock_lm_head(x: Any) -> mx.array:
            call_log.append("lm_head")
            return mx.array(np.zeros((1, 1, 50), dtype=np.float32))

        vec = mx.array(np.random.randn(64).astype(np.float32))
        result = _norm_project(mock_norm, mock_lm_head, vec)
        assert result.shape == (50,)
        assert call_log == ["norm", "lm_head"]

    def test_without_norm(self) -> None:
        """When final_norm is None, skips normalization."""

        def mock_lm_head(x: Any) -> mx.array:
            return mx.array(np.zeros((1, 1, 50), dtype=np.float32))

        vec = mx.array(np.random.randn(64).astype(np.float32))
        result = _norm_project(None, mock_lm_head, vec)
        assert result.shape == (50,)


# ============================================================================
# 2. Tool error path tests
# ============================================================================


class TestResidualDecomposition:
    """Error path tests for residual_decomposition."""

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
    async def test_multiple_layers_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await residual_decomposition(prompt="hello", layers=[0, 100, 200])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await residual_decomposition(prompt="hello", layers=[-1])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            side_effect=RuntimeError("boom"),
        ):
            result = await residual_decomposition(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "boom" in result["message"]

    @pytest.mark.asyncio
    async def test_success_with_decomposition_mock(self, loaded_model_state: MagicMock) -> None:
        """Mock _run_decomposition_forward to return controlled data."""
        hidden_dim = 64

        embeddings = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))
        prev_hidden_0 = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))
        hidden_0 = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))
        attn_out_0 = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))
        ffn_out_0 = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))

        captured = {
            "embeddings": embeddings,
            "hidden_states": {0: hidden_0},
            "prev_hidden": {0: prev_hidden_0},
            "attn_outputs": {0: attn_out_0},
            "ffn_outputs": {0: ffn_out_0},
        }

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            return_value=captured,
        ):
            result = await residual_decomposition(prompt="hello", layers=[0])

        assert "error" not in result
        assert result["prompt"] == "hello"
        assert "layers" in result
        assert len(result["layers"]) == 1
        layer = result["layers"][0]
        assert layer["layer"] == 0
        assert "total_norm" in layer
        assert "attention_norm" in layer
        assert "ffn_norm" in layer
        assert "attention_fraction" in layer
        assert "ffn_fraction" in layer
        assert "dominant_component" in layer
        assert layer["dominant_component"] in ("attention", "ffn")
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_success_all_layers_none(self, loaded_model_state: MagicMock) -> None:
        """When layers=None, should default to all layers (0..num_layers-1)."""
        hidden_dim = 64
        num_layers = 4  # from mock_metadata

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in range(num_layers):
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            return_value=captured,
        ):
            result = await residual_decomposition(prompt="hello", layers=None)

        assert "error" not in result
        assert len(result["layers"]) == num_layers

    @pytest.mark.asyncio
    async def test_success_no_decomposition(self, loaded_model_state: MagicMock) -> None:
        """When attn/ffn outputs are None, uses 50/50 split."""
        hidden_dim = 64

        # Create hidden states such that the delta is not zero
        prev_h = np.zeros((1, 5, hidden_dim), dtype=np.float32)
        curr_h = np.ones((1, 5, hidden_dim), dtype=np.float32) * 2.0

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {0: mx.array(curr_h)},
            "prev_hidden": {0: mx.array(prev_h)},
            "attn_outputs": {0: None},
            "ffn_outputs": {0: None},
        }

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            return_value=captured,
        ):
            result = await residual_decomposition(prompt="hello", layers=[0])

        assert "error" not in result
        layer = result["layers"][0]
        # 50/50 split when decomposition is not available
        assert abs(layer["attention_fraction"] - 0.5) < 1e-5
        assert abs(layer["ffn_fraction"] - 0.5) < 1e-5

    @pytest.mark.asyncio
    async def test_summary_fields(self, loaded_model_state: MagicMock) -> None:
        """Verify summary has expected keys."""
        hidden_dim = 64

        # Create data where attn dominates layer 0, ffn dominates layer 1
        captured = {
            "embeddings": mx.array(np.zeros((1, 5, hidden_dim), dtype=np.float32)),
            "hidden_states": {
                0: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32)),
                1: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32) * 2),
            },
            "prev_hidden": {
                0: mx.array(np.zeros((1, 5, hidden_dim), dtype=np.float32)),
                1: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32)),
            },
            "attn_outputs": {
                0: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32) * 3),
                1: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32) * 0.1),
            },
            "ffn_outputs": {
                0: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32) * 0.1),
                1: mx.array(np.ones((1, 5, hidden_dim), dtype=np.float32) * 3),
            },
        }

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            return_value=captured,
        ):
            result = await residual_decomposition(prompt="hello", layers=[0, 1])

        summary = result["summary"]
        assert "attention_dominant_count" in summary
        assert "ffn_dominant_count" in summary
        assert "peak_layer" in summary
        assert "peak_total_norm" in summary
        assert "peak_component" in summary


class TestLayerClustering:
    """Error path tests for layer_clustering."""

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
    async def test_too_many_prompts(self, loaded_model_state: MagicMock) -> None:
        result = await layer_clustering(prompts=["a"] * 9)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_labels_length_mismatch(self, loaded_model_state: MagicMock) -> None:
        result = await layer_clustering(
            prompts=["a", "b", "c"],
            labels=["x", "y"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await layer_clustering(prompts=["a", "b"], layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        """Exception inside the try block should yield ExtractionFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._tokenize",
            side_effect=RuntimeError("tokenization failed"),
        ):
            result = await layer_clustering(prompts=["a", "b"], layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    @pytest.mark.asyncio
    async def test_success_with_labels(self, loaded_model_state: MagicMock) -> None:
        """Successful clustering with labels should include separation scores."""
        # Mock ModelHooks to return controlled hidden states
        hidden_dim = 64
        mock_hidden_states = {0: mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))}

        mock_hooks = MagicMock()
        mock_hooks.state.hidden_states = mock_hidden_states

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = await layer_clustering(
                prompts=["hello", "world"],
                layers=[0],
                labels=["A", "B"],
            )

        assert "error" not in result
        assert result["prompts"] == ["hello", "world"]
        assert result["labels"] == ["A", "B"]
        assert "layers" in result
        assert len(result["layers"]) == 1
        layer = result["layers"][0]
        assert "similarity_matrix" in layer
        assert "mean_similarity" in layer
        assert "within_cluster_similarity" in layer
        assert "between_cluster_similarity" in layer
        assert "separation_score" in layer

    @pytest.mark.asyncio
    async def test_success_without_labels(self, loaded_model_state: MagicMock) -> None:
        """Successful clustering without labels: no separation scores."""
        hidden_dim = 64
        mock_hidden_states = {0: mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))}

        mock_hooks = MagicMock()
        mock_hooks.state.hidden_states = mock_hidden_states

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = await layer_clustering(
                prompts=["hello", "world"],
                layers=[0],
            )

        assert "error" not in result
        # labels=None is excluded by model_dump(exclude_none=True)
        assert result.get("labels") is None
        layer = result["layers"][0]
        assert "similarity_matrix" in layer
        # Without labels, these should be None (excluded by exclude_none)
        assert "within_cluster_similarity" not in layer
        assert "separation_score" not in layer

    @pytest.mark.asyncio
    async def test_summary_keys(self, loaded_model_state: MagicMock) -> None:
        """Verify summary has expected keys."""
        hidden_dim = 64
        mock_hidden_states = {
            0: mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            1: mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
        }

        mock_hooks = MagicMock()
        mock_hooks.state.hidden_states = mock_hidden_states

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = await layer_clustering(
                prompts=["hello", "world"],
                layers=[0, 1],
                labels=["A", "B"],
            )

        assert "error" not in result
        summary = result["summary"]
        assert "most_similar_layer" in summary
        assert "most_similar_value" in summary
        assert "least_similar_layer" in summary
        assert "least_similar_value" in summary
        assert "best_separation_layer" in summary
        assert "best_separation_score" in summary
        assert "separation_trend" in summary

    @pytest.mark.asyncio
    async def test_default_layers_selection(self, loaded_model_state: MagicMock) -> None:
        """When layers=None, should select key layers automatically."""
        hidden_dim = 64
        # The default for 4 layers: set([0, 1, 2, 3]) = [0, 1, 2, 3]
        all_layers = {}
        for i in range(4):
            all_layers[i] = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks.state.hidden_states = all_layers

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = await layer_clustering(
                prompts=["hello", "world"],
                layers=None,
            )

        assert "error" not in result
        # With 4 layers: 0, 4//4=1, 4//2=2, 3*4//4=3, 4-1=3 -> unique = {0,1,2,3}
        assert result["num_layers_analyzed"] == 4


class TestLogitAttribution:
    """Error path tests for logit_attribution."""

    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await logit_attribution(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await logit_attribution(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: MagicMock) -> None:
        result = await logit_attribution(prompt="hello", layers=[])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            side_effect=RuntimeError("explosion"),
        ):
            result = await logit_attribution(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "explosion" in result["message"]

    @pytest.mark.asyncio
    async def test_success_normalized_mode(self, loaded_model_state: MagicMock) -> None:
        """Test successful logit attribution in normalized mode."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        last_layer = num_layers - 1

        embeddings = mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))

        captured = {
            "embeddings": embeddings,
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        # Need layer 0 and last_layer=3 (tool always includes last layer)
        for i in [0, last_layer]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        # Mock lm_head to return [1, 1, vocab_size]
        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        # Mock final_norm as identity
        def mock_final_norm(x: Any) -> mx.array:
            return x

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = mock_final_norm

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(
                prompt="hello",
                layers=[0],
                normalized=True,
            )

        assert "error" not in result
        assert result["prompt"] == "hello"
        assert "target_token" in result
        assert "target_token_id" in result
        assert "model_logit" in result
        assert "model_probability" in result
        assert "embedding_logit" in result
        assert "layers" in result
        assert "attribution_sum" in result
        assert "summary" in result
        summary = result["summary"]
        assert summary["mode"] == "normalized"

    @pytest.mark.asyncio
    async def test_success_raw_dla_mode(self, loaded_model_state: MagicMock) -> None:
        """Test successful logit attribution in raw DLA mode."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        last_layer = num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in [0, last_layer]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        def mock_final_norm(x: Any) -> mx.array:
            return x

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = mock_final_norm

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(
                prompt="hello",
                layers=[0],
                normalized=False,
            )

        assert "error" not in result
        summary = result["summary"]
        assert summary["mode"] == "raw_dla"

    @pytest.mark.asyncio
    async def test_lm_head_none_returns_error(self, loaded_model_state: MagicMock) -> None:
        """When _get_lm_projection returns None, should return ExtractionFailed."""
        hidden_dim = 64
        num_layers = 4
        last_layer = num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {
                last_layer: mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32))
            },
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=None,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(prompt="hello", layers=[0])

        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    @pytest.mark.asyncio
    async def test_non_decomposable_layers(self, loaded_model_state: MagicMock) -> None:
        """Layers with None attn/ffn outputs should produce zero contributions."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        last_layer = num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in [0, last_layer]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            # Only layer 0 has None outputs (non-decomposable)
            if i == 0:
                captured["attn_outputs"][i] = None
                captured["ffn_outputs"][i] = None
            else:
                captured["attn_outputs"][i] = mx.array(
                    np.random.randn(1, 5, hidden_dim).astype(np.float32)
                )
                captured["ffn_outputs"][i] = mx.array(
                    np.random.randn(1, 5, hidden_dim).astype(np.float32)
                )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(prompt="hello", layers=[0], normalized=False)

        assert "error" not in result
        # Layer 0 has non-decomposable block: attention and ffn logits should be 0
        layer_data = result["layers"][0]
        assert layer_data["attention_logit"] == 0.0
        assert layer_data["ffn_logit"] == 0.0
        assert layer_data["attention_top_token"] == "?"
        assert layer_data["ffn_top_token"] == "?"

    @pytest.mark.asyncio
    async def test_default_layers_auto_selection(self, loaded_model_state: MagicMock) -> None:
        """When layers=None with few layers, all layers are used."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in range(num_layers):
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(prompt="hello", layers=None, normalized=True)

        assert "error" not in result
        # 4 layers <= 12, so all layers should be included
        assert len(result["layers"]) == num_layers

    @pytest.mark.asyncio
    async def test_with_target_token(self, loaded_model_state: MagicMock) -> None:
        """Test that specifying a target_token works correctly."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        last_layer = num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in [0, last_layer]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(
                prompt="hello",
                layers=[0],
                target_token="world",
                normalized=True,
            )

        assert "error" not in result
        # The target_token should be resolved from the tokenizer
        assert "target_token" in result
        assert "target_token_id" in result

    @pytest.mark.asyncio
    async def test_summary_structure(self, loaded_model_state: MagicMock) -> None:
        """Verify all expected summary fields are present."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        last_layer = num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in [0, 1, last_layer]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(
                prompt="hello",
                layers=[0, 1],
                normalized=False,
            )

        assert "error" not in result
        summary = result["summary"]
        expected_keys = [
            "mode",
            "top_positive_layer",
            "top_positive_logit",
            "top_negative_layer",
            "top_negative_logit",
            "total_attention_logit",
            "total_ffn_logit",
            "dominant_component",
            "embedding_logit",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing summary key: {key}"


class TestHeadAttribution:
    """Error path and success tests for head_attribution."""

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
    async def test_negative_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await head_attribution(prompt="hello", layer=-1)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "token_position": -1,
            "token_text": "tok5",
            "target_token": "world",
            "target_token_id": 42,
            "num_heads": 4,
            "layer_total_logit": 1.5,
            "heads": [
                {
                    "head": 0,
                    "logit_contribution": 0.5,
                    "fraction_of_layer": 0.333,
                    "top_token": "foo",
                },
                {
                    "head": 1,
                    "logit_contribution": 0.8,
                    "fraction_of_layer": 0.533,
                    "top_token": "bar",
                },
                {
                    "head": 2,
                    "logit_contribution": -0.2,
                    "fraction_of_layer": -0.133,
                    "top_token": "baz",
                },
                {
                    "head": 3,
                    "logit_contribution": 0.4,
                    "fraction_of_layer": 0.267,
                    "top_token": "qux",
                },
            ],
            "summary": {
                "top_positive_head": 1,
                "top_positive_logit": 0.8,
                "top_negative_head": 2,
                "top_negative_logit": -0.2,
                "positive_head_count": 3,
                "negative_head_count": 1,
                "concentration": 0.533,
            },
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._head_attribution_impl",
            return_value=mock_result,
        ):
            result = await head_attribution(prompt="hello", layer=0)
        assert "error" not in result
        assert result["prompt"] == "hello"
        assert result["layer"] == 0
        assert len(result["heads"]) == 4

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._head_attribution_impl",
            side_effect=RuntimeError("head_crash"),
        ):
            result = await head_attribution(prompt="hello", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "head_crash" in result["message"]

    @pytest.mark.asyncio
    async def test_with_target_token(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 1,
            "token_position": -1,
            "token_text": "tok5",
            "target_token": "cat",
            "target_token_id": 10,
            "num_heads": 4,
            "layer_total_logit": 2.0,
            "heads": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._head_attribution_impl",
            return_value=mock_result,
        ):
            result = await head_attribution(prompt="hello", layer=1, target_token="cat")
        assert "error" not in result
        assert result["target_token"] == "cat"


class TestTopNeurons:
    """Error path and success tests for top_neurons."""

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
    async def test_negative_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await top_neurons(prompt="hello", layer=-1)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "token_position": -1,
            "token_text": "tok5",
            "target_token": "world",
            "target_token_id": 42,
            "mlp_type": "swiglu",
            "intermediate_size": 256,
            "top_k": 10,
            "top_positive": [
                {
                    "neuron_index": 5,
                    "activation": 1.5,
                    "logit_contribution": 0.8,
                    "top_token": "foo",
                },
            ],
            "top_negative": [
                {
                    "neuron_index": 10,
                    "activation": -0.5,
                    "logit_contribution": -0.3,
                    "top_token": "bar",
                },
            ],
            "total_neuron_logit": 0.5,
            "summary": {
                "top_neuron_index": 5,
                "top_neuron_logit": 0.8,
                "positive_neuron_count": 128,
                "negative_neuron_count": 128,
                "concentration_top10": 0.4,
                "sparsity": 0.1,
            },
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._top_neurons_impl",
            return_value=mock_result,
        ):
            result = await top_neurons(prompt="hello", layer=0)
        assert "error" not in result
        assert result["mlp_type"] == "swiglu"
        assert len(result["top_positive"]) == 1
        assert len(result["top_negative"]) == 1

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._top_neurons_impl",
            side_effect=RuntimeError("neuron_boom"),
        ):
            result = await top_neurons(prompt="hello", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "neuron_boom" in result["message"]

    @pytest.mark.asyncio
    async def test_top_k_clamped(self, loaded_model_state: MagicMock) -> None:
        """top_k should be clamped to [1, 200]."""
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "token_position": -1,
            "token_text": "tok5",
            "target_token": "world",
            "target_token_id": 42,
            "mlp_type": "swiglu",
            "intermediate_size": 256,
            "top_k": 200,
            "top_positive": [],
            "top_negative": [],
            "total_neuron_logit": 0.0,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._top_neurons_impl",
            return_value=mock_result,
        ) as mock_impl:
            result = await top_neurons(prompt="hello", layer=0, top_k=500)
        assert "error" not in result
        # The top_k passed to impl should be clamped to 200
        call_args = mock_impl.call_args
        assert call_args[0][8] == 200  # 9th arg is top_k

    @pytest.mark.asyncio
    async def test_top_k_minimum(self, loaded_model_state: MagicMock) -> None:
        """top_k of 0 should be clamped to 1."""
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "token_position": -1,
            "token_text": "tok5",
            "target_token": "world",
            "target_token_id": 42,
            "mlp_type": "standard",
            "intermediate_size": 256,
            "top_k": 1,
            "top_positive": [],
            "top_negative": [],
            "total_neuron_logit": 0.0,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._top_neurons_impl",
            return_value=mock_result,
        ) as mock_impl:
            result = await top_neurons(prompt="hello", layer=0, top_k=0)
        assert "error" not in result
        call_args = mock_impl.call_args
        assert call_args[0][8] == 1  # top_k clamped to 1

    @pytest.mark.asyncio
    async def test_with_target_token(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 2,
            "token_position": -1,
            "token_text": "tok5",
            "target_token": "Paris",
            "target_token_id": 55,
            "mlp_type": "swiglu",
            "intermediate_size": 256,
            "top_k": 10,
            "top_positive": [],
            "top_negative": [],
            "total_neuron_logit": 0.0,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._top_neurons_impl",
            return_value=mock_result,
        ):
            result = await top_neurons(prompt="hello", layer=2, target_token="Paris")
        assert "error" not in result
        assert result["target_token"] == "Paris"


# ============================================================================
# 3. Additional edge case tests for helpers
# ============================================================================


class TestL2NormEdgeCases:
    """Additional edge cases for _l2_norm."""

    def test_very_large_values(self) -> None:
        vec = mx.array([1e6, 1e6])
        expected = float(np.sqrt(2e12))
        assert abs(_l2_norm(vec) - expected) / expected < 1e-4

    def test_very_small_values(self) -> None:
        vec = mx.array([1e-8, 1e-8])
        expected = float(np.sqrt(2e-16))
        assert (
            abs(_l2_norm(vec) - expected) < 1e-14
            or abs(_l2_norm(vec) - expected) / max(expected, 1e-20) < 0.1
        )


class TestExtractPositionEdgeCases:
    """Additional edge cases for _extract_position."""

    def test_3d_single_seq_length(self) -> None:
        data = np.random.randn(1, 1, 8).astype(np.float32)
        tensor = mx.array(data)
        result = _extract_position(tensor, 0)
        expected = data[0, 0, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)

    def test_2d_single_seq_length(self) -> None:
        data = np.random.randn(1, 8).astype(np.float32)
        tensor = mx.array(data)
        result = _extract_position(tensor, 0)
        expected = data[0, :]
        np.testing.assert_array_almost_equal(np.array(result.tolist()), expected)


class TestComputeClusteringScoresEdgeCases:
    """Additional edge cases for _compute_clustering_scores."""

    def test_two_prompts_same_label(self) -> None:
        sim = [[1.0, 0.7], [0.7, 1.0]]
        labels = ["A", "A"]
        within, between, separation = _compute_clustering_scores(labels, sim)
        assert abs(within["A"] - 0.7) < 1e-5
        assert len(between) == 0

    def test_all_singletons(self) -> None:
        """Each prompt has a unique label."""
        sim = [
            [1.0, 0.3, 0.5],
            [0.3, 1.0, 0.4],
            [0.5, 0.4, 1.0],
        ]
        labels = ["A", "B", "C"]
        within, between, separation = _compute_clustering_scores(labels, sim)
        # All within = 1.0 (singletons)
        assert within["A"] == 1.0
        assert within["B"] == 1.0
        assert within["C"] == 1.0
        # 3 between pairs
        assert len(between) == 3


class TestGetEmbedWeightEdgeCases:
    """Additional edge cases for _get_embed_weight."""

    def test_double_nested_wrapper_no_inner_array(self) -> None:
        """weight.weight exists but is not mx.array -> returns None."""
        model = MagicMock()
        inner_wrapper = MagicMock()
        inner_wrapper.weight = "not_mx_array"
        model.model.embed_tokens.weight = inner_wrapper
        result = _get_embed_weight(model)
        assert result is None


class TestGetUnembedVectorEdgeCases:
    """Additional edge cases for _get_unembed_vector."""

    def test_lm_head_weight_not_mx_array(self) -> None:
        """lm_head.weight exists but isn't an mx.array."""
        model = MagicMock()
        model.tie_word_embeddings = False
        model.lm_head.weight = "not_array"
        model.model.embed_tokens.weight = mx.array(np.random.randn(100, 64).astype(np.float32))
        # Falls through to embed_weight path
        result = _get_unembed_vector(model, 5)
        assert result is not None

    def test_lm_head_none_with_embed(self) -> None:
        """lm_head is None but embed_tokens is available."""
        embed_w = mx.array(np.random.randn(100, 64).astype(np.float32))
        model = MagicMock()
        model.tie_word_embeddings = False
        model.lm_head = None
        model.model.embed_tokens.weight = embed_w
        result = _get_unembed_vector(model, 3)
        assert result is not None


class TestResolveTargetTokenEdgeCases:
    """Additional edge cases for _resolve_target_token."""

    def test_empty_encode_for_first_variant_only(self) -> None:
        """If bare string gives empty but space-prefixed gives ids, should work."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda s, add_special_tokens=False: {
            "rare": [],
            " rare": [55],
        }[s]
        tokenizer.decode.return_value = " rare"

        logits_data = np.zeros(100, dtype=np.float32)
        logits_data[55] = 3.0
        logits = mx.array(logits_data)

        tid, text = _resolve_target_token(tokenizer, logits, "rare")
        assert tid == 55

    def test_both_variants_empty_raises(self) -> None:
        """If both variants give empty, should raise ValueError."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []

        logits = mx.array(np.zeros(100, dtype=np.float32))

        with pytest.raises(ValueError, match="Could not encode"):
            _resolve_target_token(tokenizer, logits, "impossible")


# ============================================================================
# 4. Tests for _run_decomposition_forward
# ============================================================================


class TestRunDecompositionForward:
    """Tests for _run_decomposition_forward."""

    def test_returns_expected_keys(self) -> None:
        """Verify the return dict has all expected keys."""
        hidden_dim = 64
        seq_len = 5

        # Build minimal model mock
        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)

        mock_embed = MagicMock(return_value=mx.array(embed_data))

        # Create a layer with sublayers
        layer = MagicMock()
        layer.self_attn = MagicMock()
        layer.input_layernorm = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        attn_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer.self_attn.return_value = (attn_out, None)
        layer.post_attention_layernorm = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        layer.mlp = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        # Make sure it has the sublayer attrs
        layer.dropout = None

        # Remove four-norm attrs
        if hasattr(layer, "pre_feedforward_layernorm"):
            del layer.pre_feedforward_layernorm
        if hasattr(layer, "post_feedforward_layernorm"):
            del layer.post_feedforward_layernorm

        mock_model = MagicMock()
        mock_config = MagicMock()

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([1, 2, 3, 4, 5]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(mock_model, mock_config, input_ids, [0])

        assert "embeddings" in result
        assert "hidden_states" in result
        assert "prev_hidden" in result
        assert "attn_outputs" in result
        assert "ffn_outputs" in result
        assert 0 in result["hidden_states"]
        assert 0 in result["prev_hidden"]
        assert 0 in result["attn_outputs"]
        assert 0 in result["ffn_outputs"]

    def test_1d_input_ids_expanded(self) -> None:
        """1D input_ids should be expanded to [1, seq]."""
        hidden_dim = 8
        seq_len = 3

        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = []  # No layers
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([1, 2, 3]))  # 1D
        assert input_ids.ndim == 1

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [])

        assert "embeddings" in result

    def test_with_embedding_scale(self) -> None:
        """Verify embedding scale is applied."""
        hidden_dim = 8
        seq_len = 3

        ones_data = np.ones((1, seq_len, hidden_dim), dtype=np.float32)
        mock_embed = MagicMock(return_value=mx.array(ones_data))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = []
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = 2.0

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [])

        # Embeddings should be scaled by 2.0
        embed_vals = np.array(result["embeddings"].tolist())
        np.testing.assert_array_almost_equal(embed_vals, ones_data * 2.0, decimal=4)

    def test_non_sublayer_block(self) -> None:
        """Layer without standard sublayers: runs full block."""
        hidden_dim = 8
        seq_len = 3

        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        # Layer without sublayer attrs
        layer = MagicMock(spec=["__call__"])
        out_data = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer.return_value = out_data

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

        # Non-decomposable: attn/ffn outputs should be None
        assert result["attn_outputs"][0] is None
        assert result["ffn_outputs"][0] is None
        assert 0 in result["hidden_states"]

    def test_layers_beyond_max_skipped(self) -> None:
        """Layers beyond max requested layer are skipped."""
        hidden_dim = 8
        seq_len = 3

        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        layer0 = MagicMock(spec=["__call__"])
        layer0.return_value = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer1 = MagicMock(spec=["__call__"])
        layer1.return_value = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer0, layer1]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            # Only request layer 0; layer 1 should not be called
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

        assert 0 in result["hidden_states"]
        assert 1 not in result["hidden_states"]

    def test_uncaptured_layer_runs_but_not_stored(self) -> None:
        """Layers not in the requested set run but don't store outputs."""
        hidden_dim = 8
        seq_len = 3

        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        layer0 = MagicMock(spec=["__call__"])
        layer0.return_value = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer1 = MagicMock(spec=["__call__"])
        layer1.return_value = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer0, layer1]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            # Request only layer 1; layer 0 runs but not captured
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [1])

        assert 0 not in result["hidden_states"]
        assert 1 in result["hidden_states"]
        # Both layers should have been called
        layer0.assert_called_once()
        layer1.assert_called_once()


# ============================================================================
# 5. Pydantic result model sanity tests
# ============================================================================


class TestResultModels:
    """Quick sanity checks that Pydantic models accept expected data."""

    def test_layer_contribution_valid(self) -> None:
        from chuk_mcp_lazarus.tools.residual_tools import LayerContribution

        lc = LayerContribution(
            layer=0,
            total_norm=1.0,
            attention_norm=0.6,
            ffn_norm=0.4,
            attention_fraction=0.6,
            ffn_fraction=0.4,
            dominant_component="attention",
        )
        d = lc.model_dump()
        assert d["layer"] == 0
        assert d["dominant_component"] == "attention"

    def test_head_contribution_valid(self) -> None:
        from chuk_mcp_lazarus.tools.residual_tools import HeadContribution

        hc = HeadContribution(
            head=2,
            logit_contribution=-0.5,
            fraction_of_layer=-0.25,
            top_token="hello",
        )
        d = hc.model_dump()
        assert d["head"] == 2
        assert d["logit_contribution"] == -0.5

    def test_neuron_contribution_valid(self) -> None:
        from chuk_mcp_lazarus.tools.residual_tools import NeuronContribution

        nc = NeuronContribution(
            neuron_index=42,
            activation=1.5,
            logit_contribution=0.8,
            top_token="world",
        )
        d = nc.model_dump()
        assert d["neuron_index"] == 42

    def test_layer_similarity_valid(self) -> None:
        from chuk_mcp_lazarus.tools.residual_tools import LayerSimilarity

        ls = LayerSimilarity(
            layer=0,
            similarity_matrix=[[1.0, 0.5], [0.5, 1.0]],
            mean_similarity=0.5,
        )
        d = ls.model_dump()
        assert d["layer"] == 0
        assert d["within_cluster_similarity"] is None

    def test_layer_attribution_valid(self) -> None:
        from chuk_mcp_lazarus.tools.residual_tools import LayerAttribution

        la = LayerAttribution(
            layer=3,
            attention_logit=0.5,
            ffn_logit=0.3,
            total_logit=0.8,
            cumulative_logit=1.2,
            attention_top_token="cat",
            ffn_top_token="dog",
        )
        d = la.model_dump()
        assert d["total_logit"] == 0.8


# ============================================================================
# 6. _run_decomposition_forward — fallback / exception paths
# ============================================================================


class TestRunDecompositionForwardFallbacks:
    """Tests for exception-handling paths in _run_decomposition_forward."""

    def _make_sublayer_layer(
        self,
        hidden_dim: int = 8,
        seq_len: int = 3,
        four_norms: bool = False,
        attn_fail_mask: bool = False,
        attn_fail_cache: bool = False,
        has_dropout: bool = False,
    ) -> MagicMock:
        """Build a layer mock with standard sublayer attrs.

        attn_fail_mask: self_attn(normed, mask=..., cache=...) raises TypeError
        attn_fail_cache: self_attn(normed, cache=...) also raises TypeError
        """
        layer = MagicMock()

        # Sublayer attrs that _has_sublayers checks (via hasattr)
        layer.input_layernorm = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        layer.post_attention_layernorm = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        layer.mlp = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )

        attn_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        if attn_fail_mask and attn_fail_cache:
            # Both (normed, mask=mask, cache=None) and (normed, cache=None) fail
            # Third fallback: self_attn(normed) returns raw array
            layer.self_attn = MagicMock(
                side_effect=[TypeError("no mask"), TypeError("no cache"), attn_out],
            )
        elif attn_fail_mask:
            # Only (normed, mask=mask, cache=None) fails
            layer.self_attn = MagicMock(
                side_effect=[TypeError("no mask"), (attn_out, None)],
            )
        else:
            layer.self_attn = MagicMock(return_value=(attn_out, None))

        if has_dropout:
            layer.dropout = MagicMock(side_effect=lambda x: x)
        else:
            layer.dropout = None

        if four_norms:
            layer.pre_feedforward_layernorm = MagicMock(
                return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
            )
            layer.post_feedforward_layernorm = MagicMock(
                return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
            )
        else:
            # Remove four-norm attrs so _has_four_norms returns False
            # Use spec to control what attrs exist
            pass

        return layer

    def _run_with_layer(self, layer: Any, hidden_dim: int = 8, seq_len: int = 3) -> dict:
        """Run _run_decomposition_forward with a single-layer model."""
        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            return _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

    def test_attn_fallback_no_mask(self) -> None:
        """self_attn(normed, mask=mask, cache=None) raises TypeError ->
        falls back to self_attn(normed, cache=None)."""
        layer = self._make_sublayer_layer(attn_fail_mask=True)
        # Remove four-norm attrs
        del layer.pre_feedforward_layernorm
        del layer.post_feedforward_layernorm

        result = self._run_with_layer(layer)
        assert 0 in result["attn_outputs"]
        assert result["attn_outputs"][0] is not None

    def test_attn_fallback_no_mask_no_cache(self) -> None:
        """Both mask+cache and cache-only calls fail ->
        falls back to self_attn(normed) returning raw result."""
        layer = self._make_sublayer_layer(attn_fail_mask=True, attn_fail_cache=True)
        del layer.pre_feedforward_layernorm
        del layer.post_feedforward_layernorm

        result = self._run_with_layer(layer)
        assert 0 in result["attn_outputs"]
        assert result["attn_outputs"][0] is not None

    def test_four_norms_path(self) -> None:
        """Layer with Gemma-style 4-norm pattern."""
        layer = self._make_sublayer_layer(four_norms=True)
        result = self._run_with_layer(layer)
        assert 0 in result["attn_outputs"]
        assert 0 in result["ffn_outputs"]
        # Both should be non-None (four-norm path stores post-normed outputs)
        assert result["attn_outputs"][0] is not None
        assert result["ffn_outputs"][0] is not None

    def test_dropout_on_attn(self) -> None:
        """Layer with dropout attribute applied after attn."""
        layer = self._make_sublayer_layer(has_dropout=True)
        del layer.pre_feedforward_layernorm
        del layer.post_feedforward_layernorm

        result = self._run_with_layer(layer)
        assert 0 in result["attn_outputs"]

    def test_dropout_on_ffn(self) -> None:
        """Standard path with dropout applied after FFN."""
        layer = self._make_sublayer_layer(has_dropout=True)
        del layer.pre_feedforward_layernorm
        del layer.post_feedforward_layernorm

        result = self._run_with_layer(layer)
        assert 0 in result["ffn_outputs"]
        assert result["ffn_outputs"][0] is not None

    def test_non_sublayer_tuple_output(self) -> None:
        """Non-sublayer block returns tuple -> extract first element."""
        hidden_dim = 8
        seq_len = 3
        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        out_data = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer = MagicMock(spec=["__call__"])
        layer.return_value = (out_data, None)  # tuple output

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

        assert 0 in result["hidden_states"]
        assert result["attn_outputs"][0] is None

    def test_non_sublayer_hidden_states_output(self) -> None:
        """Non-sublayer block returns object with .hidden_states attr."""
        hidden_dim = 8
        seq_len = 3
        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        out_data = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer_output = MagicMock()
        layer_output.hidden_states = out_data
        layer = MagicMock(spec=["__call__"])
        layer.return_value = layer_output

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

        assert 0 in result["hidden_states"]

    def test_non_sublayer_fallback_no_mask(self) -> None:
        """Non-sublayer block: (h, mask=mask, cache=None) fails -> (h, cache=None)."""
        hidden_dim = 8
        seq_len = 3
        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        out_data = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer = MagicMock(spec=["__call__"])
        layer.side_effect = [TypeError("no mask"), out_data]

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

        assert 0 in result["hidden_states"]

    def test_non_sublayer_fallback_no_mask_no_cache(self) -> None:
        """Non-sublayer block: all three call patterns tried before succeeding."""
        hidden_dim = 8
        seq_len = 3
        embed_data = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        mock_embed = MagicMock(return_value=mx.array(embed_data))

        out_data = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        layer = MagicMock(spec=["__call__"])
        layer.side_effect = [TypeError("no mask"), TypeError("no cache"), out_data]

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [layer]
        mock_hooks._get_embed_tokens.return_value = mock_embed
        mock_hooks._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([[1, 2, 3]]))

        with patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            return_value=mock_hooks,
        ):
            result = _run_decomposition_forward(MagicMock(), MagicMock(), input_ids, [0])

        assert 0 in result["hidden_states"]

    def test_attn_fallback_cache_only_raises_value_error(self) -> None:
        """self_attn(normed, cache=None) raises ValueError (not TypeError) ->
        falls back to self_attn(normed) raw call."""
        hidden_dim = 8
        seq_len = 3

        attn_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        layer = MagicMock()
        layer.input_layernorm = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        layer.post_attention_layernorm = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        layer.mlp = MagicMock(
            return_value=mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        )
        layer.dropout = None
        # mask call -> TypeError, cache call -> ValueError, raw call -> result
        layer.self_attn = MagicMock(
            side_effect=[TypeError("no mask"), ValueError("no cache"), attn_out]
        )
        del layer.pre_feedforward_layernorm
        del layer.post_feedforward_layernorm

        result = self._run_with_layer(layer)
        assert 0 in result["attn_outputs"]
        assert result["attn_outputs"][0] is not None


# ============================================================================
# 7. Zero-norm edge case in residual_decomposition
# ============================================================================


class TestResidualDecompositionZeroNorm:
    """Test zero-norm edge cases in residual_decomposition."""

    @pytest.mark.asyncio
    async def test_zero_norm_delta(self, loaded_model_state: MagicMock) -> None:
        """When attn_norm + ffn_norm == 0, fractions should be 0.5/0.5."""
        hidden_dim = 64

        # Same hidden state for prev and curr -> zero delta
        same_h = np.zeros((1, 5, hidden_dim), dtype=np.float32)

        captured = {
            "embeddings": mx.array(np.zeros((1, 5, hidden_dim), dtype=np.float32)),
            "hidden_states": {0: mx.array(same_h)},
            "prev_hidden": {0: mx.array(same_h)},
            "attn_outputs": {0: mx.array(np.zeros((1, 5, hidden_dim), dtype=np.float32))},
            "ffn_outputs": {0: mx.array(np.zeros((1, 5, hidden_dim), dtype=np.float32))},
        }

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            return_value=captured,
        ):
            result = await residual_decomposition(prompt="hello", layers=[0])

        assert "error" not in result
        layer = result["layers"][0]
        assert abs(layer["attention_fraction"] - 0.5) < 1e-5
        assert abs(layer["ffn_fraction"] - 0.5) < 1e-5

    @pytest.mark.asyncio
    async def test_missing_hidden_state_skipped(self, loaded_model_state: MagicMock) -> None:
        """If hidden_states or prev_hidden is missing for a layer, it's skipped."""
        hidden_dim = 64

        captured = {
            "embeddings": mx.array(np.zeros((1, 5, hidden_dim), dtype=np.float32)),
            "hidden_states": {},  # No hidden states
            "prev_hidden": {},  # No prev hidden
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
            return_value=captured,
        ):
            result = await residual_decomposition(prompt="hello", layers=[0])

        assert "error" not in result
        # No layers in output since hidden states are missing
        assert len(result["layers"]) == 0


# ============================================================================
# 8. Default layer selection for logit_attribution with >12 layers
# ============================================================================


class TestLogitAttributionDefaultLayers:
    """Test default layer selection when num_layers > 12."""

    @pytest.mark.asyncio
    async def test_default_layers_with_many_layers(self) -> None:
        """When num_layers > 12, should sample ~12 layers plus last."""
        from chuk_mcp_lazarus.model_state import ModelMetadata

        # Create metadata with 24 layers
        metadata = ModelMetadata(
            model_id="test/model",
            family="test",
            architecture="test_arch",
            num_layers=24,
            hidden_dim=64,
            num_attention_heads=4,
            num_kv_heads=4,
            vocab_size=100,
            intermediate_size=256,
            max_position_embeddings=512,
            head_dim=16,
            parameter_count=1000,
        )

        state = MagicMock()
        state.is_loaded = True
        state.model = MagicMock()
        state.tokenizer = MagicMock()
        state.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        state.tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        state.config = MagicMock()
        state.metadata = metadata

        hidden_dim = 64
        vocab_size = 100

        # Create captured data for all possible layers (0..23)
        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in range(24):
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.model_state.ModelState.get",
                return_value=state,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
        ):
            result = await logit_attribution(prompt="hello", layers=None, normalized=True)

        assert "error" not in result
        # With 24 layers, step = max(1, 24//12) = 2
        # range(0, 24, 2) = [0,2,4,6,8,10,12,14,16,18,20,22]
        # plus 23 if not already there -> 13 layers
        assert len(result["layers"]) == 13
        # Last layer should be 23
        layer_indices = [lay["layer"] for lay in result["layers"]]
        assert 23 in layer_indices


# ============================================================================
# 9. ValueError catch in logit_attribution target token resolution
# ============================================================================


class TestLogitAttributionValueError:
    """Test ValueError from _resolve_target_token in logit_attribution."""

    @pytest.mark.asyncio
    async def test_invalid_target_token_returns_error(self, loaded_model_state: MagicMock) -> None:
        """When _resolve_target_token raises ValueError, should return InvalidInput."""
        hidden_dim = 64
        vocab_size = 100
        num_layers = 4
        last_layer = num_layers - 1

        captured = {
            "embeddings": mx.array(np.random.randn(1, 5, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in [0, last_layer]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, 5, hidden_dim).astype(np.float32)
            )

        def mock_lm_head(x: Any) -> mx.array:
            batch = x.shape[0]
            seq = x.shape[1]
            return mx.array(np.random.randn(batch, seq, vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                side_effect=ValueError("Could not encode target token 'xyz'"),
            ),
        ):
            result = await logit_attribution(prompt="hello", layers=[0], target_token="xyz")

        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "Could not encode" in result["message"]


# ============================================================================
# 10. _head_attribution_impl tests
# ============================================================================


class _PlainAttn:
    """Minimal attention module without q_norm/k_norm/rope (plain Python class
    so ``hasattr`` returns False for missing attributes)."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, head_dim: int = 16) -> None:
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Projection callables that return correctly-shaped arrays
        self.q_proj = lambda x: mx.array(
            np.random.randn(*x.shape[:-1], num_heads * head_dim).astype(np.float32)
        )
        self.k_proj = lambda x: mx.array(
            np.random.randn(*x.shape[:-1], num_heads * head_dim).astype(np.float32)
        )
        self.v_proj = lambda x: mx.array(
            np.random.randn(*x.shape[:-1], num_heads * head_dim).astype(np.float32)
        )

        # o_proj has a weight matrix
        self.o_proj = MagicMock()
        self.o_proj.weight = mx.array(
            np.random.randn(hidden_dim, num_heads * head_dim).astype(np.float32)
        )

        self.scale = head_dim**-0.5


class _PlainAttnWithNorms(_PlainAttn):
    """Attention module with q_norm, k_norm, and rope."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__(hidden_dim, num_heads, head_dim)
        self.q_norm = lambda x: x
        self.k_norm = lambda x: x
        self.rope = lambda x: x


class _PlainLayer:
    """Minimal transformer layer (plain Python class for correct hasattr)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        head_dim: int = 16,
        attn_norms: bool = False,
    ) -> None:
        if attn_norms:
            self.self_attn = _PlainAttnWithNorms(hidden_dim, num_heads, head_dim)
        else:
            self.self_attn = _PlainAttn(hidden_dim, num_heads, head_dim)
        self.input_layernorm = lambda x: x
        self.post_attention_layernorm = lambda x: x
        self.mlp = lambda x: x

    def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> tuple:
        return (h, None)


class TestHeadAttributionImpl:
    """Tests for _head_attribution_impl."""

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        return tok

    def _make_metadata(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        head_dim: int = 16,
        hidden_dim: int = 64,
    ) -> MagicMock:
        m = MagicMock()
        m.num_layers = num_layers
        m.num_attention_heads = num_heads
        m.num_kv_heads = num_kv_heads
        m.head_dim = head_dim
        m.hidden_dim = hidden_dim
        return m

    def _make_model_layers(
        self,
        num_layers: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        head_dim: int = 16,
        attn_norms: bool = False,
    ) -> list:
        return [
            _PlainLayer(hidden_dim, num_heads, head_dim, attn_norms=attn_norms)
            for _ in range(num_layers)
        ]

    def _setup_sdpa_mock(self, num_heads: int, seq_len: int, head_dim: int) -> Any:
        """Return a patched sdpa function that returns correct shape."""

        def mock_sdpa(q: Any, k: Any, v: Any, scale: float = 1.0, mask: Any = None) -> Any:
            return mx.array(np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float32))

        return mock_sdpa

    def test_basic_success(self) -> None:
        """Basic _head_attribution_impl with 4 heads, no norms/rope."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello world",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result
        assert result["prompt"] == "hello world"
        assert result["layer"] == 0
        assert result["num_heads"] == num_heads
        assert len(result["heads"]) == num_heads
        assert "layer_total_logit" in result
        assert "summary" in result
        summary = result["summary"]
        assert "top_positive_head" in summary
        assert "top_negative_head" in summary
        assert "positive_head_count" in summary
        assert "negative_head_count" in summary
        assert "concentration" in summary

    def test_with_q_norm_k_norm_rope(self) -> None:
        """_head_attribution_impl with q_norm, k_norm, rope."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers(attn_norms=True)

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result
        assert len(result["heads"]) == num_heads

    def test_gqa_kv_heads_less_than_q_heads(self) -> None:
        """GQA: num_kv_heads < num_heads triggers key/value repeat."""
        hidden_dim = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata(num_kv_heads=num_kv_heads)

        # Create layers with kv_heads = 2 projections
        layers = []
        for _ in range(4):
            layer = _PlainLayer(hidden_dim, num_heads, head_dim)
            # Override k_proj and v_proj to output kv_heads * head_dim
            layer.self_attn.k_proj = lambda x: mx.array(
                np.random.randn(*x.shape[:-1], num_kv_heads * head_dim).astype(np.float32)
            )
            layer.self_attn.v_proj = lambda x: mx.array(
                np.random.randn(*x.shape[:-1], num_kv_heads * head_dim).astype(np.float32)
            )
            layers.append(layer)

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result
        assert result["num_heads"] == num_heads

    def test_with_embedding_scale(self) -> None:
        """_head_attribution_impl with embedding scale factor."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.ones((1, seq_len, hidden_dim), dtype=np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = 50.0  # Gemma-like scale
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_lm_head_none_returns_error(self) -> None:
        """When _get_lm_projection returns None, should return error dict."""
        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, 5, 64).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = None

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=None,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    def test_unembed_vector_none_returns_error(self) -> None:
        """When _get_unembed_vector returns None, should return error dict."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=None,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "unembedding" in result["message"]

    def test_target_token_resolve_error(self) -> None:
        """When _resolve_target_token raises ValueError."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                side_effect=ValueError("Could not encode target token 'zzz'"),
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token="zzz",
                position=-1,
            )

        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    def test_with_explicit_target_token(self) -> None:
        """Specifying a target_token string."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                return_value=(42, "world"),
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token="world",
                position=-1,
            )

        assert "error" not in result
        assert result["target_token"] == "world"
        assert result["target_token_id"] == 42

    def test_layer_gt_zero(self) -> None:
        """Test attribution on a layer > 0 (exercises forward through earlier layers)."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=2,  # Layer 2 (forward through 0,1 first)
                target_token=None,
                position=-1,
            )

        assert "error" not in result
        assert result["layer"] == 2

    def test_lm_head_with_logits_attr(self) -> None:
        """lm_head returns an object with .logits attribute (HeadOutput wrapper)."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        class HeadOutput:
            def __init__(self, x: Any) -> None:
                self.logits = mx.array(
                    np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32)
                )

        def lm_head_fn(x: Any) -> HeadOutput:
            return HeadOutput(x)

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_lm_head_tuple_output(self) -> None:
        """lm_head returns tuple -> extracts first element."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x: Any) -> tuple:
            return (
                mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32)),
                None,
            )

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_with_final_norm_none(self) -> None:
        """When final_norm is None, should still work."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = None  # No final norm

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_position_zero(self) -> None:
        """Using explicit position=0 instead of -1."""
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model_layers = self._make_model_layers()

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = model_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock(num_heads, seq_len, head_dim)

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=0,
            )

        assert "error" not in result
        assert result["token_position"] == 0


# ============================================================================
# 11. _top_neurons_impl tests
# ============================================================================


class _PlainMLP:
    """Minimal MLP module (plain Python class so hasattr works correctly)."""

    def __init__(
        self, hidden_dim: int = 64, intermediate_size: int = 128, has_gate: bool = True
    ) -> None:
        self.up_proj = lambda x: mx.array(
            np.random.randn(*x.shape[:-1], intermediate_size).astype(np.float32)
        )
        self.down_proj = MagicMock()
        self.down_proj.weight = mx.array(
            np.random.randn(hidden_dim, intermediate_size).astype(np.float32)
        )

        if has_gate:
            self.gate_proj = lambda x: mx.array(
                np.random.randn(*x.shape[:-1], intermediate_size).astype(np.float32)
            )


class _PlainMLP_Standard:
    """MLP without gate_proj (standard, non-SwiGLU)."""

    def __init__(self, hidden_dim: int = 64, intermediate_size: int = 128) -> None:
        self.up_proj = lambda x: mx.array(
            np.random.randn(*x.shape[:-1], intermediate_size).astype(np.float32)
        )
        self.down_proj = MagicMock()
        self.down_proj.weight = mx.array(
            np.random.randn(hidden_dim, intermediate_size).astype(np.float32)
        )


class _PlainLayerForNeurons:
    """Transformer layer for top_neurons tests."""

    def __init__(
        self,
        hidden_dim: int = 64,
        intermediate_size: int = 128,
        has_gate: bool = True,
        four_norms: bool = False,
    ) -> None:
        self.self_attn = MagicMock()
        self.input_layernorm = lambda x: x
        self.post_attention_layernorm = lambda x: x
        if has_gate:
            self.mlp = _PlainMLP(hidden_dim, intermediate_size, has_gate=True)
        else:
            self.mlp = _PlainMLP_Standard(hidden_dim, intermediate_size)

        if four_norms:
            self.pre_feedforward_layernorm = lambda x: x
            self.post_feedforward_layernorm = lambda x: x

    def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> tuple:
        return (h, None)


class TestTopNeuronsImpl:
    """Tests for _top_neurons_impl."""

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        return tok

    def _make_metadata(self, num_layers: int = 4) -> MagicMock:
        m = MagicMock()
        m.num_layers = num_layers
        m.num_attention_heads = 4
        m.num_kv_heads = 4
        m.head_dim = 16
        m.hidden_dim = 64
        return m

    def _make_captured(
        self,
        hidden_dim: int = 64,
        seq_len: int = 5,
        layers: list[int] | None = None,
    ) -> dict:
        """Build captured dict as returned by _run_decomposition_forward."""
        if layers is None:
            layers = [0, 3]
        captured: dict[str, Any] = {
            "embeddings": mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in layers:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
        return captured

    def test_swiglu_basic_success(self) -> None:
        """Test _top_neurons_impl with SwiGLU MLP (has gate_proj)."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["mlp_type"] == "swiglu"
        assert result["intermediate_size"] == intermediate_size
        assert result["top_k"] == 5
        assert "top_positive" in result
        assert "top_negative" in result
        assert "total_neuron_logit" in result
        assert "summary" in result
        summary = result["summary"]
        assert "top_neuron_index" in summary
        assert "positive_neuron_count" in summary
        assert "negative_neuron_count" in summary
        assert "concentration_top10" in summary
        assert "sparsity" in summary

    def test_standard_mlp_success(self) -> None:
        """Test _top_neurons_impl with standard MLP (no gate_proj, uses gelu)."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=False) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["mlp_type"] == "standard"

    def test_four_norms_path(self) -> None:
        """Test with Gemma-style 4-norm layer."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True, four_norms=True)
            for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result

    def test_lm_head_none_returns_error(self) -> None:
        """When _get_lm_projection returns None, should return error."""
        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured()

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [MagicMock()] * 4
        mock_hooks._get_final_norm.return_value = None

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=None,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    def test_unembed_vector_none_returns_error(self) -> None:
        """When _get_unembed_vector returns None, should return error."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]
        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=None,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert result["error"] is True
        assert "unembedding" in result["message"]

    def test_attn_output_none_returns_error(self) -> None:
        """When attn_output for requested layer is None, should return error."""
        hidden_dim = 64
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model = MagicMock()
        config = MagicMock()

        # Captured with attn_outputs[0] = None (non-decomposable layer)
        captured = self._make_captured(hidden_dim, seq_len, [0, 3])
        captured["attn_outputs"][0] = None

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [MagicMock()] * 4
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert result["error"] is True
        assert "decomposable" in result["message"]

    def test_target_token_resolve_error(self) -> None:
        """When _resolve_target_token raises ValueError."""
        hidden_dim = 64
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()
        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = [MagicMock()] * 4
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                side_effect=ValueError("Could not encode"),
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token="zzz",
                position=-1,
                top_k=5,
            )

        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    def test_with_explicit_target_token(self) -> None:
        """Providing an explicit target_token string."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                return_value=(42, "world"),
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token="world",
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["target_token"] == "world"
        assert result["target_token_id"] == 42

    def test_final_norm_none(self) -> None:
        """When final_norm is None, should still succeed."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = None  # No final norm

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result

    def test_position_zero(self) -> None:
        """Using explicit position=0."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=0,
                top_k=5,
            )

        assert "error" not in result
        assert result["token_position"] == 0

    def test_mlp_with_custom_activation_swiglu(self) -> None:
        """SwiGLU MLP with custom act attribute."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]
        # Add custom activation
        for lay in layers:
            lay.mlp.act = nn.silu

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["mlp_type"] == "swiglu"

    def test_mlp_with_custom_activation_standard(self) -> None:
        """Standard MLP with custom activation attribute."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=False) for _ in range(4)
        ]
        # Add custom activation
        for lay in layers:
            lay.mlp.activation = nn.gelu

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["mlp_type"] == "standard"

    def test_lm_head_logits_attr(self) -> None:
        """lm_head returns object with .logits attribute."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        class HeadOutput:
            def __init__(self, x: Any) -> None:
                self.logits = mx.array(
                    np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32)
                )

        def lm_head_fn(x: Any) -> HeadOutput:
            return HeadOutput(x)

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result

    def test_lm_head_tuple_output(self) -> None:
        """lm_head returns tuple."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        layers = [
            _PlainLayerForNeurons(hidden_dim, intermediate_size, has_gate=True) for _ in range(4)
        ]

        model = MagicMock()
        config = MagicMock()

        captured = self._make_captured(hidden_dim, seq_len, [0, 3])

        def lm_head_fn(x: Any) -> tuple:
            return (
                mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32)),
                None,
            )

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result  # final test in TestTopNeuronsImpl


# ============================================================================
# 12. Additional coverage: forward-pass fallbacks in _head_attribution_impl
# ============================================================================


class _TypeErrorLayer:
    """Layer that raises TypeError on the first two calls (mask+cache and
    cache-only) but succeeds on the third (bare call).

    Uses a call counter rather than argument inspection because the fallback
    code passes ``cache=None`` which doesn't distinguish from a missing arg.
    """

    def __init__(self, hidden_dim: int = 64, seq_len: int = 5) -> None:
        self._output = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))
        self._call_count = 0

    def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> Any:
        self._call_count += 1
        if self._call_count <= 2:
            raise TypeError(f"fail call #{self._call_count}")
        return self._output


class _HiddenStatesOutputLayer:
    """Layer that returns an object with .hidden_states attribute."""

    def __init__(self, hidden_dim: int = 64, seq_len: int = 5) -> None:
        self._output = MagicMock()
        self._output.hidden_states = mx.array(
            np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
        )

    def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> Any:
        return self._output


class TestHeadAttributionImplForwardFallbacks:
    """Test forward-pass fallback paths in _head_attribution_impl."""

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        return tok

    def _make_metadata(self) -> MagicMock:
        m = MagicMock()
        m.num_layers = 4
        m.num_attention_heads = 4
        m.num_kv_heads = 4
        m.head_dim = 16
        m.hidden_dim = 64
        return m

    def _setup_sdpa_mock(self) -> Any:
        def mock_sdpa(q: Any, k: Any, v: Any, scale: float = 1.0, mask: Any = None) -> Any:
            return mx.array(np.random.randn(1, 4, 5, 16).astype(np.float32))

        return mock_sdpa

    def test_forward_layers_raise_type_error(self) -> None:
        """Earlier layers raise TypeError -> fallback through three call patterns."""
        hidden_dim = 64
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        error_layers = [_TypeErrorLayer(hidden_dim, seq_len) for _ in range(2)]
        target_layer = _PlainLayer(hidden_dim, 4, 16)
        remaining_layer = _PlainLayer(hidden_dim, 4, 16)
        all_layers = error_layers + [target_layer, remaining_layer]

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = all_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock()

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=2,
                target_token=None,
                position=-1,
            )

        assert "error" not in result
        assert result["layer"] == 2

    def test_forward_layers_hidden_states_output(self) -> None:
        """Earlier layers return object with .hidden_states attribute."""
        hidden_dim = 64
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        hs_layer = _HiddenStatesOutputLayer(hidden_dim, seq_len)
        target_layer = _PlainLayer(hidden_dim, 4, 16)
        remaining_layers = [_PlainLayer(hidden_dim, 4, 16) for _ in range(2)]
        all_layers = [hs_layer, target_layer] + remaining_layers

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = all_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock()

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=1,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_forward_layers_plain_array_output(self) -> None:
        """Earlier layers return plain array (not tuple, no .hidden_states)."""
        hidden_dim = 64
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        class PlainOutputLayer:
            """Returns plain MockMxArray — no .hidden_states, not a tuple."""

            def __init__(self) -> None:
                self._out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

            def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> Any:
                return self._out

        plain_layer = PlainOutputLayer()
        target_layer = _PlainLayer(hidden_dim, 4, 16)
        remaining_layers = [_PlainLayer(hidden_dim, 4, 16) for _ in range(2)]
        all_layers = [plain_layer, target_layer] + remaining_layers

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = all_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock()

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=1,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_full_forward_layers_type_error_fallback(self) -> None:
        """Full-model forward (lines 1358-1372) also exercises TypeError fallback."""
        hidden_dim = 64
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        # target is layer 0, remaining layers (1, 2, 3) fail with TypeError
        target_layer = _PlainLayer(hidden_dim, 4, 16)
        error_layers = [_TypeErrorLayer(hidden_dim, seq_len) for _ in range(3)]
        all_layers = [target_layer] + error_layers

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = all_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock()

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result

    def test_full_forward_hidden_states_and_plain_output(self) -> None:
        """Full forward pass with mixed layer output types."""
        hidden_dim = 64
        seq_len = 5
        vocab_size = 100

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        target_layer = _PlainLayer(hidden_dim, 4, 16)
        hs_layer = _HiddenStatesOutputLayer(hidden_dim, seq_len)

        class PlainOutputLayer2:
            def __init__(self) -> None:
                self._out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

            def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> Any:
                return self._out

        all_layers = [target_layer, hs_layer, PlainOutputLayer2(), _PlainLayer(hidden_dim, 4, 16)]

        model = MagicMock()
        config = MagicMock()

        embed_out = mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = all_layers
        mock_hooks._get_embed_tokens.return_value = lambda ids: embed_out
        mock_hooks._get_embedding_scale.return_value = None
        mock_hooks._get_final_norm.return_value = lambda x: x

        sdpa_mock = self._setup_sdpa_mock()

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
            patch(
                "mlx.core.fast.scaled_dot_product_attention",
                side_effect=sdpa_mock,
            ),
        ):
            result = _head_attribution_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
            )

        assert "error" not in result


# ============================================================================
# 13. Additional coverage: hidden_act dimensions in _top_neurons_impl
# ============================================================================


class TestTopNeuronsImplHiddenActDimensions:
    """Cover 2D and 1D hidden_act dimension paths in _top_neurons_impl."""

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        return tok

    def _make_metadata(self) -> MagicMock:
        m = MagicMock()
        m.num_layers = 4
        m.num_attention_heads = 4
        m.num_kv_heads = 4
        m.head_dim = 16
        m.hidden_dim = 64
        return m

    def _make_captured(self, hidden_dim: int = 64, seq_len: int = 5) -> dict:
        captured: dict[str, Any] = {
            "embeddings": mx.array(np.random.randn(1, seq_len, hidden_dim).astype(np.float32)),
            "hidden_states": {},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }
        for i in [0, 3]:
            captured["hidden_states"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
            captured["prev_hidden"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
            captured["attn_outputs"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
            captured["ffn_outputs"][i] = mx.array(
                np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
            )
        return captured

    def test_2d_hidden_act(self) -> None:
        """MLP projections return [seq, intermediate_size] (2D) tensors."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        class MLP2D:
            """MLP returning 2D outputs."""

            def __init__(self) -> None:
                self.gate_proj = lambda x: mx.array(
                    np.random.randn(seq_len, intermediate_size).astype(np.float32)
                )
                self.up_proj = lambda x: mx.array(
                    np.random.randn(seq_len, intermediate_size).astype(np.float32)
                )
                self.down_proj = MagicMock()
                self.down_proj.weight = mx.array(
                    np.random.randn(hidden_dim, intermediate_size).astype(np.float32)
                )

        class LayerWith2DMLP:
            """Layer with 2D MLP output."""

            def __init__(self) -> None:
                self.self_attn = MagicMock()
                self.input_layernorm = lambda x: x
                self.post_attention_layernorm = lambda x: x
                self.mlp = MLP2D()

            def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> tuple:
                return (h, None)

        layers = [LayerWith2DMLP() for _ in range(4)]

        model = MagicMock()
        config = MagicMock()
        captured = self._make_captured(hidden_dim, seq_len)

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["intermediate_size"] == intermediate_size

    def test_1d_hidden_act(self) -> None:
        """MLP projections return [intermediate_size] (1D) tensors."""
        hidden_dim = 64
        intermediate_size = 128
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        class MLP1D:
            """MLP returning 1D outputs."""

            def __init__(self) -> None:
                self.gate_proj = lambda x: mx.array(
                    np.random.randn(intermediate_size).astype(np.float32)
                )
                self.up_proj = lambda x: mx.array(
                    np.random.randn(intermediate_size).astype(np.float32)
                )
                self.down_proj = MagicMock()
                self.down_proj.weight = mx.array(
                    np.random.randn(hidden_dim, intermediate_size).astype(np.float32)
                )

        class LayerWith1DMLP:
            """Layer with 1D MLP output."""

            def __init__(self) -> None:
                self.self_attn = MagicMock()
                self.input_layernorm = lambda x: x
                self.post_attention_layernorm = lambda x: x
                self.mlp = MLP1D()

            def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> tuple:
                return (h, None)

        layers = [LayerWith1DMLP() for _ in range(4)]

        model = MagicMock()
        config = MagicMock()
        captured = self._make_captured(hidden_dim, seq_len)

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert result["intermediate_size"] == intermediate_size

    def test_all_zero_activations(self) -> None:
        """All neuron contributions are zero -> empty all_top_indices (line 1784)."""
        hidden_dim = 64
        intermediate_size = 8
        vocab_size = 100
        seq_len = 5

        tokenizer = self._make_tokenizer()
        metadata = self._make_metadata()

        class ZeroMLP:
            """MLP that returns all-zero activations."""

            def __init__(self) -> None:
                self.gate_proj = lambda x: mx.array(
                    np.zeros((1, seq_len, intermediate_size), dtype=np.float32)
                )
                self.up_proj = lambda x: mx.array(
                    np.zeros((1, seq_len, intermediate_size), dtype=np.float32)
                )
                self.down_proj = MagicMock()
                self.down_proj.weight = mx.array(
                    np.random.randn(hidden_dim, intermediate_size).astype(np.float32)
                )

        class LayerWithZeroMLP:
            """Layer with zero-output MLP."""

            def __init__(self) -> None:
                self.self_attn = MagicMock()
                self.input_layernorm = lambda x: x
                self.post_attention_layernorm = lambda x: x
                self.mlp = ZeroMLP()

            def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> tuple:
                return (h, None)

        layers = [LayerWithZeroMLP() for _ in range(4)]

        model = MagicMock()
        config = MagicMock()
        captured = self._make_captured(hidden_dim, seq_len)

        def lm_head_fn(x):
            return mx.array(np.random.randn(*x.shape[:-1], vocab_size).astype(np.float32))

        unembed_vec = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        mock_hooks = MagicMock()
        mock_hooks._get_layers.return_value = layers
        mock_hooks._get_final_norm.return_value = lambda x: x

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=captured,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_hooks,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=unembed_vec,
            ),
        ):
            result = _top_neurons_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=0,
                target_token=None,
                position=-1,
                top_k=5,
            )

        assert "error" not in result
        assert len(result["top_positive"]) == 0
        assert len(result["top_negative"]) == 0
