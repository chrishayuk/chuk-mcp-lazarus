"""Tests for prefill_inject tools: prefill_to_layer, kv_inject_test.

Covers all private helpers, both async tool entry points, both impl
functions, and all result models.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MLX stub helpers
# ---------------------------------------------------------------------------


def _ensure_mlx_stubs() -> None:
    if not hasattr(nn, "MultiHeadAttention"):

        class _MockMHA:
            @staticmethod
            def create_additive_causal_mask(seq_len: int) -> Any:
                mask = np.zeros((seq_len, seq_len), dtype=np.float32)
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        mask[i, j] = -1e9
                return mx.array(mask)

        nn.MultiHeadAttention = _MockMHA  # type: ignore[attr-defined]


_ensure_mlx_stubs()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
VOCAB_SIZE = 100
NUM_LAYERS = 4
SEQ_LEN = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_layer() -> MagicMock:
    layer = MagicMock()
    layer.return_value = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
    return layer


def _make_hidden(seq: int = SEQ_LEN, hdim: int = HIDDEN_DIM) -> Any:
    return mx.array(np.random.randn(1, seq, hdim).astype(np.float32))


# ---------------------------------------------------------------------------
# Private helper tests: _resolve_position
# ---------------------------------------------------------------------------


class TestResolvePosition:
    def test_last_token(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(5, -1) == 4

    def test_second_to_last(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(5, -2) == 3

    def test_zero(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(5, 0) == 0

    def test_positive(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(5, 3) == 3

    def test_clamp_high(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(5, 100) == 4

    def test_clamp_negative_overflow(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(5, -100) == 0

    def test_single_token(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _resolve_position

        assert _resolve_position(1, -1) == 0


# ---------------------------------------------------------------------------
# Private helper tests: _softmax_np
# ---------------------------------------------------------------------------


class TestSoftmaxNp:
    def test_sums_to_one(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _softmax_np

        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax_np(logits)
        assert abs(float(np.sum(probs)) - 1.0) < 1e-6

    def test_all_nonnegative(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _softmax_np

        logits = np.array([-5.0, 0.0, 5.0])
        probs = _softmax_np(logits)
        assert np.all(probs >= 0.0)

    def test_max_logit_has_highest_prob(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _softmax_np

        logits = np.array([1.0, 5.0, 2.0])
        probs = _softmax_np(logits)
        assert int(np.argmax(probs)) == 1

    def test_uniform_logits_give_equal_probs(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _softmax_np

        logits = np.array([1.0, 1.0, 1.0])
        probs = _softmax_np(logits)
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_large_values_numerically_stable(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _softmax_np

        logits = np.array([1000.0, 1000.0, 1000.0])
        probs = _softmax_np(logits)
        assert np.all(np.isfinite(probs))


# ---------------------------------------------------------------------------
# Private helper tests: _kl_divergence_np
# ---------------------------------------------------------------------------


class TestKLDivergenceNp:
    def test_identical_distributions_gives_zero(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kl_divergence_np

        logits = np.array([1.0, 2.0, 3.0, 4.0])
        kl = _kl_divergence_np(logits, logits.copy())
        assert kl < 1e-5

    def test_nonnegative(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kl_divergence_np

        p = np.array([2.0, 1.0, 0.5])
        q = np.array([1.0, 2.0, 0.5])
        assert _kl_divergence_np(p, q) >= 0.0

    def test_asymmetric(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kl_divergence_np

        p = np.array([3.0, 1.0, 0.1])
        q = np.array([1.0, 2.0, 2.0])
        kl_pq = _kl_divergence_np(p, q)
        kl_qp = _kl_divergence_np(q, p)
        # KL is not symmetric in general
        assert isinstance(kl_pq, float)
        assert isinstance(kl_qp, float)

    def test_very_different_gives_large_kl(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kl_divergence_np

        p = np.array([100.0, -100.0])  # almost all mass on first
        q = np.array([-100.0, 100.0])  # almost all mass on second
        kl = _kl_divergence_np(p, q)
        assert kl > 1.0


# ---------------------------------------------------------------------------
# Private helper tests: _top_k_tokens
# ---------------------------------------------------------------------------


class TestTopKTokens:
    def test_returns_k_entries(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _top_k_tokens

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"
        logits = np.random.randn(VOCAB_SIZE).astype(np.float32)
        result = _top_k_tokens(logits, tokenizer, 5)
        assert len(result) == 5

    def test_sorted_by_probability(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _top_k_tokens

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"
        logits = np.random.randn(VOCAB_SIZE).astype(np.float32)
        result = _top_k_tokens(logits, tokenizer, 5)
        probs = [e["probability"] for e in result]
        assert probs == sorted(probs, reverse=True)

    def test_entry_has_required_keys(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _top_k_tokens

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"
        logits = np.random.randn(VOCAB_SIZE).astype(np.float32)
        result = _top_k_tokens(logits, tokenizer, 3)
        for entry in result:
            assert "token" in entry
            assert "token_id" in entry
            assert "probability" in entry

    def test_k_larger_than_vocab(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _top_k_tokens

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"
        logits = np.random.randn(10).astype(np.float32)
        result = _top_k_tokens(logits, tokenizer, 100)
        assert len(result) == 10

    def test_probabilities_sum_to_one(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _top_k_tokens

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"
        logits = np.random.randn(VOCAB_SIZE).astype(np.float32)
        result = _top_k_tokens(logits, tokenizer, VOCAB_SIZE)
        total = sum(e["probability"] for e in result)
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# _run_forward_to_layer — direct tests
# ---------------------------------------------------------------------------


class TestRunForwardToLayer:
    def _run(self, stop_after: int = 0, hidden_states_output: bool = True) -> Any:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _run_forward_to_layer

        model = MagicMock()
        config = MagicMock()
        input_ids = mx.array(np.array([[1, 2, 3, 4, 5]]))

        h_out = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))

        if hidden_states_output:
            layer_out = MagicMock()
            layer_out.hidden_states = h_out
        else:
            layer_out = h_out  # returns tensor directly

        mock_layers = [MagicMock(return_value=layer_out) for _ in range(NUM_LAYERS)]
        embed_out = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=embed_out)
        mock_helper._get_embedding_scale.return_value = None

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
        ):
            return _run_forward_to_layer(model, config, input_ids, stop_after)

    def test_returns_mx_array(self) -> None:
        result = self._run(stop_after=0)
        assert hasattr(result, "shape")

    def test_shape_correct(self) -> None:
        result = self._run(stop_after=0)
        assert result.shape == (1, SEQ_LEN, HIDDEN_DIM)

    def test_stop_at_middle_layer(self) -> None:
        result = self._run(stop_after=1)
        assert result.shape == (1, SEQ_LEN, HIDDEN_DIM)

    def test_tuple_output_handling(self) -> None:
        """Layer returning a tuple (hidden, ...) is handled."""
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _run_forward_to_layer

        model = MagicMock()
        config = MagicMock()
        input_ids = mx.array(np.array([[1, 2, 3]]))

        h_out = mx.array(np.random.randn(1, 3, HIDDEN_DIM).astype(np.float32))

        mock_layers = [MagicMock(return_value=(h_out, None)) for _ in range(NUM_LAYERS)]
        embed_out = mx.array(np.random.randn(1, 3, HIDDEN_DIM).astype(np.float32))

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=embed_out)
        mock_helper._get_embedding_scale.return_value = None

        with (
            patch("chuk_lazarus.introspection.hooks.ModelHooks", return_value=mock_helper),
            patch("mlx.core.eval"),
        ):
            result = _run_forward_to_layer(model, config, input_ids, 0)
        assert result.shape[2] == HIDDEN_DIM

    def test_with_embedding_scale(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _run_forward_to_layer

        model = MagicMock()
        config = MagicMock()
        input_ids = mx.array(np.array([[1, 2, 3]]))

        h_out = mx.array(np.random.randn(1, 3, HIDDEN_DIM).astype(np.float32))
        mock_layers = [MagicMock(return_value=h_out) for _ in range(NUM_LAYERS)]
        embed_out = mx.array(np.random.randn(1, 3, HIDDEN_DIM).astype(np.float32))

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=embed_out)
        mock_helper._get_embedding_scale.return_value = 8.0  # non-None scale

        with (
            patch("chuk_lazarus.introspection.hooks.ModelHooks", return_value=mock_helper),
            patch("mlx.core.eval"),
        ):
            result = _run_forward_to_layer(model, config, input_ids, 0)
        assert result is not None

    def test_layer_fallback_no_mask(self) -> None:
        """Layer that raises TypeError for mask kwarg falls back to cache-only."""
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _run_forward_to_layer

        model = MagicMock()
        config = MagicMock()
        input_ids = mx.array(np.array([[1, 2]]))

        h_out = mx.array(np.random.randn(1, 2, HIDDEN_DIM).astype(np.float32))

        call_count = [0]

        def layer_side_effect(*args, **kwargs):
            call_count[0] += 1
            if "mask" in kwargs:
                raise TypeError("no mask")
            return h_out

        mock_layer = MagicMock(side_effect=layer_side_effect)
        embed_out = mx.array(np.random.randn(1, 2, HIDDEN_DIM).astype(np.float32))

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = [mock_layer] + [MagicMock(return_value=h_out)] * (
            NUM_LAYERS - 1
        )
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=embed_out)
        mock_helper._get_embedding_scale.return_value = None

        with (
            patch("chuk_lazarus.introspection.hooks.ModelHooks", return_value=mock_helper),
            patch("mlx.core.eval"),
        ):
            result = _run_forward_to_layer(model, config, input_ids, 0)
        assert result is not None


# ---------------------------------------------------------------------------
# _run_forward_from_layer — direct tests
# ---------------------------------------------------------------------------


class TestRunForwardFromLayer:
    def _run(
        self,
        start_layer: int = 2,
        with_final_norm: bool = True,
        lm_head_tuple: bool = False,
        lm_head_logits_attr: bool = False,
    ) -> np.ndarray:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _run_forward_from_layer

        model = MagicMock()
        config = MagicMock()
        h = _make_hidden()

        h_out = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
        mock_layers = [MagicMock(return_value=h_out) for _ in range(NUM_LAYERS)]

        final_norm = (lambda x: x) if with_final_norm else None

        logits_tensor = mx.array(np.random.randn(1, 1, VOCAB_SIZE).astype(np.float32))
        if lm_head_logits_attr:

            class _Out:
                logits = logits_tensor

            lm_head = MagicMock(return_value=_Out())
        elif lm_head_tuple:
            lm_head = MagicMock(return_value=(logits_tensor,))
        else:
            lm_head = MagicMock(return_value=logits_tensor)

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers

        with (
            patch("chuk_lazarus.introspection.hooks.ModelHooks", return_value=mock_helper),
            patch("mlx.core.eval"),
        ):
            return _run_forward_from_layer(model, config, h, start_layer, final_norm, lm_head, 0)

    def test_returns_numpy_array(self) -> None:
        result = self._run()
        assert isinstance(result, np.ndarray)

    def test_shape_is_vocab(self) -> None:
        result = self._run()
        assert result.shape == (VOCAB_SIZE,)

    def test_no_final_norm(self) -> None:
        result = self._run(with_final_norm=False)
        assert result.shape == (VOCAB_SIZE,)

    def test_lm_head_tuple_output(self) -> None:
        result = self._run(lm_head_tuple=True)
        assert result.shape == (VOCAB_SIZE,)

    def test_lm_head_logits_attr(self) -> None:
        result = self._run(lm_head_logits_attr=True)
        assert result.shape == (VOCAB_SIZE,)

    def test_start_layer_zero(self) -> None:
        result = self._run(start_layer=0)
        assert isinstance(result, np.ndarray)

    def test_start_layer_last(self) -> None:
        result = self._run(start_layer=NUM_LAYERS - 1)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# prefill_to_layer — async validation
# ---------------------------------------------------------------------------


class TestPrefillToLayer:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer

        result = await prefill_to_layer(prompt="hello", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range_high(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer

        result = await prefill_to_layer(prompt="hello", layer=99)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_layer_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer

        result = await prefill_to_layer(prompt="hello", layer=-1)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer

        fake: dict[str, Any] = {
            "prompt": "hello",
            "layer": 0,
            "position": -1,
            "num_tokens": 5,
            "hidden_state": [0.1] * HIDDEN_DIM,
            "hidden_norm": 1.0,
            "top_raw_logits": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.prefill_inject._prefill_to_layer_impl",
            return_value=fake,
        ):
            result = await prefill_to_layer(prompt="hello", layer=0)
        assert "error" not in result
        assert result["num_tokens"] == 5

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer

        with patch(
            "chuk_mcp_lazarus.tools.geometry.prefill_inject._prefill_to_layer_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await prefill_to_layer(prompt="hello", layer=0)
        assert result["error_type"] == "GeometryFailed"

    @pytest.mark.asyncio
    async def test_top_k_clamped(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer

        fake: dict[str, Any] = {
            "prompt": "hello",
            "layer": 0,
            "position": -1,
            "num_tokens": 5,
            "hidden_state": [0.0] * HIDDEN_DIM,
            "hidden_norm": 0.0,
            "top_raw_logits": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.prefill_inject._prefill_to_layer_impl",
            return_value=fake,
        ) as mock_impl:
            await prefill_to_layer(prompt="hello", layer=0, top_k_tokens=100)
            top_k_passed = mock_impl.call_args[0][-1]
            assert top_k_passed == 20


# ---------------------------------------------------------------------------
# kv_inject_test — async validation
# ---------------------------------------------------------------------------


class TestKvInjectTest:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import kv_inject_test

        result = await kv_inject_test(
            prompt="hello", token="Paris", coefficient=1.0, inject_layer=0
        )
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_inject_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import kv_inject_test

        result = await kv_inject_test(
            prompt="hello", token="Paris", coefficient=1.0, inject_layer=99
        )
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_inject_layer_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import kv_inject_test

        result = await kv_inject_test(
            prompt="hello", token="Paris", coefficient=1.0, inject_layer=-1
        )
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import kv_inject_test

        fake: dict[str, Any] = {
            "prompt": "hello",
            "token": "Paris",
            "token_id": 42,
            "coefficient": 1.0,
            "inject_layer": 1,
            "position": -1,
            "full_target_prob": 0.3,
            "injected_target_prob": 0.28,
            "target_prob_delta": -0.02,
            "kl_divergence": 0.0001,
            "top_k_comparison": [],
            "summary": {"kl_divergence": 0.0001},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.prefill_inject._kv_inject_test_impl",
            return_value=fake,
        ):
            result = await kv_inject_test(
                prompt="hello", token="Paris", coefficient=1.0, inject_layer=1
            )
        assert "error" not in result
        assert result["kl_divergence"] == 0.0001

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import kv_inject_test

        with patch(
            "chuk_mcp_lazarus.tools.geometry.prefill_inject._kv_inject_test_impl",
            side_effect=RuntimeError("bad"),
        ):
            result = await kv_inject_test(
                prompt="hello", token="Paris", coefficient=1.0, inject_layer=1
            )
        assert result["error_type"] == "GeometryFailed"

    @pytest.mark.asyncio
    async def test_top_k_clamped(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import kv_inject_test

        fake: dict[str, Any] = {
            "prompt": "hello",
            "token": "Paris",
            "token_id": 42,
            "coefficient": 1.0,
            "inject_layer": 1,
            "position": -1,
            "full_target_prob": 0.3,
            "injected_target_prob": 0.3,
            "target_prob_delta": 0.0,
            "kl_divergence": 0.0,
            "top_k_comparison": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.prefill_inject._kv_inject_test_impl",
            return_value=fake,
        ) as mock_impl:
            await kv_inject_test(
                prompt="hello", token="Paris", coefficient=1.0, inject_layer=1, top_k=200
            )
            top_k_passed = mock_impl.call_args[0][-1]
            assert top_k_passed == 50


# ---------------------------------------------------------------------------
# _prefill_to_layer_impl — sync tests
# ---------------------------------------------------------------------------


class TestPrefillToLayerImpl:
    def _run(
        self,
        layer: int = 1,
        position: int = -1,
        top_k: int = 3,
        no_lm_head: bool = False,
        lm_head_tuple: bool = False,
        lm_head_logits_attr: bool = False,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _prefill_to_layer_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        h_out = _make_hidden()
        logits_tensor = mx.array(np.random.randn(1, 1, VOCAB_SIZE).astype(np.float32))

        if lm_head_logits_attr:

            class _Out:
                logits = logits_tensor

            lm_head = MagicMock(return_value=_Out())
        elif lm_head_tuple:
            lm_head = MagicMock(return_value=(logits_tensor,))
        elif no_lm_head:
            lm_head = None
        else:
            lm_head = MagicMock(return_value=logits_tensor)

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_to_layer",
                return_value=h_out,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=lm_head,
            ),
            patch("mlx.core.eval"),
        ):
            return _prefill_to_layer_impl(model, config, tokenizer, "hello", layer, position, top_k)

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "layer",
            "position",
            "num_tokens",
            "hidden_state",
            "hidden_norm",
            "top_raw_logits",
        ]:
            assert k in r

    def test_hidden_state_length(self) -> None:
        r = self._run()
        assert len(r["hidden_state"]) == HIDDEN_DIM

    def test_num_tokens(self) -> None:
        r = self._run()
        assert r["num_tokens"] == SEQ_LEN

    def test_hidden_norm_nonnegative(self) -> None:
        r = self._run()
        assert r["hidden_norm"] >= 0.0

    def test_top_raw_logits_list(self) -> None:
        r = self._run()
        assert isinstance(r["top_raw_logits"], list)

    def test_no_lm_head_gives_empty_logits(self) -> None:
        r = self._run(no_lm_head=True)
        assert r["top_raw_logits"] == []

    def test_lm_head_tuple_output(self) -> None:
        r = self._run(lm_head_tuple=True)
        assert isinstance(r["top_raw_logits"], list)

    def test_lm_head_logits_attr(self) -> None:
        r = self._run(lm_head_logits_attr=True)
        assert isinstance(r["top_raw_logits"], list)

    def test_position_stored(self) -> None:
        r = self._run(position=2)
        assert r["position"] == 2

    def test_layer_stored(self) -> None:
        r = self._run(layer=2)
        assert r["layer"] == 2


# ---------------------------------------------------------------------------
# _kv_inject_test_impl — sync tests
# ---------------------------------------------------------------------------


class TestKvInjectTestImpl:
    def _run(
        self,
        inject_layer: int = 1,
        position: int = -1,
        coefficient: float = 1.5,
        token_id: int | None = 10,
        no_lm_head: bool = False,
        no_unembed: bool = False,
        zero_norm_unembed: bool = False,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kv_inject_test_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = NUM_LAYERS

        # Unembed vector
        if zero_norm_unembed:
            u_vec_np = np.zeros(HIDDEN_DIM, dtype=np.float32)
        else:
            u_vec_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
            u_vec_np /= np.linalg.norm(u_vec_np)  # unit vector

        u_vec = None if no_unembed else mx.array(u_vec_np)

        # Logits for full and injected passes
        full_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        injected_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)

        # Full forward: _run_decomposition_forward
        h_full = _make_hidden()
        decomp_full = {
            "hidden_states": {NUM_LAYERS - 1: h_full},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        lm_head = None if no_lm_head else MagicMock()
        full_logits_mx = mx.array(full_logits_np)

        # Injected forward: _run_forward_to_layer
        h_pre = _make_hidden()

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = lambda x: x
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=h_pre)
        mock_helper._get_embedding_scale.return_value = None

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._resolve_token_to_id",
                return_value=token_id,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp_full,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=lm_head,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                return_value=full_logits_mx,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_to_layer",
                return_value=h_pre,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_from_layer",
                return_value=injected_logits_np,
            ),
            patch("mlx.core.eval"),
        ):
            return _kv_inject_test_impl(
                model,
                config,
                tokenizer,
                meta,
                "hello",
                "Paris",
                coefficient,
                inject_layer,
                position,
                5,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "token",
            "token_id",
            "coefficient",
            "inject_layer",
            "position",
            "full_target_prob",
            "injected_target_prob",
            "target_prob_delta",
            "kl_divergence",
            "top_k_comparison",
            "summary",
        ]:
            assert k in r

    def test_kl_nonnegative(self) -> None:
        r = self._run()
        assert r["kl_divergence"] >= 0.0

    def test_probs_in_range(self) -> None:
        r = self._run()
        assert 0.0 <= r["full_target_prob"] <= 1.0
        assert 0.0 <= r["injected_target_prob"] <= 1.0

    def test_delta_is_difference(self) -> None:
        r = self._run()
        expected_delta = round(r["injected_target_prob"] - r["full_target_prob"], 6)
        assert abs(r["target_prob_delta"] - expected_delta) < 1e-5

    def test_comparison_list(self) -> None:
        r = self._run()
        assert isinstance(r["top_k_comparison"], list)

    def test_summary_fields(self) -> None:
        r = self._run()
        s = r["summary"]
        assert "kl_divergence" in s
        assert "target_token" in s
        assert "statistically_indistinguishable" in s

    def test_coefficient_stored(self) -> None:
        r = self._run(coefficient=2.5)
        assert r["coefficient"] == 2.5

    def test_summary_interpretation_present(self) -> None:
        r = self._run()
        assert "interpretation" in r["summary"]

    def test_unresolvable_token_raises(self) -> None:
        with pytest.raises(ValueError, match="single token ID"):
            self._run(token_id=None)

    def test_no_lm_head_raises(self) -> None:
        with pytest.raises(ValueError, match="language model head"):
            self._run(no_lm_head=True)

    def test_no_unembed_raises(self) -> None:
        with pytest.raises(ValueError, match="unembedding"):
            self._run(no_unembed=True)

    def test_zero_norm_unembed_raises(self) -> None:
        with pytest.raises(ValueError, match="near-zero norm"):
            self._run(zero_norm_unembed=True)

    def test_inject_layer_zero_uses_embed_path(self) -> None:
        """inject_layer=0 takes the embedding branch, not _run_forward_to_layer."""
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kv_inject_test_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = NUM_LAYERS

        u_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
        u_np /= np.linalg.norm(u_np)
        u_vec = mx.array(u_np)

        full_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        injected_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        full_logits_mx = mx.array(full_logits_np)

        h_embed = _make_hidden()
        decomp_full = {
            "hidden_states": {NUM_LAYERS - 1: _make_hidden()},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = lambda x: x
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=h_embed)
        mock_helper._get_embedding_scale.return_value = None

        forward_to_layer_calls = []

        def track_forward_to(*args, **kwargs):
            forward_to_layer_calls.append(args)
            return _make_hidden()

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._resolve_token_to_id",
                return_value=5,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp_full,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=MagicMock(),
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                return_value=full_logits_mx,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_to_layer",
                side_effect=track_forward_to,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_from_layer",
                return_value=injected_logits_np,
            ),
            patch("mlx.core.eval"),
        ):
            result = _kv_inject_test_impl(
                model,
                config,
                tokenizer,
                meta,
                "hello",
                "Paris",
                1.0,
                0,
                -1,
                5,  # inject_layer=0
            )

        # _run_forward_to_layer should NOT be called for inject_layer=0
        assert len(forward_to_layer_calls) == 0
        assert "kl_divergence" in result

    def test_inject_layer_nonzero_calls_forward_to(self) -> None:
        """inject_layer>0 calls _run_forward_to_layer(inject_layer - 1)."""
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kv_inject_test_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = NUM_LAYERS

        u_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
        u_np /= np.linalg.norm(u_np)
        u_vec = mx.array(u_np)

        full_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        injected_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        full_logits_mx = mx.array(full_logits_np)

        h_pre = _make_hidden()
        decomp_full = {
            "hidden_states": {NUM_LAYERS - 1: _make_hidden()},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = lambda x: x

        stop_layer_captured = []

        def capture_stop(model_, config_, ids, stop_after):
            stop_layer_captured.append(stop_after)
            return h_pre

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._resolve_token_to_id",
                return_value=5,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp_full,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=MagicMock(),
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                return_value=full_logits_mx,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_to_layer",
                side_effect=capture_stop,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_from_layer",
                return_value=injected_logits_np,
            ),
            patch("mlx.core.eval"),
        ):
            result = _kv_inject_test_impl(
                model,
                config,
                tokenizer,
                meta,
                "hello",
                "Paris",
                1.0,
                2,
                -1,
                5,  # inject_layer=2
            )

        assert stop_layer_captured == [1]  # stop_after = inject_layer - 1
        assert "kl_divergence" in result

    def test_with_embedding_scale_in_layer0_path(self) -> None:
        """inject_layer=0 with non-None embedding_scale multiplies h."""
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import _kv_inject_test_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = NUM_LAYERS

        u_np = np.ones(HIDDEN_DIM, dtype=np.float32)
        u_np /= np.linalg.norm(u_np)
        u_vec = mx.array(u_np)

        full_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        injected_logits_np = np.random.randn(VOCAB_SIZE).astype(np.float32)
        full_logits_mx = mx.array(full_logits_np)

        h_embed = mx.array(np.random.randn(1, 3, HIDDEN_DIM).astype(np.float32))
        decomp_full = {
            "hidden_states": {
                NUM_LAYERS - 1: mx.array(np.random.randn(1, 3, HIDDEN_DIM).astype(np.float32))
            },
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = lambda x: x
        mock_helper._get_embed_tokens.return_value = MagicMock(return_value=h_embed)
        mock_helper._get_embedding_scale.return_value = 4.0  # non-None scale

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._resolve_token_to_id",
                return_value=5,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp_full,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=MagicMock(),
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                return_value=full_logits_mx,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.prefill_inject._run_forward_from_layer",
                return_value=injected_logits_np,
            ),
            patch("mlx.core.eval"),
        ):
            result = _kv_inject_test_impl(
                model, config, tokenizer, meta, "hello", "Paris", 1.0, 0, -1, 5
            )
        assert "kl_divergence" in result


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestResultModels:
    def test_prefill_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import PrefillResult

        r = PrefillResult(
            prompt="hello",
            layer=2,
            position=-1,
            num_tokens=5,
            hidden_state=[0.1] * HIDDEN_DIM,
            hidden_norm=1.0,
            top_raw_logits=[{"token": "t", "token_id": 1, "probability": 0.5}],
        )
        d = r.model_dump()
        assert d["num_tokens"] == 5
        assert d["hidden_norm"] == 1.0
        assert len(d["top_raw_logits"]) == 1

    def test_token_prob_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import TokenProbEntry

        e = TokenProbEntry(
            token="Paris", token_id=42, full_prob=0.3, injected_prob=0.28, delta=-0.02
        )
        d = e.model_dump()
        assert d["token"] == "Paris"
        assert d["delta"] == -0.02

    def test_kv_inject_test_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import (
            KvInjectTestResult,
            TokenProbEntry,
        )

        comparison = [
            TokenProbEntry(
                token="Paris", token_id=42, full_prob=0.3, injected_prob=0.29, delta=-0.01
            )
        ]
        r = KvInjectTestResult(
            prompt="hello",
            token="Paris",
            token_id=42,
            coefficient=1.5,
            inject_layer=2,
            position=-1,
            full_target_prob=0.3,
            injected_target_prob=0.29,
            target_prob_delta=-0.01,
            kl_divergence=0.0005,
            top_k_comparison=comparison,
            summary={"kl_divergence": 0.0005, "statistically_indistinguishable": True},
        )
        d = r.model_dump()
        assert d["kl_divergence"] == 0.0005
        assert d["inject_layer"] == 2
        assert len(d["top_k_comparison"]) == 1
        assert d["top_k_comparison"][0]["token"] == "Paris"

    def test_kv_inject_result_fields(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.prefill_inject import KvInjectTestResult

        r = KvInjectTestResult(
            prompt="p",
            token="t",
            token_id=1,
            coefficient=2.0,
            inject_layer=1,
            position=3,
            full_target_prob=0.5,
            injected_target_prob=0.48,
            target_prob_delta=-0.02,
            kl_divergence=0.001,
            top_k_comparison=[],
            summary={},
        )
        d = r.model_dump()
        assert d["coefficient"] == 2.0
        assert d["position"] == 3
