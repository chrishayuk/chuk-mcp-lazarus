"""Tests for _compare.py — comparison kernels."""

from unittest.mock import MagicMock

import numpy as np

import mlx.core as mx

from chuk_mcp_lazarus._compare import (
    _compute_attention_weights,
    _get_hidden_states,
    _js_divergence,
    activation_divergence,
    attention_divergence,
    get_layer_weights,
    weight_divergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer(hidden_size: int = 64, num_heads: int = 4, head_dim: int = 16) -> MagicMock:
    """Build a mock layer with self_attn (q/k/v/o projections) and mlp."""
    layer = MagicMock()

    # Attention projection weights
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        getattr(layer.self_attn, proj).weight = mx.array(
            np.random.randn(hidden_size, hidden_size).astype(np.float32)
        )

    # MLP weights
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        getattr(layer.mlp, proj).weight = mx.array(
            np.random.randn(hidden_size, hidden_size).astype(np.float32)
        )

    return layer


def _make_model(
    num_layers: int = 4,
    hidden_size: int = 64,
    num_heads: int = 4,
    head_dim: int = 16,
) -> MagicMock:
    """Build a mock model with `model.model.layers`."""
    model = MagicMock()
    layers = [_make_layer(hidden_size, num_heads, head_dim) for _ in range(num_layers)]
    model.model.layers = layers
    return model


class _SimpleAttn:
    """Minimal attention mock with q_proj/k_proj callables.

    Uses a plain object (not MagicMock) so that ``hasattr`` returns False
    for attributes that are intentionally absent (q_norm, k_norm, rope).
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self._nh = num_heads
        self._hd = head_dim
        # scale used by the source code via getattr(attn, "scale", ...)
        self.scale = head_dim**-0.5

    def q_proj(self, h):
        batch, seq, _ = h.shape
        return mx.array(np.random.randn(batch, seq, self._nh * self._hd).astype(np.float32))

    def k_proj(self, h):
        batch, seq, _ = h.shape
        return mx.array(np.random.randn(batch, seq, self._nh * self._hd).astype(np.float32))


class _SimpleLayer:
    """Minimal layer mock with input_layernorm and self_attn."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.self_attn = _SimpleAttn(num_heads, head_dim)

    def input_layernorm(self, h):
        return h


def _make_attn_model(
    num_layers: int = 2,
    hidden_size: int = 64,
    num_heads: int = 4,
    head_dim: int = 16,
) -> MagicMock:
    """Build a mock model whose layers have callable q_proj / k_proj.

    _compute_attention_weights projects hidden states through q_proj / k_proj,
    so these must be callables that return arrays with the right shape.
    Uses plain objects (not MagicMock) for attn/layer so that ``hasattr``
    correctly reports False for absent attributes (q_norm, k_norm, rope).
    """
    model = MagicMock()
    layers = [_SimpleLayer(num_heads, head_dim) for _ in range(num_layers)]
    model.model.layers = layers
    return model


def _make_config(
    num_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 16,
    hidden_size: int = 64,
) -> MagicMock:
    """Build a mock config for attention divergence tests."""
    config = MagicMock()
    config.num_attention_heads = num_heads
    config.num_key_value_heads = num_kv_heads
    config.head_dim = head_dim
    config.hidden_size = hidden_size
    return config


def _make_tokenizer() -> MagicMock:
    """Mock tokenizer whose encode returns [1, 2, 3, 4, 5]."""
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
    tok.eos_token_id = 0
    return tok


# ---------------------------------------------------------------------------
# TestGetLayerWeights
# ---------------------------------------------------------------------------


class TestGetLayerWeights:
    def test_returns_dict(self) -> None:
        model = MagicMock()
        layer = MagicMock()
        layer.self_attn.q_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        layer.self_attn.k_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        layer.self_attn.v_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        layer.self_attn.o_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        layer.mlp.gate_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        layer.mlp.up_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        layer.mlp.down_proj.weight = mx.array(np.ones((4, 4), dtype=np.float32))
        model.model.layers = [layer]

        weights = get_layer_weights(model, 0)
        assert isinstance(weights, dict)
        assert "attn_q" in weights
        assert "attn_k" in weights
        assert "mlp_gate" in weights


# ---------------------------------------------------------------------------
# TestWeightDivergence
# ---------------------------------------------------------------------------


class TestWeightDivergence:
    def test_identical_models(self) -> None:
        model_a = MagicMock()
        model_b = MagicMock()

        w = mx.array(np.ones((4, 4), dtype=np.float32))

        for model in [model_a, model_b]:
            layer = MagicMock()
            layer.self_attn.q_proj.weight = w
            layer.self_attn.k_proj.weight = w
            layer.self_attn.v_proj.weight = w
            layer.self_attn.o_proj.weight = w
            layer.mlp.gate_proj.weight = w
            layer.mlp.up_proj.weight = w
            layer.mlp.down_proj.weight = w
            model.model.layers = [layer]

        results = weight_divergence(model_a, model_b, [0])
        assert len(results) > 0
        for r in results:
            assert r["frobenius_norm_diff"] < 1e-5
            assert r["cosine_similarity"] > 0.999

    def test_different_models(self) -> None:
        model_a = MagicMock()
        model_b = MagicMock()

        w_a = mx.array(np.ones((4, 4), dtype=np.float32))
        w_b = mx.array(np.ones((4, 4), dtype=np.float32) * 2.0)

        for model, w in [(model_a, w_a), (model_b, w_b)]:
            layer = MagicMock()
            layer.self_attn.q_proj.weight = w
            layer.self_attn.k_proj.weight = w
            layer.self_attn.v_proj.weight = w
            layer.self_attn.o_proj.weight = w
            layer.mlp.gate_proj.weight = w
            layer.mlp.up_proj.weight = w
            layer.mlp.down_proj.weight = w
            model.model.layers = [layer]

        results = weight_divergence(model_a, model_b, [0])
        assert len(results) > 0
        # Cosine sim should still be 1.0 (same direction, different magnitude)
        for r in results:
            assert r["cosine_similarity"] > 0.999

    def test_result_shape(self) -> None:
        model_a = MagicMock()
        model_b = MagicMock()

        w = mx.array(np.ones((4, 4), dtype=np.float32))

        for model in [model_a, model_b]:
            layer = MagicMock()
            layer.self_attn.q_proj.weight = w
            layer.self_attn.k_proj.weight = w
            layer.self_attn.v_proj.weight = w
            layer.self_attn.o_proj.weight = w
            layer.mlp.gate_proj.weight = w
            layer.mlp.up_proj.weight = w
            layer.mlp.down_proj.weight = w
            model.model.layers = [layer]

        results = weight_divergence(model_a, model_b, [0])
        for r in results:
            assert "layer" in r
            assert "component" in r
            assert "frobenius_norm_diff" in r
            assert "cosine_similarity" in r


# ---------------------------------------------------------------------------
# TestJsDivergence
# ---------------------------------------------------------------------------


class TestJsDivergence:
    def test_identical(self) -> None:
        p = mx.array([0.5, 0.3, 0.2])
        q = mx.array([0.5, 0.3, 0.2])
        js = _js_divergence(p, q)
        assert js < 1e-5

    def test_different(self) -> None:
        p = mx.array([0.9, 0.05, 0.05])
        q = mx.array([0.05, 0.05, 0.9])
        js = _js_divergence(p, q)
        assert js > 0.0

    def test_symmetric(self) -> None:
        p = mx.array([0.7, 0.2, 0.1])
        q = mx.array([0.1, 0.2, 0.7])
        js_pq = _js_divergence(p, q)
        js_qp = _js_divergence(q, p)
        assert abs(js_pq - js_qp) < 1e-5


# ---------------------------------------------------------------------------
# TestGetHiddenStates
# ---------------------------------------------------------------------------


class TestGetHiddenStates:
    """Test _get_hidden_states using the stubbed ModelHooks from conftest."""

    def test_returns_dict_of_arrays(self) -> None:
        model = _make_model()
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _get_hidden_states(model, config, tokenizer, "hello", layers=[0, 1])

        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        # Each value should be a 1-D array (token_position=-1 slices 3-D -> 1-D)
        for layer_idx, arr in result.items():
            assert arr.ndim == 1, f"Expected 1-D, got {arr.ndim}-D for layer {layer_idx}"

    def test_single_layer(self) -> None:
        model = _make_model()
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _get_hidden_states(model, config, tokenizer, "test", layers=[2])
        assert 2 in result
        assert len(result) == 1

    def test_token_position_zero(self) -> None:
        model = _make_model()
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _get_hidden_states(model, config, tokenizer, "test", layers=[0], token_position=0)
        assert 0 in result
        assert result[0].ndim == 1

    def test_all_layers(self) -> None:
        model = _make_model(num_layers=4)
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _get_hidden_states(model, config, tokenizer, "test", layers=[0, 1, 2, 3])
        assert len(result) == 4
        for layer_idx in [0, 1, 2, 3]:
            assert layer_idx in result


# ---------------------------------------------------------------------------
# TestActivationDivergence
# ---------------------------------------------------------------------------


class TestActivationDivergence:
    """Test activation_divergence using stubbed ModelHooks."""

    def test_returns_list_of_dicts(self) -> None:
        model_a = _make_model()
        model_b = _make_model()
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = activation_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompts=["hello world"],
            layers=[0, 1],
        )

        assert isinstance(results, list)
        assert len(results) == 2  # 1 prompt x 2 layers

        for r in results:
            assert "layer" in r
            assert "prompt" in r
            assert "cosine_similarity" in r
            assert "l2_distance" in r
            assert "relative_l2" in r

    def test_multiple_prompts(self) -> None:
        model_a = _make_model()
        model_b = _make_model()
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = activation_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompts=["prompt one", "prompt two"],
            layers=[0],
        )

        # 2 prompts x 1 layer = 2 results
        assert len(results) == 2
        prompts_in_results = {r["prompt"] for r in results}
        assert "prompt one" in prompts_in_results
        assert "prompt two" in prompts_in_results

    def test_cosine_similarity_range(self) -> None:
        model_a = _make_model()
        model_b = _make_model()
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = activation_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompts=["test"],
            layers=[0],
        )

        for r in results:
            assert -1.0 <= r["cosine_similarity"] <= 1.0

    def test_l2_distance_non_negative(self) -> None:
        model_a = _make_model()
        model_b = _make_model()
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = activation_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompts=["test"],
            layers=[0, 1],
        )

        for r in results:
            assert r["l2_distance"] >= 0.0
            assert r["relative_l2"] >= 0.0


# ---------------------------------------------------------------------------
# TestComputeAttentionWeights
# ---------------------------------------------------------------------------


class TestComputeAttentionWeights:
    """Test _compute_attention_weights using mock models with callable projections."""

    def test_returns_dict_of_arrays(self) -> None:
        model = _make_attn_model(num_layers=2)
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _compute_attention_weights(model, config, tokenizer, "hello", layers=[0, 1])

        assert isinstance(result, dict)
        # Should get attention weights for both layers
        for layer_idx in result:
            w = result[layer_idx]
            # Expected shape: [batch, heads, seq, seq]
            assert w.ndim == 4, f"Expected 4-D, got {w.ndim}-D for layer {layer_idx}"
            # batch=1, heads=4, seq=5
            assert w.shape[0] == 1
            assert w.shape[1] == 4  # num_heads
            assert w.shape[2] == w.shape[3]  # seq x seq

    def test_single_layer(self) -> None:
        model = _make_attn_model(num_layers=2)
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _compute_attention_weights(model, config, tokenizer, "hello", layers=[0])

        assert 0 in result

    def test_attention_weights_are_probabilities(self) -> None:
        """Attention weights should sum to ~1 along the last axis (softmax output)."""
        model = _make_attn_model(num_layers=1)
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _compute_attention_weights(model, config, tokenizer, "hello", layers=[0])

        if 0 in result:
            w = result[0]
            # Sum along the last axis should be approximately 1
            sums = mx.sum(w, axis=-1)
            mx.eval(sums)
            for val in sums.tolist()[0][0]:  # batch=0, head=0
                assert abs(val - 1.0) < 1e-4, f"Softmax sum {val} != 1.0"

    def test_layer_zero_uses_self_hidden_state(self) -> None:
        """For layer 0, the code falls back to using layer 0's own hidden state."""
        model = _make_attn_model(num_layers=1)
        config = _make_config()
        tokenizer = _make_tokenizer()

        result = _compute_attention_weights(model, config, tokenizer, "hello", layers=[0])

        # Should still produce output even for layer 0
        assert 0 in result

    def test_handles_exception_in_layer(self) -> None:
        """If a layer's projection fails, that layer is skipped (logged warning)."""
        model = _make_attn_model(num_layers=2)
        config = _make_config()
        tokenizer = _make_tokenizer()

        # Break one layer's q_proj to raise an exception
        def _bad_proj(h):
            raise RuntimeError("broken projection")

        model.model.layers[1].self_attn.q_proj = _bad_proj  # type: ignore[attr-defined]

        result = _compute_attention_weights(model, config, tokenizer, "hello", layers=[0, 1])

        # Layer 0 should still be present; layer 1 should be skipped
        assert 0 in result
        assert 1 not in result


# ---------------------------------------------------------------------------
# TestAttentionDivergence
# ---------------------------------------------------------------------------


class TestAttentionDivergence:
    """Test attention_divergence end-to-end with mock models."""

    def test_returns_list_of_dicts(self) -> None:
        model_a = _make_attn_model(num_layers=2)
        model_b = _make_attn_model(num_layers=2)
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = attention_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompt="hello",
            layers=[0, 1],
        )

        assert isinstance(results, list)
        # 2 layers x 4 heads = 8 results
        assert len(results) == 8

        for r in results:
            assert "layer" in r
            assert "head" in r
            assert "js_divergence" in r
            assert "cosine_similarity" in r

    def test_js_divergence_non_negative(self) -> None:
        model_a = _make_attn_model(num_layers=1)
        model_b = _make_attn_model(num_layers=1)
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = attention_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompt="test",
            layers=[0],
        )

        for r in results:
            assert r["js_divergence"] >= 0.0

    def test_cosine_similarity_range(self) -> None:
        model_a = _make_attn_model(num_layers=1)
        model_b = _make_attn_model(num_layers=1)
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = attention_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompt="test",
            layers=[0],
        )

        for r in results:
            assert -1.0 <= r["cosine_similarity"] <= 1.0

    def test_single_layer(self) -> None:
        model_a = _make_attn_model(num_layers=2)
        model_b = _make_attn_model(num_layers=2)
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        results = attention_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompt="hello",
            layers=[1],
        )

        # 1 layer x 4 heads = 4 results
        assert len(results) == 4
        for r in results:
            assert r["layer"] == 1

    def test_missing_layer_skipped(self) -> None:
        """If one model fails to compute attention for a layer, that layer is skipped."""
        model_a = _make_attn_model(num_layers=2)
        model_b = _make_attn_model(num_layers=2)
        config_a = _make_config()
        config_b = _make_config()
        tokenizer = _make_tokenizer()

        # Break model_b's layer 0 projection
        def _bad_proj(h):
            raise RuntimeError("broken")

        model_b.model.layers[0].self_attn.q_proj = _bad_proj  # type: ignore[attr-defined]

        results = attention_divergence(
            model_a,
            config_a,
            model_b,
            config_b,
            tokenizer,
            prompt="hello",
            layers=[0, 1],
        )

        # Layer 0 missing from model_b -> skipped; layer 1 should have 4 heads
        layers_in_results = {r["layer"] for r in results}
        assert 0 not in layers_in_results
        assert 1 in layers_in_results
