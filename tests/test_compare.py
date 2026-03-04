"""Tests for _compare.py — comparison kernels."""

from unittest.mock import MagicMock

import numpy as np

import mlx.core as mx

from chuk_mcp_lazarus._compare import (
    _js_divergence,
    activation_divergence,
    attention_divergence,
    get_layer_weights,
    weight_divergence,
)


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
