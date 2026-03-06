"""Tests for weight_geometry tool.

Tests cover:
  - weight_geometry: async validation (6)
  - _weight_geometry_impl: output keys, heads, neurons, PCA, summary (14)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock weight geometry
# ---------------------------------------------------------------------------

HIDDEN_DIM = 8
NUM_HEADS = 2
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 4
INTERMEDIATE_SIZE = 16
VOCAB_SIZE = 20

RNG = np.random.default_rng(77)


def _make_mock_layer() -> MagicMock:
    """Build a mock layer with o_proj and down_proj weights."""
    import mlx.core as mx

    layer = MagicMock()

    # o_proj.weight: [hidden_dim, num_heads * head_dim] = [8, 8]
    o_weight = RNG.standard_normal((HIDDEN_DIM, NUM_HEADS * HEAD_DIM)).astype(np.float32)
    layer.self_attn.o_proj.weight = mx.array(o_weight.tolist())

    # down_proj.weight: [hidden_dim, intermediate_size] = [8, 16]
    down_weight = RNG.standard_normal((HIDDEN_DIM, INTERMEDIATE_SIZE)).astype(np.float32)
    layer.mlp.down_proj.weight = mx.array(down_weight.tolist())

    return layer


def _fake_lm_head(x: Any) -> Any:
    """Mock lm_head: [1, N, hidden_dim] -> [1, N, vocab_size]."""
    import mlx.core as mx

    d = x._data if hasattr(x, "_data") else np.array(x)
    batch, n, hdim = d.shape
    rng = np.random.default_rng(42)
    W = rng.standard_normal((VOCAB_SIZE, hdim)).astype(np.float32)
    out = d.reshape(n, hdim) @ W.T  # [N, vocab]
    return mx.array(out.reshape(1, n, VOCAB_SIZE).tolist())


# ---------------------------------------------------------------------------
# weight_geometry — async validation
# ---------------------------------------------------------------------------


class TestWeightGeometry:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry

        result = await weight_geometry(0)
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry

        result = await weight_geometry(99)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry

        result = await weight_geometry(-1)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_top_k_neurons_too_high(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry

        result = await weight_geometry(0, top_k_neurons=501)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_top_k_neurons_too_low(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry

        result = await weight_geometry(0, top_k_neurons=0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_top_k_vocab_too_high(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry

        result = await weight_geometry(0, top_k_vocab=21)
        assert result["error_type"] == "InvalidInput"


# ---------------------------------------------------------------------------
# _weight_geometry_impl tests
# ---------------------------------------------------------------------------


def _run(
    layer: int = 1,
    top_k_neurons: int = 5,
    top_k_vocab: int = 3,
    include_pca: bool = True,
) -> dict:
    """Run _weight_geometry_impl with mock weights."""
    from chuk_mcp_lazarus.tools.geometry.weight_geometry import _weight_geometry_impl

    mock_layer = _make_mock_layer()

    meta = MagicMock()
    meta.num_layers = 4
    meta.hidden_dim = HIDDEN_DIM
    meta.num_attention_heads = NUM_HEADS
    meta.intermediate_size = INTERMEDIATE_SIZE

    tok = MagicMock()
    tok.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

    # ModelHooks._get_layers() returns list with our mock layer
    model_layers = [MagicMock() for _ in range(4)]
    model_layers[layer] = mock_layer

    with (
        patch(
            "chuk_lazarus.introspection.hooks.ModelHooks._get_layers",
            return_value=model_layers,
        ),
        patch(
            "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
            return_value=_fake_lm_head,
        ),
    ):
        return _weight_geometry_impl(
            MagicMock(),
            MagicMock(),
            meta,
            tok,
            layer,
            top_k_neurons,
            top_k_vocab,
            include_pca,
        )


class TestWeightGeometryImpl:
    def test_output_keys(self) -> None:
        result = _run()
        for key in [
            "layer",
            "num_heads",
            "num_neurons_analyzed",
            "intermediate_size",
            "hidden_dim",
            "head_directions",
            "neuron_directions",
            "summary",
        ]:
            assert key in result

    def test_head_count(self) -> None:
        result = _run()
        assert len(result["head_directions"]) == NUM_HEADS

    def test_neuron_count(self) -> None:
        result = _run(top_k_neurons=5)
        assert result["num_neurons_analyzed"] == 5

    def test_neuron_count_capped(self) -> None:
        result = _run(top_k_neurons=500)
        assert result["num_neurons_analyzed"] == INTERMEDIATE_SIZE

    def test_head_norms_positive(self) -> None:
        result = _run()
        for hd in result["head_directions"]:
            assert hd["norm"] > 0

    def test_neuron_norms_descending(self) -> None:
        result = _run(top_k_neurons=10)
        norms = [nd["norm"] for nd in result["neuron_directions"]]
        for i in range(1, len(norms)):
            assert norms[i] <= norms[i - 1] + 1e-6

    def test_top_tokens_per_head(self) -> None:
        result = _run(top_k_vocab=3)
        for hd in result["head_directions"]:
            assert len(hd["top_tokens"]) == 3
            assert hd["top_token"] != ""

    def test_top_tokens_per_neuron(self) -> None:
        result = _run(top_k_vocab=3)
        for nd in result["neuron_directions"]:
            assert len(nd["top_tokens"]) == 3
            assert nd["top_token"] != ""

    def test_pca_effective_rank_ordering(self) -> None:
        result = _run(include_pca=True)
        ss = result["supply_subspace"]
        assert ss is not None
        assert ss["effective_rank_50"] <= ss["effective_rank_80"]
        assert ss["effective_rank_80"] <= ss["effective_rank_95"]

    def test_no_pca_when_disabled(self) -> None:
        result = _run(include_pca=False)
        assert result["supply_subspace"] is None

    def test_summary_fields(self) -> None:
        result = _run()
        s = result["summary"]
        assert "strongest_head" in s
        assert "strongest_neuron" in s
        assert "supply_effective_rank_80" in s

    def test_hidden_dim_in_result(self) -> None:
        result = _run()
        assert result["hidden_dim"] == HIDDEN_DIM

    def test_layer_in_result(self) -> None:
        result = _run(layer=2)
        assert result["layer"] == 2

    def test_pca_singular_values_normalised(self) -> None:
        result = _run(include_pca=True)
        svs = result["supply_subspace"]["top_singular_values"]
        assert svs[0] == 1.0  # first is always 1.0 after normalisation
        for sv in svs:
            assert 0.0 <= sv <= 1.0 + 1e-6
