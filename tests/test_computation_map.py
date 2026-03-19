"""Tests for computation_map tool.

Tests cover:
  - computation_map: async validation (7) + impl (16) + helpers (3)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Ensure MLX stubs exist for non-Apple platforms
# ---------------------------------------------------------------------------


def _ensure_mlx_stubs() -> None:
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


_ensure_mlx_stubs()


# ---------------------------------------------------------------------------
# computation_map — async validation
# ---------------------------------------------------------------------------


class TestComputationMap:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        result = await computation_map(prompt="test", candidates=["a"])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_empty_candidates(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        result = await computation_map(prompt="test", candidates=[])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_candidates(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        result = await computation_map(prompt="test", candidates=["t"] * 11)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        result = await computation_map(prompt="test", candidates=["a"], layers=[99])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        result = await computation_map(prompt="test", candidates=["a"], layers=[-1])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        fake = {
            "prompt": "t",
            "token_position": -1,
            "token_text": "x",
            "hidden_dim": 64,
            "candidates": [],
            "candidate_pairwise_angles": [],
            "layers": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.computation_map._computation_map_impl",
            return_value=fake,
        ):
            result = await computation_map(prompt="test", candidates=["a"])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        with patch(
            "chuk_mcp_lazarus.tools.geometry.computation_map._computation_map_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await computation_map(prompt="test", candidates=["a"])
        assert result["error_type"] == "GeometryFailed"

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map

        fake = {
            "prompt": "t",
            "token_position": -1,
            "token_text": "x",
            "hidden_dim": 64,
            "candidates": [],
            "candidate_pairwise_angles": [],
            "layers": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.computation_map._computation_map_impl",
            return_value=fake,
        ):
            result = await computation_map(prompt="test", candidates=["a"], layers=None)
        assert "error" not in result


# ---------------------------------------------------------------------------
# computation_map — impl tests
# ---------------------------------------------------------------------------


def _build_mock_mx_array(data: list[float]) -> MagicMock:
    """Build a mock that behaves like mx.array for tolist()."""
    arr = MagicMock()
    arr.tolist.return_value = data
    arr.ndim = 1
    arr.shape = (len(data),)
    return arr


def _build_mock_hidden(dim: int, values: list[float] | None = None) -> MagicMock:
    """Build a mock hidden state [1, seq, dim]."""
    if values is None:
        values = [float(i) / dim for i in range(dim)]
    arr = MagicMock()
    arr.ndim = 3
    arr.shape = (1, 3, dim)
    # _extract_position returns arr[0, pos, :]
    inner = MagicMock()
    inner.tolist.return_value = values
    inner.ndim = 1
    inner.shape = (dim,)
    inner.reshape.return_value = MagicMock()
    arr.__getitem__ = MagicMock(return_value=inner)
    return arr


class TestComputationMapImpl:
    """Sync impl tests with mocked model components."""

    def _run(
        self,
        candidates: list[str] | None = None,
        layers: list[int] | None = None,
        top_k_heads: int = 2,
        top_k_neurons: int = 3,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.computation_map import (
            _computation_map_impl,
        )

        if candidates is None:
            candidates = ["alpha", "beta"]
        if layers is None:
            layers = [0, 1]

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = 8
        meta.num_layers = 4
        meta.num_attention_heads = 4
        meta.num_kv_heads = 4
        meta.head_dim = 2
        meta.intermediate_size = 16

        DIM = 8

        # Unembed vectors
        unembed_map_np = {
            "alpha": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        unembed_map_mx = {10: MagicMock(), 20: MagicMock()}

        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return {"alpha": 10, "beta": 20}.get(tok, None)

        def fake_unembed_np(model_obj: Any, tid: int) -> np.ndarray | None:
            return {10: unembed_map_np["alpha"], 20: unembed_map_np["beta"]}.get(tid)

        def fake_unembed_mx(model_obj: Any, tid: int) -> Any:
            return unembed_map_mx.get(tid)

        # Build hidden states that look reasonable
        h0 = np.array([3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        h1 = np.array([2.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Mock _run_decomposition_forward return
        mock_hidden_0 = MagicMock()
        mock_hidden_1 = MagicMock()
        mock_prev_0 = MagicMock()
        mock_prev_1 = MagicMock()
        mock_attn_0 = MagicMock()
        mock_attn_1 = MagicMock()
        mock_ffn_0 = MagicMock()
        mock_ffn_1 = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.dtype = "float32"

        decomp = {
            "embeddings": mock_embeddings,
            "hidden_states": {0: mock_hidden_0, 1: mock_hidden_1},
            "prev_hidden": {0: mock_prev_0, 1: mock_prev_1},
            "attn_outputs": {0: mock_attn_0, 1: mock_attn_1},
            "ffn_outputs": {0: mock_ffn_0, 1: mock_ffn_1},
        }

        # Mock _extract_position: returns a mock with .tolist() and .item()
        extract_calls = [0]

        def fake_extract_position(tensor: Any, position: int) -> MagicMock:
            vec = MagicMock()
            data = h0.tolist() if extract_calls[0] % 6 < 3 else h1.tolist()
            extract_calls[0] += 1
            vec.tolist.return_value = data
            vec.reshape.return_value = vec
            vec.ndim = 1
            vec.shape = (DIM,)
            vec.__add__ = lambda s, o: vec
            vec.__radd__ = lambda s, o: vec
            return vec

        # Mock _norm_project: returns a mock logits array
        def fake_norm_project(final_norm: Any, lm_head: Any, vec: Any) -> MagicMock:
            logits = MagicMock()
            # Make item() return different values for different tokens
            logit_vals = {10: 5.0, 20: 3.0}

            def getitem(self_mock: Any, idx: int) -> MagicMock:
                v = MagicMock()
                v.item.return_value = logit_vals.get(idx, 0.0)
                return v

            logits.__getitem__ = getitem
            return logits

        # Mock ModelHooks
        mock_layer_obj = MagicMock()
        mock_layers_list = [mock_layer_obj] * 4

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers_list
        mock_helper._get_final_norm.return_value = MagicMock()

        mock_lm_head = MagicMock()

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.computation_map._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.computation_map._get_unembed_vec_np",
                side_effect=fake_unembed_np,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                side_effect=fake_unembed_mx,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._extract_position",
                side_effect=fake_extract_position,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._norm_project",
                side_effect=fake_norm_project,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._has_four_norms",
                return_value=False,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.computation_map._compute_top_heads",
                return_value=[],
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.computation_map._compute_top_neurons",
                return_value=[],
            ),
        ):
            return _computation_map_impl(
                model,
                config,
                tokenizer,
                meta,
                "The capital of Australia is",
                candidates,
                layers,
                top_k_heads,
                top_k_neurons,
                -1,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "prompt" in r
        assert "candidates" in r
        assert "candidate_pairwise_angles" in r
        assert "layers" in r
        assert "summary" in r
        assert "token_position" in r
        assert "token_text" in r
        assert "hidden_dim" in r

    def test_layer_count(self) -> None:
        r = self._run(layers=[0, 1])
        assert len(r["layers"]) == 2

    def test_candidate_count(self) -> None:
        r = self._run()
        assert len(r["candidates"]) == 2

    def test_pairwise_angle_count(self) -> None:
        r = self._run()
        # 2 candidates = 1 pair
        assert len(r["candidate_pairwise_angles"]) == 1

    def test_layer_geometry_keys(self) -> None:
        r = self._run()
        geo = r["layers"][0]["geometry"]
        assert "residual_norm" in geo
        assert "angles" in geo
        assert "orthogonal_fraction" in geo
        assert geo["residual_norm"] > 0

    def test_layer_attribution_keys(self) -> None:
        r = self._run()
        attr = r["layers"][0]["attribution"]
        assert "attention_logit" in attr
        assert "ffn_logit" in attr
        assert "per_candidate" in attr
        assert len(attr["per_candidate"]) == 2

    def test_layer_race_state(self) -> None:
        r = self._run()
        race = r["layers"][0]["race"]
        assert len(race) == 2
        for entry in race:
            assert "token" in entry
            assert "logit" in entry
            assert "probability" in entry
            assert "rank" in entry

    def test_top_heads_present(self) -> None:
        r = self._run()
        tc = r["layers"][0]["top_components"]
        assert "top_heads" in tc

    def test_top_neurons_present(self) -> None:
        r = self._run()
        tc = r["layers"][0]["top_components"]
        assert "top_neurons" in tc

    def test_rotation_first_layer(self) -> None:
        r = self._run()
        geo = r["layers"][0]["geometry"]
        assert geo["rotation_from_previous"] is None

    def test_rotation_subsequent(self) -> None:
        r = self._run()
        geo = r["layers"][1]["geometry"]
        assert geo["rotation_from_previous"] is not None
        assert geo["rotation_from_previous"] >= 0

    def test_probabilities_valid(self) -> None:
        r = self._run()
        for layer_data in r["layers"]:
            for race_entry in layer_data["race"]:
                assert 0.0 <= race_entry["probability"] <= 1.0

    def test_ranks_unique(self) -> None:
        r = self._run()
        for layer_data in r["layers"]:
            ranks = [e["rank"] for e in layer_data["race"]]
            assert len(ranks) == len(set(ranks))

    def test_summary_fields(self) -> None:
        r = self._run()
        s = r["summary"]
        assert "final_prediction" in s
        assert "final_probability" in s
        assert "total_rotation" in s
        assert "biggest_rotation_layer" in s
        assert "crossing_events" in s
        assert "primary_target" in s
        assert "primary_target_id" in s

    def test_single_candidate(self) -> None:
        r = self._run(candidates=["alpha"])
        assert len(r["candidates"]) == 1
        assert len(r["candidate_pairwise_angles"]) == 0

    def test_pairwise_angle_values(self) -> None:
        r = self._run()
        pair = r["candidate_pairwise_angles"][0]
        assert "token_a" in pair
        assert "token_b" in pair
        assert "angle" in pair
        assert "cosine_similarity" in pair
        # alpha=[1,0,...] and beta=[0,1,...] should be 90 degrees
        assert abs(pair["angle"] - 90.0) < 1.0


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestComputeTopHeads:
    """Test _compute_top_heads returns correct structure."""

    def test_returns_list(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import TopHead

        # Just verify TopHead model creates correctly
        h = TopHead(head=0, logit_contribution=1.5, top_token="hello")
        assert h.head == 0
        assert h.logit_contribution == 1.5
        assert h.top_token == "hello"


class TestComputeTopNeurons:
    """Test _compute_top_neurons returns correct structure."""

    def test_returns_list(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import TopNeuron

        n = TopNeuron(neuron_index=42, activation=0.5, logit_contribution=2.3, top_token="world")
        assert n.neuron_index == 42
        assert n.activation == 0.5
        assert n.logit_contribution == 2.3
        assert n.top_token == "world"


class TestResultModels:
    """Test Pydantic result model construction."""

    def test_crossing_event(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import CrossingEvent

        ce = CrossingEvent(layer=5, from_token="Sydney", to_token="Canberra")
        d = ce.model_dump()
        assert d["layer"] == 5
        assert d["from_token"] == "Sydney"
        assert d["to_token"] == "Canberra"

    def test_candidate_info(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import CandidateInfo

        ci = CandidateInfo(token="test", token_id=42, unembed_norm=1.5)
        d = ci.model_dump()
        assert d["token"] == "test"
        assert d["token_id"] == 42

    def test_layer_geometry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import (
            CandidateAngle,
            LayerGeometry,
        )

        geo = LayerGeometry(
            residual_norm=3.14,
            angles=[CandidateAngle(token="a", angle=45.0, projection=2.0, fraction=0.5)],
            orthogonal_fraction=0.3,
            rotation_from_previous=None,
        )
        d = geo.model_dump()
        assert d["residual_norm"] == 3.14
        assert d["rotation_from_previous"] is None
        assert len(d["angles"]) == 1

    def test_computation_map_summary(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import (
            ComputationMapSummary,
            CrossingEvent,
        )

        s = ComputationMapSummary(
            final_prediction="Canberra",
            final_probability=0.85,
            total_rotation=120.5,
            biggest_rotation_layer=15,
            crossing_events=[
                CrossingEvent(layer=10, from_token="Sydney", to_token="Canberra"),
            ],
            primary_target="Canberra",
            primary_target_id=42,
        )
        d = s.model_dump()
        assert d["final_prediction"] == "Canberra"
        assert len(d["crossing_events"]) == 1


# ---------------------------------------------------------------------------
# Stub for chuk_mcp_lazarus.tools._serialize (computation_map uses `.._serialize`
# from tools/geometry/, which resolves to tools._serialize — not the top-level one)
# ---------------------------------------------------------------------------


def _install_tools_serialize_stub() -> None:
    import sys
    import types

    mod_name = "chuk_mcp_lazarus.tools._serialize"
    if mod_name in sys.modules:
        return

    stub = types.ModuleType(mod_name)

    def to_pylist(arr):
        return arr.tolist()

    stub.to_pylist = to_pylist  # type: ignore[attr-defined]
    sys.modules[mod_name] = stub


_install_tools_serialize_stub()


def _ensure_mx_stack() -> None:
    """Ensure mx.stack is available in the stub."""
    import mlx.core as mx

    if not hasattr(mx, "stack"):

        def _stack(arrays, axis=0):
            arrs = [a._data if hasattr(a, "_data") else np.array(a) for a in arrays]
            return mx.array(np.stack(arrs, axis=axis))

        mx.stack = _stack  # type: ignore[attr-defined]


_ensure_mx_stack()


def _ensure_nn_activations() -> None:
    """Ensure mlx.nn has silu and gelu for the neurons path."""
    import mlx.nn as nn_mod

    if not hasattr(nn_mod, "silu"):
        nn_mod.silu = lambda x: x  # type: ignore[attr-defined]
    if not hasattr(nn_mod, "gelu"):
        nn_mod.gelu = lambda x: x  # type: ignore[attr-defined]


_ensure_nn_activations()


# ---------------------------------------------------------------------------
# TestComputeTopHeads — internal helper function
# ---------------------------------------------------------------------------


def _make_mock_layer_for_heads(
    num_heads: int = 2,
    head_dim: int = 4,
    hidden_dim: int = 8,
    num_kv_heads: int | None = None,
    with_rope: bool = False,
    with_norms: bool = False,
) -> MagicMock:
    """Build a mock transformer layer suitable for _compute_top_heads."""
    if num_kv_heads is None:
        num_kv_heads = num_heads

    layer = MagicMock()
    normed_out = mx.array(np.ones((1, 3, hidden_dim), dtype=np.float32))
    layer.input_layernorm.return_value = normed_out

    attn = MagicMock()
    attn.q_proj.return_value = mx.array(np.ones((1, 3, num_heads * head_dim), dtype=np.float32))
    attn.k_proj.return_value = mx.array(np.ones((1, 3, num_kv_heads * head_dim), dtype=np.float32))
    attn.v_proj.return_value = mx.array(np.ones((1, 3, num_kv_heads * head_dim), dtype=np.float32))
    attn.o_proj.weight = mx.array(
        np.random.randn(hidden_dim, num_heads * head_dim).astype(np.float32)
    )
    attn.scale = head_dim**-0.5

    if with_rope:
        attn.rope = MagicMock(side_effect=lambda x: x)
    else:
        attn.rope = None

    if with_norms:
        attn.q_norm = MagicMock(side_effect=lambda x: x)
        attn.k_norm = MagicMock(side_effect=lambda x: x)
    else:
        attn.q_norm = None
        attn.k_norm = None

    layer.self_attn = attn
    return layer


class TestComputeTopHeadsFunction:
    """Tests for _compute_top_heads internal helper."""

    NUM_HEADS = 2
    HEAD_DIM = 4
    HIDDEN_DIM = 8
    SEQ_LEN = 3

    def _run(
        self,
        top_k: int = 2,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        hidden_dim: int = HIDDEN_DIM,
        with_rope: bool = False,
        with_norms: bool = False,
        num_kv_heads: int | None = None,
    ) -> list:
        from chuk_mcp_lazarus.tools.geometry.computation_map import _compute_top_heads

        if num_kv_heads is None:
            num_kv_heads = num_heads

        layer = _make_mock_layer_for_heads(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            num_kv_heads=num_kv_heads,
            with_rope=with_rope,
            with_norms=with_norms,
        )

        prev_h = mx.array(np.ones((1, self.SEQ_LEN, hidden_dim), dtype=np.float32))
        mask = mx.array(np.zeros((self.SEQ_LEN, self.SEQ_LEN), dtype=np.float32))
        unembed_mx = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        # lm_head returns logits [1, top_k, vocab]
        VOCAB = 10
        mock_lm_head = MagicMock()
        mock_lm_head.return_value = mx.array(np.random.randn(1, top_k, VOCAB).astype(np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        with patch(
            "mlx.core.fast.scaled_dot_product_attention",
            return_value=mx.array(
                np.ones((1, num_heads, self.SEQ_LEN, head_dim), dtype=np.float32)
            ),
        ):
            return _compute_top_heads(
                layer,
                prev_h,
                pos=self.SEQ_LEN - 1,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mask=mask,
                unembed_mx=unembed_mx,
                lm_head=mock_lm_head,
                tokenizer=tokenizer,
                top_k=top_k,
            )

    def test_returns_list(self) -> None:
        result = self._run()
        assert isinstance(result, list)

    def test_length_equals_top_k(self) -> None:
        result = self._run(top_k=2)
        assert len(result) == 2

    def test_top_k_clamped_to_num_heads(self) -> None:
        """top_k larger than num_heads should return at most num_heads entries."""
        result = self._run(top_k=10, num_heads=2)
        assert len(result) <= 2

    def test_result_has_tophead_fields(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import TopHead

        result = self._run(top_k=1)
        assert len(result) == 1
        h = result[0]
        assert isinstance(h, TopHead)
        assert isinstance(h.head, int)
        assert isinstance(h.logit_contribution, float)
        assert isinstance(h.top_token, str)

    def test_head_index_in_range(self) -> None:
        result = self._run(top_k=2, num_heads=2)
        for h in result:
            assert 0 <= h.head < 2

    def test_with_rope(self) -> None:
        """Rope path should apply rope and return valid result."""
        result = self._run(with_rope=True, top_k=1)
        assert len(result) == 1

    def test_with_qk_norms(self) -> None:
        """q_norm / k_norm path should apply norms and return valid result."""
        result = self._run(with_norms=True, top_k=1)
        assert len(result) == 1

    def test_grouped_query_attention(self) -> None:
        """GQA (num_kv_heads < num_heads) should repeat k/v and return results."""
        result = self._run(
            num_heads=4,
            head_dim=4,
            hidden_dim=16,
            num_kv_heads=2,
            top_k=2,
        )
        assert len(result) == 2

    def test_zero_top_k_returns_empty(self) -> None:
        """top_k=0 should return empty list."""
        result = self._run(top_k=0)
        assert result == []

    def test_lm_head_called_with_stacked_vecs(self) -> None:
        """lm_head should be called once with the stacked head vectors."""
        from chuk_mcp_lazarus.tools.geometry.computation_map import _compute_top_heads

        num_heads = 2
        head_dim = 4
        hidden_dim = 8
        top_k = 2

        layer = _make_mock_layer_for_heads(
            num_heads=num_heads, head_dim=head_dim, hidden_dim=hidden_dim
        )
        prev_h = mx.array(np.ones((1, 3, hidden_dim), dtype=np.float32))
        mask = mx.array(np.zeros((3, 3), dtype=np.float32))
        unembed_mx = mx.array(np.ones(hidden_dim, dtype=np.float32))

        mock_lm_head = MagicMock()
        mock_lm_head.return_value = mx.array(np.random.randn(1, top_k, 10).astype(np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: f"t{ids[0]}"

        with patch(
            "mlx.core.fast.scaled_dot_product_attention",
            return_value=mx.array(np.ones((1, num_heads, 3, head_dim), dtype=np.float32)),
        ):
            _compute_top_heads(
                layer,
                prev_h,
                pos=2,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                mask=mask,
                unembed_mx=unembed_mx,
                lm_head=mock_lm_head,
                tokenizer=tokenizer,
                top_k=top_k,
            )

        mock_lm_head.assert_called_once()


# ---------------------------------------------------------------------------
# TestComputeTopNeuronsFunction — internal helper function
# ---------------------------------------------------------------------------


def _make_mock_layer_for_neurons(
    hidden_dim: int = 8,
    intermediate_size: int = 16,
    with_gate: bool = True,
    with_act_fn: bool = False,
    four_norms: bool = False,
) -> MagicMock:
    """Build a mock transformer layer suitable for _compute_top_neurons."""
    layer = MagicMock()

    ffn_input = mx.array(np.ones((1, 3, hidden_dim), dtype=np.float32))
    layer.post_attention_layernorm.return_value = ffn_input
    layer.pre_feedforward_layernorm.return_value = ffn_input

    mlp = MagicMock()
    gate_out = mx.array(np.ones((1, 3, intermediate_size), dtype=np.float32) * 0.5)
    up_out = mx.array(np.ones((1, 3, intermediate_size), dtype=np.float32) * 0.3)

    if with_gate:
        mlp.gate_proj.return_value = gate_out
        mlp.up_proj.return_value = up_out
    else:
        mlp.up_proj.return_value = up_out
        del mlp.gate_proj

    mlp.down_proj.weight = mx.array(
        np.random.randn(hidden_dim, intermediate_size).astype(np.float32)
    )

    if with_act_fn:
        mlp.act = MagicMock(side_effect=lambda x: x)
    else:
        mlp.act = None
        mlp.activation = None

    layer.mlp = mlp
    return layer


class TestComputeTopNeuronsFunction:
    """Tests for _compute_top_neurons internal helper."""

    HIDDEN_DIM = 8
    INTERMEDIATE = 16
    SEQ_LEN = 3

    def _run(
        self,
        top_k: int = 3,
        with_gate: bool = True,
        with_act_fn: bool = False,
        four_norms: bool = False,
        hidden_dim: int = HIDDEN_DIM,
        intermediate_size: int = INTERMEDIATE,
    ) -> list:
        from chuk_mcp_lazarus.tools.geometry.computation_map import _compute_top_neurons

        layer = _make_mock_layer_for_neurons(
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
            with_gate=with_gate,
            with_act_fn=with_act_fn,
            four_norms=four_norms,
        )

        prev_h = mx.array(np.ones((1, self.SEQ_LEN, hidden_dim), dtype=np.float32))
        attn_out = mx.array(np.ones((1, self.SEQ_LEN, hidden_dim), dtype=np.float32) * 0.1)
        pos = self.SEQ_LEN - 1
        unembed_mx = mx.array(np.random.randn(hidden_dim).astype(np.float32))

        VOCAB = 10
        mock_lm_head = MagicMock()
        mock_lm_head.return_value = mx.array(np.random.randn(1, top_k, VOCAB).astype(np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        return _compute_top_neurons(
            layer,
            prev_h,
            attn_out,
            pos=pos,
            unembed_mx=unembed_mx,
            lm_head=mock_lm_head,
            tokenizer=tokenizer,
            top_k=top_k,
            four_norms=four_norms,
        )

    def test_returns_list(self) -> None:
        result = self._run()
        assert isinstance(result, list)

    def test_length_equals_top_k(self) -> None:
        result = self._run(top_k=3)
        assert len(result) == 3

    def test_top_k_clamped_to_intermediate_size(self) -> None:
        """top_k larger than intermediate_size should return at most intermediate_size."""
        result = self._run(top_k=100, intermediate_size=16)
        assert len(result) <= 16

    def test_result_has_topneuron_fields(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.computation_map import TopNeuron

        result = self._run(top_k=1)
        assert len(result) == 1
        n = result[0]
        assert isinstance(n, TopNeuron)
        assert isinstance(n.neuron_index, int)
        assert isinstance(n.activation, float)
        assert isinstance(n.logit_contribution, float)
        assert isinstance(n.top_token, str)

    def test_neuron_index_in_range(self) -> None:
        result = self._run(top_k=3)
        for n in result:
            assert 0 <= n.neuron_index < self.INTERMEDIATE

    def test_gated_path(self) -> None:
        """SwiGLU (gate_proj present) path should return valid neurons."""
        result = self._run(with_gate=True, top_k=2)
        assert len(result) == 2

    def test_non_gated_path(self) -> None:
        """Non-gated (no gate_proj) path should use gelu and return valid neurons."""
        result = self._run(with_gate=False, top_k=2)
        assert len(result) == 2

    def test_with_custom_act_fn_gated(self) -> None:
        """When mlp.act is set and gate_proj exists, it should use mlp.act."""
        result = self._run(with_gate=True, with_act_fn=True, top_k=2)
        assert len(result) == 2

    def test_with_custom_act_fn_non_gated(self) -> None:
        """When mlp.act is set and no gate_proj, it should use mlp.act."""
        result = self._run(with_gate=False, with_act_fn=True, top_k=2)
        assert len(result) == 2

    def test_four_norms_path(self) -> None:
        """four_norms=True should use pre_feedforward_layernorm instead."""
        result = self._run(four_norms=True, top_k=2)
        assert len(result) == 2

    def test_standard_norms_path(self) -> None:
        """four_norms=False should use post_attention_layernorm."""
        result = self._run(four_norms=False, top_k=2)
        assert len(result) == 2

    def test_zero_top_k_returns_empty(self) -> None:
        """top_k=0 should return empty list."""
        result = self._run(top_k=0)
        assert result == []

    def test_lm_head_called_with_stacked_vecs(self) -> None:
        """lm_head should be called once when top_k > 0."""
        from chuk_mcp_lazarus.tools.geometry.computation_map import _compute_top_neurons

        layer = _make_mock_layer_for_neurons()
        prev_h = mx.array(np.ones((1, 3, self.HIDDEN_DIM), dtype=np.float32))
        attn_out = mx.array(np.zeros((1, 3, self.HIDDEN_DIM), dtype=np.float32))
        unembed_mx = mx.array(np.ones(self.HIDDEN_DIM, dtype=np.float32))

        mock_lm_head = MagicMock()
        mock_lm_head.return_value = mx.array(np.random.randn(1, 2, 10).astype(np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: f"t{ids[0]}"

        _compute_top_neurons(
            layer,
            prev_h,
            attn_out,
            pos=2,
            unembed_mx=unembed_mx,
            lm_head=mock_lm_head,
            tokenizer=tokenizer,
            top_k=2,
            four_norms=False,
        )

        mock_lm_head.assert_called_once()

    def test_logit_contributions_are_floats(self) -> None:
        """All logit_contribution values should be Python floats."""
        result = self._run(top_k=3)
        for n in result:
            assert isinstance(n.logit_contribution, float)

    def test_activations_are_floats(self) -> None:
        """All activation values should be Python floats."""
        result = self._run(top_k=3)
        for n in result:
            assert isinstance(n.activation, float)
