"""Tests for decode_residual tool.

Tests cover:
  - decode_residual: async validation (7) + impl (15) + models (3)
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
# decode_residual — async validation
# ---------------------------------------------------------------------------


class TestDecodeResidual:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        result = await decode_residual(prompt="test", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        result = await decode_residual(prompt="test", layers=[])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        result = await decode_residual(prompt="test", layers=[99])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        result = await decode_residual(prompt="test", layers=[-1])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        result = await decode_residual(prompt="test", layers=[0], top_k=0)
        assert result["error_type"] == "InvalidInput"

        result2 = await decode_residual(prompt="test", layers=[0], top_k=101)
        assert result2["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        fake = {
            "prompt": "t",
            "token_position": -1,
            "token_text": "x",
            "hidden_dim": 64,
            "vocab_size": 100,
            "layers": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.decode_residual._decode_residual_impl",
            return_value=fake,
        ):
            result = await decode_residual(prompt="test", layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

        with patch(
            "chuk_mcp_lazarus.tools.geometry.decode_residual._decode_residual_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await decode_residual(prompt="test", layers=[0])
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# decode_residual — impl tests
# ---------------------------------------------------------------------------

# Mock vocab: 5 tokens with known rankings
# raw_logits =  [5.0, 3.0, 1.0, 2.0, 4.0]  → order: 0,4,1,3,2
# norm_logits = [2.0, 4.0, 1.0, 5.0, 3.0]  → order: 3,1,4,0,2
# raw_ranks (1-idx):  {0:1, 4:2, 1:3, 3:4, 2:5}
# norm_ranks (1-idx): {3:1, 1:2, 4:3, 0:4, 2:5}
# rank_change (raw-norm): {0:-3, 1:1, 2:0, 3:3, 4:-1}
# Biggest gainer: token 3 (rc=3)
# Biggest loser: token 0 (rc=-3)

RAW_LOGITS = [5.0, 3.0, 1.0, 2.0, 4.0]
NORM_LOGITS = [2.0, 4.0, 1.0, 5.0, 3.0]
MEAN_LOGITS = [1.0, 2.0, 3.0, 4.0, 5.0]
HIDDEN = [3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
UNEMBED = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestDecodeResidualImpl:
    """Sync impl tests with mocked model components."""

    def _run(
        self,
        layers: list[int] | None = None,
        top_k: int = 3,
        include_mean: bool = True,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import (
            _decode_residual_impl,
        )

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

        # Mock decomposition
        mock_hidden_0 = MagicMock()
        mock_hidden_1 = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.dtype = "float32"

        decomp = {
            "embeddings": mock_embeddings,
            "hidden_states": {0: mock_hidden_0, 1: mock_hidden_1},
            "prev_hidden": {0: MagicMock(), 1: MagicMock()},
            "attn_outputs": {0: MagicMock(), 1: MagicMock()},
            "ffn_outputs": {0: MagicMock(), 1: MagicMock()},
        }

        # _extract_position returns a mock with .tolist() → HIDDEN
        def fake_extract_position(tensor: Any, position: int) -> MagicMock:
            vec = MagicMock()
            vec.tolist.return_value = list(HIDDEN)
            vec.reshape.return_value = vec
            return vec

        # Track call count for _project_to_logits to distinguish raw vs mean
        proj_calls = [0]

        def fake_project(lm_head: Any, vec: Any) -> MagicMock:
            proj_calls[0] += 1
            result = MagicMock()
            # Every other call after the first two raw calls is mean decode
            result.tolist.return_value = list(RAW_LOGITS)
            return result

        def fake_norm_project(final_norm: Any, lm_head: Any, vec: Any) -> MagicMock:
            result = MagicMock()
            result.tolist.return_value = list(NORM_LOGITS)
            return result

        def fake_unembed(model_obj: Any, tid: int) -> MagicMock:
            vec = MagicMock()
            vec.tolist.return_value = list(UNEMBED)
            return vec

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = MagicMock()
        mock_lm_head = MagicMock()

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._extract_position",
                side_effect=fake_extract_position,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._project_to_logits",
                side_effect=fake_project,
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
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                side_effect=fake_unembed,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
        ):
            return _decode_residual_impl(
                model,
                config,
                tokenizer,
                meta,
                "The capital of Australia is",
                layers,
                top_k,
                -1,
                include_mean,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "prompt" in r
        assert "token_position" in r
        assert "token_text" in r
        assert "hidden_dim" in r
        assert "vocab_size" in r
        assert "layers" in r

    def test_layer_count(self) -> None:
        r = self._run(layers=[0, 1])
        assert len(r["layers"]) == 2

    def test_single_layer(self) -> None:
        r = self._run(layers=[0])
        assert len(r["layers"]) == 1
        assert r["layers"][0]["layer"] == 0

    def test_raw_top_k_count(self) -> None:
        r = self._run(top_k=3)
        assert len(r["layers"][0]["raw_top_k"]) == 3

    def test_norm_top_k_count(self) -> None:
        r = self._run(top_k=3)
        assert len(r["layers"][0]["norm_top_k"]) == 3

    def test_raw_ranking_order(self) -> None:
        r = self._run(top_k=5)
        raw = r["layers"][0]["raw_top_k"]
        # Raw logits [5,3,1,2,4] → top is token 0 (5.0), then token 4 (4.0)
        assert raw[0]["raw_rank"] == 1
        assert raw[1]["raw_rank"] == 2
        assert raw[0]["dot_product"] >= raw[1]["dot_product"]

    def test_norm_ranking_order(self) -> None:
        r = self._run(top_k=5)
        norm = r["layers"][0]["norm_top_k"]
        # Norm logits [2,4,1,5,3] → top is token 3 (5.0), then token 1 (4.0)
        assert norm[0]["norm_rank"] == 1
        assert norm[1]["norm_rank"] == 2
        assert norm[0]["logit"] >= norm[1]["logit"]

    def test_probabilities_valid(self) -> None:
        r = self._run(top_k=5)
        norm = r["layers"][0]["norm_top_k"]
        for entry in norm:
            assert 0.0 <= entry["probability"] <= 1.0
        # Sum of all probs should be ~1.0
        total = sum(e["probability"] for e in norm)
        assert abs(total - 1.0) < 0.01

    def test_top1_changed(self) -> None:
        r = self._run()
        s = r["layers"][0]["summary"]
        # Raw top1 is token 0, norm top1 is token 3 → different
        assert s["top1_changed"] is True

    def test_gainers_present(self) -> None:
        r = self._run()
        g = r["layers"][0]["biggest_gainers"]
        assert len(g) > 0
        # Biggest gainer should have positive rank_change
        assert g[0]["rank_change"] > 0

    def test_losers_present(self) -> None:
        r = self._run()
        lo = r["layers"][0]["biggest_losers"]
        assert len(lo) > 0
        # Biggest loser should have negative rank_change
        assert lo[0]["rank_change"] < 0

    def test_gap_entry_fields(self) -> None:
        r = self._run()
        g = r["layers"][0]["biggest_gainers"][0]
        assert "token" in g
        assert "token_id" in g
        assert "raw_rank" in g
        assert "norm_rank" in g
        assert "rank_change" in g
        assert "angle_to_mean" in g

    def test_mean_decode_present(self) -> None:
        r = self._run(include_mean=True)
        md = r["layers"][0]["mean_decode"]
        assert md is not None
        assert "mean_norm" in md
        assert "mean_fraction" in md
        assert "norm_type" in md
        assert "mean_top_k" in md
        assert "interpretation" in md

    def test_mean_decode_absent(self) -> None:
        r = self._run(include_mean=False)
        md = r["layers"][0]["mean_decode"]
        assert md is None

    def test_summary_fields(self) -> None:
        r = self._run()
        s = r["layers"][0]["summary"]
        assert "raw_top1" in s
        assert "norm_top1" in s
        assert "top1_changed" in s
        assert "raw_norm_rank_correlation" in s
        assert "mean_energy_fraction" in s

    def test_vocab_size(self) -> None:
        r = self._run()
        assert r["vocab_size"] == 5  # length of RAW_LOGITS


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestDecodeResidualModels:
    def test_raw_token_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import RawTokenEntry

        e = RawTokenEntry(token="hello", token_id=1, dot_product=3.14, raw_rank=1)
        d = e.model_dump()
        assert d["token"] == "hello"
        assert d["raw_rank"] == 1

    def test_gap_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import GapEntry

        e = GapEntry(
            token="test",
            token_id=5,
            raw_rank=8,
            norm_rank=1,
            rank_change=7,
            angle_to_mean=87.0,
        )
        d = e.model_dump()
        assert d["rank_change"] == 7
        assert d["angle_to_mean"] == 87.0

    def test_mean_decode(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.decode_residual import (
            MeanDecode,
            MeanDecodeEntry,
        )

        md = MeanDecode(
            mean_norm=5.0,
            mean_fraction=0.12,
            norm_type="RMSNorm",
            mean_top_k=[MeanDecodeEntry(token="Sydney", token_id=1, dot_product=100.0)],
            interpretation="Mean direction dominated by: Sydney",
        )
        d = md.model_dump()
        assert d["norm_type"] == "RMSNorm"
        assert len(d["mean_top_k"]) == 1
