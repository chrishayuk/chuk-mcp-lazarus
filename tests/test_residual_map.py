"""Tests for residual_map tool.

Tests cover:
  - residual_map: async validation (6)
  - _auto_layers: edge cases (3)
  - _residual_map_impl: output keys, layers, ranks, variance, summary (9)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock data: 20 prompts x 8-dim activations
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(88)
MOCK_ACTIVATIONS = [RNG.standard_normal(8).astype(np.float32) for _ in range(20)]


# ---------------------------------------------------------------------------
# residual_map — async validation
# ---------------------------------------------------------------------------


def _prompts(n: int = 20) -> list[str]:
    return [f"prompt_{i}" for i in range(n)]


class TestResidualMap:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        result = await residual_map(_prompts())
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        result = await residual_map(_prompts(), layers=[99])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        result = await residual_map(["a", "b", "c"], layers=[0])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        result = await residual_map([f"p{i}" for i in range(2001)], layers=[0])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_components_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        result = await residual_map(_prompts(), layers=[0], max_components=0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        result = await residual_map(_prompts(), layers=[])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_auto_layers_default(self, loaded_model_state: Any) -> None:
        """layers=None should auto-select layers without error."""
        from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map

        with patch(
            "chuk_mcp_lazarus._extraction.extract_activations_all_layers",
            side_effect=lambda m, c, t, p, lyrs, pos: {
                lyr: MOCK_ACTIVATIONS[i % len(MOCK_ACTIVATIONS)].tolist()
                for i, lyr in enumerate(lyrs)
            },
        ):
            result = await residual_map(_prompts(), layers=None)

        assert "error_type" not in result
        assert "layers" in result
        assert len(result["layers"]) > 0


# ---------------------------------------------------------------------------
# _auto_layers tests
# ---------------------------------------------------------------------------


class TestAutoLayers:
    def test_small_model(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import _auto_layers

        result = _auto_layers(4)
        assert result == [0, 1, 2, 3]

    def test_large_model(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import _auto_layers

        result = _auto_layers(34)
        assert 0 in result
        assert 33 in result
        assert len(result) <= 12

    def test_includes_endpoints(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_map import _auto_layers

        result = _auto_layers(100)
        assert result[0] == 0
        assert result[-1] == 99


# ---------------------------------------------------------------------------
# _residual_map_impl tests
# ---------------------------------------------------------------------------


def _run(
    prompts: list[str] | None = None,
    layers: list[int] | None = None,
    max_components: int = 10,
    token_position: int = -1,
) -> dict:
    """Run _residual_map_impl with mock activations."""
    from chuk_mcp_lazarus.tools.geometry.residual_map import _residual_map_impl

    if prompts is None:
        prompts = [f"prompt_{i}" for i in range(20)]
    if layers is None:
        layers = [1]

    call_idx = [0]

    def fake_extract_single(
        model: Any, config: Any, tokenizer: Any, prompt: str, ly: int, pos: int
    ) -> list:
        idx = call_idx[0]
        call_idx[0] += 1
        return MOCK_ACTIVATIONS[idx % len(MOCK_ACTIVATIONS)].tolist()

    def fake_extract_all(
        model: Any,
        config: Any,
        tokenizer: Any,
        prompt: str,
        lyrs: list[int],
        pos: int,
    ) -> dict:
        result = {}
        for lyr in lyrs:
            idx = call_idx[0]
            call_idx[0] += 1
            result[lyr] = MOCK_ACTIVATIONS[idx % len(MOCK_ACTIVATIONS)].tolist()
        return result

    meta = MagicMock()
    meta.num_layers = 4
    meta.hidden_dim = 8

    with (
        patch(
            "chuk_mcp_lazarus._extraction.extract_activation_at_layer",
            side_effect=fake_extract_single,
        ),
        patch(
            "chuk_mcp_lazarus._extraction.extract_activations_all_layers",
            side_effect=fake_extract_all,
        ),
    ):
        return _residual_map_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            meta,
            prompts,
            layers,
            token_position,
            max_components,
        )


class TestResidualMapImpl:
    def test_output_keys(self) -> None:
        result = _run()
        assert "layers" in result
        assert "num_prompts" in result
        assert "hidden_dim" in result
        assert "summary" in result

    def test_single_layer(self) -> None:
        result = _run(layers=[1])
        assert len(result["layers"]) == 1
        assert result["layers"][0]["layer"] == 1

    def test_multi_layer(self) -> None:
        result = _run(layers=[0, 2])
        assert len(result["layers"]) == 2

    def test_effective_rank_keys(self) -> None:
        result = _run()
        ls = result["layers"][0]
        for key in [
            "effective_rank_50",
            "effective_rank_80",
            "effective_rank_90",
            "effective_rank_95",
            "effective_rank_99",
        ]:
            assert key in ls
            assert isinstance(ls[key], int)

    def test_rank_monotone(self) -> None:
        result = _run()
        ls = result["layers"][0]
        assert ls["effective_rank_50"] <= ls["effective_rank_80"]
        assert ls["effective_rank_80"] <= ls["effective_rank_90"]
        assert ls["effective_rank_90"] <= ls["effective_rank_95"]
        assert ls["effective_rank_95"] <= ls["effective_rank_99"]

    def test_singular_values_normalised(self) -> None:
        result = _run()
        svs = result["layers"][0]["top_singular_values"]
        assert svs[0] == 1.0
        for sv in svs:
            assert 0.0 <= sv <= 1.0 + 1e-6

    def test_variance_captured(self) -> None:
        result = _run()
        assert result["layers"][0]["total_variance_captured"] <= 1.0 + 1e-6
        assert result["layers"][0]["total_variance_captured"] > 0

    def test_summary_keys(self) -> None:
        result = _run(layers=[0, 2])
        s = result["summary"]
        assert "rank_progression_90pct" in s
        assert "peak_rank_layer" in s
        assert "most_compact_layer" in s

    def test_hidden_dim(self) -> None:
        result = _run()
        assert result["hidden_dim"] == 8
