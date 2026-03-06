"""Tests for residual_atlas tool.

Tests cover:
  - residual_atlas: async validation (10)
  - _residual_atlas_impl: output keys, components, variance, vocab, storage (14)
  - Summary: per_layer_rank, concentrated/distributed, stored flag (3)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.subspace_registry import SubspaceRegistry

# ---------------------------------------------------------------------------
# Mock data: 20 prompts x 8-dim activations
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)
MOCK_ACTIVATIONS = [RNG.standard_normal(8).astype(np.float32) for _ in range(20)]

VOCAB_SIZE = 16


def _fake_lm_head(x: Any) -> Any:
    """Mock lm_head: [1, N, hidden_dim] -> [1, N, vocab_size]."""
    import mlx.core as mx

    d = x._data if hasattr(x, "_data") else np.array(x)
    batch, n, hdim = d.shape
    rng = np.random.default_rng(42)
    W = rng.standard_normal((VOCAB_SIZE, hdim)).astype(np.float32)
    out = d.reshape(n, hdim) @ W.T  # [N, vocab]
    return mx.array(out.reshape(1, n, VOCAB_SIZE))


# ---------------------------------------------------------------------------
# residual_atlas — async validation
# ---------------------------------------------------------------------------


def _prompts(n: int = 20) -> list[str]:
    return [f"prompt_{i}" for i in range(n)]


class TestResidualAtlas:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), 0)
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), 99)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), -1)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(["a", "b", "c"], 0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas([f"p{i}" for i in range(2001)], 0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_components_too_large(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), 0, max_components=201)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_components_too_small(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), 0, max_components=0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_components_exceeds_prompts(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(20), 0, max_components=20)
        assert result["error_type"] == "InvalidInput"
        assert "must be <" in result["message"]

    @pytest.mark.asyncio
    async def test_top_k_tokens_too_large(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), 0, top_k_tokens=51)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas

        result = await residual_atlas(_prompts(), [])
        assert result["error_type"] == "InvalidInput"


# ---------------------------------------------------------------------------
# _residual_atlas_impl tests
# ---------------------------------------------------------------------------


def _run(
    prompts: list[str] | None = None,
    layers: list[int] | None = None,
    max_components: int = 5,
    top_k_tokens: int = 3,
    store_subspace: str | None = None,
    token_position: int = -1,
) -> dict:
    """Run _residual_atlas_impl with mock activations."""
    from chuk_mcp_lazarus.tools.geometry.residual_atlas import _residual_atlas_impl

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

    tok = MagicMock()
    tok.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

    with (
        patch(
            "chuk_mcp_lazarus._extraction.extract_activation_at_layer",
            side_effect=fake_extract_single,
        ),
        patch(
            "chuk_mcp_lazarus._extraction.extract_activations_all_layers",
            side_effect=fake_extract_all,
        ),
        patch(
            "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
            return_value=_fake_lm_head,
        ),
    ):
        return _residual_atlas_impl(
            MagicMock(),
            MagicMock(),
            tok,
            meta,
            prompts,
            layers,
            token_position,
            max_components,
            top_k_tokens,
            store_subspace,
        )


class TestResidualAtlasImpl:
    def test_output_keys(self) -> None:
        result = _run()
        assert "layers" in result
        assert "num_prompts" in result
        assert "max_components" in result
        assert "top_k_tokens" in result
        assert "summary" in result

    def test_single_layer(self) -> None:
        result = _run(layers=[1])
        assert len(result["layers"]) == 1
        assert result["layers"][0]["layer"] == 1

    def test_multi_layer(self) -> None:
        result = _run(layers=[0, 2])
        assert len(result["layers"]) == 2
        layer_ids = [lr["layer"] for lr in result["layers"]]
        assert 0 in layer_ids
        assert 2 in layer_ids

    def test_components_count(self) -> None:
        result = _run(max_components=3)
        comps = result["layers"][0]["components"]
        assert len(comps) == 3

    def test_cumulative_variance_monotone(self) -> None:
        result = _run(max_components=5)
        comps = result["layers"][0]["components"]
        for i in range(1, len(comps)):
            assert comps[i]["cumulative_variance"] >= comps[i - 1]["cumulative_variance"]

    def test_variance_sums_to_at_most_one(self) -> None:
        result = _run()
        assert result["layers"][0]["total_variance_captured"] <= 1.0 + 1e-6

    def test_effective_dimensionality_keys(self) -> None:
        result = _run()
        eff = result["layers"][0]["effective_dimensionality"]
        for key in [
            "dims_for_50pct",
            "dims_for_80pct",
            "dims_for_90pct",
            "dims_for_95pct",
            "dims_for_99pct",
        ]:
            assert key in eff
            assert isinstance(eff[key], int)
            assert eff[key] >= 1

    def test_effective_dimensionality_monotone(self) -> None:
        result = _run()
        eff = result["layers"][0]["effective_dimensionality"]
        assert eff["dims_for_50pct"] <= eff["dims_for_80pct"]
        assert eff["dims_for_80pct"] <= eff["dims_for_90pct"]
        assert eff["dims_for_90pct"] <= eff["dims_for_95pct"]
        assert eff["dims_for_95pct"] <= eff["dims_for_99pct"]

    def test_top_positive_tokens_count(self) -> None:
        result = _run(top_k_tokens=3)
        comp = result["layers"][0]["components"][0]
        assert len(comp["top_positive_tokens"]) == 3

    def test_top_negative_tokens_count(self) -> None:
        result = _run(top_k_tokens=3)
        comp = result["layers"][0]["components"][0]
        assert len(comp["top_negative_tokens"]) == 3

    def test_component_norms_positive(self) -> None:
        result = _run()
        for comp in result["layers"][0]["components"]:
            assert comp["norm"] > 0

    def test_store_subspace_single_layer(self) -> None:
        result = _run(layers=[1], store_subspace="atlas_test")
        reg = SubspaceRegistry.get()
        assert reg.exists("atlas_test")
        assert result["layers"][0]["stored_subspace_name"] == "atlas_test"

    def test_store_subspace_multi_layer(self) -> None:
        result = _run(layers=[0, 2], store_subspace="atlas_multi")
        reg = SubspaceRegistry.get()
        assert reg.exists("atlas_multi_layer0")
        assert reg.exists("atlas_multi_layer2")
        layer_names = [lr["stored_subspace_name"] for lr in result["layers"]]
        assert "atlas_multi_layer0" in layer_names
        assert "atlas_multi_layer2" in layer_names

    def test_no_store_default(self) -> None:
        result = _run()
        assert result["layers"][0]["stored_subspace_name"] is None
        reg = SubspaceRegistry.get()
        assert reg.count == 0

    def test_component_idx_one_indexed(self) -> None:
        result = _run(max_components=3)
        comps = result["layers"][0]["components"]
        assert comps[0]["component_idx"] == 1
        assert comps[1]["component_idx"] == 2
        assert comps[2]["component_idx"] == 3


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestResidualAtlasSummary:
    def test_per_layer_rank(self) -> None:
        result = _run(layers=[0, 2])
        s = result["summary"]
        assert "per_layer_effective_rank_90pct" in s
        assert 0 in s["per_layer_effective_rank_90pct"]
        assert 2 in s["per_layer_effective_rank_90pct"]

    def test_concentrated_distributed(self) -> None:
        result = _run(layers=[0, 2])
        s = result["summary"]
        assert "most_concentrated_layer" in s
        assert "most_distributed_layer" in s
        assert s["most_concentrated_layer"] in [0, 2]
        assert s["most_distributed_layer"] in [0, 2]

    def test_stored_flag(self) -> None:
        result_no = _run()
        assert result_no["summary"]["stored"] is False

        result_yes = _run(store_subspace="flag_test")
        assert result_yes["summary"]["stored"] is True
