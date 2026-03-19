"""Tests for residual_match tool.

Tests cover:
  - residual_match: async validation (6) + impl (11) + models (3)
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
# residual_match — async validation
# ---------------------------------------------------------------------------


class TestResidualMatch:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match

        result = await residual_match(target_prompt="A", candidate_prompts=["B"], layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match

        result = await residual_match(target_prompt="A", candidate_prompts=["B"], layer=99)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_candidates(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match

        result = await residual_match(target_prompt="A", candidate_prompts=[], layer=0)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_candidates(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match

        result = await residual_match(
            target_prompt="A",
            candidate_prompts=[f"c{i}" for i in range(21)],
            layer=0,
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match

        fake = {
            "target_prompt": "A",
            "layer": 0,
            "token_position": -1,
            "num_candidates": 1,
            "matches": [],
            "best_full_match": "B",
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.residual_match._residual_match_impl",
            return_value=fake,
        ):
            result = await residual_match(target_prompt="A", candidate_prompts=["B"], layer=0)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match

        with patch(
            "chuk_mcp_lazarus.tools.geometry.residual_match._residual_match_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await residual_match(target_prompt="A", candidate_prompts=["B"], layer=0)
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# residual_match — impl tests
# ---------------------------------------------------------------------------

# Mock data: 8-dim hidden, prompts with distinguishable directions
DIM = 8
TARGET_HIDDEN = [3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
SIMILAR_HIDDEN = [2.8, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # close to target
DIFFERENT_HIDDEN = [0.1, 0.2, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # far from target
UNEMBED_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
UNEMBED_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestResidualMatchImpl:
    """Sync impl tests with mocked model components."""

    def _run(
        self,
        candidate_prompts: list[str] | None = None,
        candidate_hiddens: list[list[float]] | None = None,
        subspace_tokens: list[str] | None = None,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.residual_match import (
            _residual_match_impl,
        )

        if candidate_prompts is None:
            candidate_prompts = ["similar prompt", "different prompt"]
        if candidate_hiddens is None:
            candidate_hiddens = [SIMILAR_HIDDEN, DIFFERENT_HIDDEN]

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        meta = MagicMock()
        meta.hidden_dim = DIM
        meta.num_layers = 4

        # Each call to _run_decomposition_forward returns a decomp dict
        # First call is for target, subsequent calls for candidates
        mock_hidden = MagicMock()
        decomp = {
            "embeddings": MagicMock(),
            "hidden_states": {0: mock_hidden},
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        # _extract_position: target first, then each candidate
        extract_calls = [0]
        all_hiddens = [TARGET_HIDDEN] + candidate_hiddens

        def fake_extract_position(tensor: Any, position: int) -> MagicMock:
            idx = min(extract_calls[0], len(all_hiddens) - 1)
            data = all_hiddens[idx]
            extract_calls[0] += 1
            vec = MagicMock()
            vec.tolist.return_value = list(data)
            return vec

        # Unembed vector lookup for subspace mode
        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return {"alpha": 10, "beta": 20}.get(tok)

        def fake_unembed_np(model_obj: Any, tid: int) -> np.ndarray | None:
            mapping = {10: UNEMBED_A, 20: UNEMBED_B}
            v = mapping.get(tid)
            return np.array(v, dtype=np.float32) if v else None

        with (
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._extract_position",
                side_effect=fake_extract_position,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.residual_match._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.residual_match._get_unembed_vec_np",
                side_effect=fake_unembed_np,
            ),
        ):
            return _residual_match_impl(
                model,
                config,
                tokenizer,
                meta,
                "target prompt",
                candidate_prompts,
                0,
                -1,
                subspace_tokens,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for key in [
            "target_prompt",
            "layer",
            "token_position",
            "num_candidates",
            "matches",
            "best_full_match",
            "summary",
        ]:
            assert key in r

    def test_match_count(self) -> None:
        r = self._run()
        assert r["num_candidates"] == 2
        assert len(r["matches"]) == 2

    def test_similarity_range(self) -> None:
        r = self._run()
        for m in r["matches"]:
            assert -1.0 <= m["full_cosine_similarity"] <= 1.0

    def test_angle_range(self) -> None:
        r = self._run()
        for m in r["matches"]:
            assert 0.0 <= m["full_angle"] <= 180.0

    def test_ordering(self) -> None:
        r = self._run()
        sims = [m["full_cosine_similarity"] for m in r["matches"]]
        assert sims == sorted(sims, reverse=True)

    def test_best_match(self) -> None:
        r = self._run()
        # SIMILAR_HIDDEN is closer to TARGET_HIDDEN than DIFFERENT_HIDDEN
        assert r["best_full_match"] == "similar prompt"

    def test_subspace_present(self) -> None:
        r = self._run(subspace_tokens=["alpha", "beta"])
        assert r["subspace_dim"] == 2
        assert r["subspace_tokens"] == ["alpha", "beta"]
        for m in r["matches"]:
            assert m["subspace"] is not None
            assert "subspace_cosine_similarity" in m["subspace"]
            assert "subspace_angle" in m["subspace"]
            assert "orthogonal_cosine_similarity" in m["subspace"]
            assert "orthogonal_angle" in m["subspace"]

    def test_subspace_absent(self) -> None:
        r = self._run(subspace_tokens=None)
        assert r["subspace_dim"] is None
        assert r["subspace_tokens"] is None
        for m in r["matches"]:
            assert m["subspace"] is None

    def test_self_match(self) -> None:
        """Target compared to itself should have cos~1, angle~0."""
        r = self._run(
            candidate_prompts=["self"],
            candidate_hiddens=[TARGET_HIDDEN],
        )
        m = r["matches"][0]
        assert m["full_cosine_similarity"] > 0.999
        assert m["full_angle"] < 1.0

    def test_similar_beats_different(self) -> None:
        r = self._run()
        sim_match = next(m for m in r["matches"] if m["prompt"] == "similar prompt")
        diff_match = next(m for m in r["matches"] if m["prompt"] == "different prompt")
        assert sim_match["full_cosine_similarity"] > diff_match["full_cosine_similarity"]

    def test_summary_fields(self) -> None:
        r = self._run()
        s = r["summary"]
        assert "best_similarity" in s
        assert "best_angle" in s
        assert "worst_similarity" in s
        assert s["best_similarity"] >= s["worst_similarity"]


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestResidualMatchModels:
    def test_match_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import MatchEntry

        m = MatchEntry(
            prompt="test prompt",
            full_cosine_similarity=0.95,
            full_angle=18.2,
        )
        d = m.model_dump()
        assert d["prompt"] == "test prompt"
        assert d["full_cosine_similarity"] == 0.95
        assert d["subspace"] is None

    def test_subspace_match_info(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import SubspaceMatchInfo

        m = SubspaceMatchInfo(
            subspace_cosine_similarity=0.92,
            subspace_angle=23.1,
            orthogonal_cosine_similarity=0.45,
            orthogonal_angle=63.3,
        )
        d = m.model_dump()
        assert d["subspace_cosine_similarity"] == 0.92
        assert d["orthogonal_angle"] == 63.3

    def test_residual_match_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_match import (
            MatchEntry,
            ResidualMatchResult,
        )

        m = ResidualMatchResult(
            target_prompt="target",
            layer=5,
            token_position=-1,
            num_candidates=2,
            matches=[
                MatchEntry(
                    prompt="c1",
                    full_cosine_similarity=0.95,
                    full_angle=18.2,
                ),
                MatchEntry(
                    prompt="c2",
                    full_cosine_similarity=0.80,
                    full_angle=36.9,
                ),
            ],
            best_full_match="c1",
            summary={"best_similarity": 0.95},
        )
        d = m.model_dump()
        assert d["num_candidates"] == 2
        assert len(d["matches"]) == 2
        assert d["best_full_match"] == "c1"
