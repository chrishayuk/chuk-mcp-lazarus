"""Tests for branch_and_collapse tool.

Tests cover:
  - branch_and_collapse: async validation (8) + impl (12) + models (3)
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
# branch_and_collapse — async validation
# ---------------------------------------------------------------------------


class TestBranchAndCollapse:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        result = await branch_and_collapse(donor_prompt="A", branch_prompts=["B", "C"], layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        result = await branch_and_collapse(donor_prompt="A", branch_prompts=["B", "C"], layer=99)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_donor_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        result = await branch_and_collapse(
            donor_prompt="A", branch_prompts=["B", "C"], layer=0, donor_layer=99
        )
        assert result["error_type"] == "LayerOutOfRange"
        assert "donor_layer" in result["message"]

    @pytest.mark.asyncio
    async def test_too_few_branches(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        result = await branch_and_collapse(donor_prompt="A", branch_prompts=["B"], layer=0)
        assert result["error_type"] == "InvalidInput"
        assert "2" in result["message"]

    @pytest.mark.asyncio
    async def test_too_many_branches(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        result = await branch_and_collapse(
            donor_prompt="A",
            branch_prompts=[f"T{i}" for i in range(21)],
            layer=0,
        )
        assert result["error_type"] == "InvalidInput"
        assert "20" in result["message"]

    @pytest.mark.asyncio
    async def test_top_k_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        result = await branch_and_collapse(
            donor_prompt="A", branch_prompts=["B", "C"], layer=0, top_k=0
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success_returns_dict(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        fake = {
            "donor_prompt": "A",
            "layer": 0,
            "donor_layer": None,
            "num_branches": 2,
            "donor_baseline": {},
            "branches": [],
            "collapse": {},
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.branch_and_collapse._branch_and_collapse_impl",
            return_value=fake,
        ):
            result = await branch_and_collapse(donor_prompt="A", branch_prompts=["B", "C"], layer=0)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_handling(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            branch_and_collapse,
        )

        with patch(
            "chuk_mcp_lazarus.tools.geometry.branch_and_collapse._branch_and_collapse_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await branch_and_collapse(donor_prompt="A", branch_prompts=["B", "C"], layer=0)
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# branch_and_collapse — impl tests
# ---------------------------------------------------------------------------

# Mock data: 8-dim hidden, 5-token vocab
DIM = 8
VOCAB_SIZE = 5

# Donor logits — moderate confidence (top-1 = token 0 at ~60%)
DONOR_LOGITS = [5.0, 3.0, 1.0, 2.0, 4.0]

# Branch logits — different force fields produce different confidences
# Branch 0: high confidence (top-1 = token 0 at ~85%)
BRANCH_0_LOGITS = [7.0, 1.0, 0.5, 0.5, 2.0]
# Branch 1: lower confidence (top-1 = token 3)
BRANCH_1_LOGITS = [2.0, 1.0, 0.5, 6.0, 3.0]
# Branch 2: medium confidence (top-1 = token 0)
BRANCH_2_LOGITS = [5.5, 2.0, 1.0, 1.5, 3.0]

DONOR_HIDDEN = [3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestBranchAndCollapseImpl:
    """Sync impl tests with mocked model components."""

    def _run(
        self,
        num_branches: int = 3,
        donor_layer: int | None = None,
        top_k: int = 5,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            _branch_and_collapse_impl,
        )

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = DIM
        meta.num_layers = 4

        # Decomposition: returns hidden states at any requested layer
        mock_hidden_states: dict[int, Any] = {i: MagicMock() for i in range(meta.num_layers)}
        decomp = {
            "embeddings": MagicMock(),
            "hidden_states": mock_hidden_states,
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        # _extract_position cycling:
        # Call 0: donor at capture layer (for injection)
        # Call 1: donor at final layer (for baseline)
        # Call 2..N+1: branch injected finals
        extract_calls = [0]

        def fake_extract_position(tensor: Any, position: int) -> MagicMock:
            extract_calls[0] += 1
            vec = MagicMock()
            vec.tolist.return_value = list(DONOR_HIDDEN)
            vec.reshape.return_value = vec
            return vec

        # _norm_project cycling:
        # Call 0: donor baseline logits
        # Call 1..N: branch logits
        branch_logit_pool = [BRANCH_0_LOGITS, BRANCH_1_LOGITS, BRANCH_2_LOGITS]
        norm_calls = [0]

        def fake_norm_project(final_norm: Any, lm_head: Any, vec: Any) -> MagicMock:
            idx = norm_calls[0]
            norm_calls[0] += 1
            if idx == 0:
                data = DONOR_LOGITS
            else:
                branch_idx = (idx - 1) % len(branch_logit_pool)
                data = branch_logit_pool[branch_idx]
            result = MagicMock()
            result.tolist.return_value = list(data)
            return result

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = MagicMock()
        mock_lm_head = MagicMock()

        # Mock injected forward pass result
        mock_injected_hidden = MagicMock()

        branch_prompts = [f"Template {i}" for i in range(num_branches)]

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
                "chuk_mcp_lazarus.tools.residual_tools._norm_project",
                side_effect=fake_norm_project,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.branch_and_collapse._run_forward_with_injection",
                return_value=mock_injected_hidden,
            ),
        ):
            return _branch_and_collapse_impl(
                model,
                config,
                tokenizer,
                meta,
                "What is the capital of Belgium?",
                branch_prompts,
                0,
                donor_layer,
                -1,
                top_k,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for key in [
            "donor_prompt",
            "layer",
            "donor_layer",
            "num_branches",
            "donor_baseline",
            "branches",
            "collapse",
            "summary",
        ]:
            assert key in r

    def test_branch_count(self) -> None:
        r = self._run(num_branches=3)
        assert len(r["branches"]) == 3

    def test_branch_structure(self) -> None:
        r = self._run()
        for b in r["branches"]:
            assert "branch_idx" in b
            assert "template" in b
            assert "top_prediction" in b
            assert "top_k" in b
            assert "confidence" in b
            assert "entropy" in b

    def test_collapse_selects_highest_confidence(self) -> None:
        r = self._run()
        collapse_conf = r["collapse"]["confidence"]
        for b in r["branches"]:
            assert collapse_conf >= b["confidence"]

    def test_donor_baseline_present(self) -> None:
        r = self._run()
        db = r["donor_baseline"]
        assert "top_prediction" in db
        assert "top_k" in db
        assert "confidence" in db
        assert "entropy" in db

    def test_confidence_range(self) -> None:
        r = self._run()
        assert 0.0 <= r["donor_baseline"]["confidence"] <= 1.0
        for b in r["branches"]:
            assert 0.0 <= b["confidence"] <= 1.0
        assert 0.0 <= r["collapse"]["confidence"] <= 1.0

    def test_entropy_positive(self) -> None:
        r = self._run()
        assert r["donor_baseline"]["entropy"] >= 0.0
        for b in r["branches"]:
            assert b["entropy"] >= 0.0

    def test_summary_keys(self) -> None:
        r = self._run()
        s = r["summary"]
        for key in [
            "donor_token",
            "collapsed_token",
            "collapse_matches_donor",
            "collapsed_confidence",
            "donor_confidence",
            "confidence_gain",
            "mean_branch_confidence",
            "branches_agreeing_with_collapse",
            "entropy_reduction",
        ]:
            assert key in s

    def test_donor_layer_passthrough(self) -> None:
        r = self._run(donor_layer=2)
        assert r["donor_layer"] == 2

    def test_donor_layer_none(self) -> None:
        r = self._run(donor_layer=None)
        assert r["donor_layer"] is None

    def test_two_branches(self) -> None:
        r = self._run(num_branches=2)
        assert len(r["branches"]) == 2
        assert r["num_branches"] == 2

    def test_num_branches_in_result(self) -> None:
        r = self._run(num_branches=4)
        assert r["num_branches"] == 4


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestBranchAndCollapseModels:
    def test_branch_result_model(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import BranchResult
        from chuk_mcp_lazarus.tools.geometry.inject_residual import TokenPrediction

        tp = TokenPrediction(token="hello", token_id=1, probability=0.8)
        br = BranchResult(
            branch_idx=0,
            template="test",
            top_prediction=tp,
            top_k=[tp],
            confidence=0.8,
            entropy=1.2,
        )
        d = br.model_dump()
        assert d["branch_idx"] == 0
        assert d["confidence"] == 0.8
        assert d["entropy"] == 1.2

    def test_collapse_result_model(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import CollapseResult

        cr = CollapseResult(
            selected_branch=2,
            template="The answer is",
            token="Brussels",
            token_id=42,
            confidence=0.95,
            entropy=0.5,
        )
        d = cr.model_dump()
        assert d["selected_branch"] == 2
        assert d["token"] == "Brussels"
        assert d["confidence"] == 0.95

    def test_full_result_model(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
            BranchAndCollapseResult,
            BranchResult,
            CollapseResult,
            DonorBaseline,
        )
        from chuk_mcp_lazarus.tools.geometry.inject_residual import TokenPrediction

        tp = TokenPrediction(token="X", token_id=0, probability=0.5)
        result = BranchAndCollapseResult(
            donor_prompt="test",
            layer=10,
            donor_layer=None,
            num_branches=2,
            donor_baseline=DonorBaseline(
                top_prediction=tp,
                top_k=[tp],
                confidence=0.5,
                entropy=1.0,
            ),
            branches=[
                BranchResult(
                    branch_idx=0,
                    template="A",
                    top_prediction=tp,
                    top_k=[tp],
                    confidence=0.5,
                    entropy=1.0,
                ),
                BranchResult(
                    branch_idx=1,
                    template="B",
                    top_prediction=tp,
                    top_k=[tp],
                    confidence=0.7,
                    entropy=0.8,
                ),
            ],
            collapse=CollapseResult(
                selected_branch=1,
                template="B",
                token="X",
                token_id=0,
                confidence=0.7,
                entropy=0.8,
            ),
            summary={"donor_token": "X", "collapsed_token": "X"},
        )
        d = result.model_dump()
        assert d["num_branches"] == 2
        assert len(d["branches"]) == 2
        assert d["collapse"]["selected_branch"] == 1
