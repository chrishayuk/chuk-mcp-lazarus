"""Tests for inject_residual tool.

Tests cover:
  - inject_residual: async validation (8) + impl (18) + models (3)
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
# inject_residual — async validation
# ---------------------------------------------------------------------------


class TestInjectResidual:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(donor_prompt="A", recipient_prompt="B", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(donor_prompt="A", recipient_prompt="B", layer=99)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(donor_prompt="A", recipient_prompt="B", layer=-1)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens_zero(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A", recipient_prompt="B", layer=0, max_new_tokens=0
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens_too_high(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A", recipient_prompt="B", layer=0, max_new_tokens=501
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_negative_temperature(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A", recipient_prompt="B", layer=0, temperature=-0.5
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_subspace_only_without_tokens(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A",
            recipient_prompt="B",
            layer=0,
            subspace_only=True,
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        fake = {
            "donor_prompt": "A",
            "recipient_prompt": "B",
            "layer": 0,
            "donor_position": -1,
            "recipient_position": -1,
            "subspace_only": False,
            "donor_output": {},
            "recipient_output": {},
            "injected_output": {},
            "comparison": {},
            "residual_similarity": {},
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.inject_residual._inject_residual_impl",
            return_value=fake,
        ):
            result = await inject_residual(donor_prompt="A", recipient_prompt="B", layer=0)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        with patch(
            "chuk_mcp_lazarus.tools.geometry.inject_residual._inject_residual_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await inject_residual(donor_prompt="A", recipient_prompt="B", layer=0)
        assert result["error_type"] == "GeometryFailed"

    @pytest.mark.asyncio
    async def test_subspace_name_not_found(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A",
            recipient_prompt="B",
            layer=0,
            subspace_name="nonexistent",
        )
        assert result["error_type"] == "VectorNotFound"

    @pytest.mark.asyncio
    async def test_subspace_name_mutual_exclusion(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A",
            recipient_prompt="B",
            layer=0,
            subspace_tokens=["alpha"],
            subspace_name="some_sub",
        )
        assert result["error_type"] == "InvalidInput"
        assert "mutually exclusive" in result["message"]

    @pytest.mark.asyncio
    async def test_patch_all_with_subspace_name(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A",
            recipient_prompt="B",
            layer=0,
            subspace_name="some_sub",
            patch_all_positions=True,
        )
        assert result["error_type"] == "InvalidInput"
        assert "incompatible" in result["message"]

    @pytest.mark.asyncio
    async def test_subspace_name_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.subspace_registry import SubspaceRegistry
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        # Pre-store a subspace
        reg = SubspaceRegistry.get()
        basis = np.eye(3, 64, dtype=np.float32)
        from chuk_mcp_lazarus.subspace_registry import SubspaceMetadata

        meta = SubspaceMetadata(
            name="test_sub",
            layer=0,
            rank=3,
            num_prompts=5,
            hidden_dim=64,
            variance_explained=[0.5, 0.3, 0.2],
            total_variance_explained=1.0,
            computed_at="2024-01-01T00:00:00+00:00",
        )
        reg.store("test_sub", basis, meta)

        fake = {
            "donor_prompt": "A",
            "recipient_prompt": "B",
            "layer": 0,
            "donor_position": -1,
            "recipient_position": -1,
            "subspace_only": False,
            "patch_all_positions": False,
            "donor_output": {},
            "recipient_output": {},
            "injected_output": {},
            "comparison": {},
            "residual_similarity": {},
            "subspace_analysis": {"subspace_name": "test_sub"},
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.inject_residual._inject_residual_impl",
            return_value=fake,
        ):
            result = await inject_residual(
                donor_prompt="A",
                recipient_prompt="B",
                layer=0,
                subspace_name="test_sub",
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_patch_all_with_subspace_only(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A",
            recipient_prompt="B",
            layer=0,
            subspace_only=True,
            subspace_tokens=["alpha"],
            patch_all_positions=True,
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_donor_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A", recipient_prompt="B", layer=0, donor_layer=99
        )
        assert result["error_type"] == "LayerOutOfRange"
        assert "donor_layer" in result["message"]

    @pytest.mark.asyncio
    async def test_donor_layer_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A", recipient_prompt="B", layer=0, donor_layer=-1
        )
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_donor_layer_as_string(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        fake = {
            "donor_prompt": "A",
            "recipient_prompt": "B",
            "layer": 0,
            "donor_layer": 2,
            "donor_position": -1,
            "recipient_position": -1,
            "subspace_only": False,
            "donor_output": {},
            "recipient_output": {},
            "injected_output": {},
            "comparison": {},
            "residual_similarity": {},
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.inject_residual._inject_residual_impl",
            return_value=fake,
        ):
            result = await inject_residual(
                donor_prompt="A", recipient_prompt="B", layer=0, donor_layer="2"
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_patch_all_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        fake = {
            "donor_prompt": "A",
            "recipient_prompt": "B",
            "layer": 0,
            "donor_position": -1,
            "recipient_position": -1,
            "subspace_only": False,
            "patch_all_positions": True,
            "donor_output": {},
            "recipient_output": {},
            "injected_output": {},
            "comparison": {},
            "residual_similarity": {},
            "summary": {"donor_seq_len": 3, "recipient_seq_len": 3},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.inject_residual._inject_residual_impl",
            return_value=fake,
        ):
            result = await inject_residual(
                donor_prompt="A",
                recipient_prompt="B",
                layer=0,
                patch_all_positions=True,
            )
        assert "error" not in result


# ---------------------------------------------------------------------------
# inject_residual — impl tests
# ---------------------------------------------------------------------------

# Mock data: 8-dim hidden, 5-token vocab
DIM = 8
DONOR_HIDDEN = [3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
RECIP_HIDDEN = [0.5, 3.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
DONOR_LOGITS = [5.0, 3.0, 1.0, 2.0, 4.0]  # top-1 = token 0
RECIP_LOGITS = [2.0, 4.0, 1.0, 5.0, 3.0]  # top-1 = token 3
INJECTED_LOGITS = [4.8, 3.1, 1.0, 2.1, 3.9]  # top-1 = token 0 (matches donor)
UNEMBED_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
UNEMBED_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestInjectResidualImpl:
    """Sync impl tests with mocked model components."""

    def _run(
        self,
        subspace_only: bool = False,
        subspace_tokens: list[str] | None = None,
        subspace_name: str | None = None,
        max_new_tokens: int = 5,
        patch_all_positions: bool = False,
        donor_layer: int | None = None,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import (
            _inject_residual_impl,
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
        # Use a defaultdict-like pattern so any layer key works
        mock_hidden_states: dict[int, Any] = {i: MagicMock() for i in range(meta.num_layers)}

        decomp = {
            "embeddings": MagicMock(),
            "hidden_states": mock_hidden_states,
            "prev_hidden": {},
            "attn_outputs": {},
            "ffn_outputs": {},
        }

        # _extract_position cycling: donor_layer, donor_final, recip_layer, recip_final, injected_final
        extract_calls = [0]
        hidden_sequence = [
            DONOR_HIDDEN,  # donor at injection layer
            DONOR_HIDDEN,  # donor at final layer
            RECIP_HIDDEN,  # recipient at injection layer
            RECIP_HIDDEN,  # recipient at final layer
            DONOR_HIDDEN,  # injected final (matches donor for Markov test)
        ]

        def fake_extract_position(tensor: Any, position: int) -> MagicMock:
            idx = min(extract_calls[0], len(hidden_sequence) - 1)
            data = hidden_sequence[idx]
            extract_calls[0] += 1
            vec = MagicMock()
            vec.tolist.return_value = list(data)
            vec.reshape.return_value = vec
            return vec

        # _norm_project cycling: donor_logits, recip_logits, injected_logits
        norm_calls = [0]
        logit_sequence = [DONOR_LOGITS, RECIP_LOGITS, INJECTED_LOGITS]

        def fake_norm_project(final_norm: Any, lm_head: Any, vec: Any) -> MagicMock:
            idx = min(norm_calls[0], len(logit_sequence) - 1)
            data = logit_sequence[idx]
            norm_calls[0] += 1
            result = MagicMock()
            result.tolist.return_value = list(data)
            return result

        # Unembed vector lookup for subspace mode
        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return {"alpha": 10, "beta": 20}.get(tok)

        def fake_unembed_np(model_obj: Any, tid: int) -> np.ndarray | None:
            mapping = {10: UNEMBED_A, 20: UNEMBED_B}
            v = mapping.get(tid)
            return np.array(v, dtype=np.float32) if v else None

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = MagicMock()
        mock_lm_head = MagicMock()

        # Mock injected forward pass result
        mock_injected_hidden = MagicMock()

        with (
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._extract_position",
                side_effect=fake_extract_position,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                side_effect=fake_norm_project,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual._run_forward_with_injection",
                return_value=mock_injected_hidden,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual._generate_from_hidden",
                return_value=("injected text", 3),
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual.generate_text",
                return_value=("generated text", 5),
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual._get_unembed_vec_np",
                side_effect=fake_unembed_np,
            ),
        ):
            return _inject_residual_impl(
                model,
                config,
                tokenizer,
                meta,
                "The capital of Australia is",
                "The answer is",
                0,
                donor_layer,
                max_new_tokens,
                0.0,
                -1,
                -1,
                subspace_only,
                subspace_tokens,
                subspace_name,
                patch_all_positions,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for key in [
            "donor_prompt",
            "recipient_prompt",
            "layer",
            "donor_output",
            "recipient_output",
            "injected_output",
            "comparison",
            "residual_similarity",
            "summary",
        ]:
            assert key in r

    def test_donor_output_fields(self) -> None:
        r = self._run()
        d = r["donor_output"]
        assert "text" in d
        assert "top_prediction" in d
        assert "top_prediction_id" in d
        assert "probability" in d
        assert "top_k" in d

    def test_recipient_output_fields(self) -> None:
        r = self._run()
        d = r["recipient_output"]
        assert "text" in d
        assert "top_prediction" in d
        assert "probability" in d

    def test_injected_output_fields(self) -> None:
        r = self._run()
        d = r["injected_output"]
        assert "text" in d
        assert "top_prediction" in d
        assert "probability" in d
        assert "top_k" in d

    def test_comparison_metrics_keys(self) -> None:
        r = self._run()
        c = r["comparison"]
        assert "injected_matches_donor" in c
        assert "injected_matches_recipient" in c
        assert "donor_recipient_kl" in c
        assert "donor_injected_kl" in c
        assert "recipient_injected_kl" in c

    def test_residual_similarity_keys(self) -> None:
        r = self._run()
        s = r["residual_similarity"]
        assert "cosine_similarity" in s
        assert "angle" in s
        assert "donor_norm" in s
        assert "recipient_norm" in s

    def test_kl_divergence_nonnegative(self) -> None:
        r = self._run()
        c = r["comparison"]
        assert c["donor_recipient_kl"] >= 0.0
        assert c["donor_injected_kl"] >= 0.0
        assert c["recipient_injected_kl"] >= 0.0

    def test_injected_matches_donor(self) -> None:
        r = self._run()
        # INJECTED_LOGITS top-1 is token 0, same as DONOR_LOGITS
        assert r["comparison"]["injected_matches_donor"] is True

    def test_injected_does_not_match_recipient(self) -> None:
        r = self._run()
        # RECIP_LOGITS top-1 is token 3, INJECTED_LOGITS top-1 is token 0
        assert r["comparison"]["injected_matches_recipient"] is False

    def test_cosine_similarity_range(self) -> None:
        r = self._run()
        cos = r["residual_similarity"]["cosine_similarity"]
        assert -1.0 <= cos <= 1.0

    def test_angle_range(self) -> None:
        r = self._run()
        angle = r["residual_similarity"]["angle"]
        assert 0.0 <= angle <= 180.0

    def test_donor_norm_positive(self) -> None:
        r = self._run()
        assert r["residual_similarity"]["donor_norm"] > 0.0

    def test_subspace_analysis_present(self) -> None:
        r = self._run(subspace_only=True, subspace_tokens=["alpha", "beta"])
        sa = r["subspace_analysis"]
        assert sa is not None
        assert "subspace_dim" in sa
        assert "tokens_used" in sa
        assert "donor_subspace_fraction" in sa
        assert "recipient_subspace_fraction" in sa
        assert "subspace_cosine_similarity" in sa
        assert "orthogonal_cosine_similarity" in sa

    def test_subspace_analysis_absent(self) -> None:
        r = self._run(subspace_only=False)
        assert r["subspace_analysis"] is None

    def test_subspace_fractions_range(self) -> None:
        r = self._run(subspace_only=True, subspace_tokens=["alpha", "beta"])
        sa = r["subspace_analysis"]
        assert 0.0 <= sa["donor_subspace_fraction"] <= 1.0
        assert 0.0 <= sa["recipient_subspace_fraction"] <= 1.0

    def test_text_generation_present(self) -> None:
        r = self._run()
        assert isinstance(r["injected_output"]["text"], str)
        assert isinstance(r["donor_output"]["text"], str)

    def test_summary_present(self) -> None:
        r = self._run()
        s = r["summary"]
        assert "markov_holds" in s
        assert "donor_injected_kl" in s
        assert "donor_top1" in s
        assert "injected_top1" in s

    def test_probabilities_valid(self) -> None:
        r = self._run()
        assert 0.0 <= r["donor_output"]["probability"] <= 1.0
        assert 0.0 <= r["recipient_output"]["probability"] <= 1.0
        assert 0.0 <= r["injected_output"]["probability"] <= 1.0

    def test_patch_all_output_keys(self) -> None:
        r = self._run(patch_all_positions=True)
        assert "patch_all_positions" in r
        assert r["patch_all_positions"] is True
        assert "donor_seq_len" in r["summary"]
        assert "recipient_seq_len" in r["summary"]

    def test_patch_all_matches_donor(self) -> None:
        r = self._run(patch_all_positions=True)
        # INJECTED_LOGITS top-1 == token 0, DONOR_LOGITS top-1 == token 0
        assert r["comparison"]["injected_matches_donor"] is True

    def test_subspace_name_produces_analysis(self) -> None:
        from chuk_mcp_lazarus.subspace_registry import SubspaceMetadata, SubspaceRegistry

        reg = SubspaceRegistry.get()
        basis = np.eye(2, DIM, dtype=np.float32)
        meta = SubspaceMetadata(
            name="impl_test",
            layer=0,
            rank=2,
            num_prompts=5,
            hidden_dim=DIM,
            variance_explained=[0.6, 0.4],
            total_variance_explained=1.0,
            computed_at="2024-01-01T00:00:00+00:00",
        )
        reg.store("impl_test", basis, meta)
        r = self._run(subspace_name="impl_test")
        sa = r["subspace_analysis"]
        assert sa is not None
        assert sa["subspace_name"] == "impl_test"
        assert sa["subspace_dim"] == 2
        assert sa["tokens_used"] == []

    def test_subspace_name_fractions_range(self) -> None:
        from chuk_mcp_lazarus.subspace_registry import SubspaceMetadata, SubspaceRegistry

        reg = SubspaceRegistry.get()
        basis = np.eye(2, DIM, dtype=np.float32)
        meta = SubspaceMetadata(
            name="frac_test",
            layer=0,
            rank=2,
            num_prompts=5,
            hidden_dim=DIM,
            variance_explained=[0.6, 0.4],
            total_variance_explained=1.0,
            computed_at="2024-01-01T00:00:00+00:00",
        )
        reg.store("frac_test", basis, meta)
        r = self._run(subspace_name="frac_test")
        sa = r["subspace_analysis"]
        assert 0.0 <= sa["donor_subspace_fraction"] <= 1.0
        assert 0.0 <= sa["recipient_subspace_fraction"] <= 1.0

    def test_donor_layer_in_output(self) -> None:
        r = self._run(donor_layer=2)
        assert r["donor_layer"] == 2

    def test_donor_layer_none_defaults(self) -> None:
        r = self._run(donor_layer=None)
        assert r["donor_layer"] is None

    def test_donor_layer_same_as_layer(self) -> None:
        r = self._run(donor_layer=0)
        assert r["donor_layer"] == 0
        # Should work identically to donor_layer=None
        assert "error" not in r

    def test_donor_layer_summary(self) -> None:
        r = self._run(donor_layer=2)
        assert r["summary"]["donor_layer"] == 2

    def test_donor_layer_no_summary_when_same(self) -> None:
        r = self._run(donor_layer=0)
        assert "donor_layer" not in r["summary"]


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestInjectResidualModels:
    def test_comparison_metrics(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import ComparisonMetrics

        m = ComparisonMetrics(
            injected_matches_donor=True,
            injected_matches_recipient=False,
            donor_recipient_kl=1.5,
            donor_injected_kl=0.01,
            recipient_injected_kl=1.4,
        )
        d = m.model_dump()
        assert d["injected_matches_donor"] is True
        assert d["donor_injected_kl"] == 0.01

    def test_residual_similarity(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import ResidualSimilarity

        m = ResidualSimilarity(
            cosine_similarity=0.95,
            angle=18.2,
            donor_norm=5.0,
            recipient_norm=4.8,
        )
        d = m.model_dump()
        assert d["angle"] == 18.2

    def test_subspace_analysis(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.inject_residual import SubspaceAnalysis

        m = SubspaceAnalysis(
            subspace_dim=3,
            tokens_used=["Sydney", "Canberra", "Melbourne"],
            donor_subspace_fraction=0.006,
            recipient_subspace_fraction=0.003,
            subspace_cosine_similarity=0.92,
            orthogonal_cosine_similarity=0.45,
        )
        d = m.model_dump()
        assert d["subspace_dim"] == 3
        assert len(d["tokens_used"]) == 3


# ---------------------------------------------------------------------------
# _run_forward_with_injection — direct unit tests (lines 187-247)
# ---------------------------------------------------------------------------


class TestRunForwardWithInjection:
    """Direct tests for _run_forward_with_injection covering lines 187-247."""

    def _make_model_and_config(self, num_layers: int = 3, seq_len: int = 4, hidden: int = 8):
        """Build a minimal mock model whose layers accept (h, mask, cache) calls."""
        import mlx.core as mx

        model = MagicMock()
        config = MagicMock()

        # embed_tokens returns a [1, seq, hidden] array
        embed_out = mx.array(np.ones((1, seq_len, hidden), dtype=np.float32) * 0.1)
        embed_tokens = MagicMock()
        embed_tokens.return_value = embed_out

        # Layers — each one returns h unchanged (tuple form)
        layers = []
        for _ in range(num_layers):
            layer = MagicMock()
            # Return tuple: (h,) so that layer_out[0] is used
            layer.side_effect = lambda h, **kw: (h,)
            layers.append(layer)

        # ModelHooks wiring
        helper = MagicMock()
        helper._get_layers.return_value = layers
        helper._get_embed_tokens.return_value = embed_tokens
        helper._get_embedding_scale.return_value = None

        return model, config, helper, layers, hidden, seq_len

    def _call(
        self, model, config, helper, input_ids_np, target_layer, target_position, injection_vec
    ):
        from chuk_mcp_lazarus.tools.geometry.inject_residual import (
            _run_forward_with_injection,
        )
        import mlx.core as mx

        input_ids = mx.array(input_ids_np)
        metadata = MagicMock()

        with patch("chuk_lazarus.introspection.hooks.ModelHooks", return_value=helper):
            return _run_forward_with_injection(
                model, config, input_ids, target_layer, target_position, injection_vec, metadata
            )

    def test_single_position_injection_returns_array(self):
        """1-D injection_vec: splice at last position, returns hidden state."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )
        # 1-D injection vector
        injection_vec = mx.array(np.array([1.0] * hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result is not None
        assert result.ndim == 3  # [batch, seq, hidden]
        assert result.shape[2] == hidden

    def test_single_position_injection_seq_len_preserved(self):
        """Splicing at a valid position preserves sequence length."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=4, hidden=8
        )
        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3, 4])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=2,
            injection_vec=injection_vec,
        )
        # seq_len must stay the same after splice
        assert result.shape[1] == seq_len

    def test_single_position_injection_negative_position(self):
        """Negative target_position resolves to seq_len + pos."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=4, hidden=8
        )
        injection_vec = mx.array(np.ones(hidden, dtype=np.float32))
        input_ids_np = np.array([10, 20, 30, 40])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result.shape == (1, seq_len, hidden)

    def test_all_position_injection_3d_replaces_hidden(self):
        """3-D injection_vec: entire hidden state replaced at target_layer."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=4, hidden=8
        )
        # 3-D injection: same seq_len — no mask resize needed
        injection_3d = mx.array(np.full((1, seq_len, hidden), 7.0, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3, 4])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_3d,
        )
        assert result.ndim == 3
        assert result.shape[1] == seq_len
        assert result.shape[2] == hidden

    def test_all_position_injection_different_seq_len_resizes_mask(self):
        """3-D injection whose seq_len differs from recipient triggers mask resize."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=4, hidden=8
        )
        # Donor has seq_len=6 — different from recipient (4)
        new_seq = 6
        injection_3d = mx.array(np.full((1, new_seq, hidden), 3.0, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3, 4])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_3d,
        )
        assert result.ndim == 3
        # After replacement, seq dimension must match the injected tensor
        assert result.shape[1] == new_seq

    def test_embedding_scale_applied_when_not_none(self):
        """When embedding_scale is set, it is applied to the embedding output."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )
        # Override scale
        helper._get_embedding_scale.return_value = 2.0
        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        # Should run without error; scale multiplication path exercised
        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result is not None

    def test_layer_tuple_output_unpacked(self):
        """Layer that returns tuple — h = layer_out[0] path is exercised."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )
        # Already configured as tuple via side_effect in _make_model_and_config
        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=1,
            target_position=0,
            injection_vec=injection_vec,
        )
        assert result.ndim == 3

    def test_layer_hidden_states_attr_unpacked(self):
        """Layer that returns an object with .hidden_states attribute."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )
        # Patch first layer to return object with .hidden_states
        embed_out = mx.array(np.ones((1, seq_len, hidden), dtype=np.float32) * 0.1)
        hs_obj = MagicMock()
        hs_obj.hidden_states = embed_out
        layers[0].side_effect = lambda h, **kw: hs_obj

        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result.ndim == 3

    def test_layer_plain_tensor_output(self):
        """Layer that returns a plain tensor (no tuple, no .hidden_states)."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )
        plain_h = mx.array(np.ones((1, seq_len, hidden), dtype=np.float32) * 0.5)
        layers[0].side_effect = lambda h, **kw: plain_h

        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result.ndim == 3

    def test_layer_raises_typeerror_fallback_no_mask(self):
        """Layer raises TypeError on (h, mask=...) — fallback to (h, cache=None)."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )

        def _layer_no_mask(h, **kw):
            if "mask" in kw:
                raise TypeError("no mask")
            return (h,)

        layers[0].side_effect = _layer_no_mask

        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result.ndim == 3

    def test_layer_raises_typeerror_fallback_no_cache(self):
        """Layer raises TypeError on (h, cache=None) — fallback to plain (h)."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=2, seq_len=3, hidden=8
        )

        def _layer_plain_only(h, **kw):
            if kw:
                raise TypeError("no kwargs")
            return (h,)

        layers[0].side_effect = _layer_plain_only

        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result.ndim == 3

    def test_input_ids_1d_expanded_to_2d(self):
        """1-D input_ids are expanded to [1, seq] inside the function."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=1, seq_len=3, hidden=8
        )
        injection_vec = mx.array(np.zeros(hidden, dtype=np.float32))
        # Pass 1-D array; the function does input_ids = input_ids[None, :]
        input_ids_np = np.array([5, 6, 7])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=0,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result is not None

    def test_injection_on_layer_1_of_3(self):
        """Injection fires on the correct middle layer."""
        import mlx.core as mx

        model, config, helper, layers, hidden, seq_len = self._make_model_and_config(
            num_layers=3, seq_len=3, hidden=8
        )
        injection_vec = mx.array(np.ones(hidden, dtype=np.float32) * 9.0)
        input_ids_np = np.array([1, 2, 3])

        result = self._call(
            model,
            config,
            helper,
            input_ids_np,
            target_layer=1,
            target_position=-1,
            injection_vec=injection_vec,
        )
        assert result.ndim == 3


# ---------------------------------------------------------------------------
# _generate_from_hidden — direct unit tests (lines 271-318)
# ---------------------------------------------------------------------------


class TestGenerateFromHidden:
    """Direct tests for _generate_from_hidden covering lines 271-318."""

    def _make_inputs(self, seq_len: int = 3, hidden: int = 8, vocab: int = 10):
        """Return (final_hidden, input_ids) as MockMxArrays."""
        import mlx.core as mx

        final_hidden = mx.array(np.random.randn(1, seq_len, hidden).astype(np.float32))
        input_ids = mx.array(np.array([1, 2, 3], dtype=np.int32))
        return final_hidden, input_ids

    def _make_tokenizer(self, eos_token_id: int = 0):
        tok = MagicMock()
        tok.eos_token_id = eos_token_id
        tok.decode.side_effect = lambda ids, **kw: " ".join(str(i) for i in ids)
        return tok

    def _make_model_returns_logits(self, vocab: int = 10, seq_len: int = 4, next_token_id: int = 5):
        """Model call returns logits shaped [1, seq, vocab] with argmax = next_token_id."""
        import mlx.core as mx

        logits_data = np.zeros((1, seq_len, vocab), dtype=np.float32)
        logits_data[0, -1, next_token_id] = 10.0  # force argmax
        logits = mx.array(logits_data)

        model = MagicMock()
        model.return_value = logits
        return model

    def _call(
        self,
        model,
        tokenizer,
        final_hidden,
        input_ids,
        position=-1,
        max_new_tokens=3,
        temperature=0.0,
        first_token_id=2,
        vocab=10,
    ):
        from chuk_mcp_lazarus.tools.geometry.inject_residual import _generate_from_hidden
        import mlx.core as mx

        # first_logits returned by _norm_project — shape [vocab]
        first_logits_data = np.zeros(vocab, dtype=np.float32)
        first_logits_data[first_token_id] = 10.0
        first_logits = mx.array(first_logits_data)

        # vec returned by _extract_position (irrelevant — mocked)
        mock_vec = MagicMock()
        final_norm = MagicMock()
        lm_head = MagicMock()

        with (
            patch(
                "chuk_mcp_lazarus._residual_helpers._extract_position",
                return_value=mock_vec,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                return_value=first_logits,
            ),
            patch("mlx.core.eval"),
        ):
            return _generate_from_hidden(
                model,
                tokenizer,
                final_hidden,
                final_norm,
                lm_head,
                input_ids,
                position,
                max_new_tokens,
                temperature,
            )

    def test_greedy_returns_string_and_count(self):
        """temperature=0 path returns (str, int)."""

        model = self._make_model_returns_logits(next_token_id=5)
        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=2, temperature=0.0, first_token_id=2
        )
        assert isinstance(text, str)
        assert isinstance(count, int)
        assert count >= 1

    def test_greedy_first_token_is_eos_returns_immediately(self):
        """When first token == eos_token_id, generation stops after 1 token."""

        model = self._make_model_returns_logits(next_token_id=5)
        eos_id = 7
        tok = self._make_tokenizer(eos_token_id=eos_id)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=5, temperature=0.0, first_token_id=eos_id
        )
        assert count == 1

    def test_temperature_sampling_path(self):
        """temperature > 0 exercises the mx.softmax + mx.random.categorical branch."""

        model = self._make_model_returns_logits(next_token_id=3)
        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=2, temperature=1.0, first_token_id=2
        )
        assert isinstance(text, str)
        assert count >= 1

    def test_generation_stops_at_max_new_tokens(self):
        """Generated token count is at most max_new_tokens."""

        # next_token_id != eos so generation runs to max_new_tokens
        model = self._make_model_returns_logits(next_token_id=5)
        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        max_n = 3
        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=max_n, temperature=0.0, first_token_id=2
        )
        assert count <= max_n

    def test_eos_mid_generation_stops_loop(self):
        """EOS token encountered mid-generation stops the loop early."""
        import mlx.core as mx

        eos_id = 0
        # Model always returns eos as the next prediction
        logits_data = np.zeros((1, 4, 10), dtype=np.float32)
        logits_data[0, -1, eos_id] = 10.0
        next_logits = mx.array(logits_data)
        model = MagicMock()
        model.return_value = next_logits

        tok = self._make_tokenizer(eos_token_id=eos_id)
        fh, ids = self._make_inputs()

        max_n = 5
        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=max_n, temperature=0.0, first_token_id=2
        )  # first_token != eos
        # Loop must have stopped before max_n - 1 additional tokens
        assert count < max_n

    def test_model_returns_tuple_logits(self):
        """Model returns (logits_tensor,) tuple — tuple unwrap path exercised."""
        import mlx.core as mx

        logits_data = np.zeros((1, 4, 10), dtype=np.float32)
        logits_data[0, -1, 5] = 10.0
        logits = mx.array(logits_data)

        model = MagicMock()
        model.return_value = (logits,)  # tuple

        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=2, temperature=0.0, first_token_id=2
        )
        assert isinstance(text, str)

    def test_model_returns_object_with_logits_attr(self):
        """Model returns object with .logits attribute — .logits unwrap path."""
        import mlx.core as mx

        logits_data = np.zeros((1, 4, 10), dtype=np.float32)
        logits_data[0, -1, 5] = 10.0
        logits = mx.array(logits_data)

        output_obj = MagicMock()
        output_obj.logits = logits
        # Make sure isinstance(output_obj, tuple) is False
        model = MagicMock()
        model.return_value = output_obj

        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=2, temperature=0.0, first_token_id=2
        )
        assert isinstance(text, str)

    def test_model_logits_2d_last_token(self):
        """2-D logits from model use logits[-1, :] path."""
        import mlx.core as mx

        logits_2d = np.zeros((4, 10), dtype=np.float32)
        logits_2d[-1, 5] = 10.0
        logits = mx.array(logits_2d)

        model = MagicMock()
        model.return_value = logits

        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=2, temperature=0.0, first_token_id=2
        )
        assert isinstance(text, str)

    def test_input_ids_2d_flattened_in_loop(self):
        """2-D input_ids are flattened via reshape(-1) before concatenation."""
        import mlx.core as mx

        model = self._make_model_returns_logits(next_token_id=5)
        tok = self._make_tokenizer(eos_token_id=0)

        # 2-D input_ids shape [1, 3]
        ids_2d = mx.array(np.array([[1, 2, 3]], dtype=np.int32))
        fh, _ = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids_2d, max_new_tokens=2, temperature=0.0, first_token_id=4
        )
        assert isinstance(text, str)

    def test_temperature_sampling_mid_generation(self):
        """temperature > 0 also applies during the subsequent-token loop."""

        model = self._make_model_returns_logits(next_token_id=3)
        tok = self._make_tokenizer(eos_token_id=0)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=3, temperature=0.5, first_token_id=2
        )
        assert count >= 1

    def test_temperature_sampling_loop_non_zero_eos(self):
        """temperature > 0 in subsequent-token loop with eos != categorical result.

        mx.random.categorical always returns 0 in the stub.  Setting eos_token_id
        to 99 (never returned by the stub) means the loop runs to completion
        without an early break, so the softmax+categorical path (lines 310-311)
        is definitely executed and measured.
        """

        model = self._make_model_returns_logits(next_token_id=3, vocab=10, seq_len=4)
        # eos=99 is never returned by the stub's categorical (always returns 0)
        tok = MagicMock()
        tok.eos_token_id = 99
        tok.decode.side_effect = lambda ids, **kw: " ".join(str(i) for i in ids)
        fh, ids = self._make_inputs()

        text, count = self._call(
            model, tok, fh, ids, max_new_tokens=3, temperature=0.7, first_token_id=2
        )
        assert count >= 1
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# inject_residual (async) — additional validation branches (lines 439, 521, …)
# ---------------------------------------------------------------------------


class TestInjectResidualAdditionalValidation:
    """Cover remaining async validation branches: line 439 (too many tokens),
    and impl-level error returns (lines 521, 559, 566, 574, 615)."""

    @pytest.mark.asyncio
    async def test_subspace_tokens_too_many(self, loaded_model_state: Any) -> None:
        """More than 20 subspace_tokens triggers InvalidInput (line 439)."""
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        result = await inject_residual(
            donor_prompt="A",
            recipient_prompt="B",
            layer=0,
            subspace_tokens=[f"tok{i}" for i in range(21)],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "20" in result["message"]

    @pytest.mark.asyncio
    async def test_subspace_tokens_exactly_20_allowed(self, loaded_model_state: Any) -> None:
        """Exactly 20 subspace_tokens passes the count guard and reaches impl."""
        from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual

        fake = {
            "donor_prompt": "A",
            "recipient_prompt": "B",
            "layer": 0,
            "donor_position": -1,
            "recipient_position": -1,
            "subspace_only": False,
            "donor_output": {},
            "recipient_output": {},
            "injected_output": {},
            "comparison": {},
            "residual_similarity": {},
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.inject_residual._inject_residual_impl",
            return_value=fake,
        ):
            result = await inject_residual(
                donor_prompt="A",
                recipient_prompt="B",
                layer=0,
                subspace_tokens=[f"tok{i}" for i in range(20)],
            )
        assert "error" not in result


class TestInjectResidualImplErrorBranches:
    """Test error return paths inside _inject_residual_impl (lines 521, 559, 566, 574, 615)."""

    # Shared minimal plumbing for calling _inject_residual_impl directly.
    def _base_patches(
        self,
        *,
        lm_head_return=MagicMock(),
        resolve_side_effect=None,
        unembed_side_effect=None,
        subspace_registry_fetch=None,
    ):
        """Return a context-manager stack that wires the impl to controllable stubs."""
        import mlx.core as mx

        mock_hidden = mx.array(np.zeros((1, 3, DIM), dtype=np.float32))
        mock_vec = MagicMock()
        mock_vec.tolist.return_value = [0.1] * DIM

        def _fake_decomp(model, config, ids, layers):
            return {
                "embeddings": MagicMock(),
                "hidden_states": {i: mock_hidden for i in range(8)},
                "prev_hidden": {},
                "attn_outputs": {},
                "ffn_outputs": {},
            }

        def _fake_extract(tensor, pos):
            v = MagicMock()
            v.tolist.return_value = [0.1] * DIM
            v.reshape.return_value = v
            return v

        def _fake_norm_project(fn, lm, vec):
            r = MagicMock()
            r.tolist.return_value = [1.0, 2.0, 3.0, 0.5, 0.5]
            return r

        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = MagicMock()

        patches = [
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                side_effect=_fake_decomp,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._extract_position",
                side_effect=_fake_extract,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                side_effect=_fake_norm_project,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=lm_head_return,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
        ]
        if resolve_side_effect is not None:
            patches.append(
                patch(
                    "chuk_mcp_lazarus.tools.geometry.inject_residual._resolve_token_to_id",
                    side_effect=resolve_side_effect,
                )
            )
        if unembed_side_effect is not None:
            patches.append(
                patch(
                    "chuk_mcp_lazarus.tools.geometry.inject_residual._get_unembed_vec_np",
                    side_effect=unembed_side_effect,
                )
            )
        return patches

    def _run_impl(
        self,
        patches_list,
        subspace_only=False,
        subspace_tokens=None,
        subspace_name=None,
        patch_all_positions=False,
    ):
        from chuk_mcp_lazarus.tools.geometry.inject_residual import _inject_residual_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"t{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = DIM
        meta.num_layers = 4

        # Stack context managers
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches_list:
                stack.enter_context(p)
            return _inject_residual_impl(
                model,
                config,
                tokenizer,
                meta,
                "donor prompt",
                "recipient prompt",
                0,  # layer
                None,  # donor_layer
                5,  # max_new_tokens
                0.0,  # temperature
                -1,  # donor_position
                -1,  # recipient_position
                subspace_only,
                subspace_tokens,
                subspace_name,
                patch_all_positions,
            )

    def test_lm_head_none_returns_error(self):
        """When _get_lm_projection returns None, impl returns ExtractionFailed (line 521)."""
        patches = self._base_patches(lm_head_return=None)
        result = self._run_impl(patches)
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    def test_no_valid_unembed_vecs_returns_error(self):
        """All tokens resolve to None unembed — ExtractionFailed (line 565-570)."""
        # _resolve_token_to_id returns an id but _get_unembed_vec_np returns None
        patches = self._base_patches(
            resolve_side_effect=lambda tok_obj, tok: 99,
            unembed_side_effect=lambda model, tid: None,
        )
        result = self._run_impl(
            patches,
            subspace_only=True,
            subspace_tokens=["alpha"],
        )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "No valid unembed vectors" in result["message"]

    def test_all_tokens_resolve_to_none_returns_error(self):
        """All tokens fail to resolve (tid=None) — falls through to no-unembed-vecs error."""
        patches = self._base_patches(
            resolve_side_effect=lambda tok_obj, tok: None,
        )
        result = self._run_impl(
            patches,
            subspace_only=True,
            subspace_tokens=["alpha", "beta"],
        )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    def test_empty_basis_after_gram_schmidt_returns_error(self):
        """Gram-Schmidt produces empty basis — ExtractionFailed (line 573-578)."""
        # Provide a valid unembed vec, but patch _gram_schmidt to return []
        patches = self._base_patches(
            resolve_side_effect=lambda tok_obj, tok: 10,
            unembed_side_effect=lambda model, tid: np.array([1.0] * DIM, dtype=np.float32),
        )
        patches.append(
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual._gram_schmidt",
                return_value=[],
            )
        )
        result = self._run_impl(
            patches,
            subspace_only=True,
            subspace_tokens=["alpha"],
        )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "Gram-Schmidt" in result["message"]

    def test_subspace_name_registry_fetch_none_returns_error(self):
        """SubspaceRegistry.fetch returns None inside impl — VectorNotFound (line 615)."""
        patches = self._base_patches()
        # Patch SubspaceRegistry.fetch to return None
        patches.append(
            patch(
                "chuk_mcp_lazarus.tools.geometry.inject_residual.SubspaceRegistry",
            )
        )
        # Use a fresh approach: patch the registry instance fetch directly
        from unittest.mock import patch as _patch

        mock_reg = MagicMock()
        mock_reg.fetch.return_value = None

        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in self._base_patches():
                stack.enter_context(p)
            stack.enter_context(
                _patch(
                    "chuk_mcp_lazarus.tools.geometry.inject_residual.SubspaceRegistry",
                    **{"get.return_value": mock_reg},
                )
            )
            from chuk_mcp_lazarus.tools.geometry.inject_residual import _inject_residual_impl

            model = MagicMock()
            config = MagicMock()
            tokenizer = MagicMock()
            tokenizer.encode.return_value = [1, 2, 3]
            tokenizer.decode.side_effect = lambda ids, **kw: f"t{ids[0]}"
            meta = MagicMock()
            meta.hidden_dim = DIM
            meta.num_layers = 4

            result = _inject_residual_impl(
                model,
                config,
                tokenizer,
                meta,
                "donor",
                "recipient",
                0,
                None,
                5,
                0.0,
                -1,
                -1,
                False,
                None,
                "ghost_sub",
                False,
            )
        assert result["error"] is True
        assert result["error_type"] == "VectorNotFound"
