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
