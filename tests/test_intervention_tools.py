"""Tests for tools/intervention_tools.py — component_intervention."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.tools.intervention_tools import (
    _component_intervention_impl,
    component_intervention,
)


# ---------------------------------------------------------------------------
# TestComponentIntervention (async tool)
# ---------------------------------------------------------------------------


class TestComponentIntervention:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await component_intervention(prompt="hello", layer=0, component="attention")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_component(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(prompt="hello", layer=0, component="invalid")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_intervention(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(
            prompt="hello", layer=0, component="attention", intervention="invalid"
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_head_without_index(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(prompt="hello", layer=0, component="head")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_head_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(prompt="hello", layer=0, component="head", head=99)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(prompt="hello", layer=99, component="attention")
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(
            prompt="hello", layer=0, component="attention", top_k=0
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_top_k_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await component_intervention(
            prompt="hello", layer=0, component="attention", top_k=51
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success_zero_attn(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "component": "attention",
            "intervention": "zero",
            "scale_factor": 0.0,
            "token_position": -1,
            "token_text": "hello",
            "original_top_k": [],
            "intervened_top_k": [],
            "kl_divergence": 0.5,
            "target_delta": -1.0,
            "original_top1": "world",
            "intervened_top1": "foo",
            "top1_changed": True,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.intervention_tools._component_intervention_impl",
            return_value=mock_result,
        ):
            result = await component_intervention(prompt="hello", layer=0, component="attention")
        assert "error" not in result
        assert result["component"] == "attention"

    @pytest.mark.asyncio
    async def test_success_zero_ffn(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 1,
            "component": "ffn",
            "intervention": "zero",
            "scale_factor": 0.0,
            "token_position": -1,
            "token_text": "hello",
            "original_top_k": [],
            "intervened_top_k": [],
            "kl_divergence": 0.3,
            "target_delta": -0.5,
            "original_top1": "world",
            "intervened_top1": "world",
            "top1_changed": False,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.intervention_tools._component_intervention_impl",
            return_value=mock_result,
        ):
            result = await component_intervention(prompt="hello", layer=1, component="ffn")
        assert "error" not in result
        assert result["component"] == "ffn"

    @pytest.mark.asyncio
    async def test_success_zero_head(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "component": "head",
            "intervention": "zero",
            "scale_factor": 0.0,
            "head": 2,
            "token_position": -1,
            "token_text": "hello",
            "original_top_k": [],
            "intervened_top_k": [],
            "kl_divergence": 0.1,
            "target_delta": -0.2,
            "original_top1": "world",
            "intervened_top1": "world",
            "top1_changed": False,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.intervention_tools._component_intervention_impl",
            return_value=mock_result,
        ):
            result = await component_intervention(prompt="hello", layer=0, component="head", head=2)
        assert "error" not in result
        assert result["component"] == "head"
        assert result["head"] == 2

    @pytest.mark.asyncio
    async def test_success_scale(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "layer": 0,
            "component": "attention",
            "intervention": "scale",
            "scale_factor": 2.0,
            "token_position": -1,
            "token_text": "hello",
            "original_top_k": [],
            "intervened_top_k": [],
            "kl_divergence": 0.2,
            "target_delta": 0.5,
            "original_top1": "world",
            "intervened_top1": "world",
            "top1_changed": False,
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.intervention_tools._component_intervention_impl",
            return_value=mock_result,
        ):
            result = await component_intervention(
                prompt="hello",
                layer=0,
                component="attention",
                intervention="scale",
                scale_factor=2.0,
            )
        assert "error" not in result
        assert result["scale_factor"] == 2.0

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.intervention_tools._component_intervention_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await component_intervention(prompt="hello", layer=0, component="attention")
        assert result["error"] is True
        assert result["error_type"] == "InterventionFailed"

    @pytest.mark.asyncio
    async def test_value_error_returns_invalid_input(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.intervention_tools._component_intervention_impl",
            side_effect=ValueError("bad token"),
        ):
            result = await component_intervention(prompt="hello", layer=0, component="attention")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"


# ---------------------------------------------------------------------------
# TestComponentInterventionImpl (sync helper)
# ---------------------------------------------------------------------------


class TestComponentInterventionImpl:
    """Test _component_intervention_impl directly."""

    def _make_norm_project(self):
        """Create a mock _norm_project that returns vocab-sized logits."""
        import mlx.core as mx

        call_count = [0]

        def _norm_project(final_norm, lm_head, vec):
            logits = np.zeros(100, dtype=np.float32)
            # Clean pass: token 42 has highest logit
            # Intervention pass: token 43 has highest logit (simulates effect)
            if call_count[0] == 0:
                logits[42] = 5.0
                logits[43] = 2.0
            else:
                logits[42] = 2.0
                logits[43] = 5.0
            call_count[0] += 1
            return mx.array(logits)

        return _norm_project

    def _make_norm_project_identical(self):
        """_norm_project that returns identical logits (scale_factor=1.0 test)."""
        import mlx.core as mx

        def _norm_project(final_norm, lm_head, vec):
            logits = np.zeros(100, dtype=np.float32)
            logits[42] = 5.0
            logits[43] = 2.0
            return mx.array(logits)

        return _norm_project

    def _run(
        self,
        component: str = "attention",
        intervention: str = "zero",
        scale_factor: float = 0.0,
        head: int | None = None,
        norm_project_fn=None,
    ) -> dict:
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        metadata = MagicMock()
        metadata.num_layers = 4
        metadata.num_attention_heads = 4
        metadata.num_kv_heads = 4
        metadata.head_dim = 16
        metadata.hidden_dim = 64

        if norm_project_fn is None:
            norm_project_fn = self._make_norm_project()

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=MagicMock(),
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._norm_project",
                side_effect=norm_project_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.intervention_tools._run_forward_with_intervention",
                return_value=MagicMock(
                    ndim=3,
                    shape=(1, 5, 64),
                    __getitem__=lambda self, key: MagicMock(
                        ndim=1,
                        shape=(64,),
                        reshape=lambda *a: MagicMock(),
                    ),
                ),
            ),
        ):
            return _component_intervention_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=1,
                component=component,
                intervention=intervention,
                scale_factor=scale_factor,
                head=head,
                top_k=5,
                token_position=-1,
            )

    def test_output_structure(self) -> None:
        result = self._run()
        assert isinstance(result, dict)
        assert result["prompt"] == "hello"
        assert result["layer"] == 1
        assert result["component"] == "attention"
        assert result["intervention"] == "zero"
        assert "original_top_k" in result
        assert "intervened_top_k" in result
        assert "kl_divergence" in result
        assert "target_delta" in result
        assert "original_top1" in result
        assert "intervened_top1" in result
        assert "top1_changed" in result
        assert "summary" in result

    def test_top_k_entries(self) -> None:
        result = self._run()
        assert len(result["original_top_k"]) == 5
        assert len(result["intervened_top_k"]) == 5
        for entry in result["original_top_k"]:
            assert "token" in entry
            assert "token_id" in entry
            assert "probability" in entry

    def test_kl_non_negative(self) -> None:
        result = self._run()
        assert result["kl_divergence"] >= 0.0

    def test_kl_zero_when_identical(self) -> None:
        """When clean and intervened logits are identical, KL should be ~0."""
        result = self._run(
            intervention="scale",
            scale_factor=1.0,
            norm_project_fn=self._make_norm_project_identical(),
        )
        assert result["kl_divergence"] < 0.01

    def test_top1_changed_flag(self) -> None:
        """When intervention changes the top-1, flag should be True."""
        result = self._run()
        # Our mock changes top-1 from tok42 to tok43
        assert result["top1_changed"] is True

    def test_top1_unchanged(self) -> None:
        """When intervention doesn't change top-1, flag should be False."""
        result = self._run(
            intervention="scale",
            scale_factor=1.0,
            norm_project_fn=self._make_norm_project_identical(),
        )
        assert result["top1_changed"] is False

    def test_target_delta(self) -> None:
        """target_delta should be the change in original top-1's logit."""
        result = self._run()
        # Clean: tok42 = 5.0, Intervention: tok42 = 2.0 → delta = -3.0
        assert abs(result["target_delta"] - (-3.0)) < 0.01

    def test_ffn_component(self) -> None:
        result = self._run(component="ffn")
        assert result["component"] == "ffn"

    def test_head_component(self) -> None:
        result = self._run(component="head", head=2)
        assert result["component"] == "head"
        assert result["head"] == 2

    def test_summary_fields(self) -> None:
        result = self._run()
        summary = result["summary"]
        assert "original_top1_probability" in summary
        assert "intervened_top1_probability" in summary
        assert "probability_change" in summary
        assert "kl_divergence" in summary
        assert "effect_magnitude" in summary
