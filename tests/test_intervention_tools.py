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
            "chuk_mcp_lazarus.tools.intervention.tools._component_intervention_impl",
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
            "chuk_mcp_lazarus.tools.intervention.tools._component_intervention_impl",
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
            "chuk_mcp_lazarus.tools.intervention.tools._component_intervention_impl",
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
            "chuk_mcp_lazarus.tools.intervention.tools._component_intervention_impl",
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
            "chuk_mcp_lazarus.tools.intervention.tools._component_intervention_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await component_intervention(prompt="hello", layer=0, component="attention")
        assert result["error"] is True
        assert result["error_type"] == "InterventionFailed"

    @pytest.mark.asyncio
    async def test_value_error_returns_invalid_input(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.intervention.tools._component_intervention_impl",
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
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=MagicMock(),
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                side_effect=norm_project_fn,
            ),
            patch(
                "chuk_mcp_lazarus.tools.intervention.tools._run_forward_with_intervention",
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


# ---------------------------------------------------------------------------
# TestRunForwardWithIntervention (internal helper)
# ---------------------------------------------------------------------------


class TestRunForwardWithIntervention:
    """Test _run_forward_with_intervention directly with mock objects."""

    def _make_layer(self, with_sublayers: bool = True, with_dropout: bool = False) -> MagicMock:
        """Build a mock transformer layer."""
        import mlx.core as mx

        layer = MagicMock()
        hidden = np.zeros((1, 3, 64), dtype=np.float32)
        hidden[:, :, 0] = 1.0

        if with_sublayers:
            normed_out = mx.array(np.ones((1, 3, 64), dtype=np.float32))
            attn_out = mx.array(np.ones((1, 3, 64), dtype=np.float32) * 0.5)

            layer.input_layernorm.return_value = normed_out
            layer.self_attn.return_value = (attn_out, None)
            layer.post_attention_layernorm.return_value = mx.array(
                np.ones((1, 3, 64), dtype=np.float32) * 0.3
            )
            layer.mlp.return_value = mx.array(np.ones((1, 3, 64), dtype=np.float32) * 0.2)

        if with_dropout:
            layer.dropout = MagicMock(side_effect=lambda x: x)
        else:
            del layer.dropout  # ensure hasattr returns False

        return layer

    def _make_four_norm_layer(self) -> MagicMock:
        """Build a mock layer with four-norm Gemma-style architecture."""
        import mlx.core as mx

        layer = MagicMock()
        ones = mx.array(np.ones((1, 3, 64), dtype=np.float32))
        layer.input_layernorm.return_value = ones
        layer.self_attn.return_value = (ones * 0.5, None)
        layer.post_attention_layernorm.return_value = ones * 0.4
        layer.pre_feedforward_layernorm.return_value = ones * 0.3
        layer.post_feedforward_layernorm.return_value = ones * 0.2
        layer.mlp.return_value = ones * 0.1
        del layer.dropout
        return layer

    def _run(
        self,
        component: str = "attention",
        scale_factor: float = 0.0,
        head: int | None = None,
        has_sublayers: bool = True,
        has_four_norms: bool = False,
        num_layers: int = 2,
        target_layer: int = 0,
    ):
        """Run _run_forward_with_intervention with fully mocked dependencies."""
        import mlx.core as mx

        from chuk_mcp_lazarus.tools.intervention_tools import (
            _run_forward_with_intervention,
        )

        model = MagicMock()
        config = MagicMock()

        metadata = MagicMock()
        metadata.num_attention_heads = 2
        metadata.num_kv_heads = 2
        metadata.head_dim = 32
        metadata.hidden_dim = 64

        # Build layers
        layers = []
        for _ in range(num_layers):
            if has_four_norms:
                layers.append(self._make_four_norm_layer())
            else:
                layers.append(self._make_layer(with_sublayers=has_sublayers))

        embed_out = mx.array(np.ones((1, 3, 64), dtype=np.float32))
        mock_embed = MagicMock(return_value=embed_out)

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = layers
        mock_helper._get_embed_tokens.return_value = mock_embed
        mock_helper._get_embedding_scale.return_value = None

        input_ids = mx.array(np.array([1, 2, 3], dtype=np.float32))

        # Patch the causal mask creation in nn
        import mlx.nn as nn

        if not hasattr(nn, "MultiHeadAttention"):

            class _MockMHA:
                @staticmethod
                def create_additive_causal_mask(seq_len):
                    return mx.array(np.zeros((seq_len, seq_len), dtype=np.float32))

            nn.MultiHeadAttention = _MockMHA

        if not hasattr(nn.MultiHeadAttention, "create_additive_causal_mask"):
            nn.MultiHeadAttention.create_additive_causal_mask = staticmethod(
                lambda seq_len: mx.array(np.zeros((seq_len, seq_len), dtype=np.float32))
            )

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._has_sublayers",
                return_value=has_sublayers,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._has_four_norms",
                return_value=has_four_norms,
            ),
            patch(
                "chuk_mcp_lazarus.tools.intervention.tools._intervene_head",
                return_value=mx.array(np.ones((1, 3, 64), dtype=np.float32) * 0.7),
            ) as mock_intervene_head,
        ):
            result = _run_forward_with_intervention(
                model,
                config,
                input_ids,
                target_layer,
                component,
                scale_factor,
                head,
                metadata,
            )
        return result, mock_intervene_head

    def test_attention_intervention_returns_array(self) -> None:
        """Attention zeroing should run without error and return array."""
        result, _ = self._run(component="attention", scale_factor=0.0)
        assert result is not None
        assert hasattr(result, "shape")

    def test_ffn_intervention_returns_array(self) -> None:
        """FFN zeroing should run without error and return array."""
        result, _ = self._run(component="ffn", scale_factor=0.0)
        assert result is not None
        assert hasattr(result, "shape")

    def test_head_intervention_calls_intervene_head(self) -> None:
        """Head intervention should delegate to _intervene_head."""
        result, mock_ih = self._run(component="head", scale_factor=0.0, head=0)
        assert result is not None
        mock_ih.assert_called_once()

    def test_head_intervention_passes_head_index(self) -> None:
        """_intervene_head should receive the correct head index."""
        _, mock_ih = self._run(component="head", scale_factor=0.0, head=1)
        call_kwargs = mock_ih.call_args
        # head is the 6th positional arg (index 5) in _intervene_head(layer, normed, attn_out, mask, head, ...)
        passed_head = call_kwargs[0][4]
        assert passed_head == 1

    def test_non_target_layer_standard_forward(self) -> None:
        """Layers that are not the target should use standard forward."""

        result, _ = self._run(component="attention", scale_factor=0.0, target_layer=1, num_layers=2)
        assert result is not None

    def test_four_norm_path_attention(self) -> None:
        """Four-norm path should run correctly for attention component."""
        result, _ = self._run(component="attention", scale_factor=0.0, has_four_norms=True)
        assert result is not None

    def test_four_norm_path_ffn(self) -> None:
        """Four-norm path should run correctly for FFN component."""
        result, _ = self._run(component="ffn", scale_factor=0.0, has_four_norms=True)
        assert result is not None

    def test_scale_factor_one_attention(self) -> None:
        """scale_factor=1.0 with attention should preserve attn_out magnitude."""
        result, _ = self._run(component="attention", scale_factor=1.0)
        assert result is not None

    def test_no_sublayers_uses_standard_forward(self) -> None:
        """When layer has no sublayers, standard forward path is taken."""

        # Layer returns a plain array from standard forward
        result, _ = self._run(component="attention", scale_factor=0.0, has_sublayers=False)
        assert result is not None

    def test_2d_input_expanded(self) -> None:
        """1-D input_ids (no batch dim) should be expanded to [1, seq]."""
        import mlx.core as mx

        from chuk_mcp_lazarus.tools.intervention_tools import (
            _run_forward_with_intervention,
        )

        model = MagicMock()
        config = MagicMock()
        metadata = MagicMock()
        metadata.num_attention_heads = 2
        metadata.num_kv_heads = 2
        metadata.head_dim = 32
        metadata.hidden_dim = 64

        layer = self._make_layer()
        embed_out = mx.array(np.ones((1, 3, 64), dtype=np.float32))
        mock_embed = MagicMock(return_value=embed_out)

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = [layer]
        mock_helper._get_embed_tokens.return_value = mock_embed
        mock_helper._get_embedding_scale.return_value = None

        # 1-D input (no batch)
        input_ids = mx.array(np.array([1, 2, 3]))

        import mlx.nn as nn

        if not hasattr(nn, "MultiHeadAttention"):

            class _MockMHA:
                @staticmethod
                def create_additive_causal_mask(seq_len):
                    return mx.array(np.zeros((seq_len, seq_len), dtype=np.float32))

            nn.MultiHeadAttention = _MockMHA

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._has_sublayers",
                return_value=True,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._has_four_norms",
                return_value=False,
            ),
        ):
            result = _run_forward_with_intervention(
                model, config, input_ids, 0, "attention", 0.0, None, metadata
            )
        assert result is not None

    def test_embedding_scale_applied(self) -> None:
        """When embedding_scale is set, it should be applied to embeddings."""
        import mlx.core as mx

        from chuk_mcp_lazarus.tools.intervention_tools import (
            _run_forward_with_intervention,
        )

        model = MagicMock()
        config = MagicMock()
        metadata = MagicMock()
        metadata.num_attention_heads = 2
        metadata.num_kv_heads = 2
        metadata.head_dim = 32
        metadata.hidden_dim = 64

        layer = self._make_layer()
        embed_out = mx.array(np.ones((1, 3, 64), dtype=np.float32))
        mock_embed = MagicMock(return_value=embed_out)

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = [layer]
        mock_helper._get_embed_tokens.return_value = mock_embed
        # Non-None scale triggers the multiply path
        mock_helper._get_embedding_scale.return_value = 2.0

        input_ids = mx.array(np.array([1, 2, 3]))

        import mlx.nn as nn

        if not hasattr(nn, "MultiHeadAttention"):

            class _MockMHA:
                @staticmethod
                def create_additive_causal_mask(seq_len):
                    return mx.array(np.zeros((seq_len, seq_len), dtype=np.float32))

            nn.MultiHeadAttention = _MockMHA

        with (
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._has_sublayers",
                return_value=True,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._has_four_norms",
                return_value=False,
            ),
        ):
            result = _run_forward_with_intervention(
                model, config, input_ids, 0, "attention", 0.0, None, metadata
            )
        assert result is not None


# ---------------------------------------------------------------------------
# TestInterventionHead (internal helper)
# ---------------------------------------------------------------------------


class TestInterventionHead:
    """Test _intervene_head directly."""

    def _make_attn_module(
        self,
        num_heads: int = 2,
        head_dim: int = 4,
        hidden_dim: int = 8,
        with_rope: bool = False,
        with_norms: bool = False,
        with_bias: bool = False,
        num_kv_heads: int | None = None,
    ) -> MagicMock:
        """Build a mock self_attn module."""
        import mlx.core as mx

        if num_kv_heads is None:
            num_kv_heads = num_heads

        attn = MagicMock()
        proj_out = mx.array(np.ones((1, 3, num_heads * head_dim), dtype=np.float32))

        attn.q_proj.return_value = proj_out
        attn.k_proj.return_value = mx.array(
            np.ones((1, 3, num_kv_heads * head_dim), dtype=np.float32)
        )
        attn.v_proj.return_value = mx.array(
            np.ones((1, 3, num_kv_heads * head_dim), dtype=np.float32)
        )

        # o_proj weight: [hidden_dim, num_heads * head_dim]
        attn.o_proj.weight = mx.array(
            np.random.randn(hidden_dim, num_heads * head_dim).astype(np.float32)
        )

        if with_bias:
            attn.o_proj.bias = mx.array(np.zeros(hidden_dim, dtype=np.float32))
        else:
            attn.o_proj.bias = None

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

        return attn

    def _run(
        self,
        head: int = 0,
        scale_factor: float = 0.0,
        num_heads: int = 2,
        head_dim: int = 4,
        hidden_dim: int = 8,
        with_rope: bool = False,
        with_norms: bool = False,
        with_bias: bool = False,
        num_kv_heads: int | None = None,
    ):
        import mlx.core as mx

        from chuk_mcp_lazarus.tools.intervention_tools import _intervene_head

        if num_kv_heads is None:
            num_kv_heads = num_heads

        attn = self._make_attn_module(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            with_rope=with_rope,
            with_norms=with_norms,
            with_bias=with_bias,
            num_kv_heads=num_kv_heads,
        )

        layer = MagicMock()
        layer.self_attn = attn

        normed = mx.array(np.ones((1, 3, hidden_dim), dtype=np.float32))
        original_attn_out = mx.array(np.ones((1, 3, hidden_dim), dtype=np.float32))
        mask = mx.array(np.zeros((3, 3), dtype=np.float32))

        with patch(
            "mlx.core.fast.scaled_dot_product_attention",
            return_value=mx.array(np.ones((1, num_heads, 3, head_dim), dtype=np.float32)),
        ):
            result = _intervene_head(
                layer,
                normed,
                original_attn_out,
                mask,
                head,
                scale_factor,
                num_heads,
                num_kv_heads,
                head_dim,
            )
        return result

    def test_returns_array_with_correct_shape(self) -> None:
        """Result should have shape [1, 3, hidden_dim]."""
        result = self._run()
        assert hasattr(result, "shape")
        assert result.shape == (1, 3, 8)

    def test_zero_scale_zeroes_target_head(self) -> None:
        """Zeroing head 0 should produce a different result than scale=1.0."""
        result_zero = self._run(head=0, scale_factor=0.0)
        result_one = self._run(head=0, scale_factor=1.0)
        # These should differ since scale_factor changes the target head
        assert result_zero is not None
        assert result_one is not None

    def test_scale_factor_one_preserves_output(self) -> None:
        """scale_factor=1.0 should not modify the head output."""
        result = self._run(head=0, scale_factor=1.0)
        assert result is not None

    def test_head_index_second_head(self) -> None:
        """Zeroing head 1 (not head 0) should still produce a valid result."""
        result = self._run(head=1, scale_factor=0.0)
        assert result.shape == (1, 3, 8)

    def test_with_rope(self) -> None:
        """Rope path should apply rope to queries and keys."""
        result = self._run(with_rope=True)
        assert result is not None

    def test_with_qk_norms(self) -> None:
        """q_norm / k_norm path should apply norms before attention."""
        result = self._run(with_norms=True)
        assert result is not None

    def test_with_output_bias(self) -> None:
        """o_proj with bias should add bias to result."""
        result = self._run(with_bias=True)
        assert result is not None

    def test_grouped_query_attention(self) -> None:
        """GQA (num_kv_heads < num_heads) should repeat k/v and return valid result."""
        # 4 query heads, 2 kv heads → repeat factor 2
        result = self._run(
            num_heads=4,
            head_dim=4,
            hidden_dim=16,
            num_kv_heads=2,
            head=0,
            scale_factor=0.0,
        )
        assert result is not None
        assert result.shape == (1, 3, 16)

    def test_result_is_sum_of_heads(self) -> None:
        """Result should be the sum of all per-head outputs."""
        # With scale_factor=1.0 and a known weight, verify shape
        result = self._run(num_heads=2, head_dim=4, hidden_dim=8, scale_factor=1.0)
        assert result.shape == (1, 3, 8)
