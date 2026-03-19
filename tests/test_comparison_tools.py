"""Tests for tools/comparison_tools.py — two-model comparison tools."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.model_state import ModelMetadata
from chuk_mcp_lazarus.tools.comparison_tools import (
    _require_comparison_models,
    _validate_layers,
    compare_attention,
    compare_generations,
    compare_representations,
    compare_weights,
    load_comparison_model,
    unload_comparison_model,
)


class TestValidateLayers:
    def test_none_returns_all(self) -> None:
        result = _validate_layers(None, 4)
        assert result == [0, 1, 2, 3]

    def test_valid_layers(self) -> None:
        result = _validate_layers([0, 2], 4)
        assert result == [0, 2]

    def test_deduplicates_and_sorts(self) -> None:
        result = _validate_layers([2, 0, 2, 1], 4)
        assert result == [0, 1, 2]

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _validate_layers([99], 4)


class TestRequireComparisonModels:
    def test_primary_not_loaded(self) -> None:
        primary = MagicMock()
        primary.is_loaded = False
        with patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary):
            result = _require_comparison_models("test_tool")
        assert isinstance(result, dict)
        assert result["error_type"] == "ModelNotLoaded"

    def test_comparison_not_loaded(self) -> None:
        primary = MagicMock()
        primary.is_loaded = True
        comp = MagicMock()
        comp.is_loaded = False
        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
        ):
            result = _require_comparison_models("test_tool")
        assert isinstance(result, dict)
        assert result["error_type"] == "ModelNotLoaded"

    def test_incompatible(self) -> None:
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64)
        comp = MagicMock()
        comp.is_loaded = True
        comp.require_compatible.side_effect = ValueError("mismatch")
        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
        ):
            result = _require_comparison_models("test_tool")
        assert isinstance(result, dict)
        assert result["error_type"] == "ComparisonIncompatible"

    def test_success(self) -> None:
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64)
        comp = MagicMock()
        comp.is_loaded = True
        comp.require_compatible.return_value = None
        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
        ):
            result = _require_comparison_models("test_tool")
        assert isinstance(result, tuple)
        assert result[0] is primary
        assert result[1] is comp


class TestLoadComparisonModel:
    @pytest.mark.asyncio
    async def test_invalid_dtype(self) -> None:
        result = await load_comparison_model(model_id="test", dtype="invalid")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        meta = ModelMetadata(
            model_id="comp/model",
            family="test",
            architecture="test_arch",
            num_layers=4,
            hidden_dim=64,
            num_attention_heads=4,
            parameter_count=1000,
        )
        comp = MagicMock()
        comp.load.return_value = meta
        with patch(
            "chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get",
            return_value=comp,
        ):
            result = await load_comparison_model(model_id="comp/model")
        assert "error" not in result
        assert result["model_id"] == "comp/model"
        assert result["status"] == "loaded"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self) -> None:
        """Exception path: comp.load raises -> LoadFailed."""
        comp = MagicMock()
        comp.load.side_effect = RuntimeError("download failed")
        with patch(
            "chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get",
            return_value=comp,
        ):
            result = await load_comparison_model(model_id="bad/model")
        assert result["error"] is True
        assert result["error_type"] == "LoadFailed"
        assert "download failed" in result["message"]


class TestCompareWeights:
    @pytest.mark.asyncio
    async def test_primary_not_loaded(self) -> None:
        primary = MagicMock()
        primary.is_loaded = False
        with patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary):
            result = await compare_weights()
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """compare_weights success path: mock weight_divergence to return data."""
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="primary")
        primary.model = MagicMock()
        primary.config = MagicMock()
        primary.tokenizer = MagicMock()

        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="comp")
        comp.model = MagicMock()
        comp.config = MagicMock()
        comp.tokenizer = MagicMock()
        comp.require_compatible.return_value = None

        mock_divergences = [
            {
                "layer": 0,
                "component": "attn_q",
                "frobenius_norm_diff": 0.01,
                "cosine_similarity": 0.99,
            },
            {
                "layer": 0,
                "component": "attn_k",
                "frobenius_norm_diff": 0.02,
                "cosine_similarity": 0.98,
            },
            {
                "layer": 1,
                "component": "attn_q",
                "frobenius_norm_diff": 0.05,
                "cosine_similarity": 0.95,
            },
        ]

        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
            patch(
                "chuk_mcp_lazarus.tools.comparison.tools.weight_divergence",
                return_value=mock_divergences,
            ),
        ):
            result = await compare_weights(layers=[0, 1])

        assert "error" not in result
        assert result["primary_model"] == "primary"
        assert result["comparison_model"] == "comp"
        assert result["num_layers_compared"] == 2
        assert len(result["divergences"]) == 3
        assert "summary" in result
        assert "top_divergent_layers" in result["summary"]
        assert result["summary"]["total_components"] == 3


class TestCompareRepresentations:
    @pytest.mark.asyncio
    async def test_primary_not_loaded(self) -> None:
        primary = MagicMock()
        primary.is_loaded = False
        with patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary):
            result = await compare_representations(prompts=["a", "b"])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_too_many_prompts(self) -> None:
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64)
        comp = MagicMock()
        comp.is_loaded = True
        comp.require_compatible.return_value = None
        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
        ):
            result = await compare_representations(prompts=["a"] * 9)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """compare_representations success path: mock activation_divergence."""
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="primary")
        primary.model = MagicMock()
        primary.config = MagicMock()
        primary.tokenizer = MagicMock()

        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="comp")
        comp.model = MagicMock()
        comp.config = MagicMock()
        comp.tokenizer = MagicMock()
        comp.require_compatible.return_value = None

        mock_divergences = [
            {
                "layer": 0,
                "prompt": "hello",
                "cosine_similarity": 0.95,
                "l2_distance": 1.2,
                "relative_l2": 0.05,
            },
            {
                "layer": 1,
                "prompt": "hello",
                "cosine_similarity": 0.90,
                "l2_distance": 2.1,
                "relative_l2": 0.10,
            },
        ]

        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
            patch(
                "chuk_mcp_lazarus.tools.comparison.tools.activation_divergence",
                return_value=mock_divergences,
            ),
        ):
            result = await compare_representations(prompts=["hello"], layers=[0, 1])

        assert "error" not in result
        assert result["primary_model"] == "primary"
        assert result["comparison_model"] == "comp"
        assert result["num_prompts"] == 1
        assert result["num_layers_compared"] == 2
        assert len(result["divergences"]) == 2
        assert len(result["layer_averages"]) == 2
        # Check layer averages are computed correctly
        avg_0 = next(la for la in result["layer_averages"] if la["layer"] == 0)
        assert avg_0["avg_cosine_similarity"] == 0.95


class TestCompareAttention:
    @pytest.mark.asyncio
    async def test_primary_not_loaded(self) -> None:
        primary = MagicMock()
        primary.is_loaded = False
        with patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary):
            result = await compare_attention(prompt="hello")
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """compare_attention success path: mock attention_divergence."""
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="primary")
        primary.model = MagicMock()
        primary.config = MagicMock()
        primary.tokenizer = MagicMock()

        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="comp")
        comp.model = MagicMock()
        comp.config = MagicMock()
        comp.tokenizer = MagicMock()
        comp.require_compatible.return_value = None

        mock_divergences = [
            {"layer": 0, "head": 0, "js_divergence": 0.05, "cosine_similarity": 0.98},
            {"layer": 0, "head": 1, "js_divergence": 0.12, "cosine_similarity": 0.92},
            {"layer": 1, "head": 0, "js_divergence": 0.08, "cosine_similarity": 0.96},
        ]

        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
            patch(
                "chuk_mcp_lazarus.tools.comparison.tools.attention_divergence",
                return_value=mock_divergences,
            ),
        ):
            result = await compare_attention(prompt="hello", layers=[0, 1])

        assert "error" not in result
        assert result["primary_model"] == "primary"
        assert result["comparison_model"] == "comp"
        assert result["prompt"] == "hello"
        assert result["num_layers_compared"] == 2
        assert len(result["divergences"]) == 3
        # top_divergent_heads sorted by js_divergence descending
        assert result["top_divergent_heads"][0]["js_divergence"] == 0.12


class TestCompareGenerations:
    @pytest.mark.asyncio
    async def test_primary_not_loaded(self) -> None:
        primary = MagicMock()
        primary.is_loaded = False
        with patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary):
            result = await compare_generations(prompt="hello")
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self) -> None:
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64)
        comp = MagicMock()
        comp.is_loaded = True
        comp.require_compatible.return_value = None
        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
        ):
            result = await compare_generations(prompt="hello", max_new_tokens=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """compare_generations success path: mock _generate.generate_text."""
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="primary")
        primary.model = MagicMock()
        primary.config = MagicMock()
        primary.tokenizer = MagicMock()

        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="comp")
        comp.model = MagicMock()
        comp.config = MagicMock()
        comp.tokenizer = MagicMock()
        comp.require_compatible.return_value = None

        # The function does `from .._generate import generate_text as _gen`
        # Patching on the _generate module works because the local import
        # resolves at call time.
        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
            patch(
                "chuk_mcp_lazarus._generate.generate_text",
                side_effect=[("primary output", 5), ("comp output", 7)],
            ),
        ):
            result = await compare_generations(prompt="hello")

        assert "error" not in result
        assert result["primary_model"] == "primary"
        assert result["comparison_model"] == "comp"
        assert result["prompt"] == "hello"
        assert result["primary_output"] == "primary output"
        assert result["comparison_output"] == "comp output"
        assert result["primary_tokens"] == 5
        assert result["comparison_tokens"] == 7
        assert result["outputs_match"] is False

    @pytest.mark.asyncio
    async def test_success_matching_outputs(self) -> None:
        """compare_generations with matching outputs -> outputs_match=True."""
        primary = MagicMock()
        primary.is_loaded = True
        primary.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="primary")
        primary.model = MagicMock()
        primary.config = MagicMock()
        primary.tokenizer = MagicMock()

        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata = ModelMetadata(num_layers=4, hidden_dim=64, model_id="comp")
        comp.model = MagicMock()
        comp.config = MagicMock()
        comp.tokenizer = MagicMock()
        comp.require_compatible.return_value = None

        with (
            patch("chuk_mcp_lazarus.tools.comparison_tools.ModelState.get", return_value=primary),
            patch("chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get", return_value=comp),
            patch(
                "chuk_mcp_lazarus._generate.generate_text",
                side_effect=[("same output", 5), ("same output", 5)],
            ),
        ):
            result = await compare_generations(prompt="hello")

        assert "error" not in result
        assert result["outputs_match"] is True


class TestUnloadComparisonModel:
    @pytest.mark.asyncio
    async def test_not_loaded(self) -> None:
        comp = MagicMock()
        comp.is_loaded = False
        with patch(
            "chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get",
            return_value=comp,
        ):
            result = await unload_comparison_model()
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata.model_id = "comp/model"
        comp.unload.return_value = None
        with patch(
            "chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get",
            return_value=comp,
        ):
            result = await unload_comparison_model()
        assert "error" not in result
        assert result["model_id"] == "comp/model"
        assert result["status"] == "unloaded"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self) -> None:
        """Exception path: comp.unload raises -> ComparisonFailed."""
        comp = MagicMock()
        comp.is_loaded = True
        comp.metadata.model_id = "comp/model"
        comp.unload.side_effect = RuntimeError("unload boom")
        with patch(
            "chuk_mcp_lazarus.tools.comparison_tools.ComparisonState.get",
            return_value=comp,
        ):
            result = await unload_comparison_model()
        assert result["error"] is True
        assert result["error_type"] == "ComparisonFailed"
        assert "unload boom" in result["message"]
