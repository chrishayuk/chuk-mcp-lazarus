"""Tests for comparison_state.py — ComparisonState singleton."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.comparison_state import ComparisonState
from chuk_mcp_lazarus.model_state import ModelMetadata, WeightDType


class TestComparisonState:
    def test_singleton(self) -> None:
        a = ComparisonState.get()
        b = ComparisonState.get()
        assert a is b

    def test_not_loaded_initially(self) -> None:
        state = ComparisonState.get()
        assert state.is_loaded is False

    def test_properties_when_not_loaded(self) -> None:
        state = ComparisonState.get()
        assert state.model is None
        assert state.tokenizer is None
        assert state.config is None

    def test_require_loaded_raises(self) -> None:
        state = ComparisonState.get()
        with pytest.raises(ValueError, match="No comparison model"):
            state.require_loaded()

    def test_require_compatible_raises_not_loaded(self) -> None:
        state = ComparisonState.get()
        primary = ModelMetadata(num_layers=4, hidden_dim=64)
        with pytest.raises(ValueError, match="No comparison model"):
            state.require_compatible(primary)

    def test_require_compatible_layer_mismatch(self) -> None:
        state = ComparisonState.get()
        # Force loaded state
        state._state.loaded = True
        state._state.metadata = ModelMetadata(num_layers=8, hidden_dim=64)

        primary = ModelMetadata(num_layers=4, hidden_dim=64)
        with pytest.raises(ValueError, match="Layer count mismatch"):
            state.require_compatible(primary)

    def test_require_compatible_hidden_dim_mismatch(self) -> None:
        state = ComparisonState.get()
        state._state.loaded = True
        state._state.metadata = ModelMetadata(num_layers=4, hidden_dim=128)

        primary = ModelMetadata(num_layers=4, hidden_dim=64)
        with pytest.raises(ValueError, match="Hidden dim mismatch"):
            state.require_compatible(primary)

    def test_require_compatible_passes(self) -> None:
        state = ComparisonState.get()
        state._state.loaded = True
        state._state.metadata = ModelMetadata(num_layers=4, hidden_dim=64)

        primary = ModelMetadata(num_layers=4, hidden_dim=64)
        state.require_compatible(primary)  # Should not raise

    def test_count_parameters(self) -> None:
        model = MagicMock()
        w = MagicMock()
        w.size = 500
        model.parameters.return_value = {"w": w}
        assert ComparisonState._count_parameters(model) == 500

    def test_count_parameters_error(self) -> None:
        model = MagicMock()
        model.parameters.side_effect = RuntimeError("fail")
        assert ComparisonState._count_parameters(model) == 0

    def _make_mock_pipeline(self) -> MagicMock:
        """Build a mock pipeline with properly typed config attributes."""
        pipeline = MagicMock()
        pipeline.config.num_hidden_layers = 34
        pipeline.config.hidden_size = 2560
        pipeline.config.num_attention_heads = 32
        pipeline.config.num_key_value_heads = 8
        pipeline.config.vocab_size = 262208
        pipeline.config.intermediate_size = 10240
        pipeline.config.max_position_embeddings = 8192
        pipeline.config.head_dim = 80
        pipeline.config.model_type = "gemma3"
        pipeline.config.num_local_experts = None
        pipeline.family.family_type.value = "gemma"
        pipeline.model.parameters.return_value = {}
        return pipeline

    def test_load_success(self) -> None:
        """Test load() calls _do_load and returns metadata."""
        state = ComparisonState.get()
        mock_pipeline = self._make_mock_pipeline()

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            metadata = state.load("test/comparison-model", WeightDType.BFLOAT16)

        assert state.is_loaded is True
        assert metadata.model_id == "test/comparison-model"
        assert state.metadata.model_id == "test/comparison-model"
        assert state.model is not None
        assert state.tokenizer is not None
        assert state.config is not None

    def test_load_idempotent(self) -> None:
        """Loading the same model_id twice returns immediately without re-loading."""
        state = ComparisonState.get()
        mock_pipeline = self._make_mock_pipeline()

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            meta1 = state.load("test/comparison-model", WeightDType.BFLOAT16)
            meta2 = state.load("test/comparison-model", WeightDType.BFLOAT16)

        assert meta1.model_id == meta2.model_id
        assert meta1 is meta2  # exact same ModelMetadata object

    def test_extract_metadata(self) -> None:
        """Test _extract_metadata builds ModelMetadata from a mock pipeline."""
        state = ComparisonState.get()

        pipeline = MagicMock()
        pipeline.config.num_hidden_layers = 34
        pipeline.config.hidden_size = 2560
        pipeline.config.num_attention_heads = 32
        pipeline.config.num_key_value_heads = 8
        pipeline.config.vocab_size = 262208
        pipeline.config.intermediate_size = 10240
        pipeline.config.max_position_embeddings = 8192
        pipeline.config.head_dim = 80
        pipeline.config.model_type = "gemma3"
        pipeline.config.num_local_experts = None
        pipeline.family.family_type.value = "gemma"
        pipeline.model.parameters.return_value = {}

        metadata = state._extract_metadata("test/gemma-3-4b", pipeline)

        assert metadata.model_id == "test/gemma-3-4b"
        assert metadata.family == "gemma"
        assert metadata.architecture == "gemma3"
        assert metadata.num_layers == 34
        assert metadata.hidden_dim == 2560
        assert metadata.num_attention_heads == 32
        assert metadata.num_kv_heads == 8
        assert metadata.vocab_size == 262208
        assert metadata.intermediate_size == 10240
        assert metadata.max_position_embeddings == 8192
        assert metadata.head_dim == 80
        assert metadata.is_moe is False
        assert metadata.num_experts is None

    def test_extract_metadata_no_family_info(self) -> None:
        """Test _extract_metadata when pipeline.family is None."""
        state = ComparisonState.get()

        pipeline = MagicMock()
        pipeline.config.num_hidden_layers = 4
        pipeline.config.hidden_size = 64
        pipeline.config.num_attention_heads = 4
        pipeline.config.num_key_value_heads = 4
        pipeline.config.vocab_size = 100
        pipeline.config.intermediate_size = 256
        pipeline.config.max_position_embeddings = 512
        pipeline.config.head_dim = 16
        pipeline.config.model_type = "unknown"
        pipeline.config.num_local_experts = None
        pipeline.family = None
        pipeline.model.parameters.return_value = {}

        metadata = state._extract_metadata("test/unknown", pipeline)
        assert metadata.family == "unknown"

    def test_unload_after_load(self) -> None:
        """Load a model, then unload it, verify is_loaded becomes False."""
        state = ComparisonState.get()
        mock_pipeline = self._make_mock_pipeline()

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            state.load("test/comparison-model", WeightDType.BFLOAT16)
        assert state.is_loaded is True

        state.unload()
        assert state.is_loaded is False
        assert state.model is None
        assert state.tokenizer is None
        assert state.config is None
        assert state.metadata.model_id == ""

    def test_unload_when_not_loaded(self) -> None:
        """Unloading when nothing is loaded should be a no-op (no error)."""
        state = ComparisonState.get()
        assert state.is_loaded is False

        # Should not raise
        state.unload()
        assert state.is_loaded is False
