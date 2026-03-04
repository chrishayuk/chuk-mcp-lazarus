"""Tests for model_state.py — ModelState singleton."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.model_state import (
    LoadModelResult,
    ModelMetadata,
    ModelState,
    WeightDType,
)


class TestWeightDType:
    def test_values(self) -> None:
        assert WeightDType.BFLOAT16 == "bfloat16"
        assert WeightDType.FLOAT16 == "float16"
        assert WeightDType.FLOAT32 == "float32"

    def test_from_string(self) -> None:
        assert WeightDType("bfloat16") == WeightDType.BFLOAT16


class TestModelMetadata:
    def test_defaults(self) -> None:
        meta = ModelMetadata()
        assert meta.model_id == ""
        assert meta.num_layers == 0
        assert meta.is_moe is False
        assert meta.num_experts is None

    def test_frozen(self) -> None:
        meta = ModelMetadata(model_id="test")
        with pytest.raises(Exception):
            meta.model_id = "other"  # type: ignore[misc]

    def test_model_dump(self) -> None:
        meta = ModelMetadata(model_id="test/model", num_layers=4)
        d = meta.model_dump()
        assert d["model_id"] == "test/model"
        assert d["num_layers"] == 4

    def test_custom_values(self) -> None:
        meta = ModelMetadata(
            model_id="google/gemma-3-4b-it",
            family="gemma",
            num_layers=34,
            hidden_dim=2560,
            num_attention_heads=32,
            vocab_size=262208,
            is_moe=True,
            num_experts=8,
        )
        assert meta.family == "gemma"
        assert meta.is_moe is True
        assert meta.num_experts == 8


class TestLoadModelResult:
    def test_defaults(self) -> None:
        r = LoadModelResult(
            model_id="test",
            family="test",
            architecture="test",
            num_layers=4,
            hidden_dim=64,
            num_attention_heads=4,
            parameter_count=1000,
        )
        assert r.status == "loaded"


class TestModelState:
    def test_singleton(self) -> None:
        a = ModelState.get()
        b = ModelState.get()
        assert a is b

    def test_not_loaded_initially(self) -> None:
        state = ModelState.get()
        assert state.is_loaded is False

    def test_properties_when_not_loaded(self) -> None:
        state = ModelState.get()
        assert state.model is None
        assert state.tokenizer is None
        assert state.config is None
        assert state.metadata.model_id == ""

    def test_require_loaded_raises(self) -> None:
        state = ModelState.get()
        with pytest.raises(ValueError, match="No model loaded"):
            state.require_loaded()

    def test_count_parameters(self) -> None:
        model = MagicMock()
        w1 = MagicMock()
        w1.size = 100
        w2 = MagicMock()
        w2.size = 200
        model.parameters.return_value = {"w1": w1, "nested": {"w2": w2}}
        assert ModelState._count_parameters(model) == 300

    def test_count_parameters_error(self) -> None:
        model = MagicMock()
        model.parameters.side_effect = RuntimeError("fail")
        assert ModelState._count_parameters(model) == 0

    def test_count_parameters_nested_dict(self) -> None:
        model = MagicMock()
        w = MagicMock()
        w.size = 50
        model.parameters.return_value = {"a": {"b": {"c": w}}}
        assert ModelState._count_parameters(model) == 50

    def _make_mock_pipeline(self) -> MagicMock:
        """Build a mock pipeline with properly typed config attributes."""
        pipeline = MagicMock()
        pipeline.config.num_hidden_layers = 4
        pipeline.config.hidden_size = 64
        pipeline.config.num_attention_heads = 4
        pipeline.config.num_key_value_heads = 4
        pipeline.config.vocab_size = 100
        pipeline.config.intermediate_size = 256
        pipeline.config.max_position_embeddings = 512
        pipeline.config.head_dim = 16
        pipeline.config.model_type = "test_arch"
        pipeline.config.num_local_experts = None
        pipeline.family.family_type.value = "test"
        pipeline.model.parameters.return_value = {}
        return pipeline

    def test_load_success(self) -> None:
        """Test load() calls _do_load and returns metadata."""
        state = ModelState.get()
        mock_pipeline = self._make_mock_pipeline()

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            metadata = state.load("test/model", WeightDType.BFLOAT16)

        assert state.is_loaded is True
        assert metadata.model_id == "test/model"
        assert state.metadata.model_id == "test/model"
        assert state.model is not None
        assert state.tokenizer is not None
        assert state.config is not None

    def test_load_idempotent(self) -> None:
        """Loading the same model_id twice returns immediately without re-loading."""
        state = ModelState.get()
        mock_pipeline = self._make_mock_pipeline()

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            meta1 = state.load("test/model", WeightDType.BFLOAT16)
            meta2 = state.load("test/model", WeightDType.BFLOAT16)

        assert meta1.model_id == meta2.model_id
        assert meta1 is meta2  # exact same ModelMetadata object

    def test_extract_metadata(self) -> None:
        """Test _extract_metadata builds ModelMetadata from a mock pipeline."""
        state = ModelState.get()

        # Build a mock pipeline with config and family
        pipeline = MagicMock()
        pipeline.config.num_hidden_layers = 12
        pipeline.config.hidden_size = 768
        pipeline.config.num_attention_heads = 12
        pipeline.config.num_key_value_heads = 4
        pipeline.config.vocab_size = 32000
        pipeline.config.intermediate_size = 3072
        pipeline.config.max_position_embeddings = 2048
        pipeline.config.head_dim = 64
        pipeline.config.model_type = "llama"
        pipeline.config.num_local_experts = None
        pipeline.family.family_type.value = "llama"
        pipeline.model.parameters.return_value = {}

        metadata = state._extract_metadata("test/llama", pipeline)

        assert metadata.model_id == "test/llama"
        assert metadata.family == "llama"
        assert metadata.architecture == "llama"
        assert metadata.num_layers == 12
        assert metadata.hidden_dim == 768
        assert metadata.num_attention_heads == 12
        assert metadata.num_kv_heads == 4
        assert metadata.vocab_size == 32000
        assert metadata.intermediate_size == 3072
        assert metadata.max_position_embeddings == 2048
        assert metadata.head_dim == 64
        assert metadata.is_moe is False
        assert metadata.num_experts is None

    def test_extract_metadata_moe(self) -> None:
        """Test _extract_metadata detects MoE models."""
        state = ModelState.get()

        pipeline = MagicMock()
        pipeline.config.num_hidden_layers = 8
        pipeline.config.hidden_size = 512
        pipeline.config.num_attention_heads = 8
        pipeline.config.num_key_value_heads = 8
        pipeline.config.vocab_size = 10000
        pipeline.config.intermediate_size = 2048
        pipeline.config.max_position_embeddings = 1024
        pipeline.config.head_dim = 64
        pipeline.config.model_type = "mixtral"
        pipeline.config.num_local_experts = 8
        pipeline.family.family_type.value = "mixtral"
        pipeline.model.parameters.return_value = {}

        metadata = state._extract_metadata("test/mixtral", pipeline)

        assert metadata.is_moe is True
        assert metadata.num_experts == 8

    def test_extract_metadata_no_family_info(self) -> None:
        """Test _extract_metadata when pipeline.family is None."""
        state = ModelState.get()

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
