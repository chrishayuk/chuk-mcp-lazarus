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
