"""Tests for tools/probe_tools.py — train_probe, evaluate_probe, etc."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.tools.probe_tools import (
    evaluate_probe,
    list_probes,
    scan_probe_across_layers,
    train_probe,
)


def _make_examples(n: int = 6) -> list[dict]:
    """Generate test examples with alternating labels."""
    return [
        {"prompt": f"prompt_{i}", "label": "a" if i % 2 == 0 else "b"}
        for i in range(n)
    ]


class TestTrainProbe:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await train_probe(
            probe_name="test", layer=0, examples=_make_examples()
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_probe_type(self, loaded_model_state: MagicMock) -> None:
        result = await train_probe(
            probe_name="test", layer=0, examples=_make_examples(), probe_type="invalid"
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await train_probe(
            probe_name="test", layer=99, examples=_make_examples()
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_examples(self, loaded_model_state: MagicMock) -> None:
        result = await train_probe(
            probe_name="test",
            layer=0,
            examples=[{"prompt": "a", "label": "x"}, {"prompt": "b", "label": "y"}],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_missing_keys(self, loaded_model_state: MagicMock) -> None:
        result = await train_probe(
            probe_name="test",
            layer=0,
            examples=[{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}, {"prompt": "d"}],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_single_label(self, loaded_model_state: MagicMock) -> None:
        result = await train_probe(
            probe_name="test",
            layer=0,
            examples=[
                {"prompt": "a", "label": "x"},
                {"prompt": "b", "label": "x"},
                {"prompt": "c", "label": "x"},
                {"prompt": "d", "label": "x"},
            ],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "probe_name": "test",
            "layer": 0,
            "probe_type": "linear",
            "num_examples": 6,
            "classes": ["a", "b"],
            "train_accuracy": 0.95,
            "val_accuracy": 0.90,
        }
        with patch(
            "chuk_mcp_lazarus.tools.probe_tools._train_probe_impl",
            return_value=mock_result,
        ):
            result = await train_probe(
                probe_name="test", layer=0, examples=_make_examples()
            )
        assert "error" not in result
        assert result["probe_name"] == "test"


class TestEvaluateProbe:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await evaluate_probe(
            probe_name="test", examples=[{"prompt": "a", "label": "x"}]
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_probe_not_found(self, loaded_model_state: MagicMock) -> None:
        result = await evaluate_probe(
            probe_name="nonexistent", examples=[{"prompt": "a", "label": "x"}]
        )
        assert result["error"] is True
        assert result["error_type"] == "ProbeNotFound"

    @pytest.mark.asyncio
    async def test_empty_examples(self, loaded_model_state: MagicMock) -> None:
        from chuk_mcp_lazarus.probe_store import ProbeMetadata, ProbeRegistry, ProbeType

        reg = ProbeRegistry.get()
        meta = ProbeMetadata(
            name="eval_test",
            layer=0,
            probe_type=ProbeType.LINEAR,
            classes=["a", "b"],
            num_examples=10,
            train_accuracy=0.9,
            val_accuracy=0.85,
            trained_at="2024-01-01",
        )
        reg.store("eval_test", MagicMock(), meta)

        result = await evaluate_probe(probe_name="eval_test", examples=[])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"


class TestScanProbeAcrossLayers:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await scan_probe_across_layers(
            probe_name_prefix="test", layers=[0], examples=_make_examples()
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_probe_type(self, loaded_model_state: MagicMock) -> None:
        result = await scan_probe_across_layers(
            probe_name_prefix="test",
            layers=[0],
            examples=_make_examples(),
            probe_type="invalid",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await scan_probe_across_layers(
            probe_name_prefix="test", layers=[99], examples=_make_examples()
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_examples(self, loaded_model_state: MagicMock) -> None:
        result = await scan_probe_across_layers(
            probe_name_prefix="test",
            layers=[0],
            examples=[{"prompt": "a", "label": "x"}],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"


class TestListProbes:
    @pytest.mark.asyncio
    async def test_empty(self) -> None:
        result = await list_probes()
        assert "error" not in result
        assert "count" in result
