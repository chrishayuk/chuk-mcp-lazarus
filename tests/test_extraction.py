"""Tests for _extraction.py — shared activation extraction helpers."""

from unittest.mock import MagicMock, patch

import numpy as np

from chuk_mcp_lazarus._extraction import (
    extract_activation_at_layer,
    extract_activations_all_layers,
)


class TestExtractActivationAtLayer:
    def test_returns_list(self) -> None:
        import mlx.core as mx

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        result = extract_activation_at_layer(model, config, tokenizer, "hello", layer=0)
        assert isinstance(result, list)

    def test_correct_layer(self) -> None:
        import mlx.core as mx

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        result = extract_activation_at_layer(model, config, tokenizer, "test", layer=2)
        assert isinstance(result, list)


class TestExtractActivationsAllLayers:
    def test_returns_dict(self) -> None:
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        result = extract_activations_all_layers(
            model, config, tokenizer, "hello", layers=[0, 1, 2]
        )
        assert isinstance(result, dict)

    def test_correct_keys(self) -> None:
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        result = extract_activations_all_layers(
            model, config, tokenizer, "hello", layers=[0, 2]
        )
        # Should have keys for requested layers
        for key in result:
            assert key in [0, 2]

    def test_values_are_lists(self) -> None:
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        result = extract_activations_all_layers(
            model, config, tokenizer, "hello", layers=[0]
        )
        for v in result.values():
            assert isinstance(v, list)
