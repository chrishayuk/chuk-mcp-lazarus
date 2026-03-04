"""Tests for comparison_state.py — ComparisonState singleton."""

from unittest.mock import MagicMock

import pytest

from chuk_mcp_lazarus.comparison_state import ComparisonState
from chuk_mcp_lazarus.model_state import ModelMetadata


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
