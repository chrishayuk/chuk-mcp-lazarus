"""Tests for tools/direction_tools.py — direction extraction tools."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.steering_store import SteeringVectorRegistry
from chuk_mcp_lazarus.tools.direction_tools import (
    _extract_direction_impl,
    extract_direction,
)


# ---------------------------------------------------------------------------
# Deterministic activation vectors for _impl tests
# ---------------------------------------------------------------------------

# Positive vectors: strong signal in first dimension (+1.0)
_POS_VEC_A = [1.0] + [0.0] * 63
_POS_VEC_B = [1.0] + [0.1] * 63

# Negative vectors: strong signal in first dimension (-1.0)
_NEG_VEC_A = [-1.0] + [0.0] * 63
_NEG_VEC_B = [-1.0] + [0.1] * 63


def _make_mock_extract(pos_vecs: list, neg_vecs: list):
    """Return a side_effect function that yields pos then neg vectors in order."""
    all_vecs = pos_vecs + neg_vecs
    call_count = {"n": 0}

    def _mock(model, config, tokenizer, prompt, layer, token_position):
        idx = call_count["n"]
        call_count["n"] += 1
        return all_vecs[idx]

    return _mock


# ---------------------------------------------------------------------------
# TestExtractDirection — async tool-level tests
# ---------------------------------------------------------------------------


class TestExtractDirection:
    """Tests for the extract_direction async tool wrapper."""

    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await extract_direction(
            direction_name="test",
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await extract_direction(
            direction_name="test",
            layer=99,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: MagicMock) -> None:
        """Negative layer index should be rejected (layer < 0)."""
        result = await extract_direction(
            direction_name="test",
            layer=-1,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_method(self, loaded_model_state: MagicMock) -> None:
        result = await extract_direction(
            direction_name="test",
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
            method="nonexistent_method",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "nonexistent_method" in result["message"]

    @pytest.mark.asyncio
    async def test_too_few_positive_prompts(self, loaded_model_state: MagicMock) -> None:
        """At least 2 positive prompts are required."""
        result = await extract_direction(
            direction_name="test",
            layer=0,
            positive_prompts=["only_one"],
            negative_prompts=["c", "d"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "positive" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_too_few_negative_prompts(self, loaded_model_state: MagicMock) -> None:
        """At least 2 negative prompts are required."""
        result = await extract_direction(
            direction_name="test",
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["only_one"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "negative" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        """When _extract_direction_impl returns a result, tool returns it."""
        mock_result = {
            "direction_name": "happy_sad",
            "layer": 2,
            "method": "diff_means",
            "separation_score": 2.5,
            "accuracy": 1.0,
            "mean_projection_positive": 0.8,
            "mean_projection_negative": -0.8,
            "positive_label": "happy",
            "negative_label": "sad",
            "vector_norm": 1.414,
            "num_positive": 2,
            "num_negative": 2,
            "stored_as_steering_vector": True,
        }
        with patch(
            "chuk_mcp_lazarus.tools.steering.tools._extract_direction_impl",
            return_value=mock_result,
        ):
            result = await extract_direction(
                direction_name="happy_sad",
                layer=2,
                positive_prompts=["I am happy", "Joy fills me"],
                negative_prompts=["I am sad", "Sorrow fills me"],
                positive_label="happy",
                negative_label="sad",
            )
        assert "error" not in result
        assert result["direction_name"] == "happy_sad"
        assert result["layer"] == 2
        assert result["method"] == "diff_means"

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state: MagicMock) -> None:
        """When _extract_direction_impl raises, tool returns ExtractionFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.steering.tools._extract_direction_impl",
            side_effect=RuntimeError("direction boom"),
        ):
            result = await extract_direction(
                direction_name="test",
                layer=0,
                positive_prompts=["a", "b"],
                negative_prompts=["c", "d"],
            )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "direction boom" in result["message"]


# ---------------------------------------------------------------------------
# TestExtractDirectionImpl — direct _impl function tests
# ---------------------------------------------------------------------------


class TestExtractDirectionImpl:
    """Tests for _extract_direction_impl (sync implementation)."""

    def _run_impl(
        self,
        direction_name: str = "test_dir",
        layer: int = 2,
        positive_prompts: list[str] | None = None,
        negative_prompts: list[str] | None = None,
        method: str = "diff_means",
        positive_label: str = "positive",
        negative_label: str = "negative",
        token_position: int = -1,
        pos_vecs: list | None = None,
        neg_vecs: list | None = None,
    ) -> dict:
        """Helper to run _extract_direction_impl with mocked extraction."""
        if positive_prompts is None:
            positive_prompts = ["pos1", "pos2"]
        if negative_prompts is None:
            negative_prompts = ["neg1", "neg2"]
        if pos_vecs is None:
            pos_vecs = [_POS_VEC_A, _POS_VEC_B]
        if neg_vecs is None:
            neg_vecs = [_NEG_VEC_A, _NEG_VEC_B]

        mock_extract = _make_mock_extract(pos_vecs, neg_vecs)

        with patch(
            "chuk_mcp_lazarus.tools.steering.tools.extract_activation_at_layer",
            side_effect=mock_extract,
        ):
            return _extract_direction_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                direction_name=direction_name,
                layer=layer,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                method=method,
                positive_label=positive_label,
                negative_label=negative_label,
                token_position=token_position,
            )

    def test_returns_expected_keys(self) -> None:
        """Result dict contains all keys from DirectionResult model."""
        result = self._run_impl()
        expected_keys = {
            "direction_name",
            "layer",
            "method",
            "separation_score",
            "accuracy",
            "mean_projection_positive",
            "mean_projection_negative",
            "positive_label",
            "negative_label",
            "vector_norm",
            "num_positive",
            "num_negative",
            "stored_as_steering_vector",
        }
        assert set(result.keys()) == expected_keys

    def test_diff_means_direction(self) -> None:
        """Diff-means with known vectors produces correct direction and metadata."""
        result = self._run_impl(direction_name="dm_test", layer=1)

        assert result["direction_name"] == "dm_test"
        assert result["layer"] == 1
        assert result["method"] == "diff_means"
        assert result["num_positive"] == 2
        assert result["num_negative"] == 2
        # Positive mean first dim = +1.0, Negative mean first dim = -1.0
        # So the direction should have a strong positive first component
        assert result["mean_projection_positive"] > result["mean_projection_negative"]
        assert result["stored_as_steering_vector"] is True

    def test_stores_in_registry(self) -> None:
        """After _impl runs, the direction is stored in SteeringVectorRegistry."""
        self._run_impl(direction_name="registry_test", layer=0)

        reg = SteeringVectorRegistry.get()
        assert reg.exists("registry_test")
        entry = reg.fetch("registry_test")
        assert entry is not None
        vector, meta = entry
        assert meta.name == "registry_test"
        assert meta.layer == 0
        assert meta.num_positive == 2
        assert meta.num_negative == 2
        assert vector.shape == (64,)

    def test_separation_score_is_float(self) -> None:
        """Separation score (Cohen's d) should be a float."""
        result = self._run_impl()
        assert isinstance(result["separation_score"], float)
        # With clearly separable vectors, separation should be positive
        assert result["separation_score"] > 0.0

    def test_vector_norm_positive(self) -> None:
        """Vector norm should be a positive float."""
        result = self._run_impl()
        assert isinstance(result["vector_norm"], float)
        assert result["vector_norm"] > 0.0

    def test_accuracy_in_range(self) -> None:
        """Accuracy should be between 0 and 1."""
        result = self._run_impl()
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_labels_passed_through(self) -> None:
        """Custom labels appear in the result."""
        result = self._run_impl(
            positive_label="English",
            negative_label="French",
        )
        assert result["positive_label"] == "English"
        assert result["negative_label"] == "French"

    def test_registry_metadata_has_correct_norm(self) -> None:
        """Stored VectorMetadata.vector_norm matches result vector_norm."""
        result = self._run_impl(direction_name="norm_check")

        reg = SteeringVectorRegistry.get()
        entry = reg.fetch("norm_check")
        assert entry is not None
        _, meta = entry
        assert meta.vector_norm == pytest.approx(result["vector_norm"], abs=1e-3)

    def test_high_separation_with_clear_clusters(self) -> None:
        """Clearly separable clusters yield high separation score and accuracy."""
        # Positive vectors: all +5.0 in dim 0
        pos = [[5.0] + [0.0] * 63, [5.0] + [0.0] * 63]
        # Negative vectors: all -5.0 in dim 0
        neg = [[-5.0] + [0.0] * 63, [-5.0] + [0.0] * 63]

        result = self._run_impl(
            direction_name="clear_clusters",
            pos_vecs=pos,
            neg_vecs=neg,
        )
        # With perfect separation, accuracy should be 1.0
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-4)
        # Separation score should be very high
        assert result["separation_score"] > 1.0
