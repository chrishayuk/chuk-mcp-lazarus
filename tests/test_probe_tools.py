"""Tests for tools/probe_tools.py — train_probe, evaluate_probe, etc."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.tools.probe_tools import (
    _coefficients_norm,
    _evaluate_probe_impl,
    _probe_at_inference_impl,
    _scan_probe_impl,
    _train_probe_impl,
    _train_sklearn_probe,
    evaluate_probe,
    list_probes,
    probe_at_inference,
    scan_probe_across_layers,
    train_probe,
)
from chuk_mcp_lazarus.probe_store import ProbeMetadata, ProbeRegistry, ProbeType


def _make_examples(n: int = 6) -> list[dict]:
    """Generate test examples with alternating labels."""
    return [{"prompt": f"prompt_{i}", "label": "a" if i % 2 == 0 else "b"} for i in range(n)]


# ---------------------------------------------------------------------------
# _train_sklearn_probe (pure function tests)
# ---------------------------------------------------------------------------


class TestTrainSklearnProbe:
    def test_linear_probe(self) -> None:
        """Train a linear probe on small data and verify return tuple."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 4).astype(np.float32)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        clf, train_acc, val_acc = _train_sklearn_probe(X, y, ProbeType.LINEAR)

        assert hasattr(clf, "predict"), "Classifier must have a predict method"
        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= val_acc <= 1.0
        # With 10 examples and 5 per class, n_folds=min(5,5,10)=5 so CV runs
        preds = clf.predict(X)
        assert preds.shape == (10,)

    def test_mlp_probe(self) -> None:
        """Train an MLP probe on small data and verify return tuple."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 4).astype(np.float32)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        clf, train_acc, val_acc = _train_sklearn_probe(X, y, ProbeType.MLP)

        assert hasattr(clf, "predict"), "Classifier must have a predict method"
        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= val_acc <= 1.0

    def test_small_dataset_fallback(self) -> None:
        """When n_folds < 2, val_accuracy falls back to train_accuracy."""
        # 1 example per class -> min_class_count=1, n_folds=min(5,1,2)=1 < 2
        X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        y = np.array([0, 1])

        clf, train_acc, val_acc = _train_sklearn_probe(X, y, ProbeType.LINEAR)

        # Fallback: val_accuracy = train_accuracy
        assert val_acc == train_acc

    def test_returns_fitted_model(self) -> None:
        """The returned clf should be fitted on the full dataset."""
        rng = np.random.RandomState(123)
        X = rng.randn(20, 4).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)

        clf, _, _ = _train_sklearn_probe(X, y, ProbeType.LINEAR)

        # A fitted LogisticRegression has coef_
        assert hasattr(clf, "coef_")
        assert clf.coef_.shape[1] == 4


# ---------------------------------------------------------------------------
# _coefficients_norm
# ---------------------------------------------------------------------------


class TestCoefficientsNorm:
    def test_with_coef_attr(self) -> None:
        """When clf has coef_, return its L2 norm."""
        mock_clf = MagicMock()
        mock_clf.coef_ = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = _coefficients_norm(mock_clf)
        assert result is not None
        expected = float(np.linalg.norm(mock_clf.coef_))
        assert result == pytest.approx(expected)

    def test_without_coef_attr(self) -> None:
        """When clf lacks coef_, return None."""
        mock_clf = MagicMock(spec=[])  # empty spec -> no coef_
        result = _coefficients_norm(mock_clf)
        assert result is None


# ---------------------------------------------------------------------------
# train_probe
# ---------------------------------------------------------------------------


class TestTrainProbe:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await train_probe(probe_name="test", layer=0, examples=_make_examples())
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
        result = await train_probe(probe_name="test", layer=99, examples=_make_examples())
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
            "chuk_mcp_lazarus.tools.probe.tools._train_probe_impl",
            return_value=mock_result,
        ):
            result = await train_probe(probe_name="test", layer=0, examples=_make_examples())
        assert "error" not in result
        assert result["probe_name"] == "test"

    @pytest.mark.asyncio
    async def test_exception_returns_training_failed(self, loaded_model_state: MagicMock) -> None:
        """When _train_probe_impl raises, train_probe returns TrainingFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._train_probe_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await train_probe(probe_name="test", layer=0, examples=_make_examples())
        assert result["error"] is True
        assert result["error_type"] == "TrainingFailed"
        assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# evaluate_probe
# ---------------------------------------------------------------------------


class TestEvaluateProbe:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await evaluate_probe(probe_name="test", examples=[{"prompt": "a", "label": "x"}])
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

    @pytest.mark.asyncio
    async def test_missing_label_key(self, loaded_model_state: MagicMock) -> None:
        """Examples missing 'label' key should return InvalidInput."""
        reg = ProbeRegistry.get()
        meta = ProbeMetadata(
            name="eval_missing",
            layer=0,
            probe_type=ProbeType.LINEAR,
            classes=["a", "b"],
            num_examples=10,
            train_accuracy=0.9,
            val_accuracy=0.85,
            trained_at="2024-01-01",
        )
        reg.store("eval_missing", MagicMock(), meta)

        result = await evaluate_probe(
            probe_name="eval_missing",
            examples=[{"prompt": "hello"}],  # missing "label"
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "label" in result["message"]

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        """evaluate_probe success path with mocked _evaluate_probe_impl."""
        reg = ProbeRegistry.get()
        meta = ProbeMetadata(
            name="eval_ok",
            layer=0,
            probe_type=ProbeType.LINEAR,
            classes=["a", "b"],
            num_examples=10,
            train_accuracy=0.9,
            val_accuracy=0.85,
            trained_at="2024-01-01",
        )
        reg.store("eval_ok", MagicMock(), meta)

        mock_result = {
            "probe_name": "eval_ok",
            "layer": 0,
            "accuracy": 1.0,
            "per_class_accuracy": {"a": 1.0, "b": 1.0},
            "confusion_matrix": [[1, 0], [0, 1]],
            "predictions": [
                {
                    "prompt": "hello",
                    "true_label": "a",
                    "predicted_label": "a",
                    "correct": True,
                    "confidence": 0.95,
                },
                {
                    "prompt": "world",
                    "true_label": "b",
                    "predicted_label": "b",
                    "correct": True,
                    "confidence": 0.90,
                },
            ],
        }
        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._evaluate_probe_impl",
            return_value=mock_result,
        ):
            result = await evaluate_probe(
                probe_name="eval_ok",
                examples=[
                    {"prompt": "hello", "label": "a"},
                    {"prompt": "world", "label": "b"},
                ],
            )
        assert "error" not in result
        assert result["probe_name"] == "eval_ok"
        assert result["accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_exception_returns_evaluation_failed(self, loaded_model_state: MagicMock) -> None:
        """When _evaluate_probe_impl raises, evaluate_probe returns EvaluationFailed."""
        reg = ProbeRegistry.get()
        meta = ProbeMetadata(
            name="eval_exc",
            layer=0,
            probe_type=ProbeType.LINEAR,
            classes=["a", "b"],
            num_examples=10,
            train_accuracy=0.9,
            val_accuracy=0.85,
            trained_at="2024-01-01",
        )
        reg.store("eval_exc", MagicMock(), meta)

        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._evaluate_probe_impl",
            side_effect=RuntimeError("eval boom"),
        ):
            result = await evaluate_probe(
                probe_name="eval_exc",
                examples=[{"prompt": "hello", "label": "a"}],
            )
        assert result["error"] is True
        assert result["error_type"] == "EvaluationFailed"
        assert "eval boom" in result["message"]


# ---------------------------------------------------------------------------
# scan_probe_across_layers
# ---------------------------------------------------------------------------


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

    @pytest.mark.asyncio
    async def test_missing_keys(self, loaded_model_state: MagicMock) -> None:
        """Examples missing 'label' key should return InvalidInput."""
        result = await scan_probe_across_layers(
            probe_name_prefix="test",
            layers=[0],
            examples=[
                {"prompt": "a", "label": "x"},
                {"prompt": "b"},  # missing label
                {"prompt": "c", "label": "y"},
                {"prompt": "d", "label": "x"},
            ],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "label" in result["message"]

    @pytest.mark.asyncio
    async def test_single_label(self, loaded_model_state: MagicMock) -> None:
        """All examples with the same label should return InvalidInput."""
        result = await scan_probe_across_layers(
            probe_name_prefix="test",
            layers=[0],
            examples=[
                {"prompt": "a", "label": "same"},
                {"prompt": "b", "label": "same"},
                {"prompt": "c", "label": "same"},
                {"prompt": "d", "label": "same"},
            ],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "2 distinct labels" in result["message"]

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        """scan_probe_across_layers success path with mocked _scan_probe_impl."""
        mock_result = {
            "probe_name_prefix": "scan_test",
            "layers_scanned": [0, 1, 2],
            "accuracy_by_layer": [
                {"layer": 0, "train_accuracy": 0.6, "val_accuracy": 0.5},
                {"layer": 1, "train_accuracy": 0.8, "val_accuracy": 0.75},
                {"layer": 2, "train_accuracy": 0.95, "val_accuracy": 0.9},
            ],
            "peak_layer": 2,
            "peak_val_accuracy": 0.9,
            "crossover_layer": 2,
            "interpretation": "Feature becomes linearly decodable at layer 2.",
        }
        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._scan_probe_impl",
            return_value=mock_result,
        ):
            result = await scan_probe_across_layers(
                probe_name_prefix="scan_test",
                layers=[0, 1, 2],
                examples=_make_examples(),
            )
        assert "error" not in result
        assert result["probe_name_prefix"] == "scan_test"
        assert result["peak_layer"] == 2
        assert len(result["accuracy_by_layer"]) == 3

    @pytest.mark.asyncio
    async def test_exception_returns_training_failed(self, loaded_model_state: MagicMock) -> None:
        """When _scan_probe_impl raises, scan returns TrainingFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._scan_probe_impl",
            side_effect=RuntimeError("scan boom"),
        ):
            result = await scan_probe_across_layers(
                probe_name_prefix="test",
                layers=[0, 1],
                examples=_make_examples(),
            )
        assert result["error"] is True
        assert result["error_type"] == "TrainingFailed"
        assert "scan boom" in result["message"]


# ---------------------------------------------------------------------------
# list_probes
# ---------------------------------------------------------------------------


class TestListProbes:
    @pytest.mark.asyncio
    async def test_empty(self) -> None:
        result = await list_probes()
        assert "error" not in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_with_stored_probes(self) -> None:
        """list_probes should return metadata for stored probes."""
        reg = ProbeRegistry.get()
        for name, layer in [("probe_a", 0), ("probe_b", 2)]:
            meta = ProbeMetadata(
                name=name,
                layer=layer,
                probe_type=ProbeType.LINEAR,
                classes=["x", "y"],
                num_examples=8,
                train_accuracy=0.9,
                val_accuracy=0.85,
                trained_at="2024-06-01",
            )
            reg.store(name, MagicMock(), meta)

        result = await list_probes()
        assert "error" not in result
        assert result["count"] == 2
        assert len(result["probes"]) == 2
        names = {p["name"] for p in result["probes"]}
        assert names == {"probe_a", "probe_b"}

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self) -> None:
        """When registry.dump() raises, list_probes returns ExtractionFailed."""
        with patch.object(ProbeRegistry, "get", side_effect=RuntimeError("registry exploded")):
            result = await list_probes()
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "registry exploded" in result["message"]


# ---------------------------------------------------------------------------
# _train_probe_impl (sync implementation)
# ---------------------------------------------------------------------------

DIM = 64
_EXTRACTION_MOD = "chuk_mcp_lazarus.tools.probe.tools"


def _fake_activation(dim: int = DIM) -> list[float]:
    """Return a deterministic fake activation vector."""
    rng = np.random.RandomState(abs(hash("fake")) % (2**31))
    return rng.randn(dim).tolist()


def _make_labeled_examples(n: int = 6) -> tuple[list[dict], list[str], list[str]]:
    """Build examples list, labels_raw, and sorted classes for impl tests."""
    examples = [
        {"prompt": f"prompt_{i}", "label": "pos" if i % 2 == 0 else "neg"} for i in range(n)
    ]
    labels_raw = [ex["label"] for ex in examples]
    classes = sorted(set(labels_raw))
    return examples, labels_raw, classes


class TestTrainProbeImpl:
    """Unit tests for _train_probe_impl (lines 245-300)."""

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_returns_expected_keys(self, mock_extract: MagicMock) -> None:
        """Result dict has all TrainProbeResult keys."""
        mock_extract.return_value = _fake_activation()
        examples, labels_raw, classes = _make_labeled_examples(8)
        model, config, tokenizer = MagicMock(), MagicMock(), MagicMock()

        result = _train_probe_impl(
            model,
            config,
            tokenizer,
            "test_probe",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        expected_keys = {
            "probe_name",
            "layer",
            "probe_type",
            "num_examples",
            "classes",
            "train_accuracy",
            "val_accuracy",
            "coefficients_norm",
        }
        assert expected_keys == set(result.keys())

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_correct_metadata_values(self, mock_extract: MagicMock) -> None:
        """Verify probe_name, layer, probe_type, num_examples, classes."""
        mock_extract.return_value = _fake_activation()
        examples, labels_raw, classes = _make_labeled_examples(6)
        model, config, tokenizer = MagicMock(), MagicMock(), MagicMock()

        result = _train_probe_impl(
            model,
            config,
            tokenizer,
            "my_probe",
            2,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["probe_name"] == "my_probe"
        assert result["layer"] == 2
        assert result["probe_type"] == "linear"
        assert result["num_examples"] == 6
        assert set(result["classes"]) == {"neg", "pos"}

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_accuracy_in_valid_range(self, mock_extract: MagicMock) -> None:
        """Train and val accuracy are between 0 and 1."""
        mock_extract.return_value = _fake_activation()
        examples, labels_raw, classes = _make_labeled_examples(8)

        result = _train_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "acc_probe",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert 0.0 <= result["train_accuracy"] <= 1.0
        assert 0.0 <= result["val_accuracy"] <= 1.0

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_stores_probe_in_registry(self, mock_extract: MagicMock) -> None:
        """After training, probe should exist in ProbeRegistry."""
        mock_extract.return_value = _fake_activation()
        examples, labels_raw, classes = _make_labeled_examples(6)

        _train_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "stored_probe",
            1,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        reg = ProbeRegistry.get()
        assert reg.exists("stored_probe")
        entry = reg.fetch("stored_probe")
        assert entry is not None
        clf, meta = entry
        assert meta.name == "stored_probe"
        assert meta.layer == 1
        assert meta.probe_type == ProbeType.LINEAR

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_mlp_probe_type(self, mock_extract: MagicMock) -> None:
        """MLP probe type trains without error and stores correctly."""
        mock_extract.return_value = _fake_activation()
        examples, labels_raw, classes = _make_labeled_examples(8)

        result = _train_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "mlp_impl_probe",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.MLP,
            -1,
        )

        assert result["probe_type"] == "mlp"
        assert result["coefficients_norm"] is None  # MLP has no coef_

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_coefficients_norm_is_float(self, mock_extract: MagicMock) -> None:
        """Linear probe should have a non-None coefficients_norm."""
        rng = np.random.RandomState(42)

        def _varied_activation(model, config, tokenizer, prompt, layer, pos):
            idx = int(prompt.split("_")[1])
            sign = 1.0 if idx % 2 == 0 else -1.0
            return (rng.randn(DIM) + sign * 2.0).tolist()

        mock_extract.side_effect = _varied_activation
        examples, labels_raw, classes = _make_labeled_examples(8)

        result = _train_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "coef_probe",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["coefficients_norm"] is not None
        assert isinstance(result["coefficients_norm"], float)
        assert result["coefficients_norm"] > 0.0

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_extraction_called_per_example(self, mock_extract: MagicMock) -> None:
        """extract_activation_at_layer is called once per example."""
        mock_extract.return_value = _fake_activation()
        examples, labels_raw, classes = _make_labeled_examples(6)
        model, config, tokenizer = MagicMock(), MagicMock(), MagicMock()

        _train_probe_impl(
            model,
            config,
            tokenizer,
            "call_count_probe",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert mock_extract.call_count == 6
        # Verify it was called with correct args for first example
        first_call = mock_extract.call_args_list[0]
        assert first_call[0] == (model, config, tokenizer, "prompt_0", 0, -1)

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_separable_data_high_accuracy(self, mock_extract: MagicMock) -> None:
        """Linearly separable data should yield high training accuracy."""
        rng = np.random.RandomState(42)

        def _separable_activation(model, config, tokenizer, prompt, layer, pos):
            # pos class: positive mean; neg class: negative mean
            label = "pos" if int(prompt.split("_")[1]) % 2 == 0 else "neg"
            center = 5.0 if label == "pos" else -5.0
            return (rng.randn(DIM) * 0.1 + center).tolist()

        mock_extract.side_effect = _separable_activation
        examples, labels_raw, classes = _make_labeled_examples(20)

        result = _train_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "sep_probe",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["train_accuracy"] >= 0.9

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_three_classes(self, mock_extract: MagicMock) -> None:
        """Train probe with 3 classes."""
        mock_extract.return_value = _fake_activation()
        examples = [{"prompt": f"p_{i}", "label": ["a", "b", "c"][i % 3]} for i in range(9)]
        labels_raw = [ex["label"] for ex in examples]
        classes = sorted(set(labels_raw))

        result = _train_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "three_class",
            0,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert set(result["classes"]) == {"a", "b", "c"}
        assert result["num_examples"] == 9


# ---------------------------------------------------------------------------
# _evaluate_probe_impl (sync implementation)
# ---------------------------------------------------------------------------


class TestEvaluateProbeImpl:
    """Unit tests for _evaluate_probe_impl (lines 374-447)."""

    def _make_clf_and_meta(
        self, classes: list[str] | None = None
    ) -> tuple[MagicMock, ProbeMetadata]:
        """Create a mock classifier and metadata for evaluation tests."""
        classes = classes or ["neg", "pos"]
        clf = MagicMock()
        meta = ProbeMetadata(
            name="eval_probe",
            layer=1,
            probe_type=ProbeType.LINEAR,
            classes=classes,
            num_examples=10,
            train_accuracy=0.9,
            val_accuracy=0.85,
            trained_at="2024-01-01T00:00:00",
        )
        return clf, meta

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_returns_expected_keys(self, mock_extract: MagicMock) -> None:
        """Result dict has all EvaluateProbeResult keys."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        clf.predict.return_value = np.array([0, 1])
        clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

        examples = [
            {"prompt": "hello", "label": "neg"},
            {"prompt": "world", "label": "pos"},
        ]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        expected_keys = {
            "probe_name",
            "layer",
            "accuracy",
            "per_class_accuracy",
            "confusion_matrix",
            "predictions",
        }
        assert expected_keys == set(result.keys())

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_perfect_predictions(self, mock_extract: MagicMock) -> None:
        """All predictions correct => accuracy 1.0."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        # Class order: ["neg", "pos"], so idx 0="neg", 1="pos"
        clf.predict.return_value = np.array([0, 1, 0, 1])
        clf.predict_proba.return_value = np.array(
            [[0.95, 0.05], [0.1, 0.9], [0.85, 0.15], [0.05, 0.95]]
        )

        examples = [
            {"prompt": "a", "label": "neg"},
            {"prompt": "b", "label": "pos"},
            {"prompt": "c", "label": "neg"},
            {"prompt": "d", "label": "pos"},
        ]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        assert result["accuracy"] == 1.0
        assert result["per_class_accuracy"]["neg"] == 1.0
        assert result["per_class_accuracy"]["pos"] == 1.0

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_half_correct_accuracy(self, mock_extract: MagicMock) -> None:
        """Two out of four correct => accuracy 0.5."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        # Predict all as class 0 ("neg")
        clf.predict.return_value = np.array([0, 0, 0, 0])
        clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])

        examples = [
            {"prompt": "a", "label": "neg"},  # correct
            {"prompt": "b", "label": "pos"},  # wrong
            {"prompt": "c", "label": "neg"},  # correct
            {"prompt": "d", "label": "pos"},  # wrong
        ]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        assert result["accuracy"] == 0.5
        assert result["per_class_accuracy"]["neg"] == 1.0
        assert result["per_class_accuracy"]["pos"] == 0.0

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_confusion_matrix_shape(self, mock_extract: MagicMock) -> None:
        """Confusion matrix should be n_classes x n_classes."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        clf.predict.return_value = np.array([0, 1])
        clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

        examples = [
            {"prompt": "a", "label": "neg"},
            {"prompt": "b", "label": "pos"},
        ]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_confusion_matrix_values(self, mock_extract: MagicMock) -> None:
        """Confusion matrix tracks correct and incorrect predictions."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        # Predict: neg, neg, pos, pos — but truth: neg, pos, neg, pos
        clf.predict.return_value = np.array([0, 0, 1, 1])
        clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]])

        examples = [
            {"prompt": "a", "label": "neg"},  # correct
            {"prompt": "b", "label": "pos"},  # wrong (pred neg, true pos)
            {"prompt": "c", "label": "neg"},  # wrong (pred pos, true neg)
            {"prompt": "d", "label": "pos"},  # correct
        ]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        # class_to_idx: neg=0, pos=1
        # confusion[true][pred]
        cm = result["confusion_matrix"]
        assert cm[0][0] == 1  # true neg, pred neg
        assert cm[0][1] == 1  # true neg, pred pos
        assert cm[1][0] == 1  # true pos, pred neg
        assert cm[1][1] == 1  # true pos, pred pos

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_predictions_contain_confidence(self, mock_extract: MagicMock) -> None:
        """Each prediction should have a confidence value from predict_proba."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        clf.predict.return_value = np.array([0])
        clf.predict_proba.return_value = np.array([[0.87, 0.13]])

        examples = [{"prompt": "test", "label": "neg"}]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        pred = result["predictions"][0]
        assert "confidence" in pred
        assert pred["confidence"] == pytest.approx(0.87, abs=0.01)

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_no_predict_proba(self, mock_extract: MagicMock) -> None:
        """When clf has no predict_proba, predictions omit confidence."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        clf.predict.return_value = np.array([1])
        # Remove predict_proba
        del clf.predict_proba

        examples = [{"prompt": "test", "label": "pos"}]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        pred = result["predictions"][0]
        assert "confidence" not in pred

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_prediction_fields(self, mock_extract: MagicMock) -> None:
        """Each prediction has prompt, true_label, predicted_label, correct."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        clf.predict.return_value = np.array([1])
        clf.predict_proba.return_value = np.array([[0.3, 0.7]])

        examples = [{"prompt": "hello", "label": "neg"}]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        pred = result["predictions"][0]
        assert pred["prompt"] == "hello"
        assert pred["true_label"] == "neg"
        assert pred["predicted_label"] == "pos"
        assert pred["correct"] is False

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_extraction_called_per_example(self, mock_extract: MagicMock) -> None:
        """extract_activation_at_layer is called once per example."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta()
        clf.predict.return_value = np.array([0, 1, 0])
        clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])

        examples = [
            {"prompt": "a", "label": "neg"},
            {"prompt": "b", "label": "pos"},
            {"prompt": "c", "label": "neg"},
        ]
        model, config, tokenizer = MagicMock(), MagicMock(), MagicMock()

        _evaluate_probe_impl(
            model,
            config,
            tokenizer,
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        assert mock_extract.call_count == 3
        # Verify layer comes from meta.layer (1)
        for call in mock_extract.call_args_list:
            assert call[0][4] == 1  # layer arg

    @patch(f"{_EXTRACTION_MOD}.extract_activation_at_layer")
    def test_per_class_accuracy_with_missing_class(self, mock_extract: MagicMock) -> None:
        """Classes with no examples in test set get accuracy 0.0."""
        mock_extract.return_value = _fake_activation()
        clf, meta = self._make_clf_and_meta(classes=["a", "b", "c"])
        # Only test examples for classes "a" and "b"
        clf.predict.return_value = np.array([0, 1])
        clf.predict_proba.return_value = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])

        examples = [
            {"prompt": "x", "label": "a"},
            {"prompt": "y", "label": "b"},
        ]

        result = _evaluate_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "eval_probe",
            clf,
            meta,
            examples,
            -1,
        )

        assert result["per_class_accuracy"]["c"] == 0.0


# ---------------------------------------------------------------------------
# _scan_probe_impl (sync implementation)
# ---------------------------------------------------------------------------


class TestScanProbeImpl:
    """Unit tests for _scan_probe_impl (lines 550-649)."""

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_returns_expected_keys(self, mock_extract_all: MagicMock) -> None:
        """Result dict has all ScanProbeResult keys."""
        mock_extract_all.return_value = {
            0: _fake_activation(),
            1: _fake_activation(),
        }
        examples, labels_raw, classes = _make_labeled_examples(8)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_test",
            [0, 1],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        expected_keys = {
            "probe_name_prefix",
            "layers_scanned",
            "accuracy_by_layer",
            "peak_layer",
            "peak_val_accuracy",
            "crossover_layer",
            "interpretation",
        }
        assert expected_keys == set(result.keys())

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_layers_scanned(self, mock_extract_all: MagicMock) -> None:
        """layers_scanned should match the input layers."""
        layers = [0, 1, 2]
        mock_extract_all.return_value = {lay: _fake_activation() for lay in layers}
        examples, labels_raw, classes = _make_labeled_examples(6)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_layers",
            layers,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["layers_scanned"] == [0, 1, 2]
        assert len(result["accuracy_by_layer"]) == 3

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_accuracy_by_layer_structure(self, mock_extract_all: MagicMock) -> None:
        """Each entry in accuracy_by_layer has layer, train_accuracy, val_accuracy."""
        layers = [0, 2]
        mock_extract_all.return_value = {lay: _fake_activation() for lay in layers}
        examples, labels_raw, classes = _make_labeled_examples(6)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_struct",
            layers,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        for entry in result["accuracy_by_layer"]:
            assert "layer" in entry
            assert "train_accuracy" in entry
            assert "val_accuracy" in entry
            assert 0.0 <= entry["train_accuracy"] <= 1.0
            assert 0.0 <= entry["val_accuracy"] <= 1.0

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_stores_probes_per_layer(self, mock_extract_all: MagicMock) -> None:
        """Each layer gets a probe stored as {prefix}_L{layer}."""
        layers = [0, 1, 3]
        mock_extract_all.return_value = {lay: _fake_activation() for lay in layers}
        examples, labels_raw, classes = _make_labeled_examples(6)

        _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_store",
            layers,
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        reg = ProbeRegistry.get()
        for lay in layers:
            name = f"scan_store_L{lay}"
            assert reg.exists(name), f"Probe {name} not found in registry"
            _, meta = reg.fetch(name)
            assert meta.layer == lay
            assert meta.probe_type == ProbeType.LINEAR

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_crossover_detection(self, mock_extract_all: MagicMock) -> None:
        """Crossover is the first layer where val_accuracy > 0.8."""
        rng = np.random.RandomState(42)

        call_count = [0]

        def _increasing_accuracy(model, config, tokenizer, prompt, layers, pos):
            """Return activations where later layers are more separable."""
            idx = int(prompt.split("_")[1])
            label_sign = 1.0 if idx % 2 == 0 else -1.0
            result = {}
            for lay in layers:
                # Increase separability with layer: layer 0 = noise, layer 2 = very separable
                signal = label_sign * (lay + 1) * 3.0
                result[lay] = (rng.randn(DIM) * 0.5 + signal).tolist()
            call_count[0] += 1
            return result

        mock_extract_all.side_effect = _increasing_accuracy
        examples, labels_raw, classes = _make_labeled_examples(20)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_cross",
            [0, 1, 2],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        # With this setup, later layers should have higher accuracy
        # Crossover should be found (some layer > 0.8)
        if result["crossover_layer"] is not None:
            assert result["crossover_layer"] in [0, 1, 2]
            assert "linearly decodable" in result["interpretation"]

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_no_crossover(self, mock_extract_all: MagicMock) -> None:
        """When no layer exceeds 0.8, crossover_layer is None."""
        # All activations are pure noise => accuracy ~0.5
        rng = np.random.RandomState(99)

        def _noisy_activations(model, config, tokenizer, prompt, layers, pos):
            return {lay: rng.randn(DIM).tolist() for lay in layers}

        mock_extract_all.side_effect = _noisy_activations
        examples, labels_raw, classes = _make_labeled_examples(6)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_nocross",
            [0, 1],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        # With pure noise and only 6 examples, accuracy unlikely to exceed 0.8
        # But even if it does by chance, we verify the interpretation string
        if result["crossover_layer"] is None:
            assert "not strongly decodable" in result["interpretation"]
        else:
            assert "linearly decodable" in result["interpretation"]

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_peak_layer_tracking(self, mock_extract_all: MagicMock) -> None:
        """peak_layer and peak_val_accuracy track the best layer."""
        mock_extract_all.return_value = {0: _fake_activation(), 1: _fake_activation()}
        examples, labels_raw, classes = _make_labeled_examples(6)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_peak",
            [0, 1],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["peak_layer"] in [0, 1]
        assert 0.0 <= result["peak_val_accuracy"] <= 1.0
        # peak_val_accuracy should match the max val_accuracy in the results
        val_accs = [e["val_accuracy"] for e in result["accuracy_by_layer"]]
        assert result["peak_val_accuracy"] == pytest.approx(max(val_accs), abs=0.001)

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_extraction_called_per_example(self, mock_extract_all: MagicMock) -> None:
        """extract_activations_all_layers is called once per example."""
        mock_extract_all.return_value = {0: _fake_activation()}
        examples, labels_raw, classes = _make_labeled_examples(6)
        model, config, tokenizer = MagicMock(), MagicMock(), MagicMock()

        _scan_probe_impl(
            model,
            config,
            tokenizer,
            "scan_calls",
            [0],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert mock_extract_all.call_count == 6

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_single_layer_scan(self, mock_extract_all: MagicMock) -> None:
        """Scanning a single layer should work."""
        mock_extract_all.return_value = {2: _fake_activation()}
        examples, labels_raw, classes = _make_labeled_examples(6)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_single",
            [2],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["layers_scanned"] == [2]
        assert len(result["accuracy_by_layer"]) == 1
        assert result["accuracy_by_layer"][0]["layer"] == 2
        assert result["peak_layer"] == 2

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_mlp_probe_type(self, mock_extract_all: MagicMock) -> None:
        """Scan with MLP probe type should store MLP probes."""
        mock_extract_all.return_value = {0: _fake_activation(), 1: _fake_activation()}
        examples, labels_raw, classes = _make_labeled_examples(8)

        _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_mlp",
            [0, 1],
            examples,
            labels_raw,
            classes,
            ProbeType.MLP,
            -1,
        )

        reg = ProbeRegistry.get()
        _, meta = reg.fetch("scan_mlp_L0")
        assert meta.probe_type == ProbeType.MLP

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_crossover_interpretation_message(self, mock_extract_all: MagicMock) -> None:
        """When crossover is found, interpretation includes layer number and accuracy."""
        rng = np.random.RandomState(0)

        def _very_separable(model, config, tokenizer, prompt, layers, pos):
            idx = int(prompt.split("_")[1])
            sign = 1.0 if idx % 2 == 0 else -1.0
            return {lay: (rng.randn(DIM) * 0.01 + sign * 10.0).tolist() for lay in layers}

        mock_extract_all.side_effect = _very_separable
        examples, labels_raw, classes = _make_labeled_examples(20)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "scan_interp",
            [0, 1, 2],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        # With very separable data, crossover should be found
        assert result["crossover_layer"] is not None
        assert "linearly decodable" in result["interpretation"]
        assert f"layer {result['crossover_layer']}" in result["interpretation"]
        assert f"layer {result['peak_layer']}" in result["interpretation"]

    @patch(f"{_EXTRACTION_MOD}.extract_activations_all_layers")
    def test_probe_name_prefix_used(self, mock_extract_all: MagicMock) -> None:
        """Result probe_name_prefix matches input."""
        mock_extract_all.return_value = {0: _fake_activation()}
        examples, labels_raw, classes = _make_labeled_examples(6)

        result = _scan_probe_impl(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "my_custom_prefix",
            [0],
            examples,
            labels_raw,
            classes,
            ProbeType.LINEAR,
            -1,
        )

        assert result["probe_name_prefix"] == "my_custom_prefix"


# ---------------------------------------------------------------------------
# TestProbeAtInference (async tool)
# ---------------------------------------------------------------------------


def _make_fake_probe():
    """Create a fake sklearn classifier and ProbeMetadata."""
    clf = MagicMock()
    # predict_proba returns 2 classes: class_0=0.3, class_1=0.7
    clf.predict_proba.return_value = np.array([[0.3, 0.7]])

    meta = ProbeMetadata(
        name="test_probe",
        layer=2,
        probe_type=ProbeType.LINEAR,
        classes=["class_0", "class_1"],
        num_examples=100,
        train_accuracy=0.9,
        val_accuracy=0.85,
        trained_at="2025-01-01T00:00:00",
    )
    return clf, meta


class TestProbeAtInference:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await probe_at_inference(prompt="hello", probe_name="test")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_probe_not_found(self, loaded_model_state: MagicMock) -> None:
        result = await probe_at_inference(prompt="hello", probe_name="nonexistent")
        assert result["error"] is True
        assert result["error_type"] == "ProbeNotFound"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        # Register a probe so we get past the probe check
        clf, meta = _make_fake_probe()
        ProbeRegistry.get().store("test_probe", clf, meta)

        result = await probe_at_inference(prompt="hello", probe_name="test_probe", max_tokens=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_tokens_too_high(self, loaded_model_state: MagicMock) -> None:
        clf, meta = _make_fake_probe()
        ProbeRegistry.get().store("test_probe", clf, meta)

        result = await probe_at_inference(prompt="hello", probe_name="test_probe", max_tokens=501)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "probe_name": "test_probe",
            "probe_layer": 2,
            "generated_text": "world",
            "tokens_generated": 1,
            "per_token": [],
            "overall_majority_class": "class_1",
            "overall_mean_confidence": 0.7,
            "class_distribution": {"class_1": 1},
        }
        clf, meta = _make_fake_probe()
        ProbeRegistry.get().store("test_probe", clf, meta)

        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._probe_at_inference_impl",
            return_value=mock_result,
        ):
            result = await probe_at_inference(prompt="hello", probe_name="test_probe")
        assert "error" not in result
        assert result["probe_name"] == "test_probe"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        clf, meta = _make_fake_probe()
        ProbeRegistry.get().store("test_probe", clf, meta)

        with patch(
            "chuk_mcp_lazarus.tools.probe.tools._probe_at_inference_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await probe_at_inference(prompt="hello", probe_name="test_probe")
        assert result["error"] is True
        assert result["error_type"] == "GenerationFailed"


# ---------------------------------------------------------------------------
# TestProbeAtInferenceImpl (sync helper)
# ---------------------------------------------------------------------------


class TestProbeAtInferenceImpl:
    """Test _probe_at_inference_impl directly.

    The conftest ModelHooks stub's forward() returns None (no logits)
    and only populates hidden_states. We patch the hooks module so
    forward() returns logits AND populates hidden_states.
    """

    @staticmethod
    def _make_hooks_patch(gen_ids: list[int] | None = None):
        """Return a context manager that patches ModelHooks.forward to return logits."""
        import mlx.core as mx

        gen_ids = gen_ids or [10, 11, 0]
        call_count = [0]

        class PatchedModelHooks:
            def __init__(self, model=None, model_config=None):
                self.model = model
                self.model_config = model_config
                from chuk_lazarus.introspection.hooks import CapturedState

                self.state = CapturedState()
                self._config = None

            def configure(self, config):
                self._config = config

            def forward(self, input_ids):
                # Populate hidden states (like the stub)
                for layer in getattr(self._config, "layers", []):
                    self.state.hidden_states[layer] = mx.array(
                        np.random.randn(1, 5, 64).astype(np.float32)
                    )
                # Also return logits
                seq_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else 5
                logits = np.zeros((1, seq_len, 100), dtype=np.float32)
                step = call_count[0]
                if step < len(gen_ids):
                    logits[0, -1, gen_ids[step]] = 10.0
                call_count[0] += 1
                return mx.array(logits)

            def _get_final_norm(self):
                return lambda x: x

        return patch(
            "chuk_lazarus.introspection.hooks.ModelHooks",
            PatchedModelHooks,
        )

    def _make_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.eos_token_id = 0
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
        return tokenizer

    def test_output_structure(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        with self._make_hooks_patch():
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=3,
                temperature=0.0,
                token_position=-1,
            )

        assert isinstance(result, dict)
        assert result["prompt"] == "hello"
        assert result["probe_name"] == "test_probe"
        assert result["probe_layer"] == 2
        assert "generated_text" in result
        assert "tokens_generated" in result
        assert "per_token" in result
        assert "overall_majority_class" in result
        assert "overall_mean_confidence" in result
        assert "class_distribution" in result

    def test_generated_text(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        with self._make_hooks_patch(gen_ids=[10, 11, 0]):
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=5,
                temperature=0.0,
                token_position=-1,
            )

        # Should generate tokens then hit EOS (token 0)
        assert result["tokens_generated"] >= 1

    def test_per_token_fields(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        with self._make_hooks_patch():
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=2,
                temperature=0.0,
                token_position=-1,
            )

        for entry in result["per_token"]:
            assert "step" in entry
            assert "token" in entry
            assert "token_id" in entry
            assert "probe_prediction" in entry
            assert "probe_confidence" in entry
            assert "probe_probabilities" in entry

    def test_prediction_classes(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        with self._make_hooks_patch():
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=2,
                temperature=0.0,
                token_position=-1,
            )

        for entry in result["per_token"]:
            assert entry["probe_prediction"] in ["class_0", "class_1"]

    def test_majority_class(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()
        # Always predicts class_1 (0.7 > 0.3)

        with self._make_hooks_patch():
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=2,
                temperature=0.0,
                token_position=-1,
            )

        assert result["overall_majority_class"] == "class_1"

    def test_mean_confidence(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        with self._make_hooks_patch():
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=2,
                temperature=0.0,
                token_position=-1,
            )

        assert 0.0 <= result["overall_mean_confidence"] <= 1.0

    def test_class_distribution(self) -> None:
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        with self._make_hooks_patch():
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=2,
                temperature=0.0,
                token_position=-1,
            )

        dist = result["class_distribution"]
        assert isinstance(dist, dict)
        total = sum(dist.values())
        assert total == result["tokens_generated"]

    def test_max_tokens_stop(self) -> None:
        """Generation should stop at max_tokens even without EOS."""
        tokenizer = self._make_tokenizer()
        tokenizer.eos_token_id = 999  # Never generated
        clf, meta = _make_fake_probe()

        # gen_ids never contains 999 (EOS)
        with self._make_hooks_patch(gen_ids=[10, 11, 12, 13, 14]):
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=3,
                temperature=0.0,
                token_position=-1,
            )

        assert result["tokens_generated"] == 3

    def test_eos_stops_generation(self) -> None:
        """Generation should stop when EOS is generated."""
        tokenizer = self._make_tokenizer()
        clf, meta = _make_fake_probe()

        # 3rd token is 0 = EOS
        with self._make_hooks_patch(gen_ids=[10, 11, 0]):
            result = _probe_at_inference_impl(
                MagicMock(),
                MagicMock(),
                tokenizer,
                (clf, meta),
                prompt="hello",
                max_tokens=100,  # High limit
                temperature=0.0,
                token_position=-1,
            )

        # Should stop at EOS (3rd generated token is token 0 = EOS)
        assert result["tokens_generated"] <= 3
