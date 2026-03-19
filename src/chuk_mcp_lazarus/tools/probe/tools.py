"""
Probe tools: train_probe, evaluate_probe, scan_probe_across_layers, list_probes.

Trains sklearn classifiers on hidden-state activations to identify which
layers encode specific features (e.g. source language identity). The
scan tool caches activations so each example runs a single forward pass
through all layers, avoiding redundant computation.
"""

import asyncio
import datetime
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..._extraction import extract_activation_at_layer, extract_activations_all_layers
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...probe_store import ProbeMetadata, ProbeRegistry, ProbeType
from ...server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class TrainProbeResult(BaseModel):
    """Result from train_probe."""

    probe_name: str
    layer: int
    probe_type: ProbeType
    num_examples: int
    classes: list[str]
    train_accuracy: float
    val_accuracy: float
    coefficients_norm: float | None = None


class EvaluateProbeResult(BaseModel):
    """Result from evaluate_probe."""

    probe_name: str
    layer: int
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: list[list[int]]
    predictions: list[dict[str, Any]]


class LayerAccuracy(BaseModel):
    """Per-layer accuracy in a scan."""

    layer: int
    train_accuracy: float
    val_accuracy: float


class ScanProbeResult(BaseModel):
    """Result from scan_probe_across_layers."""

    probe_name_prefix: str
    layers_scanned: list[int]
    accuracy_by_layer: list[LayerAccuracy]
    peak_layer: int
    peak_val_accuracy: float
    crossover_layer: int | None = Field(
        None, description="Layer where accuracy first exceeds 0.8, if any."
    )
    interpretation: str


class ListProbesResult(BaseModel):
    """Result from list_probes."""

    probes: list[dict[str, Any]]
    count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _train_sklearn_probe(
    X: np.ndarray,
    y: np.ndarray,
    probe_type: ProbeType,
    random_seed: int = 42,
) -> tuple[Any, float, float]:
    """Train a probe and return (model, train_accuracy, val_accuracy).

    Uses cross-validation for val_accuracy. Falls back to train accuracy
    when the dataset is too small for stratified CV.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    if probe_type == ProbeType.LINEAR:
        clf = LogisticRegression(max_iter=1000, random_state=random_seed, C=1.0)
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=500,
            random_state=random_seed,
        )

    # Cross-validation for val accuracy
    unique_labels = np.unique(y)
    min_class_count = min(np.sum(y == label) for label in unique_labels)
    n_folds = min(5, min_class_count, len(y))

    if n_folds >= 2:
        cv_scores = cross_val_score(clf, X, y, cv=n_folds)
        val_accuracy = float(np.mean(cv_scores))
    else:
        val_accuracy = 0.0

    # Fit on full data for the stored model
    clf.fit(X, y)
    train_accuracy = float(clf.score(X, y))

    # If CV wasn't possible, use train accuracy as fallback
    if val_accuracy == 0.0:
        val_accuracy = train_accuracy

    return clf, train_accuracy, val_accuracy


def _coefficients_norm(clf: Any) -> float | None:
    """Extract L2 norm of probe coefficients, if available."""
    if hasattr(clf, "coef_"):
        return float(np.linalg.norm(clf.coef_))
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def train_probe(
    probe_name: str,
    layer: int,
    examples: list[dict],
    probe_type: str = "linear",
    token_position: int = -1,
) -> dict:
    """
    Train a probe classifier on activations at the specified layer.

    Each example must have a "prompt" and "label" field. The probe
    learns to classify which label a prompt belongs to based on its
    hidden-state activation at the given layer.

    Args:
        probe_name:     Unique name for this probe.
        layer:          Layer to extract activations from.
        examples:       [{"prompt": str, "label": str}, ...]
        probe_type:     "linear" (LogisticRegression) or "mlp".
        token_position: Token position (default: last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "train_probe",
        )

    # Validate probe_type
    try:
        ptype = ProbeType(probe_type)
    except ValueError:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid probe_type '{probe_type}'. Use 'linear' or 'mlp'.",
            "train_probe",
        )

    # Validate layer
    num_layers = state.metadata.num_layers
    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "train_probe",
        )

    # Validate examples
    if len(examples) < 4:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need at least 4 examples, got {len(examples)}.",
            "train_probe",
        )

    for i, ex in enumerate(examples):
        if "prompt" not in ex or "label" not in ex:
            return make_error(
                ToolError.INVALID_INPUT,
                f"Example {i} must have 'prompt' and 'label' keys.",
                "train_probe",
            )

    labels_raw = [ex["label"] for ex in examples]
    classes = sorted(set(labels_raw))
    if len(classes) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            "Need at least 2 distinct labels.",
            "train_probe",
        )

    try:
        result = await asyncio.to_thread(
            _train_probe_impl,
            state.model,
            state.config,
            state.tokenizer,
            probe_name,
            layer,
            examples,
            labels_raw,
            classes,
            ptype,
            token_position,
        )
        return result

    except Exception as e:
        logger.exception("train_probe failed")
        return make_error(ToolError.TRAINING_FAILED, str(e), "train_probe")


def _train_probe_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    probe_name: str,
    layer: int,
    examples: list[dict],
    labels_raw: list[str],
    classes: list[str],
    ptype: ProbeType,
    token_position: int,
) -> dict:
    """Sync implementation of train_probe."""
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([label_to_idx[label] for label in labels_raw])

    X_list: list[list[float]] = []
    for ex in examples:
        vec = extract_activation_at_layer(
            model,
            config,
            tokenizer,
            ex["prompt"],
            layer,
            token_position,
        )
        X_list.append(vec)
    X = np.array(X_list, dtype=np.float32)

    clf, train_acc, val_acc = _train_sklearn_probe(X, y, ptype)
    coef_norm = _coefficients_norm(clf)

    metadata = ProbeMetadata(
        name=probe_name,
        layer=layer,
        probe_type=ptype,
        classes=classes,
        num_examples=len(examples),
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        coefficients_norm=coef_norm,
        trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    ProbeRegistry.get().store(probe_name, clf, metadata)

    result = TrainProbeResult(
        probe_name=probe_name,
        layer=layer,
        probe_type=ptype,
        num_examples=len(examples),
        classes=classes,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        coefficients_norm=coef_norm,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def evaluate_probe(
    probe_name: str,
    examples: list[dict],
    token_position: int = -1,
) -> dict:
    """
    Evaluate a stored probe on new (held-out) examples.

    Args:
        probe_name:     Name of a trained probe.
        examples:       [{"prompt": str, "label": str}, ...]
        token_position: Token position (default: last).

    Returns accuracy, per-class accuracy, confusion matrix, and
    per-example predictions with confidence scores.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "evaluate_probe",
        )

    registry = ProbeRegistry.get()
    entry = registry.fetch(probe_name)
    if entry is None:
        return make_error(
            ToolError.PROBE_NOT_FOUND,
            f"Probe '{probe_name}' not found. Use list_probes() to see available probes.",
            "evaluate_probe",
        )

    clf, meta = entry

    # Validate examples
    if not examples:
        return make_error(
            ToolError.INVALID_INPUT,
            "Need at least 1 example.",
            "evaluate_probe",
        )

    for i, ex in enumerate(examples):
        if "prompt" not in ex or "label" not in ex:
            return make_error(
                ToolError.INVALID_INPUT,
                f"Example {i} must have 'prompt' and 'label' keys.",
                "evaluate_probe",
            )

    try:
        result = await asyncio.to_thread(
            _evaluate_probe_impl,
            state.model,
            state.config,
            state.tokenizer,
            probe_name,
            clf,
            meta,
            examples,
            token_position,
        )
        return result

    except Exception as e:
        logger.exception("evaluate_probe failed")
        return make_error(ToolError.EVALUATION_FAILED, str(e), "evaluate_probe")


def _evaluate_probe_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    probe_name: str,
    clf: Any,
    meta: ProbeMetadata,
    examples: list[dict],
    token_position: int,
) -> dict:
    """Sync implementation of evaluate_probe."""
    class_to_idx = {label: idx for idx, label in enumerate(meta.classes)}

    X_list: list[list[float]] = []
    for ex in examples:
        vec = extract_activation_at_layer(
            model,
            config,
            tokenizer,
            ex["prompt"],
            meta.layer,
            token_position,
        )
        X_list.append(vec)
    X = np.array(X_list, dtype=np.float32)

    y_pred_idx = clf.predict(X)
    y_proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    predictions: list[dict[str, Any]] = []
    correct = 0
    for i, ex in enumerate(examples):
        predicted_label = meta.classes[int(y_pred_idx[i])]
        confidence = float(y_proba[i].max()) if y_proba is not None else None
        is_correct = predicted_label == ex["label"]
        if is_correct:
            correct += 1
        pred = {
            "prompt": ex["prompt"],
            "true_label": ex["label"],
            "predicted_label": predicted_label,
            "correct": is_correct,
        }
        if confidence is not None:
            pred["confidence"] = round(confidence, 4)
        predictions.append(pred)

    accuracy = correct / len(examples)

    per_class: dict[str, float] = {}
    for cls in meta.classes:
        cls_examples = [p for p in predictions if p["true_label"] == cls]
        if cls_examples:
            per_class[cls] = sum(1 for p in cls_examples if p["correct"]) / len(cls_examples)
        else:
            per_class[cls] = 0.0

    n_classes = len(meta.classes)
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for pred in predictions:
        true_idx = class_to_idx.get(pred["true_label"])
        pred_idx = class_to_idx.get(pred["predicted_label"])
        if true_idx is not None and pred_idx is not None:
            confusion[true_idx][pred_idx] += 1

    result = EvaluateProbeResult(
        probe_name=probe_name,
        layer=meta.layer,
        accuracy=accuracy,
        per_class_accuracy=per_class,
        confusion_matrix=confusion,
        predictions=predictions,
    )
    return result.model_dump()


@mcp.tool()
async def scan_probe_across_layers(
    probe_name_prefix: str,
    layers: list[int],
    examples: list[dict],
    probe_type: str = "linear",
    token_position: int = -1,
) -> dict:
    """
    Train and evaluate a probe at every specified layer in one call.
    Primary tool for finding the crossover layer in language
    transition experiments.

    Caches activations: each example runs one forward pass through all
    layers, then probes are trained from the cache. For a 34-layer
    model with 40 examples this avoids 34x redundant forward passes.

    Creates probes named "{probe_name_prefix}_L{layer}" for each layer.

    Args:
        probe_name_prefix: Base name; layer suffix appended.
        layers:            Layers to scan.
        examples:          [{"prompt": str, "label": str}, ...]
        probe_type:        "linear" or "mlp".
        token_position:    Token position (default: last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "scan_probe_across_layers",
        )

    # Validate probe_type
    try:
        ptype = ProbeType(probe_type)
    except ValueError:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid probe_type '{probe_type}'. Use 'linear' or 'mlp'.",
            "scan_probe_across_layers",
        )

    # Validate layers
    num_layers = state.metadata.num_layers
    out_of_range = [lay for lay in layers if lay < 0 or lay >= num_layers]
    if out_of_range:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layers {out_of_range} out of range [0, {num_layers - 1}].",
            "scan_probe_across_layers",
        )

    # Validate examples
    if len(examples) < 4:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Need at least 4 examples, got {len(examples)}.",
            "scan_probe_across_layers",
        )

    for i, ex in enumerate(examples):
        if "prompt" not in ex or "label" not in ex:
            return make_error(
                ToolError.INVALID_INPUT,
                f"Example {i} must have 'prompt' and 'label' keys.",
                "scan_probe_across_layers",
            )

    labels_raw = [ex["label"] for ex in examples]
    classes = sorted(set(labels_raw))
    if len(classes) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            "Need at least 2 distinct labels.",
            "scan_probe_across_layers",
        )

    try:
        result = await asyncio.to_thread(
            _scan_probe_impl,
            state.model,
            state.config,
            state.tokenizer,
            probe_name_prefix,
            layers,
            examples,
            labels_raw,
            classes,
            ptype,
            token_position,
        )
        return result

    except Exception as e:
        logger.exception("scan_probe_across_layers failed")
        return make_error(ToolError.TRAINING_FAILED, str(e), "scan_probe_across_layers")


def _scan_probe_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    probe_name_prefix: str,
    layers: list[int],
    examples: list[dict],
    labels_raw: list[str],
    classes: list[str],
    ptype: ProbeType,
    token_position: int,
) -> dict:
    """Sync implementation of scan_probe_across_layers."""
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([label_to_idx[label] for label in labels_raw])

    cache: dict[int, list[list[float]]] = {lay: [] for lay in layers}

    logger.info(
        "Scanning %d layers with %d examples (cached extraction)...",
        len(layers),
        len(examples),
    )
    for i, ex in enumerate(examples):
        activations = extract_activations_all_layers(
            model,
            config,
            tokenizer,
            ex["prompt"],
            layers,
            token_position,
        )
        for lay in layers:
            cache[lay].append(activations[lay])

    registry = ProbeRegistry.get()
    layer_results: list[LayerAccuracy] = []
    best_layer = layers[0]
    best_val_acc = 0.0
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for lay in layers:
        X = np.array(cache[lay], dtype=np.float32)
        clf, train_acc, val_acc = _train_sklearn_probe(X, y, ptype)
        coef_norm = _coefficients_norm(clf)

        layer_results.append(
            LayerAccuracy(
                layer=lay,
                train_accuracy=round(train_acc, 4),
                val_accuracy=round(val_acc, 4),
            )
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_layer = lay

        probe_name = f"{probe_name_prefix}_L{lay}"
        meta = ProbeMetadata(
            name=probe_name,
            layer=lay,
            probe_type=ptype,
            classes=classes,
            num_examples=len(examples),
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            coefficients_norm=coef_norm,
            trained_at=timestamp,
        )
        registry.store(probe_name, clf, meta)

    crossover = None
    for lr in layer_results:
        if lr.val_accuracy > 0.8:
            crossover = lr.layer
            break

    if crossover is not None:
        interpretation = (
            f"Feature becomes linearly decodable at layer {crossover} "
            f"(val_accuracy={next(lr.val_accuracy for lr in layer_results if lr.layer == crossover):.2%}). "
            f"Peak accuracy at layer {best_layer} ({best_val_acc:.2%})."
        )
    else:
        interpretation = (
            f"Feature not strongly decodable (peak val_accuracy={best_val_acc:.2%} "
            f"at layer {best_layer}). Consider adding more examples or trying different features."
        )

    result = ScanProbeResult(
        probe_name_prefix=probe_name_prefix,
        layers_scanned=layers,
        accuracy_by_layer=layer_results,
        peak_layer=best_layer,
        peak_val_accuracy=round(best_val_acc, 4),
        crossover_layer=crossover,
        interpretation=interpretation,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Result models for probe_at_inference
# ---------------------------------------------------------------------------


class ProbeTokenEntry(BaseModel):
    """Probe prediction for a single generated token."""

    step: int
    token: str
    token_id: int
    probe_prediction: str
    probe_confidence: float
    probe_probabilities: dict[str, float]


class ProbeAtInferenceResult(BaseModel):
    """Result from probe_at_inference."""

    prompt: str
    probe_name: str
    probe_layer: int
    generated_text: str
    tokens_generated: int
    per_token: list[ProbeTokenEntry]
    overall_majority_class: str
    overall_mean_confidence: float
    class_distribution: dict[str, int]


# ---------------------------------------------------------------------------
# Tool: probe_at_inference
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True)
async def probe_at_inference(
    prompt: str,
    probe_name: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    token_position: int = -1,
) -> dict:
    """
    Run a trained probe during autoregressive generation.

    For each generated token, extracts the hidden state at the probe's
    layer and runs the probe classifier. Returns per-token predictions,
    class distribution, and overall confidence.

    Useful for monitoring how the model's internal state evolves
    token-by-token during generation (e.g. does a hallucination-type
    probe fire consistently throughout a response?).

    Args:
        prompt:         Input text to continue generating from.
        probe_name:     Name of a previously trained probe.
        max_tokens:     Maximum tokens to generate (1–500).
        temperature:    Sampling temperature (0.0 = greedy).
        token_position: Token position to extract activation from (-1 = last).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "probe_at_inference",
        )

    registry = ProbeRegistry.get()
    probe_data = registry.fetch(probe_name)
    if probe_data is None:
        return make_error(
            ToolError.PROBE_NOT_FOUND,
            f"Probe {probe_name!r} not found. Train it first.",
            "probe_at_inference",
        )

    if max_tokens < 1 or max_tokens > 500:
        return make_error(
            ToolError.INVALID_INPUT,
            f"max_tokens must be 1–500, got {max_tokens}.",
            "probe_at_inference",
        )

    try:
        result = await asyncio.to_thread(
            _probe_at_inference_impl,
            state.model,
            state.config,
            state.tokenizer,
            probe_data,
            prompt,
            max_tokens,
            temperature,
            token_position,
        )
        return result
    except Exception as e:
        logger.exception("probe_at_inference failed")
        return make_error(ToolError.GENERATION_FAILED, str(e), "probe_at_inference")


def _probe_at_inference_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    probe_data: tuple[Any, ProbeMetadata],
    prompt: str,
    max_tokens: int,
    temperature: float,
    token_position: int,
) -> dict:
    """Sync implementation of probe_at_inference."""
    import mlx.core as mx

    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    clf, meta = probe_data
    probe_layer = meta.layer
    classes = meta.classes

    # Tokenize the prompt
    current_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    eos_id = getattr(tokenizer, "eos_token_id", None)

    generated_tokens: list[int] = []
    per_token: list[ProbeTokenEntry] = []

    for step in range(max_tokens):
        input_ids = mx.array(current_ids)

        # Forward pass with hidden state capture at probe layer
        hooks = ModelHooks(model, model_config=config)
        hooks.configure(
            CaptureConfig(
                layers=[probe_layer],
                capture_hidden_states=True,
            )
        )
        logits = hooks.forward(input_ids)
        mx.eval(hooks.state.hidden_states)

        # Get next token
        if logits is not None:
            if hasattr(logits, "logits"):
                logits = logits.logits
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                last_logits = logits[-1, :]
            else:
                last_logits = logits

            if temperature <= 0.0:
                next_token_id = int(mx.argmax(last_logits).item())
            else:
                scaled = last_logits / temperature
                probs = mx.softmax(scaled, axis=-1)
                next_token_id = int(mx.random.categorical(probs).item())
        else:
            break

        # Extract hidden state at probe layer
        h = hooks.state.hidden_states.get(probe_layer)
        if h is None:
            break

        # Extract activation at token_position
        if h.ndim == 3:
            act = h[0, token_position, :]
        elif h.ndim == 2:
            act = h[token_position, :]
        else:
            act = h

        # Convert to numpy for sklearn
        if hasattr(act, "_data"):
            act_np = np.array(act._data, dtype=np.float32).reshape(1, -1)
        elif hasattr(act, "tolist"):
            act_np = np.array(act.tolist(), dtype=np.float32).reshape(1, -1)
        else:
            act_np = np.array(act, dtype=np.float32).reshape(1, -1)

        # Run probe
        probe_probs = clf.predict_proba(act_np)[0]
        pred_idx = int(np.argmax(probe_probs))
        pred_class = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        confidence = float(probe_probs[pred_idx])

        class_probs = {
            classes[i] if i < len(classes) else str(i): round(float(probe_probs[i]), 6)
            for i in range(len(probe_probs))
        }

        token_text = tokenizer.decode([next_token_id])

        per_token.append(
            ProbeTokenEntry(
                step=step,
                token=token_text,
                token_id=next_token_id,
                probe_prediction=pred_class,
                probe_confidence=round(confidence, 6),
                probe_probabilities=class_probs,
            )
        )

        generated_tokens.append(next_token_id)
        current_ids.append(next_token_id)

        # Check EOS
        if eos_id is not None and next_token_id == eos_id:
            break

    # Aggregate
    generated_text = tokenizer.decode(generated_tokens) if generated_tokens else ""

    class_dist: dict[str, int] = {}
    confidences: list[float] = []
    for entry in per_token:
        class_dist[entry.probe_prediction] = class_dist.get(entry.probe_prediction, 0) + 1
        confidences.append(entry.probe_confidence)

    majority_class = max(class_dist, key=lambda k: class_dist[k]) if class_dist else ""
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    result = ProbeAtInferenceResult(
        prompt=prompt,
        probe_name=meta.name,
        probe_layer=probe_layer,
        generated_text=generated_text,
        tokens_generated=len(generated_tokens),
        per_token=per_token,
        overall_majority_class=majority_class,
        overall_mean_confidence=round(mean_conf, 6),
        class_distribution=class_dist,
    )
    return result.model_dump()


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_probes() -> dict:
    """
    List all probes in memory.

    Returns summary metadata for every trained probe, including
    name, layer, classes, probe_type, val_accuracy, and trained_at.
    """
    try:
        registry = ProbeRegistry.get()
        dump = registry.dump()
        return dump.model_dump()
    except Exception as e:
        logger.exception("list_probes failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "list_probes")
