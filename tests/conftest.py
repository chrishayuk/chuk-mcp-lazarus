"""Shared test fixtures for chuk-mcp-lazarus.

Provides mock MLX arrays, mock tokenizer/model/config, model state
fixtures, and autouse helpers so tests run on any platform (no Apple
Silicon / MLX required).
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. MLX stub — must be installed BEFORE any source imports
# ---------------------------------------------------------------------------


def _make_mx_stub() -> types.ModuleType:
    """Build a minimal mlx.core stub for testing."""
    mx = types.ModuleType("mlx")
    mx_core = types.ModuleType("mlx.core")
    mx_fast = types.ModuleType("mlx.core.fast")
    mx_metal = types.ModuleType("mlx.core.metal")
    mx_random = types.ModuleType("mlx.core.random")
    mx_nn = types.ModuleType("mlx.nn")

    class MockMxArray:
        """Minimal MLX array mock with arithmetic, indexing, and shape."""

        def __init__(self, data: Any, dtype: Any = None):
            if isinstance(data, MockMxArray):
                self._data = data._data
            elif isinstance(data, np.ndarray):
                self._data = data
            elif isinstance(data, (list, tuple)):
                self._data = np.array(data, dtype=np.float32)
            elif isinstance(data, (int, float)):
                self._data = np.array(data, dtype=np.float32)
            else:
                self._data = np.array(data, dtype=np.float32)
            self.dtype = dtype or "float32"

        @property
        def shape(self) -> tuple:
            return self._data.shape

        @property
        def ndim(self) -> int:
            return self._data.ndim

        @property
        def size(self) -> int:
            return self._data.size

        def tolist(self) -> Any:
            return self._data.tolist()

        def item(self) -> Any:
            return self._data.item()

        def reshape(self, *args: Any) -> MockMxArray:
            return MockMxArray(self._data.reshape(*args))

        def transpose(self, *axes: Any) -> MockMxArray:
            return MockMxArray(self._data.transpose(*axes))

        @property
        def T(self) -> MockMxArray:
            return MockMxArray(self._data.T)

        def sum(self, axis: Any = None, keepdims: bool = False) -> MockMxArray:
            return MockMxArray(self._data.sum(axis=axis, keepdims=keepdims))

        def astype(self, dtype: Any) -> MockMxArray:
            return MockMxArray(self._data, dtype=dtype)

        def __getitem__(self, key: Any) -> MockMxArray:
            # Handle MockMxArray as index (fancy indexing)
            if isinstance(key, MockMxArray):
                key = key._data.astype(int) if key._data.dtype.kind == "f" else key._data
            elif isinstance(key, tuple):
                key = tuple(
                    k._data.astype(int)
                    if isinstance(k, MockMxArray) and k._data.dtype.kind == "f"
                    else k._data
                    if isinstance(k, MockMxArray)
                    else k
                    for k in key
                )
            result = self._data[key]
            if isinstance(result, np.ndarray):
                return MockMxArray(result)
            return MockMxArray(np.array(result, dtype=np.float32))

        def __add__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data + od)

        def __radd__(self, other: Any) -> MockMxArray:
            return self.__add__(other)

        def __sub__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data - od)

        def __mul__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data * od)

        def __rmul__(self, other: Any) -> MockMxArray:
            return self.__mul__(other)

        def __truediv__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data / od)

        def __matmul__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data @ od)

        def __gt__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data > od)

        def __lt__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data < od)

        def __ge__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data >= od)

        def __le__(self, other: Any) -> MockMxArray:
            od = other._data if isinstance(other, MockMxArray) else other
            return MockMxArray(self._data <= od)

        def __neg__(self) -> MockMxArray:
            return MockMxArray(-self._data)

        def __len__(self) -> int:
            return len(self._data)

        def __float__(self) -> float:
            return float(self._data)

        def __int__(self) -> int:
            return int(self._data)

        def __repr__(self) -> str:
            return f"MockMxArray({self._data})"

    # Attach array class
    mx_core.array = MockMxArray  # type: ignore[attr-defined]

    # Functional ops
    def _softmax(x: Any, axis: int = -1) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        e = np.exp(d - np.max(d, axis=axis, keepdims=True))
        return MockMxArray(e / np.sum(e, axis=axis, keepdims=True))

    def _argmax(x: Any, axis: int | None = None) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.array(np.argmax(d, axis=axis)))

    def _argsort(x: Any, axis: int = -1) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.argsort(d, axis=axis))

    def _sum(x: Any, axis: Any = None, keepdims: bool = False) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.sum(d, axis=axis, keepdims=keepdims))

    def _sqrt(x: Any) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.sqrt(d))

    def _log(x: Any) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.log(np.clip(d, 1e-10, None)))

    def _clip(x: Any, lo: Any, hi: Any) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.clip(d, lo, hi))

    def _max(x: Any, axis: Any = None, keepdims: bool = False) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.max(d, axis=axis, keepdims=keepdims))

    def _mean(x: Any, axis: Any = None, keepdims: bool = False) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.mean(d, axis=axis, keepdims=keepdims))

    def _matmul(a: Any, b: Any) -> MockMxArray:
        da = a._data if isinstance(a, MockMxArray) else np.array(a)
        db = b._data if isinstance(b, MockMxArray) else np.array(b)
        return MockMxArray(da @ db)

    def _eval(*args: Any) -> None:
        pass  # No-op for mock

    def _concatenate(arrays: list, axis: int = 0) -> MockMxArray:
        arrs = [a._data if isinstance(a, MockMxArray) else np.array(a) for a in arrays]
        return MockMxArray(np.concatenate(arrs, axis=axis))

    def _triu(x: Any, k: int = 0) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.triu(d, k=k))

    def _full(shape: Any, fill: Any, dtype: Any = None) -> MockMxArray:
        return MockMxArray(np.full(shape, fill))

    def _repeat(x: Any, repeats: int, axis: int = 0) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.repeat(d, repeats, axis=axis))

    def _abs(x: Any) -> MockMxArray:
        d = x._data if isinstance(x, MockMxArray) else np.array(x)
        return MockMxArray(np.abs(d))

    mx_core.softmax = _softmax
    mx_core.argmax = _argmax
    mx_core.argsort = _argsort
    mx_core.sum = _sum
    mx_core.sqrt = _sqrt
    mx_core.log = _log
    mx_core.clip = _clip
    mx_core.max = _max
    mx_core.mean = _mean
    mx_core.matmul = _matmul
    mx_core.eval = _eval
    mx_core.concatenate = _concatenate
    mx_core.triu = _triu
    mx_core.full = _full
    mx_core.repeat = _repeat
    mx_core.abs = _abs

    # random
    def _categorical(logits: Any) -> MockMxArray:
        return MockMxArray(np.array(0))

    mx_random.categorical = _categorical
    mx_core.random = mx_random

    # fast
    def _scaled_dot_product_attention(q: Any, k: Any, v: Any, scale: float = 1.0) -> MockMxArray:
        return MockMxArray(np.zeros((1, 1, 1, 64)))

    mx_fast.scaled_dot_product_attention = _scaled_dot_product_attention
    mx_core.fast = mx_fast

    # metal
    mx_metal.clear_cache = lambda: None
    mx_core.metal = mx_metal

    # mlx.nn stub
    class MockEmbedding:
        def __init__(self) -> None:
            self.weight = MockMxArray(np.random.randn(100, 64).astype(np.float32))

    class MockLinear:
        def __init__(self, in_f: int = 64, out_f: int = 64) -> None:
            self.weight = MockMxArray(np.random.randn(out_f, in_f).astype(np.float32))

        def __call__(self, x: Any) -> MockMxArray:
            return MockMxArray(np.random.randn(1, 1, 64).astype(np.float32))

    mx_nn.Embedding = MockEmbedding
    mx_nn.Linear = MockLinear

    # Wire up module tree
    mx.core = mx_core
    mx.nn = mx_nn

    return mx, mx_core, mx_nn


# Install stubs if real MLX not available
if "mlx" not in sys.modules:
    _mx, _mx_core, _mx_nn = _make_mx_stub()
    sys.modules["mlx"] = _mx
    sys.modules["mlx.core"] = _mx_core
    sys.modules["mlx.core.fast"] = _mx_core.fast
    sys.modules["mlx.core.metal"] = _mx_core.metal
    sys.modules["mlx.core.random"] = _mx_core.random
    sys.modules["mlx.nn"] = _mx_nn


# ---------------------------------------------------------------------------
# 2. chuk-lazarus stubs
# ---------------------------------------------------------------------------


def _install_lazarus_stubs() -> None:
    """Install stubs for chuk_lazarus imports used in source code."""
    if "chuk_lazarus" in sys.modules:
        return

    # Base module
    chuk_lazarus = types.ModuleType("chuk_lazarus")
    sys.modules["chuk_lazarus"] = chuk_lazarus

    # inference
    inference = types.ModuleType("chuk_lazarus.inference")
    sys.modules["chuk_lazarus.inference"] = inference

    class DType:
        def __init__(self, value: str = "bfloat16") -> None:
            self.value = value

    loader = types.ModuleType("chuk_lazarus.inference.loader")
    loader.DType = DType  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.inference.loader"] = loader

    class UnifiedPipelineConfig:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class UnifiedPipeline:
        def __init__(self) -> None:
            self.model = MagicMock()
            self.tokenizer = MagicMock()
            self.config = MagicMock()
            self.family = MagicMock()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> UnifiedPipeline:
            return cls()

    inference.UnifiedPipelineConfig = UnifiedPipelineConfig  # type: ignore[attr-defined]
    inference.UnifiedPipeline = UnifiedPipeline  # type: ignore[attr-defined]

    # introspection.hooks
    introspection = types.ModuleType("chuk_lazarus.introspection")
    sys.modules["chuk_lazarus.introspection"] = introspection

    hooks_mod = types.ModuleType("chuk_lazarus.introspection.hooks")

    class PositionSelection:
        LAST = "last"
        ALL = "all"

    class CaptureConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.layers = kwargs.get("layers", [])
            self.capture_hidden_states = kwargs.get("capture_hidden_states", False)
            self.capture_attention_weights = kwargs.get("capture_attention_weights", False)
            self.positions = kwargs.get("positions", PositionSelection.LAST)

    class CapturedState:
        def __init__(self) -> None:
            self.hidden_states: dict = {}
            self.attention_weights: dict = {}

    class ModelHooks:
        def __init__(self, model: Any = None, model_config: Any = None) -> None:
            self.model = model
            self.model_config = model_config
            self.state = CapturedState()

        def configure(self, config: CaptureConfig) -> None:
            self._config = config

        def forward(self, input_ids: Any) -> None:
            import mlx.core as mx

            # Populate hidden states with mock data
            for layer in getattr(self, "_config", CaptureConfig()).layers:
                self.state.hidden_states[layer] = mx.array(
                    np.random.randn(1, 5, 64).astype(np.float32)
                )

        def _get_final_norm(self) -> Any:
            return lambda x: x

        def _get_lm_head(self) -> Any:
            return MagicMock()

        def get_layer_logits(self, layer: int, normalize: bool = False) -> Any:
            import mlx.core as mx

            return mx.array(np.random.randn(100).astype(np.float32))

    hooks_mod.CaptureConfig = CaptureConfig  # type: ignore[attr-defined]
    hooks_mod.ModelHooks = ModelHooks  # type: ignore[attr-defined]
    hooks_mod.PositionSelection = PositionSelection  # type: ignore[attr-defined]
    hooks_mod.CapturedState = CapturedState  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.hooks"] = hooks_mod
    introspection.hooks = hooks_mod  # type: ignore[attr-defined]

    # ---- ablation stubs ----
    ablation_pkg = types.ModuleType("chuk_lazarus.introspection.ablation")
    sys.modules["chuk_lazarus.introspection.ablation"] = ablation_pkg
    introspection.ablation = ablation_pkg  # type: ignore[attr-defined]

    adapter_mod = types.ModuleType("chuk_lazarus.introspection.ablation.adapter")

    class StubModelAdapter:
        def __init__(self, model: Any = None, tokenizer: Any = None, config: Any = None) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.config = config

    adapter_mod.ModelAdapter = StubModelAdapter  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.ablation.adapter"] = adapter_mod
    ablation_pkg.adapter = adapter_mod  # type: ignore[attr-defined]

    study_mod = types.ModuleType("chuk_lazarus.introspection.ablation.study")

    class StubAblationStudy:
        def __init__(self, adapter: Any = None) -> None:
            self.adapter = adapter

        def ablate_and_generate(self, **kwargs: Any) -> str:
            return "ablated output stub"

    study_mod.AblationStudy = StubAblationStudy  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.ablation.study"] = study_mod
    ablation_pkg.study = study_mod  # type: ignore[attr-defined]

    config_mod = types.ModuleType("chuk_lazarus.introspection.ablation.config")

    from enum import Enum

    class StubComponentType(str, Enum):
        MLP = "mlp"
        ATTENTION = "attention"
        BOTH = "both"

    class StubAblationConfig:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    config_mod.ComponentType = StubComponentType  # type: ignore[attr-defined]
    config_mod.AblationConfig = StubAblationConfig  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.ablation.config"] = config_mod
    ablation_pkg.config = config_mod  # type: ignore[attr-defined]

    # ---- interventions stub ----
    interventions_mod = types.ModuleType("chuk_lazarus.introspection.interventions")

    class StubPatchResult:
        def __init__(self) -> None:
            self.patched_output = "patched output stub"
            self.corrupt_output = "corrupt output stub"
            self.clean_output = "clean output stub"
            self.recovery_rate = 0.75
            self.effect_size = 0.25

    class StubCounterfactualIntervention:
        def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
            self.model = model
            self.tokenizer = tokenizer

        def patch_run(self, **kwargs: Any) -> StubPatchResult:
            return StubPatchResult()

    interventions_mod.CounterfactualIntervention = StubCounterfactualIntervention  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.interventions"] = interventions_mod
    introspection.interventions = interventions_mod  # type: ignore[attr-defined]

    # ---- circuit stubs (for direction_tools) ----
    from enum import Enum as _Enum

    circuit_pkg = types.ModuleType("chuk_lazarus.introspection.circuit")
    sys.modules["chuk_lazarus.introspection.circuit"] = circuit_pkg
    introspection.circuit = circuit_pkg  # type: ignore[attr-defined]

    # collector stub
    collector_mod = types.ModuleType("chuk_lazarus.introspection.circuit.collector")

    class StubCollectedActivations:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def get_activations_numpy(self, layer: int) -> Any:
            acts = self.hidden_states.get(layer)  # type: ignore[attr-defined]
            if acts is None:
                return None
            if hasattr(acts, "_data"):
                return np.array(acts._data, dtype=np.float32, copy=False)
            return np.array(acts, dtype=np.float32)

    collector_mod.CollectedActivations = StubCollectedActivations  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.circuit.collector"] = collector_mod
    circuit_pkg.collector = collector_mod  # type: ignore[attr-defined]

    # directions stub
    directions_mod = types.ModuleType("chuk_lazarus.introspection.circuit.directions")

    class StubDirectionMethod(str, _Enum):
        DIFFERENCE_OF_MEANS = "diff_means"
        LDA = "lda"
        PROBE_WEIGHTS = "probe_weights"
        CONTRASTIVE = "contrastive"
        PCA = "pca"

    class StubExtractedDirection:
        def __init__(self, **kwargs: Any) -> None:
            self.direction = kwargs.get("direction", np.zeros(64))
            self.separation_score = kwargs.get("separation_score", 1.0)
            self.accuracy = kwargs.get("accuracy", 0.9)
            self.mean_projection_positive = kwargs.get("mean_projection_positive", 0.5)
            self.mean_projection_negative = kwargs.get("mean_projection_negative", -0.5)
            self.positive_label = kwargs.get("positive_label", "positive")
            self.negative_label = kwargs.get("negative_label", "negative")

    class StubDirectionExtractor:
        def __init__(self, activations: Any) -> None:
            self.activations = activations

        def extract_direction(
            self,
            layer: int,
            method: Any = None,
            positive_label: int = 1,
            negative_label: int = 0,
        ) -> StubExtractedDirection:
            X = self.activations.get_activations_numpy(layer)
            labels = np.array(self.activations.labels)
            pos_acts = X[labels == positive_label]
            neg_acts = X[labels == negative_label]
            direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                normed = direction / norm
            else:
                normed = direction
            # Compute separation score (Cohen's d)
            pos_proj = pos_acts @ normed
            neg_proj = neg_acts @ normed
            pooled_std = np.sqrt((np.var(pos_proj) + np.var(neg_proj)) / 2 + 1e-8)
            sep = float(abs(np.mean(pos_proj) - np.mean(neg_proj)) / pooled_std)
            # Accuracy via midpoint threshold
            midpoint = (np.mean(pos_proj) + np.mean(neg_proj)) / 2
            all_proj = np.concatenate([pos_proj, neg_proj])
            all_labels = np.concatenate([np.ones(len(pos_proj)), np.zeros(len(neg_proj))])
            preds = (all_proj > midpoint).astype(float)
            acc = float(np.mean(preds == all_labels))
            return StubExtractedDirection(
                direction=direction,
                separation_score=sep,
                accuracy=acc,
                mean_projection_positive=float(np.mean(pos_proj)),
                mean_projection_negative=float(np.mean(neg_proj)),
            )

    directions_mod.DirectionMethod = StubDirectionMethod  # type: ignore[attr-defined]
    directions_mod.DirectionExtractor = StubDirectionExtractor  # type: ignore[attr-defined]
    directions_mod.ExtractedDirection = StubExtractedDirection  # type: ignore[attr-defined]
    sys.modules["chuk_lazarus.introspection.circuit.directions"] = directions_mod
    circuit_pkg.directions = directions_mod  # type: ignore[attr-defined]


_install_lazarus_stubs()


# ---------------------------------------------------------------------------
# 3. chuk-mcp-server stub
# ---------------------------------------------------------------------------


def _install_mcp_server_stub() -> None:
    """Install a stub for chuk_mcp_server so server.py can import."""
    if "chuk_mcp_server" in sys.modules:
        return

    mod = types.ModuleType("chuk_mcp_server")

    class ChukMCPServer:
        def __init__(self, **kwargs: Any) -> None:
            self.name = kwargs.get("name", "test")
            self.version = kwargs.get("version", "0.0.0")
            self.title = kwargs.get("title", "")
            self.description = kwargs.get("description", "")
            self._tools: dict[str, Any] = {}
            self._resources: dict[str, Any] = {}

        def tool(self, **kwargs: Any) -> Any:
            def decorator(fn: Any) -> Any:
                self._tools[fn.__name__] = fn
                return fn

            return decorator

        def resource(self, uri: str, **kwargs: Any) -> Any:
            def decorator(fn: Any) -> Any:
                self._resources[uri] = fn
                return fn

            return decorator

    mod.ChukMCPServer = ChukMCPServer  # type: ignore[attr-defined]
    sys.modules["chuk_mcp_server"] = mod


_install_mcp_server_stub()


# ---------------------------------------------------------------------------
# 4. chuk-virtual-expert stub (via bootstrap)
# ---------------------------------------------------------------------------


def _install_virtual_expert_stub() -> None:
    """Install stub for chuk_virtual_expert if missing."""
    if "chuk_virtual_expert" in sys.modules:
        return
    stub = types.ModuleType("chuk_virtual_expert")
    stub.__version__ = "0.0.0-stub"  # type: ignore[attr-defined]
    for name in (
        "VirtualExpert",
        "VirtualExpertResult",
        "VirtualExpertAction",
        "VirtualExpertPlugin",
        "VirtualExpertRegistry",
    ):
        setattr(stub, name, type(name, (), {}))
    sys.modules["chuk_virtual_expert"] = stub


_install_virtual_expert_stub()


# ---------------------------------------------------------------------------
# 5. Now import source modules (stubs are in place)
# ---------------------------------------------------------------------------

from chuk_mcp_lazarus.model_state import ModelMetadata, ModelState  # noqa: E402
from chuk_mcp_lazarus.comparison_state import ComparisonState  # noqa: E402
from chuk_mcp_lazarus.probe_store import ProbeRegistry  # noqa: E402
from chuk_mcp_lazarus.steering_store import SteeringVectorRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# 6. Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata() -> ModelMetadata:
    """Pre-filled model metadata for tests."""
    return ModelMetadata(
        model_id="test/model",
        family="test",
        architecture="test_arch",
        num_layers=4,
        hidden_dim=64,
        num_attention_heads=4,
        num_kv_heads=4,
        vocab_size=100,
        intermediate_size=256,
        max_position_embeddings=512,
        head_dim=16,
        parameter_count=1000,
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer with encode/decode/eos_token_id."""
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)
    tok.eos_token_id = 0
    return tok


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock model that returns logits when called."""
    import mlx.core as mx

    model = MagicMock()

    # Model call returns logits shape [batch, seq, vocab]
    logits = mx.array(np.random.randn(1, 5, 100).astype(np.float32))
    model.return_value = logits

    # Embed tokens weight
    model.model.embed_tokens.weight = mx.array(np.random.randn(100, 64).astype(np.float32))

    # Layers (4 layers with attention and mlp sub-modules)
    layers = []
    for _ in range(4):
        layer = MagicMock()
        # Attention weights
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(layer.self_attn, proj).weight = mx.array(
                np.random.randn(64, 64).astype(np.float32)
            )
        # MLP weights
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            getattr(layer.mlp, proj).weight = mx.array(np.random.randn(64, 64).astype(np.float32))
        layers.append(layer)
    model.model.layers = layers

    # lm_head
    model.lm_head = MagicMock()
    model.lm_head.weight = mx.array(np.random.randn(100, 64).astype(np.float32))

    # parameters()
    model.parameters.return_value = {"w": MagicMock(size=1000)}

    return model


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock model config with standard attributes."""
    config = MagicMock()
    config.num_hidden_layers = 4
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.vocab_size = 100
    config.intermediate_size = 256
    config.max_position_embeddings = 512
    config.head_dim = 16
    config.model_type = "test_arch"
    config.num_local_experts = None
    return config


@pytest.fixture
def loaded_model_state(
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
    mock_config: MagicMock,
    mock_metadata: ModelMetadata,
) -> Any:
    """Patch ModelState.get() to return a loaded state."""
    state = MagicMock()
    state.is_loaded = True
    state.model = mock_model
    state.tokenizer = mock_tokenizer
    state.config = mock_config
    state.metadata = mock_metadata

    with patch("chuk_mcp_lazarus.model_state.ModelState.get", return_value=state):
        yield state


@pytest.fixture
def unloaded_model_state() -> Any:
    """Patch ModelState.get() to return an unloaded state."""
    state = MagicMock()
    state.is_loaded = False

    with patch("chuk_mcp_lazarus.model_state.ModelState.get", return_value=state):
        yield state


@pytest.fixture(autouse=True)
def sync_to_thread() -> Any:
    """Replace asyncio.to_thread with synchronous execution."""
    original = asyncio.to_thread

    async def _sync_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    with patch("asyncio.to_thread", side_effect=_sync_to_thread):
        yield

    # Restore
    asyncio.to_thread = original


@pytest.fixture(autouse=True)
def reset_singletons() -> Any:
    """Reset all singletons after each test."""
    yield
    ModelState._instance = None
    ComparisonState._instance = None
    ProbeRegistry._instance = None
    SteeringVectorRegistry._instance = None
