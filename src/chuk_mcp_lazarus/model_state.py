"""
ModelState singleton.

Owns the loaded MLX model, tokenizer, config, and family info.
Thread-safe via threading.Lock. No MCP imports, no tool logic.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class WeightDType(str, Enum):
    """Supported weight data types."""

    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class ModelMetadata(BaseModel):
    """Immutable snapshot of loaded model metadata.

    Serialise to dict at the tool boundary with .model_dump().
    """

    model_config = ConfigDict(frozen=True)

    model_id: str = Field("", description="HuggingFace model ID or local path.")
    family: str = Field("", description="Detected model family (gemma, llama, etc.).")
    architecture: str = Field("", description="HF model_type string.")
    num_layers: int = Field(0, description="Number of transformer layers.")
    hidden_dim: int = Field(0, description="Hidden dimension size.")
    num_attention_heads: int = Field(0, description="Number of attention heads.")
    num_kv_heads: int = Field(0, description="Number of key-value heads (GQA).")
    vocab_size: int = Field(0, description="Vocabulary size.")
    intermediate_size: int = Field(0, description="FFN intermediate dimension.")
    max_position_embeddings: int = Field(0, description="Maximum sequence length.")
    head_dim: int = Field(0, description="Per-head dimension.")
    is_moe: bool = Field(False, description="Whether model uses mixture-of-experts.")
    num_experts: int | None = Field(None, description="Number of MoE experts, if applicable.")
    parameter_count: int = Field(0, description="Total parameter count.")


class LoadModelResult(BaseModel):
    """Result returned by load_model tool."""

    model_id: str
    family: str
    architecture: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    parameter_count: int
    status: str = "loaded"


class _InternalState(BaseModel):
    """Internal mutable state held by the singleton.

    Uses arbitrary_types_allowed for MLX model and HF tokenizer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = None
    tokenizer: Any = None
    config: Any = None
    family_info: Any = None
    pipeline: Any = None
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)  # type: ignore[arg-type]
    loaded: bool = False


class ModelState:
    """Singleton managing the currently loaded model.

    Usage:
        state = ModelState.get()
        state.load("google/gemma-3-4b-it")
        info = state.metadata.model_dump()
    """

    _instance: ModelState | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._state = _InternalState()
        self._load_lock = threading.Lock()

    @classmethod
    def get(cls) -> ModelState:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._state.loaded

    @property
    def model(self) -> Any:
        return self._state.model

    @property
    def tokenizer(self) -> Any:
        return self._state.tokenizer

    @property
    def config(self) -> Any:
        return self._state.config

    @property
    def metadata(self) -> ModelMetadata:
        return self._state.metadata

    def load(self, model_id: str, dtype: WeightDType = WeightDType.BFLOAT16) -> ModelMetadata:
        """Load a model. Idempotent: returns immediately if same model_id is already loaded."""
        with self._load_lock:
            if self._state.loaded and self._state.metadata.model_id == model_id:
                logger.info("Model %s already loaded, skipping.", model_id)
                return self._state.metadata

            return self._do_load(model_id, dtype)

    def _do_load(self, model_id: str, dtype: WeightDType) -> ModelMetadata:
        """Perform the actual model load."""
        from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
        from chuk_lazarus.inference.loader import DType

        dtype_enum = DType(dtype.value)
        pipeline_config = UnifiedPipelineConfig(dtype=dtype_enum)

        logger.info("Loading model %s (dtype=%s)...", model_id, dtype.value)
        pipeline = UnifiedPipeline.from_pretrained(
            model_id, pipeline_config=pipeline_config, verbose=True
        )

        metadata = self._extract_metadata(model_id, pipeline)

        self._state = _InternalState(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            config=pipeline.config,
            family_info=pipeline.family,
            pipeline=pipeline,
            metadata=metadata,
            loaded=True,
        )

        logger.info(
            "Model loaded: %s (%s, %d layers, hidden=%d)",
            model_id,
            metadata.family,
            metadata.num_layers,
            metadata.hidden_dim,
        )
        return metadata

    def _extract_metadata(self, model_id: str, pipeline: Any) -> ModelMetadata:
        """Build ModelMetadata from a loaded pipeline."""
        config = pipeline.config
        family_info = pipeline.family

        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_dim = getattr(config, "hidden_size", 0)
        num_heads = getattr(config, "num_attention_heads", 0)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        vocab_size = getattr(config, "vocab_size", 0)
        intermediate_size = getattr(config, "intermediate_size", 0)
        max_pos = getattr(config, "max_position_embeddings", 0)
        head_dim = getattr(config, "head_dim", hidden_dim // max(num_heads, 1))
        model_type = getattr(config, "model_type", "unknown")

        is_moe = getattr(config, "num_local_experts", None) is not None
        num_experts = getattr(config, "num_local_experts", None)

        param_count = self._count_parameters(pipeline.model)
        family_name = family_info.family_type.value if family_info else "unknown"

        return ModelMetadata(
            model_id=model_id,
            family=family_name,
            architecture=model_type,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_pos,
            head_dim=head_dim,
            is_moe=is_moe,
            num_experts=num_experts,
            parameter_count=param_count,
        )

    @staticmethod
    def _count_parameters(model: Any) -> int:
        """Count total parameters in the model."""

        def _count(params: dict) -> int:
            total = 0
            for v in params.values():
                if isinstance(v, dict):
                    total += _count(v)
                elif hasattr(v, "size"):
                    total += v.size
            return total

        try:
            return _count(model.parameters())
        except Exception:
            return 0

    def require_loaded(self) -> None:
        """Raise ValueError if no model is loaded."""
        if not self._state.loaded:
            raise ValueError("No model loaded. Call load_model() first.")
