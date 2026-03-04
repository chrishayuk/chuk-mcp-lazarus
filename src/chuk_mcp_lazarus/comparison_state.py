"""
ComparisonState singleton.

Holds a second model for two-model comparison operations (weight divergence,
activation divergence, attention divergence). Mirrors the ModelState pattern:
threading.Lock, double-checked locking, _InternalState Pydantic model.

The comparison model is loaded on-demand and can be freed to reclaim VRAM.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .model_state import ModelMetadata, WeightDType

logger = logging.getLogger(__name__)


class _InternalState(BaseModel):
    """Internal mutable state held by the singleton."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = None
    tokenizer: Any = None
    config: Any = None
    pipeline: Any = None
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)  # type: ignore[arg-type]
    loaded: bool = False


class ComparisonState:
    """Singleton managing a second model for two-model comparison.

    Usage:
        comp = ComparisonState.get()
        comp.load("google/translategemma-4b-it")
        comp.require_compatible(primary_metadata)
        # ... do comparison ...
        comp.unload()
    """

    _instance: ComparisonState | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._state = _InternalState()
        self._load_lock = threading.Lock()

    @classmethod
    def get(cls) -> ComparisonState:
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
        """Load a comparison model. Idempotent if same model_id."""
        with self._load_lock:
            if self._state.loaded and self._state.metadata.model_id == model_id:
                logger.info("Comparison model %s already loaded, skipping.", model_id)
                return self._state.metadata

            return self._do_load(model_id, dtype)

    def _do_load(self, model_id: str, dtype: WeightDType) -> ModelMetadata:
        """Perform the actual model load."""
        from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
        from chuk_lazarus.inference.loader import DType

        dtype_enum = DType(dtype.value)
        pipeline_config = UnifiedPipelineConfig(dtype=dtype_enum)

        logger.info("Loading comparison model %s (dtype=%s)...", model_id, dtype.value)
        pipeline = UnifiedPipeline.from_pretrained(
            model_id, pipeline_config=pipeline_config, verbose=True
        )

        metadata = self._extract_metadata(model_id, pipeline)

        self._state = _InternalState(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            config=pipeline.config,
            pipeline=pipeline,
            metadata=metadata,
            loaded=True,
        )

        logger.info(
            "Comparison model loaded: %s (%s, %d layers, hidden=%d)",
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
        """Raise ValueError if no comparison model is loaded."""
        if not self._state.loaded:
            raise ValueError("No comparison model loaded. Call load_comparison_model() first.")

    def require_compatible(self, primary: ModelMetadata) -> None:
        """Raise ValueError if comparison model is incompatible with primary."""
        self.require_loaded()
        comp = self._state.metadata

        if comp.num_layers != primary.num_layers:
            raise ValueError(
                f"Layer count mismatch: primary has {primary.num_layers}, "
                f"comparison has {comp.num_layers}."
            )
        if comp.hidden_dim != primary.hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch: primary has {primary.hidden_dim}, "
                f"comparison has {comp.hidden_dim}."
            )

    def unload(self) -> None:
        """Unload the comparison model and free VRAM."""
        import mlx.core as mx

        with self._load_lock:
            if not self._state.loaded:
                return

            model_id = self._state.metadata.model_id
            self._state = _InternalState()
            clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
            if clear:
                clear()
            logger.info("Comparison model unloaded: %s", model_id)
