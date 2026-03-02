"""
ProbeRegistry singleton.

Stores trained sklearn probe classifiers and their metadata in memory.
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProbeType(str, Enum):
    """Supported probe classifier types."""

    LINEAR = "linear"
    MLP = "mlp"


class ProbeMetadata(BaseModel):
    """Immutable metadata for a trained probe."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique probe name.")
    layer: int = Field(..., description="Layer the probe was trained on.")
    probe_type: ProbeType = Field(..., description="Classifier type.")
    classes: list[str] = Field(..., description="Class labels.")
    num_examples: int = Field(..., description="Number of training examples.")
    train_accuracy: float = Field(..., description="Training set accuracy.")
    val_accuracy: float = Field(..., description="Validation set accuracy.")
    coefficients_norm: float | None = Field(None, description="L2 norm of weight matrix.")
    trained_at: str = Field(..., description="ISO timestamp of training.")


class ProbeListEntry(BaseModel):
    """Summary entry for list_probes results."""

    name: str
    layer: int
    classes: list[str]
    probe_type: ProbeType
    val_accuracy: float
    trained_at: str


class ProbeRegistryDump(BaseModel):
    """Full registry dump for MCP resource."""

    probes: list[ProbeListEntry]
    count: int


class ProbeRegistry:
    """In-memory registry for trained probes.

    Usage:
        registry = ProbeRegistry.get()
        registry.store("lang_probe_L12", model, metadata)
        model, meta = registry.fetch("lang_probe_L12")
    """

    _instance: ProbeRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._probes: dict[str, tuple[Any, ProbeMetadata]] = {}
        self._access_lock = threading.Lock()

    @classmethod
    def get(cls) -> ProbeRegistry:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, name: str, model: Any, metadata: ProbeMetadata) -> None:
        """Store a trained probe."""
        with self._access_lock:
            self._probes[name] = (model, metadata)

    def fetch(self, name: str) -> tuple[Any, ProbeMetadata] | None:
        """Fetch a probe by name. Returns None if not found."""
        with self._access_lock:
            return self._probes.get(name)

    def exists(self, name: str) -> bool:
        """Check if a probe exists."""
        with self._access_lock:
            return name in self._probes

    def list_all(self) -> list[ProbeListEntry]:
        """Return summary entries for all probes."""
        with self._access_lock:
            return [
                ProbeListEntry(
                    name=meta.name,
                    layer=meta.layer,
                    classes=meta.classes,
                    probe_type=meta.probe_type,
                    val_accuracy=meta.val_accuracy,
                    trained_at=meta.trained_at,
                )
                for _, (_, meta) in sorted(self._probes.items())
            ]

    @property
    def count(self) -> int:
        with self._access_lock:
            return len(self._probes)

    def clear(self) -> None:
        """Remove all probes."""
        with self._access_lock:
            self._probes.clear()

    def dump(self) -> ProbeRegistryDump:
        """Full registry dump for MCP resource."""
        entries = self.list_all()
        return ProbeRegistryDump(probes=entries, count=len(entries))
