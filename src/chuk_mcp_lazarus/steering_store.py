"""
SteeringVectorRegistry singleton.

Stores computed steering vectors (numpy arrays) and their metadata in memory.
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class VectorMetadata(BaseModel):
    """Immutable metadata for a computed steering vector."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique vector name.")
    layer: int = Field(..., description="Layer the vector was computed at.")
    vector_norm: float = Field(..., description="L2 norm of the steering vector.")
    separability_score: float = Field(..., description="Cosine distance between class means.")
    num_positive: int = Field(..., description="Number of positive prompts.")
    num_negative: int = Field(..., description="Number of negative prompts.")
    computed_at: str = Field(..., description="ISO timestamp of computation.")


class VectorListEntry(BaseModel):
    """Summary entry for list_steering_vectors results."""

    name: str
    layer: int
    vector_norm: float
    separability_score: float
    computed_at: str


class VectorRegistryDump(BaseModel):
    """Full registry dump for MCP resource."""

    vectors: list[VectorListEntry]
    count: int


class SteeringVectorRegistry:
    """In-memory registry for steering vectors.

    Usage:
        registry = SteeringVectorRegistry.get()
        registry.store("en_to_de", vector, metadata)
        vector, meta = registry.fetch("en_to_de")
    """

    _instance: SteeringVectorRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[np.ndarray, VectorMetadata]] = {}
        self._access_lock = threading.Lock()

    @classmethod
    def get(cls) -> SteeringVectorRegistry:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, name: str, vector: np.ndarray, metadata: VectorMetadata) -> None:
        """Store a steering vector."""
        with self._access_lock:
            self._vectors[name] = (vector, metadata)

    def fetch(self, name: str) -> tuple[np.ndarray, VectorMetadata] | None:
        """Fetch a vector by name. Returns None if not found."""
        with self._access_lock:
            return self._vectors.get(name)

    def exists(self, name: str) -> bool:
        """Check if a vector exists."""
        with self._access_lock:
            return name in self._vectors

    def list_all(self) -> list[VectorListEntry]:
        """Return summary entries for all vectors."""
        with self._access_lock:
            return [
                VectorListEntry(
                    name=meta.name,
                    layer=meta.layer,
                    vector_norm=meta.vector_norm,
                    separability_score=meta.separability_score,
                    computed_at=meta.computed_at,
                )
                for _, (_, meta) in sorted(self._vectors.items())
            ]

    @property
    def count(self) -> int:
        with self._access_lock:
            return len(self._vectors)

    def clear(self) -> None:
        """Remove all vectors."""
        with self._access_lock:
            self._vectors.clear()

    def dump(self) -> VectorRegistryDump:
        """Full registry dump for MCP resource."""
        entries = self.list_all()
        return VectorRegistryDump(vectors=entries, count=len(entries))
