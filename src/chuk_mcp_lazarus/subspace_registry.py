"""
SubspaceRegistry singleton.

Stores computed PCA subspace bases (numpy arrays of shape [rank, hidden_dim])
and their metadata in memory.  Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SubspaceMetadata(BaseModel):
    """Immutable metadata for a computed PCA subspace."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique subspace name.")
    layer: int = Field(..., description="Layer the subspace was computed at.")
    rank: int = Field(..., description="Number of retained PCA components.")
    num_prompts: int = Field(..., description="Number of prompts used for PCA.")
    hidden_dim: int = Field(..., description="Hidden dimension of the model.")
    variance_explained: list[float] = Field(
        ..., description="Per-component variance explained (fraction)."
    )
    total_variance_explained: float = Field(
        ..., description="Cumulative variance explained by all retained components."
    )
    computed_at: str = Field(..., description="ISO timestamp of computation.")


class SubspaceListEntry(BaseModel):
    """Summary entry for list_subspaces results."""

    name: str
    layer: int
    rank: int
    num_prompts: int
    total_variance_explained: float
    computed_at: str


class SubspaceRegistryDump(BaseModel):
    """Full registry dump."""

    subspaces: list[SubspaceListEntry]
    count: int


class SubspaceRegistry:
    """In-memory registry for PCA subspace bases.

    Usage:
        registry = SubspaceRegistry.get()
        registry.store("capital_cities", basis, metadata)
        basis, meta = registry.fetch("capital_cities")
    """

    _instance: SubspaceRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._subspaces: dict[str, tuple[np.ndarray, SubspaceMetadata]] = {}
        self._access_lock = threading.Lock()

    @classmethod
    def get(cls) -> SubspaceRegistry:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, name: str, basis: np.ndarray, metadata: SubspaceMetadata) -> None:
        """Store a PCA basis.  basis shape: [rank, hidden_dim], dtype float32."""
        with self._access_lock:
            self._subspaces[name] = (basis, metadata)

    def fetch(self, name: str) -> tuple[np.ndarray, SubspaceMetadata] | None:
        """Fetch a subspace by name.  Returns None if not found."""
        with self._access_lock:
            return self._subspaces.get(name)

    def exists(self, name: str) -> bool:
        """Check if a subspace exists."""
        with self._access_lock:
            return name in self._subspaces

    def list_all(self) -> list[SubspaceListEntry]:
        """Return summary entries for all subspaces."""
        with self._access_lock:
            return [
                SubspaceListEntry(
                    name=meta.name,
                    layer=meta.layer,
                    rank=meta.rank,
                    num_prompts=meta.num_prompts,
                    total_variance_explained=meta.total_variance_explained,
                    computed_at=meta.computed_at,
                )
                for _, (_, meta) in sorted(self._subspaces.items())
            ]

    @property
    def count(self) -> int:
        with self._access_lock:
            return len(self._subspaces)

    def clear(self) -> None:
        """Remove all subspaces."""
        with self._access_lock:
            self._subspaces.clear()

    def dump(self) -> SubspaceRegistryDump:
        """Full registry dump."""
        entries = self.list_all()
        return SubspaceRegistryDump(subspaces=entries, count=len(entries))
