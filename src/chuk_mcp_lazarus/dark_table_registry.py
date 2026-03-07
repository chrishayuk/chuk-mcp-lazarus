"""
DarkTableRegistry singleton.

Stores precomputed subspace coordinate tables (numpy arrays of shape [rank])
keyed by string labels.  Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DarkTableMetadata(BaseModel):
    """Immutable metadata for a dark coordinate table."""

    model_config = ConfigDict(frozen=True)

    table_name: str = Field(..., description="Unique table name.")
    subspace_name: str = Field(..., description="Subspace used for projection.")
    layer: int = Field(..., description="Layer coordinates were extracted at.")
    rank: int = Field(..., description="Dimensionality of coordinate vectors.")
    num_entries: int = Field(..., description="Number of entries in the table.")
    token_position: int = Field(..., description="Token position used for extraction.")
    computed_at: str = Field(..., description="ISO timestamp of computation.")


class DarkTableListEntry(BaseModel):
    """Summary entry for list_dark_tables results."""

    table_name: str
    subspace_name: str
    layer: int
    rank: int
    num_entries: int
    computed_at: str


class DarkTableRegistryDump(BaseModel):
    """Full registry dump."""

    tables: list[DarkTableListEntry]
    count: int


class DarkTableRegistry:
    """In-memory registry for precomputed dark coordinate tables.

    Usage:
        registry = DarkTableRegistry.get()
        registry.store("math_results", coordinates, metadata)
        coords, meta = registry.fetch("math_results")
        vec = registry.lookup("math_results", "7")
    """

    _instance: DarkTableRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._tables: dict[str, tuple[dict[str, np.ndarray], DarkTableMetadata]] = {}
        self._access_lock = threading.Lock()

    @classmethod
    def get(cls) -> DarkTableRegistry:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(
        self,
        name: str,
        coordinates: dict[str, np.ndarray],
        metadata: DarkTableMetadata,
    ) -> None:
        """Store a coordinate table.  Each value shape: [rank], dtype float32."""
        with self._access_lock:
            self._tables[name] = (coordinates, metadata)

    def fetch(self, name: str) -> tuple[dict[str, np.ndarray], DarkTableMetadata] | None:
        """Fetch a table by name.  Returns None if not found."""
        with self._access_lock:
            return self._tables.get(name)

    def lookup(self, table_name: str, key: str) -> np.ndarray | None:
        """Fetch a single coordinate vector by table name and key."""
        with self._access_lock:
            entry = self._tables.get(table_name)
            if entry is None:
                return None
            coords, _ = entry
            return coords.get(key)

    def exists(self, name: str) -> bool:
        """Check if a table exists."""
        with self._access_lock:
            return name in self._tables

    def list_all(self) -> list[DarkTableListEntry]:
        """Return summary entries for all tables."""
        with self._access_lock:
            return [
                DarkTableListEntry(
                    table_name=meta.table_name,
                    subspace_name=meta.subspace_name,
                    layer=meta.layer,
                    rank=meta.rank,
                    num_entries=meta.num_entries,
                    computed_at=meta.computed_at,
                )
                for _, (_, meta) in sorted(self._tables.items())
            ]

    @property
    def count(self) -> int:
        with self._access_lock:
            return len(self._tables)

    def clear(self) -> None:
        """Remove all tables."""
        with self._access_lock:
            self._tables.clear()

    def dump(self) -> DarkTableRegistryDump:
        """Full registry dump."""
        entries = self.list_all()
        return DarkTableRegistryDump(tables=entries, count=len(entries))
