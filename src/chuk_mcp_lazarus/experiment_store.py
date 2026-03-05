"""
ExperimentStore singleton.

Persists experiment results to disk at ~/.chuk-lazarus/experiments/.
Thread-safe via threading.Lock. Each experiment is a JSON file containing
metadata and an ordered list of result steps.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ExperimentMetadata(BaseModel):
    """Immutable metadata for an experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(..., description="Unique experiment ID (UUID).")
    name: str = Field(..., description="Human-readable experiment name.")
    model_id: str = Field(..., description="Model ID at creation time.")
    created_at: str = Field(..., description="ISO 8601 creation timestamp.")
    description: str = Field("", description="Optional description.")
    tags: list[str] = Field(default_factory=list, description="Optional tags.")


class ExperimentStep(BaseModel):
    """A single result step in an experiment."""

    step_name: str = Field(..., description="Name of this step.")
    recorded_at: str = Field(..., description="ISO 8601 timestamp.")
    data: dict[str, Any] = Field(default_factory=dict, description="Result data.")


class ExperimentSummary(BaseModel):
    """Summary entry for list_experiments."""

    experiment_id: str
    name: str
    model_id: str
    created_at: str
    description: str
    tags: list[str]
    num_steps: int


class ExperimentDetail(BaseModel):
    """Full experiment with metadata and steps."""

    metadata: ExperimentMetadata
    steps: list[ExperimentStep]


class ExperimentListResult(BaseModel):
    """Result of list_experiments."""

    experiments: list[ExperimentSummary]
    count: int


class ExperimentStore:
    """Disk-backed experiment persistence.

    Usage:
        store = ExperimentStore.get()
        eid = store.create("my_experiment", "SmolLM2-135M")
        store.add_result(eid, "step_1", {"accuracy": 0.95})
        exp = store.get_experiment(eid)
    """

    _instance: ExperimentStore | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentDetail] = {}
        self._access_lock = threading.Lock()
        self._base_dir = Path.home() / ".chuk-lazarus" / "experiments"

    @classmethod
    def get(cls) -> ExperimentStore:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for tests)."""
        with cls._lock:
            cls._instance = None

    def create(
        self,
        name: str,
        model_id: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Create a new experiment. Returns the experiment_id."""
        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            model_id=model_id,
            created_at=now,
            description=description,
            tags=tags or [],
        )
        detail = ExperimentDetail(metadata=metadata, steps=[])
        with self._access_lock:
            self._experiments[experiment_id] = detail
        self._save_to_disk(experiment_id)
        return experiment_id

    def add_result(
        self,
        experiment_id: str,
        step_name: str,
        result_data: dict[str, Any],
    ) -> None:
        """Add a result step to an experiment. Auto-saves to disk."""
        now = datetime.now(timezone.utc).isoformat()
        step = ExperimentStep(
            step_name=step_name,
            recorded_at=now,
            data=result_data,
        )
        with self._access_lock:
            exp = self._experiments.get(experiment_id)
            if exp is None:
                raise KeyError(f"Experiment {experiment_id} not found.")
            exp.steps.append(step)
        self._save_to_disk(experiment_id)

    def get_experiment(self, experiment_id: str) -> ExperimentDetail | None:
        """Get full experiment detail. Returns None if not found."""
        with self._access_lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> dict:
        """List all experiments as a JSON-safe dict."""
        with self._access_lock:
            summaries = [
                ExperimentSummary(
                    experiment_id=exp.metadata.experiment_id,
                    name=exp.metadata.name,
                    model_id=exp.metadata.model_id,
                    created_at=exp.metadata.created_at,
                    description=exp.metadata.description,
                    tags=list(exp.metadata.tags),
                    num_steps=len(exp.steps),
                )
                for exp in self._experiments.values()
            ]
        result = ExperimentListResult(experiments=summaries, count=len(summaries))
        return result.model_dump()

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment from memory and disk."""
        with self._access_lock:
            if experiment_id not in self._experiments:
                return False
            del self._experiments[experiment_id]
        # Remove from disk
        path = self._base_dir / f"{experiment_id}.json"
        if path.exists():
            path.unlink()
        return True

    def _save_to_disk(self, experiment_id: str) -> None:
        """Persist an experiment to disk as JSON."""
        with self._access_lock:
            exp = self._experiments.get(experiment_id)
            if exp is None:
                return
            data = exp.model_dump()

        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._base_dir / f"{experiment_id}.json"
            path.write_text(json.dumps(data, indent=2, default=str))
        except OSError:
            logger.warning("Failed to save experiment %s to disk", experiment_id)

    def load_from_disk(self, experiment_id: str) -> bool:
        """Load an experiment from disk into memory."""
        path = self._base_dir / f"{experiment_id}.json"
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text())
            detail = ExperimentDetail(**raw)
            with self._access_lock:
                self._experiments[experiment_id] = detail
            return True
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning("Failed to load experiment %s from disk", experiment_id)
            return False
