"""
Experiment persistence tools: create, add results, get, list.

Experiments collect results across multiple tool calls and persist
them to disk at ~/.chuk-lazarus/experiments/. This enables multi-step
interpretability workflows that survive session restarts.
"""

import logging
from typing import Any

from ..errors import ToolError, make_error
from ..experiment_store import ExperimentStore
from ..model_state import ModelState
from ..server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
async def create_experiment(
    name: str,
    description: str = "",
    tags: list[str] | None = None,
) -> dict:
    """
    Create a new experiment to collect results across multiple tool calls.

    Captures the currently loaded model ID. Returns the experiment_id
    to use with add_experiment_result and get_experiment.

    Args:
        name:        Human-readable experiment name.
        description: Optional description of the experiment.
        tags:        Optional tags for categorization.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "create_experiment",
        )

    try:
        store = ExperimentStore.get()
        experiment_id = store.create(
            name=name,
            model_id=state.metadata.model_id,
            description=description,
            tags=tags or [],
        )
        return {
            "experiment_id": experiment_id,
            "name": name,
            "model_id": state.metadata.model_id,
            "description": description,
            "tags": tags or [],
        }
    except Exception as e:
        logger.exception("create_experiment failed")
        return make_error(ToolError.EXPERIMENT_STORE_ERROR, str(e), "create_experiment")


@mcp.tool()
async def add_experiment_result(
    experiment_id: str,
    step_name: str,
    result: dict[str, Any],
) -> dict:
    """
    Add a result step to an existing experiment. Auto-saves to disk.

    Typically called after running an analysis tool, passing the tool's
    result dict as the result parameter.

    Args:
        experiment_id: ID from create_experiment.
        step_name:     Name for this step (e.g. "logit_attribution_layer_5").
        result:        Result data dict from any tool call.
    """
    store = ExperimentStore.get()
    try:
        store.add_result(experiment_id, step_name, result)
    except KeyError:
        return make_error(
            ToolError.EXPERIMENT_NOT_FOUND,
            f"Experiment {experiment_id} not found.",
            "add_experiment_result",
        )
    except Exception as e:
        logger.exception("add_experiment_result failed")
        return make_error(ToolError.EXPERIMENT_STORE_ERROR, str(e), "add_experiment_result")

    exp = store.get_experiment(experiment_id)
    return {
        "experiment_id": experiment_id,
        "step_name": step_name,
        "total_steps": len(exp.steps) if exp else 0,
    }


@mcp.tool(read_only_hint=True)
async def get_experiment(experiment_id: str) -> dict:
    """
    Retrieve an experiment and all its results.

    Returns full metadata and all result steps.

    Args:
        experiment_id: ID from create_experiment.
    """
    store = ExperimentStore.get()
    exp = store.get_experiment(experiment_id)
    if exp is None:
        # Try loading from disk
        if store.load_from_disk(experiment_id):
            exp = store.get_experiment(experiment_id)

    if exp is None:
        return make_error(
            ToolError.EXPERIMENT_NOT_FOUND,
            f"Experiment {experiment_id} not found.",
            "get_experiment",
        )

    return exp.model_dump()


@mcp.tool(read_only_hint=True)
async def list_experiments() -> dict:
    """
    List all experiments with metadata.

    Returns experiment summaries including name, model_id, creation time,
    and number of result steps.
    """
    store = ExperimentStore.get()
    return store.list_experiments()
