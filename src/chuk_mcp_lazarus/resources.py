"""
MCP resources: read-only state exposed to clients.

Resources provide persistent context that Claude can reference
without making tool calls. They update automatically as tools
modify state (loading models, training probes, computing vectors).
"""

from __future__ import annotations

from .comparison_state import ComparisonState
from .model_state import ModelState
from .probe_store import ProbeRegistry
from .server import mcp
from .steering_store import SteeringVectorRegistry


@mcp.resource("model://info", mime_type="application/json")
def model_info_resource() -> dict:
    """Current model metadata.

    Returns architecture details for the loaded model: model_id,
    family, num_layers, hidden_dim, num_attention_heads, vocab_size,
    and more. Empty dict if no model is loaded.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return {"loaded": False}
    return {"loaded": True, **state.metadata.model_dump()}


@mcp.resource("probes://registry", mime_type="application/json")
def probes_registry_resource() -> dict:
    """All trained probes and their accuracy metrics.

    Returns a list of probe summaries: name, layer, classes,
    probe_type, val_accuracy, trained_at. Updates as probes
    are trained or removed.
    """
    return ProbeRegistry.get().dump().model_dump()


@mcp.resource("vectors://registry", mime_type="application/json")
def vectors_registry_resource() -> dict:
    """All computed steering vectors.

    Returns a list of vector summaries: name, layer, vector_norm,
    separability_score, computed_at. Updates as vectors are
    computed or removed.
    """
    return SteeringVectorRegistry.get().dump().model_dump()


@mcp.resource("comparisons://state", mime_type="application/json")
def comparison_state_resource() -> dict:
    """Current comparison model state.

    Returns metadata for the loaded comparison model, or
    loaded=False if no comparison model is active. Used by
    compare_weights, compare_representations, and compare_attention.
    """
    comp = ComparisonState.get()
    if not comp.is_loaded:
        return {"loaded": False}
    return {"loaded": True, **comp.metadata.model_dump()}
