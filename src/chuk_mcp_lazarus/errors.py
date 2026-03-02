"""
Error types and envelope model.

Tools never raise exceptions. They return a ToolErrorResult via
make_error() so the MCP session stays alive. The model is serialised
to a dict at the tool boundary with .model_dump().
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ToolError(str, Enum):
    """Enumerated error types returned by tools."""

    MODEL_NOT_LOADED = "ModelNotLoaded"
    LAYER_OUT_OF_RANGE = "LayerOutOfRange"
    PROBE_NOT_FOUND = "ProbeNotFound"
    VECTOR_NOT_FOUND = "VectorNotFound"
    INVALID_INPUT = "InvalidInput"
    EXTRACTION_FAILED = "ExtractionFailed"
    TRAINING_FAILED = "TrainingFailed"
    EVALUATION_FAILED = "EvaluationFailed"
    GENERATION_FAILED = "GenerationFailed"
    ABLATION_FAILED = "AblationFailed"
    COMPARISON_FAILED = "ComparisonFailed"
    COMPARISON_INCOMPATIBLE = "ComparisonIncompatible"
    LOAD_FAILED = "LoadFailed"


class ToolErrorResult(BaseModel):
    """Structured error envelope returned by tools on failure."""

    error: bool = Field(True, description="Always True for error responses.")
    error_type: ToolError = Field(..., description="Categorised error type.")
    message: str = Field(..., description="Human-readable error description.")
    tool: str = Field(..., description="Name of the tool that failed.")


def make_error(error_type: ToolError, message: str, tool: str) -> dict:
    """Build a structured error envelope as a JSON-safe dict."""
    return ToolErrorResult(
        error_type=error_type, message=message, tool=tool
    ).model_dump()
