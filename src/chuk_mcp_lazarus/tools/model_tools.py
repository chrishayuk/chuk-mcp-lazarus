"""
Model management tools: load_model, get_model_info.

These are the first tools called in any session. load_model must
succeed before any other tool can run.
"""

import logging

from ..errors import ToolError, make_error
from ..model_state import LoadModelResult, ModelState, WeightDType
from ..server import mcp

logger = logging.getLogger(__name__)


@mcp.tool(idempotent_hint=True)
async def load_model(
    model_id: str = "google/gemma-3-4b-it",
    dtype: str = "bfloat16",
) -> dict:
    """
    Load a HuggingFace model into memory.
    Must be called before any other tool.

    Works with any model chuk-lazarus supports: Gemma, Llama, Qwen,
    Granite, Jamba, Mamba, StarCoder2, GPT-2, and more.

    Args:
        model_id: HuggingFace model ID or local path.
        dtype:    Weight dtype. One of: bfloat16, float16, float32.
    """
    try:
        weight_dtype = WeightDType(dtype)
    except ValueError:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid dtype '{dtype}'. Use one of: bfloat16, float16, float32.",
            "load_model",
        )

    try:
        state = ModelState.get()
        metadata = state.load(model_id, dtype=weight_dtype)
        result = LoadModelResult(
            model_id=metadata.model_id,
            family=metadata.family,
            architecture=metadata.architecture,
            num_layers=metadata.num_layers,
            hidden_dim=metadata.hidden_dim,
            num_attention_heads=metadata.num_attention_heads,
            parameter_count=metadata.parameter_count,
        )
        return result.model_dump()
    except Exception as e:
        logger.exception("Failed to load model %s", model_id)
        return make_error(ToolError.LOAD_FAILED, str(e), "load_model")


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def get_model_info() -> dict:
    """
    Return architecture metadata for the currently-loaded model.

    Call load_model first. Returns num_layers, hidden_dim,
    num_attention_heads, vocab_size, and more.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "get_model_info",
        )
    try:
        return state.metadata.model_dump()
    except Exception as e:
        logger.exception("get_model_info failed")
        return make_error(ToolError.EXTRACTION_FAILED, str(e), "get_model_info")
