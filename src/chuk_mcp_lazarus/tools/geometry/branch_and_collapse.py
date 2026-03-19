"""
branch_and_collapse — non-collapsing superposition via parallel branch evolution.

Captures the residual stream from a donor prompt at a specified layer,
injects it into multiple template prompts ("force fields"), runs the
remaining layers independently for each branch, and collapses to the
highest-confidence branch.  One tool call wraps the full quantum protocol:
capture → branch → evolve → measure → collapse.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._injection_helpers import (
    TokenPrediction,
    _run_forward_with_injection,
    _softmax_np,
    _top_k_from_logits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class BranchResult(BaseModel):
    """Prediction from one branch (template + injected residual)."""

    branch_idx: int
    template: str
    top_prediction: TokenPrediction
    top_k: list[TokenPrediction]
    confidence: float = Field(..., description="Top-1 probability.")
    entropy: float = Field(..., description="Distribution entropy (nats).")


class DonorBaseline(BaseModel):
    """Donor's own prediction without branching (baseline comparison)."""

    top_prediction: TokenPrediction
    top_k: list[TokenPrediction]
    confidence: float
    entropy: float


class CollapseResult(BaseModel):
    """The collapsed branch — highest confidence wins."""

    selected_branch: int
    template: str
    token: str
    token_id: int
    confidence: float
    entropy: float


class BranchAndCollapseResult(BaseModel):
    """Result from branch_and_collapse."""

    donor_prompt: str
    layer: int
    donor_layer: int | None
    num_branches: int
    donor_baseline: DonorBaseline
    branches: list[BranchResult]
    collapse: CollapseResult
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entropy_nats(logits_np: np.ndarray) -> float:
    """Shannon entropy in nats from logits."""
    probs = _softmax_np(logits_np)
    return -float(np.sum(probs * np.log(probs + 1e-10)))


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True)
async def branch_and_collapse(
    donor_prompt: str,
    branch_prompts: list[str],
    layer: int,
    donor_layer: int | None = None,
    token_position: int = -1,
    top_k: int = 10,
) -> dict:
    """Non-collapsing superposition via parallel branch evolution.

    Captures the residual stream from a donor prompt, injects it into
    multiple template prompts (force fields), runs the remaining layers
    independently, and collapses to the highest-confidence branch.

    Use this when the donor prompt produces an uncertain prediction
    (e.g. 60/37 split) — branching through clean templates can resolve
    the superposition to 80-98% confidence.

    Args:
        donor_prompt:    Prompt containing the uncollapsed superposition.
        branch_prompts:  Template prompts (force fields) to evolve through (2-20).
        layer:           Injection layer (into each branch template).
        donor_layer:     Capture layer from donor (defaults to layer).
        token_position:  Token position to inject/read (-1 = last).
        top_k:           Predictions per branch (1-50).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "branch_and_collapse",
        )
    meta = state.metadata

    # -- MCP coercion and validation --
    layer = int(layer)
    if layer < 0 or layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of [0, {meta.num_layers - 1}].",
            "branch_and_collapse",
        )

    donor_layer = int(donor_layer) if donor_layer is not None else None
    effective_donor_layer = donor_layer if donor_layer is not None else layer
    if effective_donor_layer < 0 or effective_donor_layer >= meta.num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"donor_layer {effective_donor_layer} out of [0, {meta.num_layers - 1}].",
            "branch_and_collapse",
        )

    if len(branch_prompts) < 2:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 2 branch prompts required.",
            "branch_and_collapse",
        )
    if len(branch_prompts) > 20:
        return make_error(
            ToolError.INVALID_INPUT,
            "Maximum 20 branch prompts.",
            "branch_and_collapse",
        )
    if top_k < 1 or top_k > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k must be in [1, 50].",
            "branch_and_collapse",
        )

    try:
        return await asyncio.to_thread(
            _branch_and_collapse_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            donor_prompt,
            branch_prompts,
            layer,
            donor_layer,
            token_position,
            top_k,
        )
    except Exception as exc:
        logger.exception("branch_and_collapse failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "branch_and_collapse")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _branch_and_collapse_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    donor_prompt: str,
    branch_prompts: list[str],
    layer: int,
    donor_layer: int | None,
    token_position: int,
    top_k: int,
) -> dict:
    """Sync implementation of branch_and_collapse."""
    import mlx.core as mx

    from ..._residual_helpers import (
        _extract_position,
        _get_lm_projection,
        _norm_project,
        _run_decomposition_forward,
    )
    from chuk_lazarus.introspection.hooks import ModelHooks

    # -- Phase 1: Setup --
    donor_ids_list = tokenizer.encode(donor_prompt, add_special_tokens=True)
    donor_ids = mx.array(donor_ids_list)
    num_donor = len(donor_ids_list)
    d_pos = token_position if token_position >= 0 else num_donor + token_position
    d_pos = max(0, min(d_pos, num_donor - 1))

    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not access lm_head.",
            "branch_and_collapse",
        )

    last_layer = meta.num_layers - 1
    effective_donor_layer = donor_layer if donor_layer is not None else layer

    # -- Phase 2: Capture donor --
    donor_capture_layers = sorted(set([effective_donor_layer, last_layer]))
    donor_decomp = _run_decomposition_forward(model, config, donor_ids, donor_capture_layers)
    mx.eval(*donor_decomp["hidden_states"].values())

    # Donor vector for injection (at capture layer)
    donor_vec_at_layer = _extract_position(
        donor_decomp["hidden_states"][effective_donor_layer], d_pos
    )

    # Donor baseline prediction (at final layer)
    donor_vec_final = _extract_position(donor_decomp["hidden_states"][last_layer], d_pos)
    donor_logits = _norm_project(final_norm, lm_head, donor_vec_final)
    mx.eval(donor_logits)
    donor_logits_np = np.array(donor_logits.tolist(), dtype=np.float32)
    donor_top_k, donor_top_id, donor_top_token, donor_top_prob = _top_k_from_logits(
        donor_logits_np, tokenizer, top_k
    )
    donor_entropy = _entropy_nats(donor_logits_np)

    donor_baseline = DonorBaseline(
        top_prediction=TokenPrediction(
            token=donor_top_token,
            token_id=donor_top_id,
            probability=round(donor_top_prob, 6),
        ),
        top_k=donor_top_k,
        confidence=round(donor_top_prob, 6),
        entropy=round(donor_entropy, 6),
    )

    # -- Phase 3: Branch and evolve --
    # Convert donor vector to mx array once
    donor_vec_np = np.array(donor_vec_at_layer.tolist(), dtype=np.float32)
    donor_vec_mx = mx.array(donor_vec_np.tolist())

    branches: list[BranchResult] = []

    for i, template in enumerate(branch_prompts):
        template_ids_list = tokenizer.encode(template, add_special_tokens=True)
        template_ids = mx.array(template_ids_list)
        num_template = len(template_ids_list)

        # Resolve injection position for this template
        t_pos = token_position if token_position >= 0 else num_template + token_position
        t_pos = max(0, min(t_pos, num_template - 1))

        # Inject donor residual and run remaining layers
        final_hidden = _run_forward_with_injection(
            model, config, template_ids, layer, t_pos, donor_vec_mx, meta
        )

        # Extract logits at injection position
        branch_vec = _extract_position(final_hidden, t_pos)
        branch_logits = _norm_project(final_norm, lm_head, branch_vec)
        mx.eval(branch_logits)
        branch_logits_np = np.array(branch_logits.tolist(), dtype=np.float32)

        branch_top_k, b_top_id, b_top_token, b_top_prob = _top_k_from_logits(
            branch_logits_np, tokenizer, top_k
        )
        branch_entropy = _entropy_nats(branch_logits_np)

        branches.append(
            BranchResult(
                branch_idx=i,
                template=template,
                top_prediction=TokenPrediction(
                    token=b_top_token,
                    token_id=b_top_id,
                    probability=round(b_top_prob, 6),
                ),
                top_k=branch_top_k,
                confidence=round(b_top_prob, 6),
                entropy=round(branch_entropy, 6),
            )
        )

    # -- Phase 4: Collapse --
    best = max(branches, key=lambda b: b.confidence)
    collapse = CollapseResult(
        selected_branch=best.branch_idx,
        template=best.template,
        token=best.top_prediction.token,
        token_id=best.top_prediction.token_id,
        confidence=best.confidence,
        entropy=best.entropy,
    )

    # -- Phase 5: Summary --
    mean_conf = sum(b.confidence for b in branches) / len(branches)
    agreeing = sum(1 for b in branches if b.top_prediction.token_id == collapse.token_id)

    summary: dict[str, Any] = {
        "donor_token": donor_baseline.top_prediction.token,
        "collapsed_token": collapse.token,
        "collapse_matches_donor": (collapse.token_id == donor_baseline.top_prediction.token_id),
        "collapsed_confidence": collapse.confidence,
        "donor_confidence": donor_baseline.confidence,
        "confidence_gain": round(collapse.confidence - donor_baseline.confidence, 6),
        "mean_branch_confidence": round(mean_conf, 6),
        "branches_agreeing_with_collapse": agreeing,
        "entropy_reduction": round(donor_baseline.entropy - collapse.entropy, 6),
    }
    if donor_layer is not None and donor_layer != layer:
        summary["donor_layer"] = effective_donor_layer

    # -- Phase 6: Return --
    return BranchAndCollapseResult(
        donor_prompt=donor_prompt,
        layer=layer,
        donor_layer=donor_layer,
        num_branches=len(branch_prompts),
        donor_baseline=donor_baseline,
        branches=branches,
        collapse=collapse,
        summary=summary,
    ).model_dump()
