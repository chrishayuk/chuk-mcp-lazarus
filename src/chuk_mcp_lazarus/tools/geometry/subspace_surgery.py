"""
subspace_surgery — all-position subspace replacement.

Replaces ONLY the subspace component of the residual stream at ALL positions
while preserving the orthogonal complement.  Three source modes:

  donor:       transplant another prompt's subspace projection
  coordinates: inject an explicit [rank]-dim coordinate vector
  lookup:      inject precomputed coordinates from a dark table

Core operation (vectorised across positions):
  projections      = H @ basis.T           # [seq, rank]
  subspace_part    = projections @ basis    # [seq, hidden]
  orthogonal       = H - subspace_part     # preserved
  surgical_H       = orthogonal + new_content
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...dark_table_registry import DarkTableRegistry
from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ...subspace_registry import SubspaceRegistry
from ._helpers import _cosine_sim
from ._injection_helpers import (
    TokenPrediction,
    _generate_from_hidden,
    _run_forward_with_injection,
    _top_k_from_logits,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class SurgeryVerification(BaseModel):
    """Verification that surgery only affected the subspace."""

    orthogonal_cosine: float = Field(
        ...,
        description=("Cosine similarity of orthogonal complements before/after. Should be ~1.0."),
    )
    orthogonal_norm_ratio: float = Field(
        ...,
        description="Ratio of orthogonal norms after/before. Should be ~1.0.",
    )
    surgery_clean: bool = Field(..., description="True if orthogonal_cosine > 0.999.")


class SubspaceEnergyAnalysis(BaseModel):
    """Energy distribution analysis for the surgery."""

    recipient_subspace_energy_fraction: float = Field(
        ...,
        description="Fraction of recipient energy in the subspace (pre-surgery).",
    )
    new_content_energy_fraction: float = Field(
        ...,
        description="Fraction of surgical hidden energy in the subspace (post-surgery).",
    )
    recipient_subspace_norm: float
    recipient_orthogonal_norm: float
    new_content_norm: float


class SubspaceSurgeryResult(BaseModel):
    """Result from subspace_surgery."""

    recipient_prompt: str
    mode: str
    layer: int
    subspace_name: str
    subspace_rank: int
    donor_prompt: str | None = None
    donor_layer: int | None = None
    lookup_key: str | None = None
    table_name: str | None = None
    recipient_baseline: list[TokenPrediction]
    surgical_output: list[TokenPrediction]
    generated_text: str
    num_generated_tokens: int
    verification: SurgeryVerification
    energy_analysis: SubspaceEnergyAnalysis
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Sync implementation
# ---------------------------------------------------------------------------


def _subspace_surgery_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    recipient_prompt: str,
    layer: int,
    subspace_name: str,
    mode: str,
    donor_prompt: str | None,
    donor_layer: int | None,
    coordinates: list[float] | None,
    lookup_key: str | None,
    table_name: str | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict:
    import mlx.core as mx

    from chuk_lazarus.introspection.hooks import ModelHooks

    from ..._residual_helpers import (
        _extract_position,
        _get_lm_projection,
        _norm_project,
        _run_decomposition_forward,
    )

    # -- Phase 1: Setup --
    recip_ids = mx.array(tokenizer.encode(recipient_prompt, add_special_tokens=True))
    last_layer = meta.num_layers - 1

    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)

    # Fetch subspace basis [rank, hidden_dim]
    sub_entry = SubspaceRegistry.get().fetch(subspace_name)
    if sub_entry is None:
        return make_error(
            ToolError.VECTOR_NOT_FOUND,
            f"Subspace '{subspace_name}' not found.",
            "subspace_surgery",
        )
    basis, _sub_meta = sub_entry
    rank = int(basis.shape[0])

    effective_donor_layer = donor_layer if donor_layer is not None else layer

    # -- Phase 2: Capture recipient --
    recip_capture_layers = sorted(set([layer, last_layer]))
    recip_decomp = _run_decomposition_forward(model, config, recip_ids, recip_capture_layers)
    mx.eval(*recip_decomp["hidden_states"].values())

    # Full hidden tensor at injection layer: [1, seq, hidden_dim]
    recip_hidden_at_layer = recip_decomp["hidden_states"][layer]
    R = np.array(recip_hidden_at_layer[0].tolist(), dtype=np.float32)  # [seq, hidden]
    recip_seq = R.shape[0]

    # Baseline logits from last layer
    recip_vec_final = _extract_position(recip_decomp["hidden_states"][last_layer], -1)
    baseline_logits = _norm_project(final_norm, lm_head, recip_vec_final)
    mx.eval(baseline_logits)
    baseline_logits_np = np.array(baseline_logits.tolist(), dtype=np.float32)
    baseline_top_k, _, _, _ = _top_k_from_logits(baseline_logits_np, tokenizer, top_k)

    # -- Phase 3: Compute new subspace content (mode-dependent) --
    if mode == "donor":
        donor_ids = mx.array(tokenizer.encode(donor_prompt, add_special_tokens=True))
        donor_decomp = _run_decomposition_forward(model, config, donor_ids, [effective_donor_layer])
        mx.eval(*donor_decomp["hidden_states"].values())

        D = np.array(
            donor_decomp["hidden_states"][effective_donor_layer][0].tolist(),
            dtype=np.float32,
        )  # [donor_seq, hidden_dim]
        donor_seq = D.shape[0]

        # Project donor onto subspace at all positions
        donor_coords = D @ basis.T  # [donor_seq, rank]
        donor_content = donor_coords @ basis  # [donor_seq, hidden_dim]

        # Right-align if sequence lengths differ
        if donor_seq == recip_seq:
            aligned_content = donor_content
        elif donor_seq < recip_seq:
            aligned_content = np.zeros_like(R)
            aligned_content[recip_seq - donor_seq :] = donor_content
        else:
            aligned_content = donor_content[donor_seq - recip_seq :]

    elif mode == "coordinates":
        coords_np = np.array(coordinates, dtype=np.float32)  # [rank]
        content_vec = coords_np @ basis  # [hidden_dim]
        aligned_content = np.tile(content_vec, (recip_seq, 1))  # [seq, hidden_dim]

    elif mode == "lookup":
        lookup_coords = DarkTableRegistry.get().lookup(table_name, lookup_key)  # type: ignore[arg-type]
        if lookup_coords is None:
            return make_error(
                ToolError.VECTOR_NOT_FOUND,
                f"Key '{lookup_key}' not found in table '{table_name}'.",
                "subspace_surgery",
            )
        content_vec = lookup_coords @ basis  # [hidden_dim]
        aligned_content = np.tile(content_vec, (recip_seq, 1))  # [seq, hidden_dim]

    else:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Unknown mode '{mode}'. Use 'donor', 'coordinates', or 'lookup'.",
            "subspace_surgery",
        )

    # -- Phase 4: Surgery (vectorised, all positions) --
    projections = R @ basis.T  # [seq, rank]
    subspace_components = projections @ basis  # [seq, hidden_dim]
    orthogonal = R - subspace_components  # [seq, hidden_dim] — PRESERVED

    surgical_R = orthogonal + aligned_content  # [seq, hidden_dim]

    # -- Phase 5: Verification --
    orth_before = orthogonal[-1]  # last position
    surgical_proj = surgical_R @ basis.T
    surgical_sub = surgical_proj @ basis
    orth_after = surgical_R[-1] - surgical_sub[-1]

    orth_before_norm = float(np.linalg.norm(orth_before))
    orth_after_norm = float(np.linalg.norm(orth_after))
    orth_cos = float(_cosine_sim(orth_before, orth_after))
    orth_norm_ratio = orth_after_norm / (orth_before_norm + 1e-12)

    verification = SurgeryVerification(
        orthogonal_cosine=round(orth_cos, 6),
        orthogonal_norm_ratio=round(orth_norm_ratio, 6),
        surgery_clean=orth_cos > 0.999,
    )

    # -- Phase 6: Inject and continue --
    surgical_mx = mx.array(surgical_R[np.newaxis].tolist())  # [1, seq, hidden_dim]
    injected_hidden = _run_forward_with_injection(
        model, config, recip_ids, layer, -1, surgical_mx, meta
    )
    mx.eval(injected_hidden)

    # -- Phase 7: Get output --
    surgical_vec = _extract_position(injected_hidden, -1)
    surgical_logits = _norm_project(final_norm, lm_head, surgical_vec)
    mx.eval(surgical_logits)
    surgical_logits_np = np.array(surgical_logits.tolist(), dtype=np.float32)
    surgical_top_k, _, _, _ = _top_k_from_logits(surgical_logits_np, tokenizer, top_k)

    # Generate text
    gen_text, num_gen = _generate_from_hidden(
        model,
        tokenizer,
        injected_hidden,
        final_norm,
        lm_head,
        recip_ids,
        -1,
        max_new_tokens,
        temperature,
    )

    # -- Phase 8: Energy analysis --
    recip_sub_norm = float(np.linalg.norm(subspace_components[-1]))
    recip_orth_norm = orth_before_norm
    recip_total_sq = recip_sub_norm**2 + recip_orth_norm**2 + 1e-12
    recip_sub_frac = recip_sub_norm**2 / recip_total_sq

    new_content_norm = float(np.linalg.norm(aligned_content[-1]))
    surgical_total_sq = recip_orth_norm**2 + new_content_norm**2 + 1e-12
    new_frac = new_content_norm**2 / surgical_total_sq

    energy_analysis = SubspaceEnergyAnalysis(
        recipient_subspace_energy_fraction=round(recip_sub_frac, 6),
        new_content_energy_fraction=round(new_frac, 6),
        recipient_subspace_norm=round(recip_sub_norm, 6),
        recipient_orthogonal_norm=round(recip_orth_norm, 6),
        new_content_norm=round(new_content_norm, 6),
    )

    # -- Summary --
    baseline_token = baseline_top_k[0].token if baseline_top_k else "?"
    surgical_token = surgical_top_k[0].token if surgical_top_k else "?"

    summary: dict[str, Any] = {
        "recipient_baseline_token": baseline_token,
        "surgical_token": surgical_token,
        "prediction_changed": baseline_token != surgical_token,
        "orthogonal_cosine": round(orth_cos, 6),
        "surgery_clean": orth_cos > 0.999,
        "subspace_rank": rank,
        "mode": mode,
    }

    return SubspaceSurgeryResult(
        recipient_prompt=recipient_prompt,
        mode=mode,
        layer=layer,
        subspace_name=subspace_name,
        subspace_rank=rank,
        donor_prompt=donor_prompt,
        donor_layer=donor_layer,
        lookup_key=lookup_key,
        table_name=table_name,
        recipient_baseline=baseline_top_k,
        surgical_output=surgical_top_k,
        generated_text=gen_text,
        num_generated_tokens=num_gen,
        verification=verification,
        energy_analysis=energy_analysis,
        summary=summary,
    ).model_dump()


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------

VALID_MODES = {"donor", "coordinates", "lookup"}


@mcp.tool()
async def subspace_surgery(
    recipient_prompt: str,
    layer: int,
    subspace_name: str,
    mode: str,
    donor_prompt: str | None = None,
    donor_layer: int | None = None,
    coordinates: list[float] | None = None,
    lookup_key: str | None = None,
    table_name: str | None = None,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 10,
) -> dict:
    """All-position subspace replacement in the residual stream.

    Replaces ONLY the subspace component at ALL positions while preserving
    the orthogonal complement.  Three modes:

    * **donor** — transplant another prompt's subspace projection.
      Requires `donor_prompt`.  Optional `donor_layer` (defaults to `layer`).
    * **coordinates** — inject explicit subspace coordinates.
      Requires `coordinates` (list of floats, length = subspace rank).
    * **lookup** — inject precomputed coordinates from a dark table.
      Requires `lookup_key` and `table_name`.

    Call compute_subspace first to create the PCA subspace.
    Call build_dark_table first to create lookup tables.

    Verification: orthogonal_cosine should be ~1.0 (surgery didn't touch
    anything outside the subspace).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "subspace_surgery",
        )

    num_layers = state.metadata.num_layers
    layer = int(layer)

    if layer < 0 or layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Layer {layer} out of range [0, {num_layers - 1}].",
            "subspace_surgery",
        )

    if not SubspaceRegistry.get().exists(subspace_name):
        return make_error(
            ToolError.VECTOR_NOT_FOUND,
            f"Subspace '{subspace_name}' not found. Call compute_subspace first.",
            "subspace_surgery",
        )

    if mode not in VALID_MODES:
        return make_error(
            ToolError.INVALID_INPUT,
            f"mode must be one of {sorted(VALID_MODES)}. Got '{mode}'.",
            "subspace_surgery",
        )

    # Mode-specific validation
    if mode == "donor":
        if not donor_prompt:
            return make_error(
                ToolError.INVALID_INPUT,
                "donor_prompt required for donor mode.",
                "subspace_surgery",
            )
        if donor_layer is not None:
            donor_layer = int(donor_layer)
            if donor_layer < 0 or donor_layer >= num_layers:
                return make_error(
                    ToolError.LAYER_OUT_OF_RANGE,
                    f"donor_layer {donor_layer} out of range [0, {num_layers - 1}].",
                    "subspace_surgery",
                )

    elif mode == "coordinates":
        if coordinates is None:
            return make_error(
                ToolError.INVALID_INPUT,
                "coordinates required for coordinates mode.",
                "subspace_surgery",
            )
        # MCP may send coordinates as a JSON string
        if isinstance(coordinates, str):
            try:
                coordinates = json.loads(coordinates)
            except (json.JSONDecodeError, TypeError):
                return make_error(
                    ToolError.INVALID_INPUT,
                    "coordinates must be a JSON array of floats.",
                    "subspace_surgery",
                )
        sub_entry = SubspaceRegistry.get().fetch(subspace_name)
        if sub_entry is not None:
            expected_rank = int(sub_entry[0].shape[0])
            if len(coordinates) != expected_rank:
                return make_error(
                    ToolError.INVALID_INPUT,
                    f"coordinates length {len(coordinates)} != subspace rank {expected_rank}.",
                    "subspace_surgery",
                )

    elif mode == "lookup":
        if not lookup_key:
            return make_error(
                ToolError.INVALID_INPUT,
                "lookup_key required for lookup mode.",
                "subspace_surgery",
            )
        if not table_name:
            return make_error(
                ToolError.INVALID_INPUT,
                "table_name required for lookup mode.",
                "subspace_surgery",
            )
        if not DarkTableRegistry.get().exists(table_name):
            return make_error(
                ToolError.VECTOR_NOT_FOUND,
                f"Dark table '{table_name}' not found. Call build_dark_table first.",
                "subspace_surgery",
            )
        if DarkTableRegistry.get().lookup(table_name, lookup_key) is None:
            return make_error(
                ToolError.VECTOR_NOT_FOUND,
                f"Key '{lookup_key}' not found in table '{table_name}'.",
                "subspace_surgery",
            )

    top_k = int(top_k)
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)

    if top_k < 1 or top_k > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k must be between 1 and 50.",
            "subspace_surgery",
        )

    if max_new_tokens < 1 or max_new_tokens > 500:
        return make_error(
            ToolError.INVALID_INPUT,
            "max_new_tokens must be between 1 and 500.",
            "subspace_surgery",
        )

    try:
        return await asyncio.to_thread(
            _subspace_surgery_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            recipient_prompt,
            layer,
            subspace_name,
            mode,
            donor_prompt,
            donor_layer,
            coordinates,
            lookup_key,
            table_name,
            max_new_tokens,
            temperature,
            top_k,
        )
    except Exception as e:
        return make_error(
            ToolError.GEOMETRY_FAILED,
            f"subspace_surgery failed: {e}",
            "subspace_surgery",
        )
