"""
decode_residual — decode the residual stream into vocabulary space.

Shows both raw (dot product) and normalised (after layer norm) decoding,
revealing the common-mode rejection performed by layer norm.  Words that
lose rank after normalisation are riding the shared mean; words that gain
rank are specific signals that survive centering.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp
from ._helpers import _angle_between, coerce_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class RawTokenEntry(BaseModel):
    """A token decoded from the raw (unnormalised) residual."""

    token: str
    token_id: int
    dot_product: float = Field(..., description="Raw dot(residual, unembed).")
    raw_rank: int


class NormTokenEntry(BaseModel):
    """A token decoded from the normalised residual."""

    token: str
    token_id: int
    logit: float = Field(..., description="dot(layernorm(residual), unembed).")
    probability: float
    norm_rank: int


class GapEntry(BaseModel):
    """A token that changed rank significantly after normalisation."""

    token: str
    token_id: int
    raw_rank: int
    norm_rank: int
    rank_change: int = Field(
        ...,
        description="raw_rank - norm_rank.  Positive = gained rank from norm.",
    )
    angle_to_mean: float = Field(
        ...,
        description="Angle between token unembed and residual mean direction.",
    )


class MeanDecodeEntry(BaseModel):
    """A token decoded from the mean direction."""

    token: str
    token_id: int
    dot_product: float


class MeanDecode(BaseModel):
    """Decoded mean direction of the residual stream."""

    mean_norm: float
    mean_fraction: float = Field(
        ..., description="mean^2 / residual^2 — energy in the mean direction."
    )
    norm_type: str = Field(..., description="Type of normalisation (RMSNorm, LayerNorm, etc.).")
    mean_top_k: list[MeanDecodeEntry]
    interpretation: str


class LayerDecodeResult(BaseModel):
    """Decoded residual stream at one layer."""

    layer: int
    raw_top_k: list[RawTokenEntry]
    norm_top_k: list[NormTokenEntry]
    biggest_gainers: list[GapEntry]
    biggest_losers: list[GapEntry]
    mean_decode: MeanDecode | None = None
    summary: dict[str, Any]


class DecodeResidualResult(BaseModel):
    """Result from decode_residual."""

    prompt: str
    token_position: int
    token_text: str
    hidden_dim: int
    vocab_size: int
    layers: list[LayerDecodeResult]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def decode_residual(
    prompt: str,
    layers: list[int],
    top_k: int = 20,
    token_position: int = -1,
    include_mean_decode: bool = True,
) -> dict:
    """Decode the residual stream into vocabulary space at specified
    layers, showing both raw and normalised nearest neighbours.

    For each layer, returns three views of the same residual stream:

    1. Raw — dot product with every vocabulary word's unembedding
       vector. What the stream is "pointing at" before normalisation.

    2. Normalised — after applying the model's actual layer norm,
       then dot product with unembedding vectors. What the model
       would actually predict if computation stopped here.

    3. Gap — the difference between raw and normalised rankings.
       Words that drop in rank after normalisation are correlated
       with the residual mean. Words that rise are orthogonal to
       the mean (specific signals that survive centering).

    Optionally decodes the mean direction itself — what the
    subtracted common-mode signal "points at" in vocabulary space.

    Args:
        prompt:              Input text.
        layers:              Layer indices to decode at.
        top_k:               Number of top words per view (1-100).
        token_position:      Token position (-1 = last).
        include_mean_decode: Decode the mean direction (default: True).
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            "decode_residual",
        )
    meta = state.metadata
    layers = coerce_layers(layers) or []
    if not layers:
        return make_error(
            ToolError.INVALID_INPUT,
            "At least 1 layer required.",
            "decode_residual",
        )
    for lyr in layers:
        if lyr < 0 or lyr >= meta.num_layers:
            return make_error(
                ToolError.LAYER_OUT_OF_RANGE,
                f"Layer {lyr} out of [0, {meta.num_layers - 1}].",
                "decode_residual",
            )
    if top_k < 1 or top_k > 100:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k must be in [1, 100].",
            "decode_residual",
        )
    try:
        return await asyncio.to_thread(
            _decode_residual_impl,
            state.model,
            state.config,
            state.tokenizer,
            meta,
            prompt,
            layers,
            top_k,
            token_position,
            include_mean_decode,
        )
    except Exception as exc:
        logger.exception("decode_residual failed")
        return make_error(ToolError.GEOMETRY_FAILED, str(exc), "decode_residual")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _decode_residual_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    meta: Any,
    prompt: str,
    layers: list[int],
    top_k: int,
    token_position: int,
    include_mean_decode: bool,
) -> dict:
    """Sync implementation of decode_residual."""
    import mlx.core as mx

    from ..._residual_helpers import (
        _extract_position,
        _get_lm_projection,
        _get_unembed_vector,
        _norm_project,
        _project_to_logits,
        _run_decomposition_forward,
    )

    # -- Setup --
    tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array(tok_ids)
    num_tokens = len(tok_ids)
    pos = token_position if token_position >= 0 else num_tokens + token_position
    pos = max(0, min(pos, num_tokens - 1))
    tok_text = tokenizer.decode([tok_ids[pos]])

    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    if lm_head is None:
        return make_error(
            ToolError.EXTRACTION_FAILED,
            "Could not access lm_head.",
            "decode_residual",
        )

    # Detect norm type
    norm_type_str = type(final_norm).__name__ if final_norm else "Unknown"

    # -- Forward pass --
    layers_sorted = sorted(layers)
    decomp = _run_decomposition_forward(model, config, input_ids, layers_sorted)
    mx.eval(*decomp["hidden_states"].values())

    # -- Per-layer decode --
    vocab_size: int | None = None
    layer_results: list[LayerDecodeResult] = []

    for lyr in layers_sorted:
        hidden = decomp["hidden_states"].get(lyr)
        if hidden is None:
            continue

        h_vec = _extract_position(hidden, pos)

        # Raw logits (no norm) and normalised logits
        raw_logits = _project_to_logits(lm_head, h_vec)
        norm_logits = _norm_project(final_norm, lm_head, h_vec)
        mx.eval(raw_logits, norm_logits)

        raw_np = np.array(raw_logits.tolist(), dtype=np.float32)
        norm_np = np.array(norm_logits.tolist(), dtype=np.float32)

        if vocab_size is None:
            vocab_size = len(raw_np)

        vs = len(raw_np)

        # Rankings (descending order)
        raw_order = np.argsort(-raw_np)
        norm_order = np.argsort(-norm_np)

        # rank[token_id] = position (0-indexed)
        raw_ranks = np.empty(vs, dtype=np.int32)
        norm_ranks = np.empty(vs, dtype=np.int32)
        raw_ranks[raw_order] = np.arange(vs)
        norm_ranks[norm_order] = np.arange(vs)

        # -- Raw top_k --
        raw_top_entries: list[RawTokenEntry] = []
        for i in range(min(top_k, vs)):
            tid = int(raw_order[i])
            raw_top_entries.append(
                RawTokenEntry(
                    token=tokenizer.decode([tid]),
                    token_id=tid,
                    dot_product=round(float(raw_np[tid]), 4),
                    raw_rank=i + 1,
                )
            )

        # -- Norm top_k with softmax probabilities --
        norm_max = float(np.max(norm_np))
        exp_vals = np.exp(norm_np - norm_max)
        sum_exp = float(np.sum(exp_vals))

        norm_top_entries: list[NormTokenEntry] = []
        for i in range(min(top_k, vs)):
            tid = int(norm_order[i])
            prob = float(exp_vals[tid]) / sum_exp if sum_exp > 0 else 0.0
            norm_top_entries.append(
                NormTokenEntry(
                    token=tokenizer.decode([tid]),
                    token_id=tid,
                    logit=round(float(norm_np[tid]), 4),
                    probability=round(prob, 6),
                    norm_rank=i + 1,
                )
            )

        # -- Gap computation --
        # Mean direction for angle computation
        h_np = np.array(h_vec.tolist(), dtype=np.float32)
        mean_val = float(np.mean(h_np))
        mean_dir = np.ones(len(h_np), dtype=np.float32) * mean_val
        mean_norm_val = float(np.linalg.norm(mean_dir))
        res_norm = float(np.linalg.norm(h_np))
        mean_fraction_val = (mean_norm_val**2) / (res_norm**2 + 1e-12)

        # Union of raw top-100 and norm top-100
        gap_k = min(100, vs)
        raw_top100 = set(raw_order[:gap_k].tolist())
        norm_top100 = set(norm_order[:gap_k].tolist())
        union_ids = list(raw_top100 | norm_top100)

        # Rank changes for union
        gap_data: list[tuple[int, int, int, int]] = []
        for tid in union_ids:
            r_rank = int(raw_ranks[tid]) + 1  # 1-indexed
            n_rank = int(norm_ranks[tid]) + 1
            rank_change = r_rank - n_rank
            gap_data.append((tid, r_rank, n_rank, rank_change))

        # Gainers and losers
        gainers = sorted(gap_data, key=lambda x: x[3], reverse=True)[:top_k]
        losers = sorted(gap_data, key=lambda x: x[3])[:top_k]

        # Compute angle_to_mean for gainers + losers
        need_angle_ids = set(x[0] for x in gainers) | set(x[0] for x in losers)
        angle_cache: dict[int, float] = {}
        if mean_norm_val > 1e-8:
            for tid in need_angle_ids:
                uv = _get_unembed_vector(model, tid)
                if uv is not None:
                    mx.eval(uv)
                    uv_np = np.array(uv.tolist(), dtype=np.float32)
                    angle_cache[tid] = _angle_between(uv_np, mean_dir)
                else:
                    angle_cache[tid] = 0.0

        gainer_entries = [
            GapEntry(
                token=tokenizer.decode([int(tid)]),
                token_id=int(tid),
                raw_rank=r_rank,
                norm_rank=n_rank,
                rank_change=rc,
                angle_to_mean=round(angle_cache.get(tid, 0.0), 4),
            )
            for tid, r_rank, n_rank, rc in gainers
        ]

        loser_entries = [
            GapEntry(
                token=tokenizer.decode([int(tid)]),
                token_id=int(tid),
                raw_rank=r_rank,
                norm_rank=n_rank,
                rank_change=rc,
                angle_to_mean=round(angle_cache.get(tid, 0.0), 4),
            )
            for tid, r_rank, n_rank, rc in losers
        ]

        # -- Mean decode (optional) --
        mean_decode_result: MeanDecode | None = None
        if include_mean_decode and mean_norm_val > 1e-8:
            mean_dir_mx = mx.array(mean_dir.tolist())
            mean_logits_mx = _project_to_logits(lm_head, mean_dir_mx)
            mx.eval(mean_logits_mx)
            mean_logits_np = np.array(mean_logits_mx.tolist(), dtype=np.float32)
            mean_order = np.argsort(-mean_logits_np)

            mean_top_entries: list[MeanDecodeEntry] = []
            for i in range(min(top_k, vs)):
                tid = int(mean_order[i])
                mean_top_entries.append(
                    MeanDecodeEntry(
                        token=tokenizer.decode([tid]),
                        token_id=tid,
                        dot_product=round(float(mean_logits_np[tid]), 4),
                    )
                )

            top_words = [e.token.strip() for e in mean_top_entries[:5] if e.token.strip()]
            interp = f"Mean direction dominated by: {', '.join(top_words)}"

            mean_decode_result = MeanDecode(
                mean_norm=round(mean_norm_val, 4),
                mean_fraction=round(mean_fraction_val, 6),
                norm_type=norm_type_str,
                mean_top_k=mean_top_entries,
                interpretation=interp,
            )

        # -- Spearman rank correlation (raw top-100) --
        raw_top100_ids = raw_order[:gap_k]
        raw_top100_norm_ranks = np.array([int(norm_ranks[tid]) + 1 for tid in raw_top100_ids])
        raw_top100_raw_ranks = np.arange(1, gap_k + 1)
        if gap_k > 1:
            spearman = float(np.corrcoef(raw_top100_raw_ranks, raw_top100_norm_ranks)[0, 1])
        else:
            spearman = 1.0

        # -- Summary --
        raw_top1 = raw_top_entries[0].token if raw_top_entries else ""
        norm_top1 = norm_top_entries[0].token if norm_top_entries else ""

        summary: dict[str, Any] = {
            "raw_top1": raw_top1,
            "norm_top1": norm_top1,
            "top1_changed": raw_top1 != norm_top1,
            "raw_norm_rank_correlation": round(spearman, 6),
            "mean_energy_fraction": round(mean_fraction_val, 6),
        }

        layer_results.append(
            LayerDecodeResult(
                layer=lyr,
                raw_top_k=raw_top_entries,
                norm_top_k=norm_top_entries,
                biggest_gainers=gainer_entries,
                biggest_losers=loser_entries,
                mean_decode=mean_decode_result,
                summary=summary,
            )
        )

    return DecodeResidualResult(
        prompt=prompt,
        token_position=token_position,
        token_text=tok_text,
        hidden_dim=meta.hidden_dim,
        vocab_size=vocab_size or 0,
        layers=layer_results,
    ).model_dump()
