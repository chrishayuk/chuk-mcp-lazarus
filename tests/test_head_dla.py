"""Tests for head_dla tools.

Covers: compute_dla, batch_dla_scan, extract_attention_output,
get_token_embedding, extract_k_vector, extract_q_vector,
plus all private helpers and result models.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MLX stub helpers
# ---------------------------------------------------------------------------

_MockMxArray = type(mx.array([1.0]))


def _ensure_mlx_stubs() -> None:
    if not hasattr(nn, "MultiHeadAttention"):

        class _MockMHA:
            @staticmethod
            def create_additive_causal_mask(seq_len: int) -> Any:
                mask = np.zeros((seq_len, seq_len), dtype=np.float32)
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        mask[i, j] = -1e9
                return mx.array(mask)

        nn.MultiHeadAttention = _MockMHA  # type: ignore[attr-defined]

    if not hasattr(mx, "stack"):

        def _stack(arrays: list, axis: int = 0) -> Any:
            arrs = [
                a._data if hasattr(a, "_data") else np.array(a, dtype=np.float32) for a in arrays
            ]
            return mx.array(np.stack(arrs, axis=axis).astype(np.float32))

        mx.stack = _stack  # type: ignore[attr-defined]

    # Patch SDPA to accept mask kwarg and return shape-aware result
    import mlx.core.fast as mx_fast

    def _sdpa_with_mask(q: Any, k: Any, v: Any, scale: float = 1.0, mask: Any = None) -> Any:
        if hasattr(q, "shape") and len(q.shape) == 4:
            b, h, s, d = q.shape
            return mx.array(np.zeros((b, h, s, d), dtype=np.float32))
        return mx.array(np.zeros((1, 1, 1, 64), dtype=np.float32))

    mx_fast.scaled_dot_product_attention = _sdpa_with_mask


_ensure_mlx_stubs()


# ---------------------------------------------------------------------------
# Constants (match conftest mock_metadata)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
VOCAB_SIZE = 100
NUM_HEADS = 4
HEAD_DIM = 16
NUM_LAYERS = 4
SEQ_LEN = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_decomp(layers: list[int] | None = None) -> dict:
    if layers is None:
        layers = list(range(NUM_LAYERS))
    prev_hidden = {}
    hidden_states = {}
    for lay in layers:
        prev_hidden[lay] = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
        hidden_states[lay] = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
    return {
        "prev_hidden": prev_hidden,
        "hidden_states": hidden_states,
        "attn_outputs": {},
        "ffn_outputs": {},
    }


def _make_mock_layer() -> MagicMock:
    layer = MagicMock()
    # input_layernorm returns proper shaped array
    layer.input_layernorm.return_value = mx.array(
        np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32)
    )
    # Proj functions return proper shapes
    layer.self_attn.q_proj.return_value = mx.array(
        np.random.randn(1, SEQ_LEN, NUM_HEADS * HEAD_DIM).astype(np.float32)
    )
    layer.self_attn.k_proj.return_value = mx.array(
        np.random.randn(1, SEQ_LEN, NUM_HEADS * HEAD_DIM).astype(np.float32)
    )
    layer.self_attn.v_proj.return_value = mx.array(
        np.random.randn(1, SEQ_LEN, NUM_HEADS * HEAD_DIM).astype(np.float32)
    )
    layer.self_attn.o_proj.weight = mx.array(
        np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32)
    )
    layer.self_attn.rope = None
    layer.self_attn.q_norm = None
    layer.self_attn.k_norm = None
    layer.self_attn.scale = HEAD_DIM**-0.5
    return layer


def _make_per_head_dla(num_heads: int = NUM_HEADS) -> list[tuple[float, str]]:
    return [(float(i) * 0.1 + 0.05, f"tok{i}") for i in range(num_heads)]


# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


class TestHasSublayers:
    def test_true_when_both_present(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _has_sublayers

        layer = MagicMock()
        # MagicMock auto-creates all attrs — both will exist
        assert _has_sublayers(layer) is True

    def test_false_when_self_attn_missing(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _has_sublayers

        layer = MagicMock(spec=["input_layernorm", "mlp"])
        assert _has_sublayers(layer) is False

    def test_false_when_layernorm_missing(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _has_sublayers

        layer = MagicMock(spec=["self_attn", "mlp"])
        assert _has_sublayers(layer) is False

    def test_false_for_plain_object(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _has_sublayers

        assert _has_sublayers(object()) is False


class TestResolvePosition:
    def test_last_token(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(5, -1) == 4

    def test_second_to_last(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(5, -2) == 3

    def test_zero(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(5, 0) == 0

    def test_positive(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(5, 2) == 2

    def test_clamp_high(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(5, 100) == 4

    def test_clamp_negative_overflow(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(5, -100) == 0

    def test_single_token(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _resolve_position

        assert _resolve_position(1, -1) == 0


# ---------------------------------------------------------------------------
# Direct tests for _compute_per_head_attention and _per_head_dla
# (These require special mock setup to cover the actual implementation.)
# ---------------------------------------------------------------------------


class TestComputePerHeadAttention:
    def test_returns_context_and_weight(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _compute_per_head_attention

        layer = _make_mock_layer()
        prev_h = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
        mask = mx.array(np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32))

        context, o_weight = _compute_per_head_attention(
            layer, prev_h, mask, 0, NUM_HEADS, NUM_HEADS, HEAD_DIM
        )
        # context shape: (1, num_heads, seq_len, head_dim)
        assert context.shape == (1, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        # o_weight is the layer's o_proj.weight
        assert o_weight is layer.self_attn.o_proj.weight

    def test_gqa_expands_kv(self) -> None:
        """num_kv_heads < num_heads triggers GQA repeat."""
        from chuk_mcp_lazarus.tools.geometry.head_dla import _compute_per_head_attention

        # Use 2 kv_heads, 4 query heads
        layer = _make_mock_layer()
        layer.self_attn.k_proj.return_value = mx.array(
            np.random.randn(1, SEQ_LEN, 2 * HEAD_DIM).astype(np.float32)
        )
        layer.self_attn.v_proj.return_value = mx.array(
            np.random.randn(1, SEQ_LEN, 2 * HEAD_DIM).astype(np.float32)
        )
        prev_h = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
        mask = mx.array(np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32))

        # Should not raise
        context, _ = _compute_per_head_attention(layer, prev_h, mask, 0, NUM_HEADS, 2, HEAD_DIM)
        assert context is not None

    def test_with_q_norm_and_k_norm(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _compute_per_head_attention

        layer = _make_mock_layer()
        # Add q_norm / k_norm that return their input unchanged
        layer.self_attn.q_norm = lambda x: x
        layer.self_attn.k_norm = lambda x: x
        prev_h = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
        mask = mx.array(np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32))

        context, _ = _compute_per_head_attention(
            layer, prev_h, mask, 0, NUM_HEADS, NUM_HEADS, HEAD_DIM
        )
        assert context is not None

    def test_with_rope(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _compute_per_head_attention

        layer = _make_mock_layer()
        # rope returns the same arrays unchanged
        layer.self_attn.rope = lambda q, k: (q, k)
        prev_h = mx.array(np.random.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32))
        mask = mx.array(np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32))

        context, _ = _compute_per_head_attention(
            layer, prev_h, mask, 0, NUM_HEADS, NUM_HEADS, HEAD_DIM
        )
        assert context is not None


class TestPerHeadDla:
    def test_returns_dla_per_head(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _per_head_dla

        num_heads = 1
        head_dim = HIDDEN_DIM  # 64 so o_weight slicing works cleanly

        context = mx.array(np.zeros((1, num_heads, SEQ_LEN, head_dim), dtype=np.float32))
        o_weight = mx.array(np.eye(HIDDEN_DIM, num_heads * head_dim, dtype=np.float32))
        unembed_u = mx.array(np.ones(HIDDEN_DIM, dtype=np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        lm_head = MagicMock()
        lm_head.return_value = mx.array(
            np.random.randn(1, num_heads, VOCAB_SIZE).astype(np.float32)
        )

        result = _per_head_dla(
            context, o_weight, unembed_u, 0, num_heads, head_dim, tokenizer, lm_head
        )
        assert len(result) == num_heads
        dla_val, top_tok = result[0]
        assert isinstance(dla_val, float)
        assert isinstance(top_tok, str)

    def test_logits_as_tuple(self) -> None:
        """lm_head returning a tuple is handled correctly."""
        from chuk_mcp_lazarus.tools.geometry.head_dla import _per_head_dla

        head_dim = HIDDEN_DIM

        context = mx.array(np.zeros((1, 1, SEQ_LEN, head_dim), dtype=np.float32))
        o_weight = mx.array(np.eye(HIDDEN_DIM, dtype=np.float32))
        unembed_u = mx.array(np.ones(HIDDEN_DIM, dtype=np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        logits_inner = mx.array(np.random.randn(1, 1, VOCAB_SIZE).astype(np.float32))
        lm_head = MagicMock(return_value=(logits_inner,))

        result = _per_head_dla(context, o_weight, unembed_u, 0, 1, head_dim, tokenizer, lm_head)
        assert len(result) == 1

    def test_logits_with_logits_attr(self) -> None:
        """lm_head returning object with .logits attribute."""
        from chuk_mcp_lazarus.tools.geometry.head_dla import _per_head_dla

        head_dim = HIDDEN_DIM

        context = mx.array(np.zeros((1, 1, SEQ_LEN, head_dim), dtype=np.float32))
        o_weight = mx.array(np.eye(HIDDEN_DIM, dtype=np.float32))
        unembed_u = mx.array(np.ones(HIDDEN_DIM, dtype=np.float32))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        logits_tensor = mx.array(np.random.randn(1, 1, VOCAB_SIZE).astype(np.float32))

        class _Out:
            logits = logits_tensor

        lm_head = MagicMock(return_value=_Out())

        result = _per_head_dla(context, o_weight, unembed_u, 0, 1, head_dim, tokenizer, lm_head)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# compute_dla — async validation
# ---------------------------------------------------------------------------


class TestComputeDla:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        result = await compute_dla(prompt="hello", layer=0, head=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range_high(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        result = await compute_dla(prompt="hello", layer=99, head=0)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_layer_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        result = await compute_dla(prompt="hello", layer=-1, head=0)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_head_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        result = await compute_dla(prompt="hello", layer=0, head=99)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_head_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        result = await compute_dla(prompt="hello", layer=0, head=-1)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        fake = {
            "prompt": "hello",
            "layer": 0,
            "head": 0,
            "target_token": "tok1",
            "target_token_id": 1,
            "position": -1,
            "dla": 0.5,
            "fraction_of_layer": 0.25,
            "top_token": "tok1",
            "head_output_norm": 1.0,
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._compute_dla_impl",
            return_value=fake,
        ):
            result = await compute_dla(prompt="hello", layer=0, head=0)
        assert "error" not in result
        assert result["dla"] == 0.5

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import compute_dla

        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._compute_dla_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await compute_dla(prompt="hello", layer=0, head=0)
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# batch_dla_scan — async validation
# ---------------------------------------------------------------------------


class TestBatchDlaScan:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        result = await batch_dla_scan(prompt="hello")
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        result = await batch_dla_scan(prompt="hello", layers=[0, 99])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        result = await batch_dla_scan(prompt="hello", layers=[])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_negative_layer_in_list(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        result = await batch_dla_scan(prompt="hello", layers=[-1])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        fake: dict[str, Any] = {
            "prompt": "hello",
            "target_token": "tok1",
            "target_token_id": 1,
            "position": -1,
            "num_layers_scanned": 2,
            "num_heads": 4,
            "layers": [],
            "hot_cells": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._batch_dla_scan_impl",
            return_value=fake,
        ):
            result = await batch_dla_scan(prompt="hello")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._batch_dla_scan_impl",
            side_effect=ValueError("bad"),
        ):
            result = await batch_dla_scan(prompt="hello", layers=[0])
        assert result["error_type"] == "GeometryFailed"

    @pytest.mark.asyncio
    async def test_top_k_clamped_to_max(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        fake: dict[str, Any] = {
            "prompt": "h",
            "target_token": "t",
            "target_token_id": 1,
            "position": -1,
            "num_layers_scanned": 1,
            "num_heads": 4,
            "layers": [],
            "hot_cells": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._batch_dla_scan_impl",
            return_value=fake,
        ) as mock_impl:
            await batch_dla_scan(prompt="h", layers=[0], top_k_cells=100)
            top_k_passed = mock_impl.call_args[0][-1]
            assert top_k_passed == 20

    @pytest.mark.asyncio
    async def test_top_k_clamped_to_min(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan

        fake: dict[str, Any] = {
            "prompt": "h",
            "target_token": "t",
            "target_token_id": 1,
            "position": -1,
            "num_layers_scanned": 1,
            "num_heads": 4,
            "layers": [],
            "hot_cells": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._batch_dla_scan_impl",
            return_value=fake,
        ) as mock_impl:
            await batch_dla_scan(prompt="h", layers=[0], top_k_cells=0)
            top_k_passed = mock_impl.call_args[0][-1]
            assert top_k_passed == 1


# ---------------------------------------------------------------------------
# extract_attention_output — async validation
# ---------------------------------------------------------------------------


class TestExtractAttentionOutput:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_attention_output

        result = await extract_attention_output(prompt="hello", layer=0, head=0)
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_attention_output

        result = await extract_attention_output(prompt="hello", layer=10, head=0)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_head_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_attention_output

        result = await extract_attention_output(prompt="hello", layer=0, head=10)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_attention_output

        fake: dict[str, Any] = {
            "prompt": "hello",
            "layer": 0,
            "head": 0,
            "position": -1,
            "vector": [0.0] * HIDDEN_DIM,
            "vector_norm": 1.0,
            "top_projections": [],
            "dimensionality": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._extract_attention_output_impl",
            return_value=fake,
        ):
            result = await extract_attention_output(prompt="hello", layer=0, head=0)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_attention_output

        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._extract_attention_output_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await extract_attention_output(prompt="hello", layer=0, head=0)
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# get_token_embedding — async validation
# ---------------------------------------------------------------------------


class TestGetTokenEmbedding:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import get_token_embedding

        result = await get_token_embedding(token="Paris")
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import get_token_embedding

        fake: dict[str, Any] = {
            "token": "Paris",
            "token_id": 42,
            "unembedding": [0.1] * HIDDEN_DIM,
            "input_embedding": [0.2] * HIDDEN_DIM,
            "embeddings_tied": False,
            "unembedding_norm": 1.0,
            "input_embedding_norm": 1.0,
            "cosine_similarity": 0.9,
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._get_token_embedding_impl",
            return_value=fake,
        ):
            result = await get_token_embedding(token="Paris")
        assert "error" not in result
        assert result["token_id"] == 42

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import get_token_embedding

        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._get_token_embedding_impl",
            side_effect=ValueError("no token"),
        ):
            result = await get_token_embedding(token="!!!!")
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# extract_k_vector — async validation
# ---------------------------------------------------------------------------


class TestExtractKVector:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_k_vector

        result = await extract_k_vector(prompt="hello", layer=0, kv_head=0)
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_k_vector

        result = await extract_k_vector(prompt="hello", layer=99, kv_head=0)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_kv_head_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_k_vector

        result = await extract_k_vector(prompt="hello", layer=0, kv_head=99)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_kv_head_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_k_vector

        result = await extract_k_vector(prompt="hello", layer=0, kv_head=-1)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_k_vector

        fake: dict[str, Any] = {
            "prompt": "hello",
            "layer": 0,
            "kv_head": 0,
            "position": -1,
            "k_vector": [0.1] * HEAD_DIM,
            "k_norm": 1.0,
            "num_kv_heads": 4,
            "head_dim": HEAD_DIM,
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._extract_k_vector_impl",
            return_value=fake,
        ):
            result = await extract_k_vector(prompt="hello", layer=0, kv_head=0)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_k_vector

        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._extract_k_vector_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await extract_k_vector(prompt="hello", layer=0, kv_head=0)
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# extract_q_vector — async validation
# ---------------------------------------------------------------------------


class TestExtractQVector:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_q_vector

        result = await extract_q_vector(prompt="hello", layer=0, head=0)
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_q_vector

        result = await extract_q_vector(prompt="hello", layer=99, head=0)
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_head_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_q_vector

        result = await extract_q_vector(prompt="hello", layer=0, head=99)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_q_vector

        fake: dict[str, Any] = {
            "prompt": "hello",
            "layer": 0,
            "head": 0,
            "position": -1,
            "q_vector": [0.1] * HEAD_DIM,
            "q_norm": 1.0,
            "num_heads": 4,
            "head_dim": HEAD_DIM,
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._extract_q_vector_impl",
            return_value=fake,
        ):
            result = await extract_q_vector(prompt="hello", layer=0, head=0)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import extract_q_vector

        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._extract_q_vector_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await extract_q_vector(prompt="hello", layer=0, head=0)
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# _compute_dla_impl — sync tests
# ---------------------------------------------------------------------------


class TestComputeDlaImpl:
    def _run(
        self,
        layer: int = 0,
        head: int = 0,
        target_token: str | None = None,
        position: int = -1,
        non_decomposable: bool = False,
        no_lm_head: bool = False,
        no_unembed: bool = False,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _compute_dla_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = NUM_LAYERS
        meta.num_attention_heads = NUM_HEADS
        meta.num_kv_heads = NUM_HEADS
        meta.hidden_dim = HIDDEN_DIM
        meta.head_dim = HEAD_DIM

        decomp = _make_decomp(layers=sorted({layer, NUM_LAYERS - 1}))

        mock_layers = []
        for i in range(NUM_LAYERS):
            if non_decomposable and i == layer:
                mock_layers.append(MagicMock(spec=["mlp"]))
            else:
                mock_layers.append(_make_mock_layer())

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers
        mock_helper._get_final_norm.return_value = lambda x: x

        u_vec = None if no_unembed else mx.array(np.ones(HIDDEN_DIM, dtype=np.float32))
        lm_head = None if no_lm_head else MagicMock()
        full_logits = mx.array(np.random.randn(VOCAB_SIZE).astype(np.float32))

        per_head = _make_per_head_dla(NUM_HEADS)
        context = mx.array(np.random.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32))
        o_weight = mx.array(np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32))

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._norm_project",
                return_value=full_logits,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                return_value=(1, "tok1"),
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._compute_per_head_attention",
                return_value=(context, o_weight),
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._per_head_dla",
                return_value=per_head,
            ),
        ):
            return _compute_dla_impl(
                model, config, tokenizer, meta, "hello", layer, head, target_token, position
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "layer",
            "head",
            "target_token",
            "target_token_id",
            "position",
            "dla",
            "fraction_of_layer",
            "top_token",
            "head_output_norm",
        ]:
            assert k in r

    def test_dla_is_float(self) -> None:
        r = self._run()
        assert isinstance(r["dla"], float)

    def test_fraction_in_range(self) -> None:
        r = self._run()
        assert 0.0 <= r["fraction_of_layer"] <= 1.0

    def test_head_output_norm_nonnegative(self) -> None:
        r = self._run()
        assert r["head_output_norm"] >= 0.0

    def test_target_token_from_resolve(self) -> None:
        r = self._run()
        assert r["target_token"] == "tok1"

    def test_position_stored(self) -> None:
        r = self._run(position=2)
        assert r["position"] == 2

    def test_no_lm_head_raises(self) -> None:
        with pytest.raises(ValueError, match="language model head"):
            self._run(no_lm_head=True)

    def test_no_unembed_raises(self) -> None:
        with pytest.raises(ValueError, match="unembedding"):
            self._run(no_unembed=True)

    def test_non_decomposable_raises(self) -> None:
        with pytest.raises(ValueError, match="decomposable"):
            self._run(non_decomposable=True)

    def test_specific_head_index(self) -> None:
        r = self._run(head=2)
        assert r["head"] == 2


# ---------------------------------------------------------------------------
# _batch_dla_scan_impl — sync tests
# ---------------------------------------------------------------------------


class TestBatchDlaScanImpl:
    def _run(
        self,
        layers: list[int] | None = None,
        non_decomposable_idx: int | None = None,
        make_layer_fail: bool = False,
        no_lm_head: bool = False,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _batch_dla_scan_impl

        if layers is None:
            layers = [0, 1]

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = NUM_LAYERS
        meta.num_attention_heads = NUM_HEADS
        meta.num_kv_heads = NUM_HEADS
        meta.hidden_dim = HIDDEN_DIM
        meta.head_dim = HEAD_DIM

        all_layers = sorted(set(layers) | {NUM_LAYERS - 1})
        decomp = _make_decomp(layers=all_layers)

        mock_layers_list = []
        for i in range(NUM_LAYERS):
            if non_decomposable_idx is not None and i == non_decomposable_idx:
                mock_layers_list.append(MagicMock(spec=["mlp"]))
            else:
                mock_layers_list.append(_make_mock_layer())

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers_list
        mock_helper._get_final_norm.return_value = lambda x: x

        lm_head = None if no_lm_head else MagicMock()
        u_vec = mx.array(np.ones(HIDDEN_DIM, dtype=np.float32))
        full_logits = mx.array(np.random.randn(VOCAB_SIZE).astype(np.float32))
        per_head = _make_per_head_dla(NUM_HEADS)
        context = mx.array(np.random.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32))
        o_weight = mx.array(np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32))

        cpha_effect = RuntimeError("attn fail") if make_layer_fail else None

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._norm_project",
                return_value=full_logits,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._resolve_target_token",
                return_value=(1, "tok1"),
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._compute_per_head_attention",
                return_value=(context, o_weight),
                side_effect=cpha_effect,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._per_head_dla",
                return_value=per_head,
            ),
        ):
            return _batch_dla_scan_impl(
                model, config, tokenizer, meta, "hello", None, layers, -1, 3
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "target_token",
            "target_token_id",
            "position",
            "num_layers_scanned",
            "num_heads",
            "layers",
            "hot_cells",
            "summary",
        ]:
            assert k in r

    def test_layers_count(self) -> None:
        r = self._run(layers=[0, 1])
        assert r["num_layers_scanned"] == 2

    def test_single_layer(self) -> None:
        r = self._run(layers=[2])
        assert r["num_layers_scanned"] == 1

    def test_hot_cells_capped_at_top_k(self) -> None:
        r = self._run(layers=[0, 1])
        assert len(r["hot_cells"]) <= 3

    def test_hot_cells_ranked(self) -> None:
        r = self._run(layers=[0, 1])
        for i in range(len(r["hot_cells"]) - 1):
            assert r["hot_cells"][i]["abs_rank"] < r["hot_cells"][i + 1]["abs_rank"]

    def test_summary_fields(self) -> None:
        r = self._run()
        assert "target_token" in r["summary"]
        assert "copy_circuit" in r["summary"]
        assert "top_cell" in r["summary"]

    def test_non_decomposable_layer_flagged(self) -> None:
        r = self._run(layers=[0, 1], non_decomposable_idx=0)
        # layer 0 should be flagged as non-decomposable
        layer0 = next(e for e in r["layers"] if e["layer"] == 0)
        assert layer0["is_decomposable"] is False

    def test_layer_exception_logged_gracefully(self) -> None:
        r = self._run(make_layer_fail=True)
        # All layers fail → all non-decomposable
        assert all(not e["is_decomposable"] for e in r["layers"])

    def test_no_lm_head_raises(self) -> None:
        with pytest.raises(ValueError):
            self._run(no_lm_head=True)

    def test_all_layers_default_none(self, loaded_model_state: Any) -> None:
        """layers=None scans all layers (covered in async validation via mock)."""
        r = self._run(layers=list(range(NUM_LAYERS)))
        assert r["num_layers_scanned"] == NUM_LAYERS


# ---------------------------------------------------------------------------
# _get_token_embedding_impl — sync tests
# ---------------------------------------------------------------------------


class TestGetTokenEmbeddingImpl:
    def _run(
        self,
        token_id: int | None = 5,
        has_embed_weight: bool = True,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _get_token_embedding_impl

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        u_vec = mx.array(np.random.randn(HIDDEN_DIM).astype(np.float32))
        embed_w = (
            mx.array(np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32))
            if has_embed_weight
            else None
        )

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._resolve_token_to_id",
                return_value=token_id,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=u_vec,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_embed_weight",
                return_value=embed_w,
            ),
            patch("mlx.core.eval"),
        ):
            return _get_token_embedding_impl(model, tokenizer, "Paris")

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "token",
            "token_id",
            "unembedding",
            "input_embedding",
            "embeddings_tied",
            "unembedding_norm",
        ]:
            assert k in r

    def test_unembedding_length(self) -> None:
        r = self._run()
        assert len(r["unembedding"]) == HIDDEN_DIM

    def test_cosine_similarity_in_range(self) -> None:
        r = self._run()
        if r["cosine_similarity"] is not None:
            assert -1.0 <= r["cosine_similarity"] <= 1.0

    def test_no_embed_weight_gives_none(self) -> None:
        r = self._run(has_embed_weight=False)
        assert r["input_embedding"] is None
        assert r["input_embedding_norm"] is None
        assert r["cosine_similarity"] is None

    def test_unresolvable_token_raises(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _get_token_embedding_impl

        model = MagicMock()
        tokenizer = MagicMock()
        with patch(
            "chuk_mcp_lazarus.tools.geometry.head_dla._resolve_token_to_id",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="single token ID"):
                _get_token_embedding_impl(model, tokenizer, "!!!!")

    def test_no_unembed_vector_raises(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _get_token_embedding_impl

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "tok5"
        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._resolve_token_to_id",
                return_value=5,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_unembed_vector",
                return_value=None,
            ),
            patch("mlx.core.eval"),
        ):
            with pytest.raises(ValueError, match="unembedding"):
                _get_token_embedding_impl(model, tokenizer, "Paris")


# ---------------------------------------------------------------------------
# _extract_k_vector_impl — sync tests
# ---------------------------------------------------------------------------


class TestExtractKVectorImpl:
    def _run(
        self,
        layer: int = 0,
        kv_head: int = 0,
        position: int = -1,
        non_decomposable: bool = False,
        with_rope: bool = False,
        with_k_norm: bool = False,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _extract_k_vector_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        meta = MagicMock()
        meta.num_attention_heads = NUM_HEADS
        meta.num_kv_heads = NUM_HEADS
        meta.hidden_dim = HIDDEN_DIM
        meta.head_dim = HEAD_DIM

        mock_layer = _make_mock_layer()
        if with_rope:
            mock_layer.self_attn.rope = lambda q, k: (q, k)
        if with_k_norm:
            mock_layer.self_attn.k_norm = lambda x: x

        mock_layers = [mock_layer] + [_make_mock_layer() for _ in range(NUM_LAYERS - 1)]
        if non_decomposable:
            mock_layers[layer] = MagicMock(spec=["mlp"])

        decomp = _make_decomp(layers=[layer])

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
        ):
            return _extract_k_vector_impl(
                model, config, tokenizer, meta, "hello", layer, kv_head, position
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "layer",
            "kv_head",
            "position",
            "k_vector",
            "k_norm",
            "num_kv_heads",
            "head_dim",
        ]:
            assert k in r

    def test_k_vector_length(self) -> None:
        r = self._run()
        assert len(r["k_vector"]) == HEAD_DIM

    def test_k_norm_nonnegative(self) -> None:
        r = self._run()
        assert r["k_norm"] >= 0.0

    def test_position_stored(self) -> None:
        r = self._run(position=2)
        assert r["position"] == 2

    def test_with_rope_path(self) -> None:
        r = self._run(with_rope=True)
        assert len(r["k_vector"]) == HEAD_DIM

    def test_with_k_norm_path(self) -> None:
        r = self._run(with_k_norm=True)
        assert len(r["k_vector"]) == HEAD_DIM

    def test_non_decomposable_raises(self) -> None:
        with pytest.raises(ValueError, match="decomposable"):
            self._run(non_decomposable=True)


# ---------------------------------------------------------------------------
# _extract_q_vector_impl — sync tests
# ---------------------------------------------------------------------------


class TestExtractQVectorImpl:
    def _run(
        self,
        layer: int = 0,
        head: int = 0,
        position: int = -1,
        non_decomposable: bool = False,
        with_rope: bool = False,
        with_q_norm: bool = False,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _extract_q_vector_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        meta = MagicMock()
        meta.num_attention_heads = NUM_HEADS
        meta.num_kv_heads = NUM_HEADS
        meta.hidden_dim = HIDDEN_DIM
        meta.head_dim = HEAD_DIM

        mock_layer = _make_mock_layer()
        if with_rope:
            mock_layer.self_attn.rope = lambda q, k: (q, k)
        if with_q_norm:
            mock_layer.self_attn.q_norm = lambda x: x

        mock_layers = [mock_layer] + [_make_mock_layer() for _ in range(NUM_LAYERS - 1)]
        if non_decomposable:
            mock_layers[layer] = MagicMock(spec=["mlp"])

        decomp = _make_decomp(layers=[layer])

        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = mock_layers

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
        ):
            return _extract_q_vector_impl(
                model, config, tokenizer, meta, "hello", layer, head, position
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "layer",
            "head",
            "position",
            "q_vector",
            "q_norm",
            "num_heads",
            "head_dim",
        ]:
            assert k in r

    def test_q_vector_length(self) -> None:
        r = self._run()
        assert len(r["q_vector"]) == HEAD_DIM

    def test_q_norm_nonnegative(self) -> None:
        r = self._run()
        assert r["q_norm"] >= 0.0

    def test_with_rope_path(self) -> None:
        r = self._run(with_rope=True)
        assert len(r["q_vector"]) == HEAD_DIM

    def test_with_q_norm_path(self) -> None:
        r = self._run(with_q_norm=True)
        assert len(r["q_vector"]) == HEAD_DIM

    def test_non_decomposable_raises(self) -> None:
        with pytest.raises(ValueError, match="decomposable"):
            self._run(non_decomposable=True)


# ---------------------------------------------------------------------------
# _extract_attention_output_impl — sync tests
# ---------------------------------------------------------------------------


class TestExtractAttentionOutputImpl:
    def _run(self, no_lm_head: bool = False, non_decomposable: bool = False) -> dict:
        from chuk_mcp_lazarus.tools.geometry.head_dla import _extract_attention_output_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_attention_heads = NUM_HEADS
        meta.num_kv_heads = NUM_HEADS
        meta.hidden_dim = HIDDEN_DIM
        meta.head_dim = HEAD_DIM

        decomp = _make_decomp(layers=[0])

        mock_layer = _make_mock_layer() if not non_decomposable else MagicMock(spec=["mlp"])
        mock_helper = MagicMock()
        mock_helper._get_layers.return_value = [mock_layer] + [
            _make_mock_layer() for _ in range(NUM_LAYERS - 1)
        ]

        lm_head = (
            None
            if no_lm_head
            else MagicMock(
                return_value=mx.array(np.random.randn(1, 1, VOCAB_SIZE).astype(np.float32))
            )
        )

        context = mx.array(np.random.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32))
        o_weight = mx.array(np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32))

        with (
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._run_decomposition_forward",
                return_value=decomp,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch(
                "chuk_mcp_lazarus.tools.geometry.head_dla._compute_per_head_attention",
                return_value=(context, o_weight),
            ),
        ):
            return _extract_attention_output_impl(
                model, config, tokenizer, meta, "hello", 0, 0, -1, 5
            )

    def test_output_keys(self) -> None:
        r = self._run()
        for k in [
            "prompt",
            "layer",
            "head",
            "position",
            "vector",
            "vector_norm",
            "top_projections",
            "dimensionality",
        ]:
            assert k in r

    def test_vector_length(self) -> None:
        r = self._run()
        assert len(r["vector"]) == HIDDEN_DIM

    def test_top_projections_structure(self) -> None:
        r = self._run()
        assert isinstance(r["top_projections"], list)
        if r["top_projections"]:
            proj = r["top_projections"][0]
            assert "token" in proj
            assert "token_id" in proj
            assert "coefficient" in proj
            assert "fraction" in proj

    def test_dimensionality_keys(self) -> None:
        r = self._run()
        d = r["dimensionality"]
        assert "dims_for_99pct" in d
        assert "is_one_dimensional" in d
        assert "top1_fraction" in d
        assert "top2_fraction" in d

    def test_no_lm_head_raises(self) -> None:
        with pytest.raises(ValueError, match="language model head"):
            self._run(no_lm_head=True)

    def test_non_decomposable_raises(self) -> None:
        with pytest.raises(ValueError, match="decomposable"):
            self._run(non_decomposable=True)

    def test_vector_norm_nonnegative(self) -> None:
        r = self._run()
        assert r["vector_norm"] >= 0.0


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestResultModels:
    def test_dla_head_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import DlaHeadEntry

        e = DlaHeadEntry(head=2, dla=0.42, fraction_of_layer=0.15, top_token="Paris")
        d = e.model_dump()
        assert d["head"] == 2
        assert d["dla"] == 0.42
        assert d["top_token"] == "Paris"

    def test_dla_layer_entry(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import DlaHeadEntry, DlaLayerEntry

        heads = [
            DlaHeadEntry(head=i, dla=0.1, fraction_of_layer=0.25, top_token="t") for i in range(4)
        ]
        e = DlaLayerEntry(layer=3, is_decomposable=True, heads=heads, layer_total_dla=0.4)
        d = e.model_dump()
        assert d["layer"] == 3
        assert d["is_decomposable"] is True
        assert len(d["heads"]) == 4

    def test_hot_cell(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import HotCell

        c = HotCell(layer=5, head=2, dla=1.2, top_token="Paris", fraction_of_layer=0.8, abs_rank=1)
        d = c.model_dump()
        assert d["abs_rank"] == 1
        assert d["layer"] == 5

    def test_batch_dla_scan_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import BatchDlaScanResult

        r = BatchDlaScanResult(
            prompt="hello",
            target_token="tok",
            target_token_id=1,
            position=-1,
            num_layers_scanned=2,
            num_heads=4,
            layers=[],
            hot_cells=[],
            summary={"x": 1},
        )
        d = r.model_dump()
        assert d["num_heads"] == 4
        assert d["summary"]["x"] == 1

    def test_compute_dla_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import ComputeDlaResult

        r = ComputeDlaResult(
            prompt="hello",
            layer=0,
            head=1,
            target_token="Paris",
            target_token_id=42,
            position=-1,
            dla=0.75,
            fraction_of_layer=0.33,
            top_token="Paris",
            head_output_norm=2.1,
        )
        d = r.model_dump()
        assert d["dla"] == 0.75

    def test_attention_output_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import AttentionOutputResult

        r = AttentionOutputResult(
            prompt="hello",
            layer=0,
            head=0,
            position=-1,
            vector=[0.1] * HIDDEN_DIM,
            vector_norm=1.0,
            top_projections=[{"token": "t", "token_id": 1, "coefficient": 0.5, "fraction": 0.1}],
            dimensionality={"dims_for_99pct": 1, "is_one_dimensional": True},
        )
        d = r.model_dump()
        assert d["vector_norm"] == 1.0

    def test_embedding_result_with_both(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import EmbeddingResult

        r = EmbeddingResult(
            token="Paris",
            token_id=42,
            unembedding=[0.1] * HIDDEN_DIM,
            input_embedding=[0.2] * HIDDEN_DIM,
            embeddings_tied=False,
            unembedding_norm=1.0,
            input_embedding_norm=1.1,
            cosine_similarity=0.9,
        )
        d = r.model_dump()
        assert d["embeddings_tied"] is False
        assert d["cosine_similarity"] == 0.9

    def test_embedding_result_tied_no_input(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import EmbeddingResult

        r = EmbeddingResult(
            token="Paris",
            token_id=42,
            unembedding=[0.1] * HIDDEN_DIM,
            input_embedding=None,
            embeddings_tied=True,
            unembedding_norm=1.0,
        )
        d = r.model_dump()
        assert d["input_embedding"] is None
        assert d["embeddings_tied"] is True

    def test_k_vector_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import KVectorResult

        r = KVectorResult(
            prompt="hi",
            layer=2,
            kv_head=1,
            position=-1,
            k_vector=[0.0] * HEAD_DIM,
            k_norm=0.5,
            num_kv_heads=4,
            head_dim=HEAD_DIM,
        )
        d = r.model_dump()
        assert d["head_dim"] == HEAD_DIM
        assert d["kv_head"] == 1

    def test_q_vector_result(self) -> None:
        from chuk_mcp_lazarus.tools.geometry.head_dla import QVectorResult

        r = QVectorResult(
            prompt="hi",
            layer=1,
            head=0,
            position=-1,
            q_vector=[0.0] * HEAD_DIM,
            q_norm=0.7,
            num_heads=4,
            head_dim=HEAD_DIM,
        )
        d = r.model_dump()
        assert d["num_heads"] == 4
        assert len(d["q_vector"]) == HEAD_DIM
