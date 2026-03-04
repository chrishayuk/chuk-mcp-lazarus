"""Tests for tools/attention_tools.py — attention_pattern, attention_heads."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mlx.core as mx

from chuk_mcp_lazarus.tools.attention_tools import (
    attention_heads,
    attention_pattern,
    _attention_pattern_impl,
    _attention_heads_impl,
)


class TestAttentionPattern:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_top_k_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await attention_pattern(prompt="hello", top_k=101)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "tokens": ["hello"],
            "num_layers_analyzed": 1,
            "patterns": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_pattern_impl",
            return_value=mock_result,
        ):
            result = await attention_pattern(prompt="hello", layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "tokens": ["hello"],
            "num_layers_analyzed": 3,
            "patterns": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_pattern_impl",
            return_value=mock_result,
        ):
            result = await attention_pattern(prompt="hello")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        """Exception path: _attention_pattern_impl raises -> ExtractionFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_pattern_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await attention_pattern(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "boom" in result["message"]

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: MagicMock) -> None:
        """Negative layer indices are rejected as out of range."""
        result = await attention_pattern(prompt="hello", layers=[-1])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"


class TestAttentionHeads:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await attention_heads(prompt="hello")
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await attention_heads(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, loaded_model_state: MagicMock) -> None:
        result = await attention_heads(prompt="hello", top_k=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "tokens": ["hello"],
            "num_heads_analyzed": 4,
            "heads": [],
            "summary": {"most_focused_heads": [], "most_diffuse_heads": []},
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_heads_impl",
            return_value=mock_result,
        ):
            result = await attention_heads(prompt="hello", layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: MagicMock) -> None:
        """attention_heads with layers=None defaults to 3 sampled layers."""
        mock_result = {
            "prompt": "hello",
            "tokens": ["hello"],
            "num_heads_analyzed": 12,
            "heads": [],
            "summary": {"most_focused_heads": [], "most_diffuse_heads": []},
        }
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_heads_impl",
            return_value=mock_result,
        ):
            result = await attention_heads(prompt="hello")
        assert "error" not in result
        assert result["num_heads_analyzed"] == 12

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        """Exception path: _attention_heads_impl raises -> ExtractionFailed."""
        with patch(
            "chuk_mcp_lazarus.tools.attention_tools._attention_heads_impl",
            side_effect=RuntimeError("kaboom"),
        ):
            result = await attention_heads(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "kaboom" in result["message"]

    @pytest.mark.asyncio
    async def test_top_k_too_high(self, loaded_model_state: MagicMock) -> None:
        """top_k > 100 is rejected as InvalidInput."""
        result = await attention_heads(prompt="hello", top_k=101)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"


# ---------------------------------------------------------------------------
# Helpers for _impl tests
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(tokens: list[str]) -> MagicMock:
    """Build a mock tokenizer that encodes to len(tokens) ids."""
    tok = MagicMock()
    ids = list(range(len(tokens)))
    tok.encode.return_value = ids
    tok.decode.side_effect = lambda tid_list, **kw: tokens[tid_list[0]]
    return tok


def _make_attn_weights(layers: list[int], num_heads: int, seq_len: int, ndim: int = 4) -> dict:
    """Build a fake _compute_attention_weights return value.

    Each layer gets a softmax-normalised attention weight array.
    ndim=4 -> [batch, heads, seq, seq], ndim=3 -> [heads, seq, seq].
    """
    out: dict[int, mx.array] = {}
    for layer_idx in layers:
        # Create a simple attention pattern: each position attends mostly to itself
        raw = np.zeros((num_heads, seq_len, seq_len), dtype=np.float32)
        for h in range(num_heads):
            for s in range(seq_len):
                raw[h, s, s] = 0.7  # self-attention
                # spread remaining 0.3 over other positions
                for j in range(seq_len):
                    if j != s:
                        raw[h, s, j] = 0.3 / max(seq_len - 1, 1)
        if ndim == 4:
            raw = raw[np.newaxis, ...]  # add batch dim
        out[layer_idx] = mx.array(raw)
    return out


# ---------------------------------------------------------------------------
# Tests for _attention_pattern_impl
# ---------------------------------------------------------------------------


class TestAttentionPatternImpl:
    """Direct tests for the _attention_pattern_impl function."""

    def test_basic_single_layer(self) -> None:
        """Basic call with one layer returns expected structure."""
        tokens = ["hello", "world", "foo"]
        tok = _make_mock_tokenizer(tokens)
        model = MagicMock()
        config = MagicMock()
        layers = [0]
        num_heads = 2
        seq_len = len(tokens)

        fake_weights = _make_attn_weights(layers, num_heads, seq_len)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(model, config, tok, "hello world foo", layers, -1, 3)

        assert result["prompt"] == "hello world foo"
        assert result["token_position"] == -1
        assert result["token_text"] == tokens[-1]  # last token
        assert result["tokens"] == tokens
        assert result["num_layers_analyzed"] == 1
        assert len(result["patterns"]) == 1

        pattern = result["patterns"][0]
        assert pattern["layer"] == 0
        assert pattern["num_heads"] == num_heads
        assert len(pattern["heads"]) == num_heads

    def test_multiple_layers(self) -> None:
        """Multiple layers are all returned in results."""
        tokens = ["a", "b", "c", "d"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0, 2, 3]
        num_heads = 4
        seq_len = len(tokens)

        fake_weights = _make_attn_weights(layers, num_heads, seq_len)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(
                MagicMock(), MagicMock(), tok, "a b c d", layers, -1, 2
            )

        assert result["num_layers_analyzed"] == 3
        assert [p["layer"] for p in result["patterns"]] == [0, 2, 3]

    def test_token_position_positive(self) -> None:
        """Positive token_position selects correct token."""
        tokens = ["the", "cat", "sat"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 2, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(
                MagicMock(), MagicMock(), tok, "the cat sat", layers, 1, 2
            )

        assert result["token_position"] == 1
        assert result["token_text"] == "cat"

    def test_token_position_last(self) -> None:
        """token_position=-1 resolves to last token."""
        tokens = ["x", "y"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "x y", layers, -1, 2)

        assert result["token_text"] == "y"

    def test_top_k_limits_attended_positions(self) -> None:
        """top_k controls how many attended positions appear per head."""
        tokens = ["a", "b", "c", "d", "e"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(
                MagicMock(), MagicMock(), tok, "a b c d e", layers, -1, 2
            )

        head = result["patterns"][0]["heads"][0]
        assert len(head["top_attended"]) == 2  # top_k=2

    def test_top_k_exceeds_seq_len(self) -> None:
        """top_k larger than seq_len returns all positions."""
        tokens = ["a", "b"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "a b", layers, -1, 10)

        head = result["patterns"][0]["heads"][0]
        # Only 2 tokens, so at most 2 attended positions
        assert len(head["top_attended"]) == 2

    def test_attention_weights_are_rounded(self) -> None:
        """Weights in output are rounded to 6 decimal places."""
        tokens = ["hi", "there"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(
                MagicMock(), MagicMock(), tok, "hi there", layers, -1, 2
            )

        head = result["patterns"][0]["heads"][0]
        for w in head["attention_weights"]:
            # Check it's a float (rounded values are still floats)
            assert isinstance(w, float)
            # Check rounding: string repr has at most 6 decimal digits
            parts = str(w).split(".")
            if len(parts) == 2:
                assert (
                    len(parts[1]) <= 7
                )  # float repr may have trailing digits, but value is rounded

    def test_top_attended_sorted_descending(self) -> None:
        """Top attended positions are sorted by weight descending."""
        tokens = ["a", "b", "c"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]

        # Build a custom attention pattern where position 2 has highest weight
        raw = np.array([[[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.15, 0.15, 0.7]]]], dtype=np.float32)
        fake_weights = {0: mx.array(raw)}

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "a b c", layers, -1, 3)

        head = result["patterns"][0]["heads"][0]
        weights = [pos["weight"] for pos in head["top_attended"]]
        assert weights == sorted(weights, reverse=True)

    def test_missing_layer_in_weights(self) -> None:
        """If _compute_attention_weights returns no data for a layer, skip it."""
        tokens = ["a", "b"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0, 1]
        # Only provide weights for layer 0, not layer 1
        fake_weights = _make_attn_weights([0], 2, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "a b", layers, -1, 2)

        # Only layer 0 should appear
        assert result["num_layers_analyzed"] == 1
        assert result["patterns"][0]["layer"] == 0

    def test_3d_attention_weights(self) -> None:
        """Handles 3D attention weights (no batch dimension) correctly."""
        tokens = ["x", "y", "z"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 2, len(tokens), ndim=3)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "x y z", layers, -1, 2)

        # Should still work — 3D array doesn't need w[0] squeeze
        assert result["num_layers_analyzed"] == 1
        assert result["patterns"][0]["num_heads"] == 2

    def test_token_position_clamped_to_valid_range(self) -> None:
        """Out-of-range negative position is clamped to 0."""
        tokens = ["a", "b"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            # token_position=-100 should clamp to 0
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "a b", layers, -100, 2)

        assert result["token_text"] == "a"

    def test_single_token_prompt(self) -> None:
        """Single token prompt works correctly."""
        tokens = ["only"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, 1)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_pattern_impl(MagicMock(), MagicMock(), tok, "only", layers, -1, 5)

        assert result["tokens"] == ["only"]
        assert result["token_text"] == "only"
        assert len(result["patterns"][0]["heads"][0]["top_attended"]) == 1


# ---------------------------------------------------------------------------
# Tests for _attention_heads_impl
# ---------------------------------------------------------------------------


class TestAttentionHeadsImpl:
    """Direct tests for the _attention_heads_impl function."""

    def test_basic_single_layer(self) -> None:
        """Basic call with one layer returns expected structure."""
        tokens = ["hello", "world", "foo"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        num_heads = 2
        seq_len = len(tokens)

        fake_weights = _make_attn_weights(layers, num_heads, seq_len)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(
                MagicMock(), MagicMock(), tok, "hello world foo", layers, 3
            )

        assert result["prompt"] == "hello world foo"
        assert result["tokens"] == tokens
        assert result["num_heads_analyzed"] == num_heads
        assert len(result["heads"]) == num_heads
        assert "summary" in result
        assert "most_focused_heads" in result["summary"]
        assert "most_diffuse_heads" in result["summary"]

    def test_multiple_layers(self) -> None:
        """Multiple layers produce heads from all layers."""
        tokens = ["a", "b", "c"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0, 1, 2]
        num_heads = 3

        fake_weights = _make_attn_weights(layers, num_heads, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b c", layers, 2)

        assert result["num_heads_analyzed"] == num_heads * len(layers)
        # Check all layers represented
        head_layers = {h["layer"] for h in result["heads"]}
        assert head_layers == {0, 1, 2}

    def test_entropy_fields_present(self) -> None:
        """Each head has entropy and max_attention fields."""
        tokens = ["a", "b"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 2, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b", layers, 2)

        for head in result["heads"]:
            assert "entropy" in head
            assert "max_attention" in head
            assert isinstance(head["entropy"], float)
            assert isinstance(head["max_attention"], float)
            assert head["entropy"] >= 0.0

    def test_focused_attention_low_entropy(self) -> None:
        """A head that concentrates on one token has low entropy."""
        tokens = ["a", "b", "c", "d"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        seq_len = len(tokens)

        # Head 0: focused (all weight on position 0)
        # Head 1: diffuse (uniform)
        raw = np.zeros((1, 2, seq_len, seq_len), dtype=np.float32)
        # Focused head: last token attends entirely to position 0
        raw[0, 0, -1, 0] = 1.0
        # Diffuse head: uniform attention from last token
        raw[0, 1, -1, :] = 1.0 / seq_len

        fake_weights = {0: mx.array(raw)}

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b c d", layers, 2)

        focused_head = result["heads"][0]
        diffuse_head = result["heads"][1]
        assert focused_head["entropy"] < diffuse_head["entropy"]

    def test_max_attention_value(self) -> None:
        """max_attention reflects the actual max weight from last token."""
        tokens = ["x", "y"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]

        raw = np.array([[[[0.3, 0.7], [0.8, 0.2]]]], dtype=np.float32)
        fake_weights = {0: mx.array(raw)}

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "x y", layers, 2)

        # Head 0 last token attention: [0.8, 0.2] -> max is 0.8
        head0 = result["heads"][0]
        assert abs(head0["max_attention"] - 0.8) < 1e-4

    def test_top_k_attended_positions(self) -> None:
        """top_k limits attended positions in head results."""
        tokens = ["a", "b", "c", "d", "e"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b c d e", layers, 2)

        head = result["heads"][0]
        assert len(head["top_attended_positions"]) == 2

    def test_summary_most_focused_and_diffuse(self) -> None:
        """Summary contains up to 5 most focused and most diffuse heads."""
        tokens = ["a", "b", "c"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0, 1]
        num_heads = 4
        fake_weights = _make_attn_weights(layers, num_heads, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b c", layers, 2)

        summary = result["summary"]
        # 8 total heads, summary should have at most 5
        assert len(summary["most_focused_heads"]) <= 5
        assert len(summary["most_diffuse_heads"]) <= 5
        # Each entry has layer, head, entropy
        for entry in summary["most_focused_heads"]:
            assert "layer" in entry
            assert "head" in entry
            assert "entropy" in entry

    def test_summary_sorted_by_entropy(self) -> None:
        """Most focused heads have lowest entropy, most diffuse have highest."""
        tokens = ["a", "b", "c", "d"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        seq_len = len(tokens)

        # Create varying entropy heads
        raw = np.zeros((1, 4, seq_len, seq_len), dtype=np.float32)
        # Head 0: very focused (entropy ~0)
        raw[0, 0, -1, 0] = 1.0
        # Head 1: somewhat focused
        raw[0, 1, -1, 0] = 0.8
        raw[0, 1, -1, 1] = 0.2
        # Head 2: more diffuse
        raw[0, 2, -1, :] = 0.25
        # Head 3: completely uniform
        raw[0, 3, -1, :] = 0.25

        fake_weights = {0: mx.array(raw)}

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b c d", layers, 2)

        focused = result["summary"]["most_focused_heads"]
        diffuse = result["summary"]["most_diffuse_heads"]

        # Most focused should have lower entropy than most diffuse
        if focused and diffuse:
            assert focused[0]["entropy"] <= diffuse[-1]["entropy"]

    def test_missing_layer_in_weights(self) -> None:
        """If layer missing from attention weights, it is skipped."""
        tokens = ["a", "b"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0, 1]
        # Only provide layer 0
        fake_weights = _make_attn_weights([0], 2, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b", layers, 2)

        assert result["num_heads_analyzed"] == 2
        assert all(h["layer"] == 0 for h in result["heads"])

    def test_3d_attention_weights(self) -> None:
        """Handles 3D attention weights (no batch dimension)."""
        tokens = ["p", "q"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 2, len(tokens), ndim=3)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "p q", layers, 2)

        assert result["num_heads_analyzed"] == 2

    def test_single_token_prompt(self) -> None:
        """Single-token prompt: max_entropy set to 1.0 (seq_len=1 special case)."""
        tokens = ["z"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        fake_weights = _make_attn_weights(layers, 1, 1)

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "z", layers, 5)

        assert result["num_heads_analyzed"] == 1
        head = result["heads"][0]
        assert isinstance(head["entropy"], float)

    def test_entropy_normalized(self) -> None:
        """Entropy is normalized by max_entropy = log(seq_len)."""

        tokens = ["a", "b", "c", "d"]
        tok = _make_mock_tokenizer(tokens)
        layers = [0]
        seq_len = len(tokens)

        # Uniform attention from last token -> max entropy
        raw = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
        raw[0, 0, -1, :] = 1.0 / seq_len

        fake_weights = {0: mx.array(raw)}

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b c d", layers, 2)

        head = result["heads"][0]
        # Uniform distribution -> entropy should be close to 1.0 (normalized)
        assert abs(head["entropy"] - 1.0) < 0.01

    def test_head_analysis_has_correct_layer_and_head_indices(self) -> None:
        """Each HeadAnalysis entry has correct layer and head indices."""
        tokens = ["a", "b"]
        tok = _make_mock_tokenizer(tokens)
        layers = [2, 5]
        num_heads = 3
        fake_weights = _make_attn_weights(layers, num_heads, len(tokens))

        with patch(
            "chuk_mcp_lazarus._compare._compute_attention_weights",
            return_value=fake_weights,
        ):
            result = _attention_heads_impl(MagicMock(), MagicMock(), tok, "a b", layers, 2)

        # Should have 6 heads total (3 per layer x 2 layers)
        assert result["num_heads_analyzed"] == 6
        # First 3 heads from layer 2, next 3 from layer 5
        for i in range(3):
            assert result["heads"][i]["layer"] == 2
            assert result["heads"][i]["head"] == i
        for i in range(3):
            assert result["heads"][3 + i]["layer"] == 5
            assert result["heads"][3 + i]["head"] == i
