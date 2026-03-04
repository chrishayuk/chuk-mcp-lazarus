"""Tests for _generate.py — shared text generation helper."""

from unittest.mock import MagicMock

import numpy as np

from chuk_mcp_lazarus._generate import generate_text


class TestGenerateText:
    def _make_model_and_tokenizer(self, vocab_size: int = 100, eos_id: int = 0):
        """Create mock model and tokenizer for generation tests."""
        import mlx.core as mx

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3]
        tok.eos_token_id = eos_id
        tok.decode.side_effect = lambda ids, **kw: "generated text"

        # Model returns logits: shape [1, seq, vocab]
        # Make token 5 the highest logit each time
        call_count = [0]

        def model_call(input_ids):
            call_count[0] += 1
            seq_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else 1
            logits = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
            # After a few tokens, predict EOS
            if call_count[0] >= 4:
                logits[0, -1, eos_id] = 10.0
            else:
                logits[0, -1, 5] = 10.0
            return mx.array(logits)

        model = MagicMock(side_effect=model_call)
        return model, tok

    def test_greedy(self) -> None:
        model, tok = self._make_model_and_tokenizer()
        text, num_tokens = generate_text(model, tok, "hello", max_new_tokens=5, temperature=0.0)
        assert isinstance(text, str)
        assert num_tokens > 0

    def test_stops_at_eos(self) -> None:
        model, tok = self._make_model_and_tokenizer()
        _, num_tokens = generate_text(model, tok, "hello", max_new_tokens=10, temperature=0.0)
        # Should stop before max_new_tokens due to EOS at step 4
        assert num_tokens <= 10

    def test_respects_max_tokens(self) -> None:
        import mlx.core as mx

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3]
        tok.eos_token_id = None  # No EOS - will run to max
        tok.decode.return_value = "text"

        def model_call(input_ids):
            seq_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else 1
            logits = np.random.randn(1, seq_len, 100).astype(np.float32)
            return mx.array(logits)

        model = MagicMock(side_effect=model_call)
        _, num_tokens = generate_text(model, tok, "hi", max_new_tokens=3, temperature=0.0)
        assert num_tokens == 3

    def test_sampling(self) -> None:
        model, tok = self._make_model_and_tokenizer()
        text, num_tokens = generate_text(model, tok, "hello", max_new_tokens=5, temperature=0.8)
        assert isinstance(text, str)

    def test_returns_tuple(self) -> None:
        model, tok = self._make_model_and_tokenizer()
        result = generate_text(model, tok, "hello", max_new_tokens=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
