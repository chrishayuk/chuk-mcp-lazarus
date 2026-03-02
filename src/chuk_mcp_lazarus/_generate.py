"""
Shared text generation helper.

Simple greedy/sampling generation used by steering and ablation tools.
Extracted here to avoid code duplication (Architecture Principle 4).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> tuple[str, int]:
    """Generate text from a prompt. Returns (generated_text, num_tokens).

    Uses greedy decoding when temperature <= 0, otherwise samples.
    """
    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    generated_ids: list[int] = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        logits = model(current_ids[None, :] if current_ids.ndim == 1 else current_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        if hasattr(logits, "logits"):
            logits = logits.logits

        next_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]

        if temperature <= 0:
            next_id = mx.argmax(next_logits).item()
        else:
            probs = mx.softmax(next_logits / temperature)
            next_id = mx.random.categorical(mx.log(probs)).item()

        if next_id == eos_token_id:
            break

        generated_ids.append(next_id)
        next_token = mx.array([next_id])
        current_ids = mx.concatenate([
            current_ids.reshape(-1) if current_ids.ndim > 1 else current_ids,
            next_token,
        ])

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, len(generated_ids)
