"""
Shared activation extraction helpers.

Consolidates the duplicate _extract_activation / _extract_activation_at_layer
functions that were repeated across probe_tools.py and steering_tools.py.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from ._serialize import hidden_state_to_list


def extract_activation_at_layer(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    token_position: int = -1,
) -> list[float]:
    """Extract a single activation vector for one prompt at one layer."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    hooks = ModelHooks(model, model_config=config)
    hooks.configure(CaptureConfig(layers=[layer], capture_hidden_states=True))
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)
    return hidden_state_to_list(hooks.state.hidden_states[layer], position=token_position)


def extract_activations_all_layers(
    model: Any,
    config: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    token_position: int = -1,
) -> dict[int, list[float]]:
    """Extract activations at multiple layers from a single forward pass."""
    from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

    input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    hooks = ModelHooks(model, model_config=config)
    hooks.configure(CaptureConfig(layers=layers, capture_hidden_states=True))
    hooks.forward(input_ids)
    mx.eval(hooks.state.hidden_states)

    result: dict[int, list[float]] = {}
    for layer in layers:
        if layer in hooks.state.hidden_states:
            result[layer] = hidden_state_to_list(
                hooks.state.hidden_states[layer], position=token_position
            )
    return result
