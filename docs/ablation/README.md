# Ablation

Two tools for causal ablation experiments: zero-out layers and patch activations between prompts.

---

## `ablate_layers`

```python
async def ablate_layers(
    prompt: str,
    layers: list[int],
    max_new_tokens: int = 50,
    ablation_type: str = "zero",
) -> dict
```

Generate text with specific layers zeroed out or mean-ablated.  Compares output to baseline.

**`ablation_type`:**
- `"zero"` — set the layer's output to zero
- `"mean"` — replace with the mean activation across a reference corpus

**Returns** `{ablated_output, baseline_output, output_similarity, disruption_score}`.

---

## `patch_activations`

```python
async def patch_activations(
    source_prompt: str,
    target_prompt: str,
    layer: int,
    max_new_tokens: int = 50,
) -> dict
```

Run `target_prompt` but replace the hidden state at `layer` with the hidden state from `source_prompt`.

This directly tests: *"does knowledge from `source_prompt` flow through layer `layer`?"*

**Returns** `{patched_output, baseline_output, disruption_score}`.

---

### Relationship to `full_causal_trace`

`patch_activations` patches one (position-agnostic) layer.  `full_causal_trace` (see [Causal](../causal/)) sweeps all `(position, layer)` combinations to build a heatmap.
