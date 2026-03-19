# Causal Tracing

Two tools implementing causal tracing (Meng et al. 2022 style).

---

## `trace_token`

```python
async def trace_token(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
    effect_threshold: float = 0.05,
) -> dict
```

Which layers are causally responsible for predicting a specific token?  Ablates each layer and measures the change in the target token's probability.

**Returns** `{baseline_prob, peak_layer, peak_effect, critical_layers, layer_effects: [{layer, effect, ablated_prob}]}`.

---

## `full_causal_trace`

```python
async def full_causal_trace(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
) -> dict
```

Position × layer causal heatmap.  Tests every `(position, layer)` combination to find the full circuit.

- Reveals which subject token positions contribute to factual recall
- Returns `heatmap: [{position, layer, effect}]` and `peak: {position, layer, effect}`

---

### Relationship to `trace_token`

`trace_token` = row of the heatmap (all layers, one implicit position average).
`full_causal_trace` = the full heatmap — more expensive but reveals the complete circuit structure.
