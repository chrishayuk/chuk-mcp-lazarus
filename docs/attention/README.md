# Attention Analysis

Two tools for inspecting per-head attention weights and entropy.

> **Note:** These tools manually unroll the attention forward pass to expose weights that MLX's fused `scaled_dot_product_attention` does not return.

---

## `attention_pattern`

```python
async def attention_pattern(
    prompt: str,
    layers: list[int],
    token_position: int = -1,
    top_k: int = 5,
) -> dict
```

Extract per-head attention weights showing which tokens each head attends to.

**Returns** for each layer and head: `top_attended` — top-k source positions ranked by attention weight.

---

## `attention_heads`

```python
async def attention_heads(
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 3,
) -> dict
```

Compute per-head entropy and focus across layers.

- **Low entropy** = focused head (attends strongly to one or few positions)
- **High entropy** = diffuse head (spreads attention broadly)

**Returns** `heads: [{layer, head, entropy, max_attention, top_attended_positions}]` and `summary: {most_focused_heads, most_diffuse_heads}`.

---

### Interpretation

Focused heads with low entropy are candidates for copy heads in the copy circuit hypothesis.  Use `batch_dla_scan` (see [Geometry / DLA](../geometry/dla/)) to confirm their logit contribution.
