# Surgical Intervention

Zero out or scale a specific component (attention, FFN, or individual head) at a layer and compare predictions.

---

## `component_intervention`

```python
async def component_intervention(
    prompt: str,
    layer: int,
    component: str,
    intervention: str = "zero",
    scale_factor: float = 0.0,
    head: int | None = None,
    top_k: int = 10,
    token_position: int = -1,
) -> dict
```

**Parameters**

| Name | Default | Description |
|------|---------|-------------|
| `prompt` | — | Input text |
| `layer` | — | Layer to intervene at |
| `component` | — | `"attention"`, `"ffn"`, or `"head"` |
| `intervention` | `"zero"` | `"zero"` or `"scale"` |
| `scale_factor` | `0.0` | Scale factor (0 for zero, any float for scale) |
| `head` | `None` | Head index (required when `component="head"`) |
| `top_k` | `10` | Top predictions to compare (1–50) |
| `token_position` | `-1` | Token position (-1 = last) |

**Returns**

```json
{
  "original_top_k": [...],
  "intervened_top_k": [...],
  "kl_divergence": 0.42,
  "target_delta": -0.18,
  "original_top1": "Paris",
  "intervened_top1": "Rome",
  "top1_changed": true,
  "summary": "..."
}
```

---

### Difference from `ablate_layers`

`ablate_layers` re-runs generation (token-by-token).  `component_intervention` does a **single forward pass** and compares the next-token distribution — faster and more surgical.  Use intervention to pinpoint which component at which layer holds a specific fact; use ablation to see how generation changes over many tokens.
