# Attribution Sweep

Batch logit attribution across multiple prompts in one call.

---

## `attribution_sweep`

```python
async def attribution_sweep(
    prompts: list[str],
    layers: list[int] | None = None,
    position: int = -1,
    target_token: str | None = None,
    normalized: bool = True,
    labels: list[str] | None = None,
) -> dict
```

Run `logit_attribution` on 2–50 prompts and aggregate results.  Returns:

- **`layer_summary`** — mean attention/FFN logit per layer across all prompts
- **`prompt_summary`** — per-prompt row: `{embedding_logit, net_attention, net_ffn, final_logit, top_prediction, probability, dominant_component, top_positive_layer, top_negative_layer}`
- **`per_prompt`** — full attribution result for each prompt
- **`dominant_layer`** / **`dominant_component`** — the single most important layer/component across all prompts

**Parameters**

| Name | Default | Description |
|------|---------|-------------|
| `prompts` | — | 2–50 input texts |
| `layers` | `None` | Layers to attribute (None = all) |
| `position` | `-1` | Token position |
| `target_token` | `None` | Token to attribute (None = top per prompt) |
| `normalized` | `True` | Use calibrated `norm → lm_head` projection |
| `labels` | `None` | Optional per-prompt labels for grouping |

---

### Use case

Compare attribution profiles for semantically related prompts to identify which layers are consistently responsible for a concept:

```python
result = await attribution_sweep(
    prompts=[
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
    ],
    target_token="Paris",
    labels=["correct", "incorrect", "incorrect"],
)
```
