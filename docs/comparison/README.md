# Model Comparison

Load a second model and compare it to the primary model on weights, representations, attention, and generation.

---

## `load_comparison_model`

```python
async def load_comparison_model(
    model_id: str,
    dtype: str = "bfloat16",
) -> dict
```

Load a second model into `ComparisonState`.  Both models must share the same architecture (num_layers, hidden_dim).

---

## `compare_weights`

```python
async def compare_weights(
    layers: list[int] | None = None,
) -> dict
```

Compare weight matrices between primary and comparison models.  Returns per-layer, per-component Frobenius norm and cosine similarity.  **No inference needed** — cheapest comparison operation.

---

## `compare_representations`

```python
async def compare_representations(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict
```

Compare hidden-state activations at the same layers for the same prompts.  Shows where representations diverge between the two models.

---

## `compare_attention`

```python
async def compare_attention(
    prompt: str,
    layers: list[int] | None = None,
) -> dict
```

Compare attention patterns between both models.  Returns per-head Jensen-Shannon divergence and cosine similarity, and the most divergent heads.

---

## `compare_generations`

```python
async def compare_generations(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> dict
```

Generate text from both models and compare side-by-side.  Essential first step to see *what actually changes* before diving into internal representations.

---

## `unload_comparison_model`

```python
async def unload_comparison_model() -> dict
```

Unload the comparison model and free memory.

---

### Typical workflow

```python
# 1. Load both models
await load_model("google/gemma-3-4b")           # primary
await load_comparison_model("translate-gemma")   # fine-tuned

# 2. Quick weight comparison (no inference)
weight_diff = await compare_weights()

# 3. See what actually changed in outputs
gen_diff = await compare_generations("Translate to French: Hello")

# 4. Find where internal representations diverge
rep_diff = await compare_representations(["Paris", "Berlin"], layers=[8, 16, 24])

# 5. Unload to free VRAM
await unload_comparison_model()
```
