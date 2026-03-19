# Neuron Analysis

Discover, profile, and trace individual MLP neurons.

---

## `discover_neurons`

```python
async def discover_neurons(
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    top_k: int = 10,
    token_position: int = -1,
) -> dict
```

Find neurons that discriminate between two prompt groups using **Cohen's d** effect size.

**Returns** `top_neurons: [{neuron_idx, cohens_d, positive_mean, negative_mean}]`.

---

## `analyze_neuron`

```python
async def analyze_neuron(
    layer: int,
    neuron_indices: list[int],
    prompts: list[str],
    token_position: int = -1,
    detailed: bool = False,
) -> dict
```

Profile specific neurons across prompts.  Returns min/max/mean/std activation statistics.  With `detailed=True`, also returns per-prompt activations.

---

## `neuron_trace`

```python
async def neuron_trace(
    prompt: str,
    layer: int,
    neuron_index: int,
    target_layers: list[int] | None = None,
    token_position: int = -1,
    top_k_heads: int = 5,
) -> dict
```

Trace a neuron's output direction through downstream layers.

Computes cosine similarity between the neuron's output vector and the residual stream, attention output, and FFN output at each downstream layer.  Returns `top_aligned_heads` — the attention heads most aligned with the neuron's direction.

**Returns** `trace: [{layer, residual_alignment, attention_alignment, ffn_alignment, residual_projection, top_aligned_heads}]`.

---

### Workflow

```python
# 1. Find neurons that respond to "capital cities" vs "common nouns"
result = await discover_neurons(
    layer=15,
    positive_prompts=["Paris", "Berlin", "Tokyo", "London"],
    negative_prompts=["cat", "table", "run", "blue"],
    top_k=5,
)
top_neuron = result["top_neurons"][0]["neuron_idx"]

# 2. Profile it across more prompts
profile = await analyze_neuron(
    layer=15,
    neuron_indices=[top_neuron],
    prompts=["Paris is", "Berlin is", "cat is", "table is"],
    detailed=True,
)

# 3. Trace its influence downstream
trace = await neuron_trace(
    prompt="The capital of France is",
    layer=15,
    neuron_index=top_neuron,
)
```
