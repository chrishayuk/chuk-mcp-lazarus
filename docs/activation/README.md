# Activation Extraction

Two tools for extracting and comparing hidden-state activations.

---

## `extract_activations`

```python
async def extract_activations(
    prompt: str,
    layers: list[int],
    token_position: int = -1,
    capture_attention: bool = False,
) -> dict
```

Run a forward pass and return hidden-state activations at the specified layers for one token position.

**Parameters**

| Name | Default | Description |
|------|---------|-------------|
| `prompt` | — | Input text |
| `layers` | — | Layer indices (0-indexed) |
| `token_position` | `-1` | Token to extract (`-1` = last) |
| `capture_attention` | `False` | Also return attention weights (memory-heavy) |

**Returns** `activations: {layer_idx: [float, ...]}` — hidden states as plain Python lists.

---

## `compare_activations`

```python
async def compare_activations(
    prompts: list[str],
    layer: int,
    token_position: int = -1,
) -> dict
```

Extract activations at a single layer for 2–8 prompts.  Returns pairwise cosine similarities and a 2-D PCA projection.

**Returns**

```json
{
  "layer": 10,
  "prompts": ["Hello", "Bonjour"],
  "cosine_similarity_matrix": [[1.0, 0.82], [0.82, 1.0]],
  "pca_2d": [[-0.12, 0.04], [0.12, -0.04]],
  "centroid_distance": 0.18
}
```

---

### Usage pattern

Use `compare_activations` at early and late layers to detect where representations converge — a key signal in language transition experiments:

```python
# Layer 0: distinct language representations
result_early = await compare_activations(["Hello", "Bonjour", "Hola"], layer=0)
# Layer 33: converged multilingual representation
result_late  = await compare_activations(["Hello", "Bonjour", "Hola"], layer=33)
```
