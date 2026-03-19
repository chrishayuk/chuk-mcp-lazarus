# Activation Steering

Compute steering vectors (contrastive activation addition) and apply them at inference time.

---

## `compute_steering_vector`

```python
async def compute_steering_vector(
    vector_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int = -1,
) -> dict
```

Compute a steering vector as the **mean difference** between two sets of activations.

The vector points from the negative cluster to the positive cluster in activation space at the specified layer.

**Returns** `{vector_name, layer, vector_norm, separability_score, num_positive, num_negative}`.

---

## `steer_and_generate`

```python
async def steer_and_generate(
    prompt: str,
    vector_name: str,
    alpha: float = 20.0,
    max_new_tokens: int = 100,
) -> dict
```

Add `alpha × steering_vector` to the residual stream at the stored layer and generate.  Also runs an unsteered baseline for comparison.

**`alpha`:** Start at 10–20.  Too large distorts outputs; too small has no effect.

**Returns** `{steered_output, baseline_output, steered_tokens, baseline_tokens}`.

---

## `list_steering_vectors`

```python
async def list_steering_vectors() -> dict
```

List all vectors in `SteeringVectorRegistry`.

---

### Relationship to `extract_direction`

`compute_steering_vector` uses `mean_diff` method only.  For other extraction methods (LDA, PCA, probe-derived), use `extract_direction` (see [Direction](../direction/)).  Both store in `SteeringVectorRegistry` and are usable with `steer_and_generate`.
