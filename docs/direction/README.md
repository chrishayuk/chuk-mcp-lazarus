# Direction Extraction

Extract interpretable directions in activation space using multiple statistical methods.

---

## `extract_direction`

```python
async def extract_direction(
    direction_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    method: str = "mean_diff",
    token_position: int = -1,
) -> dict
```

Extract a direction vector from contrastive prompt pairs and store it in `SteeringVectorRegistry`.

**Methods**

| Method | Description |
|--------|-------------|
| `mean_diff` | Mean of positive activations minus mean of negative activations |
| `lda` | Linear Discriminant Analysis direction |
| `pca` | First principal component of the contrastive difference |
| `probe` | Normal vector of a trained logistic regression probe |

**Returns** `{direction_name, layer, method, vector_norm, separation_score, classification_accuracy}`.

---

### Relationship to other tools

- Stored in the same `SteeringVectorRegistry` as `compute_steering_vector` — usable with `steer_and_generate`.
- The resulting direction can be referenced in `direction_angles` (see [Geometry](../geometry/)) as type `"steering_vector"`.
- Use `probe_at_inference` to monitor the direction's activation live during generation.
