# Geometry — Angles

Four tools for measuring angles and decompositions in activation space.

---

## `token_space`

```python
async def token_space(
    prompt: str,
    layer: int,
    tokens: list[str],
    token_position: int = -1,
    include_projection: bool = False,
) -> dict
```

Map geometric relationships between token unembedding directions and the residual stream.

For each token, reports:
- Angle (degrees) between the token's unembedding vector and the residual stream
- Projection (signed scalar) of the residual stream onto the token direction
- Pairwise angles between all tokens

`include_projection=True` adds a lossy 2D PCA projection.

---

## `direction_angles`

```python
async def direction_angles(
    prompt: str,
    layer: int,
    directions: list[dict],
    token_position: int = -1,
) -> dict
```

Compute pairwise angles between arbitrary directions at a layer.

Each direction is a dict with `"type"` and optional `"value"`:

| type | value | What it extracts |
|------|-------|-----------------|
| `"token"` | token string | Token unembedding vector |
| `"neuron"` | neuron index | Neuron output direction |
| `"residual"` | — | Residual stream at this layer |
| `"ffn_output"` | — | FFN sub-layer output |
| `"attention_output"` | — | Attention sub-layer output |
| `"head_output"` | head index | Single attention head output |
| `"steering_vector"` | vector name | Stored steering vector |

---

## `subspace_decomposition`

```python
async def subspace_decomposition(
    prompt: str,
    layer: int,
    target: dict,
    basis_directions: list[dict],
    token_position: int = -1,
    orthogonalize: bool = True,
) -> dict
```

Decompose a target vector into components along basis directions plus an orthogonal residual.

`orthogonalize=True` applies Gram-Schmidt to the basis before projection — critical when basis directions are not orthogonal.

Returns `coefficients`, `explained_variance`, and the `orthogonal_residual` norm.

---

## `residual_trajectory`

```python
async def residual_trajectory(
    prompt: str,
    reference_tokens: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict
```

Track how the residual stream rotates through activation space across layers.

At each layer:
- Angle (degrees) to each reference token's unembedding direction
- Rotation from the previous layer (angular velocity)
- Fraction of the residual that is orthogonal to all reference tokens

Use this to see at which layer the residual stream starts pointing toward the correct answer.

---

### Example: factual recall geometry

```python
result = await residual_trajectory(
    prompt="The capital of France is",
    reference_tokens=["Paris", "Lyon", "Berlin"],
    layers=list(range(34)),
    token_position=-1,
)
# Watch "Paris" angle decrease as layers increase
```
