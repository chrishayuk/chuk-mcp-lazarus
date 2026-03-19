# Geometry — Navigation

Three tools for decoding, matching, and mapping the full computation flow.

---

## `decode_residual`

```python
async def decode_residual(
    prompt: str,
    layers: list[int],
    top_k: int = 20,
    token_position: int = -1,
    include_mean_decode: bool = True,
) -> dict
```

Decode the residual stream into vocabulary space at specified layers.  Three views per layer:

1. **Raw** — dot product with unembedding vectors (before normalization)
2. **Normalized** — after applying the model's actual layer norm
3. **Gap** — rank changes between raw and normalized rankings

The **gap** view reveals tokens whose logit rank is dramatically changed by layer norm — a sign of structured information in the mean direction.

`include_mean_decode=True` also decodes the mean direction (what the common-mode signal "points at").

---

## `residual_match`

```python
async def residual_match(
    target_prompt: str,
    candidate_prompts: list[str],
    layer: int,
    token_position: int = -1,
    subspace_tokens: list[str] | None = None,
) -> dict
```

Find which candidate prompts produce the most similar residual stream to a target at a given layer.

Used to find natural test cases for the Markov property: two prompts that arrive at similar hidden states through different computation paths.

`subspace_tokens` specifies tokens that define a task-relevant subspace — returns both full-space and subspace similarity separately.

---

## `computation_map`

```python
async def computation_map(
    prompt: str,
    candidates: list[str],
    layers: list[int] | None = None,
    top_k_heads: int = 3,
    top_k_neurons: int = 5,
    token_position: int = -1,
) -> dict
```

Map the complete prediction flow in a single call.

Runs one decomposition forward pass and returns, per layer:
- Residual geometry (angles to candidate tokens)
- Component attribution (attention vs FFN logit deltas)
- Logit lens race (candidate probabilities)
- Top `top_k_heads` attention heads by DLA
- Top `top_k_neurons` MLP neurons by DLA

Structured for rendering as a network flow diagram.
