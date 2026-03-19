# Residual Stream Analysis

Five tools for decomposing the residual stream: attention vs FFN contributions, layer clustering, logit attribution, per-head attribution, and neuron-level DLA.

---

## `residual_decomposition`

```python
async def residual_decomposition(
    prompt: str,
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict
```

Separate attention vs MLP contribution to the residual stream per layer.

**Returns** per layer: `{attention_norm, ffn_norm, attention_cosine_to_residual, ffn_cosine_to_residual, dominant_component}`.

---

## `layer_clustering`

```python
async def layer_clustering(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict
```

Cluster prompts by representation similarity at each layer.  Returns `peak_separation_layer` and `convergence_layer`.

---

## `logit_attribution`

```python
async def logit_attribution(
    prompt: str,
    target_token: str | None = None,
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict
```

**Direct Logit Attribution (DLA):** per-layer component contributions to the predicted token's logit.

Two modes:
- **Normalized** (default): computes logit deltas through `norm → lm_head`. Attribution sum equals model logit exactly.
- **Raw** (`normalized=False`): projects raw sub-outputs through unembedding without final norm.  Can produce large values on models with embedding scale (e.g. Gemma).

**Returns** per layer: `{attention_logit, ffn_logit, total_logit}` and `summary: {peak_attention_layer, peak_ffn_layer, total_logit}`.

---

## `head_attribution`

```python
async def head_attribution(
    prompt: str,
    layer: int,
    target_token: str | None = None,
    token_position: int = -1,
) -> dict
```

Per-head logit attribution at a specific layer via `o_proj` weight slicing.  Raw DLA; sum of heads = `layer_total_logit`.

**Returns** `heads: [{head, logit_contribution, fraction_of_layer, top_token}]`.

---

## `top_neurons`

```python
async def top_neurons(
    prompt: str,
    layer: int,
    target_token: str | None = None,
    top_k: int = 10,
    token_position: int = -1,
) -> dict
```

Per-neuron MLP identification via SwiGLU decomposition.  Efficient: computes `neuron_projections = down_proj.weight.T @ unembed_vector` then `neuron_logits = hidden * neuron_projections`.

**Returns** `top_positive` and `top_negative` neuron lists with `{neuron_index, activation, logit_contribution, top_token}`.
