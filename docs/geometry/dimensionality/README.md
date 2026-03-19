# Geometry — Dimensionality

Four tools for measuring how many dimensions are actually used in the residual stream.

---

## `feature_dimensionality`

```python
async def feature_dimensionality(
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int = -1,
    max_dims: int = 50,
) -> dict
```

Estimate the effective dimensionality of a feature at a layer.

Extracts activations for both groups, computes the difference matrix, and runs PCA.  The number of components needed to explain 90% of variance = the effective dimensionality of the feature.

- **1 dimension** = clean direction (copy circuit hypothesis)
- **~50 dimensions** = holographically distributed feature

---

## `residual_map`

```python
async def residual_map(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
    max_components: int = 100,
) -> dict
```

Compact per-layer variance spectrum across the full model.

Runs PCA at each layer and returns effective dimensionality and the variance explained by each component.  No vocabulary projections — just the geometric structure.  Use this first to understand how representation complexity evolves across layers.

`layers=None` auto-selects evenly-spaced layers.

---

## `residual_atlas`

```python
async def residual_atlas(
    prompts: list[str],
    layers: list[int] | int,
    token_position: int = -1,
    max_components: int = 50,
    top_k_tokens: int = 10,
    store_subspace: str | None = None,
) -> dict
```

Map the residual stream structure via PCA on diverse prompt activations, then decode each principal component through the unembedding matrix to see what it represents in vocabulary space.

Each PC is decoded as `{positive_tokens, negative_tokens}` — the tokens it most promotes and suppresses.

`store_subspace="my_subspace"` stores the PCA basis in `SubspaceRegistry` for use with `inject_residual`.

---

## `weight_geometry`

```python
async def weight_geometry(
    layer: int,
    top_k_neurons: int = 100,
    top_k_vocab: int = 5,
    include_pca: bool = True,
) -> dict
```

Map what directions every component **can** push toward — **supply side** (from weights alone, no forward pass).

For each attention head, extracts the output projection push direction.  For each MLP neuron, extracts the `down_proj` column.  Projects each through the unembedding matrix to see what vocabulary tokens it promotes.

`include_pca=True` also runs PCA on all push directions to compute `supply_rank` — how many directions the layer can express.
