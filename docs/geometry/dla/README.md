# Geometry — DLA and Copy Circuit (`head_dla.py`, `prefill_inject.py`)

Four tools for per-head DLA analysis and the copy circuit injection test.

Source files: `src/chuk_mcp_lazarus/tools/geometry/head_dla.py` (compute_dla, batch_dla_scan)
and `src/chuk_mcp_lazarus/tools/geometry/prefill_inject.py` (prefill_to_layer, kv_inject_test).

All DLA computations use **raw (un-normalised)** Direct Logit Attribution because the output projection (`o_proj`) is linear: the sum of per-head DLA equals the layer's total raw attention DLA.

---

## `compute_dla`

```python
async def compute_dla(
    prompt: str,
    layer: int,
    head: int,
    target_token: str | None = None,
    position: int = -1,
) -> dict
```

Compute DLA for a single `(layer, head)` pair.

Projects the attention head's output vector onto the target token's unembedding direction:
- **Positive DLA** = this head promotes the target token
- **Negative DLA** = this head suppresses the target token

**Returns** `{dla, fraction_of_layer, top_token, head_output_norm}`.

Use `batch_dla_scan` first to identify hot cells, then `compute_dla` to drill into specific heads.

---

## `batch_dla_scan`

```python
async def batch_dla_scan(
    prompt: str,
    target_token: str | None = None,
    layers: list[int] | None = None,
    position: int = -1,
    top_k_cells: int = 5,
) -> dict
```

Scan **all** `(layer, head)` pairs in one forward pass.

Builds the full layers × heads DLA matrix.  Returns `hot_cells` — the sparse set of cells ranked by `|DLA|` — and `copy_circuit` (top 3 cells, the likely copy heads for factual retrieval).

The `layers` result is the heatmap for Layer 1 of copy-circuit analysis.  Click a hot cell to drill into `extract_attention_output`.

**Returns** `{layers: [DlaLayerEntry], hot_cells: [HotCell], summary: {copy_circuit: [...]}}`.

---

## `prefill_to_layer`

```python
async def prefill_to_layer(
    prompt: str,
    layer: int,
    position: int = -1,
    top_k_tokens: int = 5,
) -> dict
```

Run the model forward to a given layer and return the hidden state.

`top_raw_logits` provides a logit-lens view at this depth: what the model would predict if it projected from here without the final layer norm.

Use this to verify that the bare-query residual is **orthogonal** to the answer token before the copy head fires — confirming the copy head is solely responsible for writing the factual signal.

---

## `kv_inject_test`

```python
async def kv_inject_test(
    prompt: str,
    token: str,
    coefficient: float,
    inject_layer: int,
    position: int = -1,
    top_k: int = 10,
) -> dict
```

**The compression test:** Replace full KV attention with a 12-byte signal and compare outputs.

Instead of running the full KV attention mechanism, sets the residual stream's component in the target token's unembedding direction to exactly `coefficient` at `inject_layer`:

```
h[pos] += (coefficient - dot(h[pos], e_unit)) * e_unit
```

Two passes are compared:
- **Full:** normal model inference (all layers, full KV cache)
- **Injected:** forward to `inject_layer`, apply formula, continue to output

**`kl_divergence < 0.001`** = "12 bytes = full KV cache for this fact."

The `summary.interpretation` field confirms the result:
- `"12 bytes = full KV cache for this fact"` — copy circuit is 1-dimensional ✓
- `"KL=X.XXXX — verify copy head and coefficient"` — something is off

---

### Full copy circuit workflow

```python
# Step 1: Find copy heads via DLA heatmap
scan = await batch_dla_scan(
    prompt="The capital of France is",
    target_token="Paris",
)
# scan["hot_cells"][0] = {layer: 15, head: 3, dla: 1.82}

# Step 2: Inspect the copy head's output vector
output = await extract_attention_output(
    prompt="The capital of France is",
    layer=15, head=3, top_k_tokens=5,
)
# output["dimensionality"]["is_one_dimensional"] = True
# output["top_projections"][0] = {"token": "Paris", "coefficient": 1.82}

# Step 3: Verify pre-copy-head residual is orthogonal to answer
prefill = await prefill_to_layer(
    prompt="The capital of France is",
    layer=15,  # just before copy head fires
    top_k_tokens=10,
)
# "Paris" should NOT appear in top_raw_logits — the info is in the KV cache, not the residual

# Step 4: Test 12-byte compression
inject_result = await kv_inject_test(
    prompt="The capital of France is",
    token="Paris",
    coefficient=1.82,          # from extract_attention_output
    inject_layer=16,           # copy_layer + 1
)
# inject_result["summary"]["kl_divergence"] < 0.001 → 12 bytes confirmed
```
