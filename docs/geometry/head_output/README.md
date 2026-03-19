# Geometry — Head Output Space (`head_output.py`)

Two tools for extracting attention head output vectors and token embedding vectors.

These tools work in **hidden_dim** space (the full residual stream dimensionality).

Source file: `src/chuk_mcp_lazarus/tools/geometry/head_output.py`

---

## `extract_attention_output`

```python
async def extract_attention_output(
    prompt: str,
    layer: int,
    head: int,
    position: int = -1,
    top_k_tokens: int = 10,
) -> dict
```

Extract the output vector of a specific attention head in hidden space.

The returned vector is what the head **contributes to the residual stream** — the result of `context @ o_proj_slice`.

Also projects onto the top-N token unembedding directions to reveal the head's content in vocabulary space.

**Dimensionality analysis:** The `dimensionality` field reports `dims_for_95pct`, `dims_for_99pct`, `dims_for_999pct`, `top1_fraction`, `top2_fraction`, and `is_one_dimensional`.

- `is_one_dimensional=True` (`dims_for_99pct <= 2`) supports the **copy circuit hypothesis**: the head's entire output can be expressed as a single scalar coefficient in a single token direction.
- This means 12 bytes = `(token_id: u32, coefficient: f64)` = the full factual content of the copy head.

**Returns**

```json
{
  "vector": [...],
  "vector_norm": 2.14,
  "top_projections": [
    {"token": " Paris", "token_id": 12345, "coefficient": 1.82, "fraction": 0.47}
  ],
  "dimensionality": {
    "dims_for_95pct": 1,
    "dims_for_99pct": 2,
    "is_one_dimensional": true
  }
}
```

---

## `get_token_embedding`

```python
async def get_token_embedding(token: str) -> dict
```

Get the input embedding and unembedding vectors for a token.

- **Unembedding vector** — used for DLA and content projection (column of the lm_head weight matrix)
- **Input embedding** — what the model places in the residual stream when this token appears as input

For most modern models (Gemma, Llama, Mistral) these are **tied** — the same weight matrix is used for both.

Both bare and space-prefixed forms are tried (e.g. `"Paris"` and `" Paris"`); the higher-norm form is used.

**Returns** `{token, token_id, unembedding, input_embedding, embeddings_tied, unembedding_norm, cosine_similarity}`.
