# Geometry — KV Space (`kv_vectors.py`)

Two tools for extracting Key and Query vectors in the QK addressing space.

Source file: `src/chuk_mcp_lazarus/tools/geometry/kv_vectors.py`

These vectors live in **head_dim** space (typically 64–128 dimensions), not hidden_dim space.  They are extracted **after** `k_norm`/`q_norm` and **after** RoPE encoding.

---

## `extract_k_vector`

```python
async def extract_k_vector(
    prompt: str,
    layer: int,
    kv_head: int,
    position: int = -1,
) -> dict
```

Extract the Key vector for a specific KV head at a given token position.

The K vector is what the model uses for **attention pattern matching** (addressing) — the "what do I have here?" signal for GQA matching.

For GQA models (e.g. Llama, Gemma 3): `kv_head` is in `[0, num_kv_heads - 1]`.  Multiple query heads share each KV head.

**Use case:** Collect K vectors for the same KV head across many prompts with the same template structure (e.g. `"X agreed to sell his Y"`) to analyze **K-space crowding** — how `W_K` maps distinct entities to potentially overlapping key vectors.

**Returns** `{k_vector: [float, ...], k_norm, num_kv_heads, head_dim}`.

---

## `extract_q_vector`

```python
async def extract_q_vector(
    prompt: str,
    layer: int,
    head: int,
    position: int = -1,
) -> dict
```

Extract the Query vector for a specific head at a given token position.

The Q vector is what the model uses to **search** the K-space — the "what am I looking for?" signal.

`head` is the full query head index (0-indexed, using `num_attention_heads`).  For GQA models, the corresponding KV head is `head // (num_heads // num_kv_heads)`.

**Returns** `{q_vector: [float, ...], q_norm, num_heads, head_dim}`.

---

### Copy circuit analysis workflow

```python
# 1. Find the copy head via batch_dla_scan
scan = await batch_dla_scan(prompt="The capital of France is", target_token="Paris")
copy_layer, copy_head = scan["hot_cells"][0]["layer"], scan["hot_cells"][0]["head"]

# 2. For GQA: compute the KV head index
# kv_head = copy_head // (num_heads // num_kv_heads)

# 3. Extract K vector at the factual token position
k = await extract_k_vector(
    prompt="The capital of France is",
    layer=copy_layer, kv_head=0,
    position=3,   # token position of "France"
)

# 4. Extract Q vector at the answer position
q = await extract_q_vector(
    prompt="The capital of France is",
    layer=copy_layer, head=copy_head,
    position=-1,  # last token (the query for what comes next)
)

# 5. Check Q·K alignment — high dot product = copy head is attending to "France"
import numpy as np
dot = np.dot(k["k_vector"], q["q_vector"])
```
