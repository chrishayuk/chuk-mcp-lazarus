# Geometry — Injection

Seven tools for residual stream injection and subspace manipulation.

---

## `inject_residual`

```python
async def inject_residual(
    donor_prompt: str,
    recipient_prompt: str,
    layer: int,
    donor_layer: int | None = None,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    donor_position: int = -1,
    recipient_position: int = -1,
    subspace_only: bool = False,
    subspace_tokens: list[str] | None = None,
) -> dict
```

Capture the donor's residual stream at `donor_layer` (or `layer`) and inject it into the recipient at `layer`, then continue generation.

Tests the **Markov property**: if downstream computation depends only on the current hidden state, the injected output should match the donor's output regardless of the recipient's earlier processing.

`subspace_only=True` + `subspace_tokens` injects only the component of the donor state that lies in the span of the specified token directions (Gram-Schmidt basis).

---

## `branch_and_collapse`

```python
async def branch_and_collapse(
    donor_prompt: str,
    branch_prompts: list[str],
    layer: int,
    donor_layer: int | None = None,
    token_position: int = -1,
    top_k: int = 10,
) -> dict
```

Non-collapsing superposition: inject a donor residual into 2–20 template prompts, evolve each independently through the remaining layers, and collapse to the highest-confidence branch.

Simulates a quantum-like "superposition of interpretations" where the donor state doesn't commit to one outcome until it encounters a contextual template that collapses it.

---

## `subspace_surgery`

```python
async def subspace_surgery(
    recipient_prompt: str,
    layer: int,
    subspace_name: str,
    mode: str,
    donor_prompt: str | None = None,
    donor_layer: int | None = None,
    coordinates: list[float] | None = None,
    lookup_key: str | None = None,
    table_name: str | None = None,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 10,
) -> dict
```

All-position subspace replacement: replace only the subspace component at **all token positions** while preserving the orthogonal complement.

**Modes:**

| mode | Required args | Description |
|------|--------------|-------------|
| `"donor"` | `donor_prompt` | Transplant another prompt's subspace projection |
| `"coordinates"` | `coordinates` | Inject explicit subspace coordinates |
| `"lookup"` | `lookup_key`, `table_name` | Inject precomputed coordinates from a dark table |

**Verification:** `orthogonal_cosine` should be ~1.0 — surgery preserved everything outside the subspace.

---

## `compute_subspace`

```python
async def compute_subspace(
    subspace_name: str,
    layer: int,
    prompts: list[str],
    rank: int = 10,
    token_position: int = -1,
) -> dict
```

Compute a PCA subspace from model activations.  Runs each prompt through the model, collects hidden states, centres them, and computes the top-`rank` principal components via SVD.  Stores the basis in `SubspaceRegistry`.

**Required before:** `inject_residual(subspace_only=True, ...)`, `subspace_surgery`, `build_dark_table`.

---

## `list_subspaces`

```python
async def list_subspaces() -> dict
```

List all named subspaces in `SubspaceRegistry` with names, layers, ranks, and variance explained.

---

## `build_dark_table`

```python
async def build_dark_table(
    table_name: str,
    subspace_name: str,
    layer: int,
    entries: dict[str, str],
    token_position: int = -1,
) -> dict
```

Precompute a dark coordinate lookup table.  For each `{key: prompt}` pair, extracts the hidden state at `layer` and projects onto the named PCA subspace.  Stores the resulting `[rank]`-dimensional coordinate vector.

After building, `subspace_surgery(mode="lookup", lookup_key=..., table_name=...)` can inject any stored entity with **zero extra forward passes**.

---

## `list_dark_tables`

```python
async def list_dark_tables() -> dict
```

List all dark tables in `DarkTableRegistry` with names, subspace associations, entry counts, and timestamps.

---

### Pipeline

```python
# 1. Collect diverse prompts and compute a low-rank subspace
await compute_subspace("entity_subspace", layer=15,
    prompts=["Paris is...", "Berlin is...", "Tokyo is...", ...], rank=8)

# 2. Pre-bake coordinates for known entities
await build_dark_table("capitals", "entity_subspace", layer=15, entries={
    "Paris": "The capital of France is",
    "Berlin": "The capital of Germany is",
})

# 3. Surgery: swap entity subspace, preserving all other directions
await subspace_surgery(
    recipient_prompt="The largest city in Italy is",
    layer=15, subspace_name="entity_subspace",
    mode="lookup", lookup_key="Paris", table_name="capitals",
)
```
