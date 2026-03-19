# Geometry Tools

26 tools for working in the full native activation space (e.g. 2560 dimensions for Gemma 3-4B).

All angles are reported in **degrees**. PCA projections are optional and always flagged as lossy.

## Subgroups

| Subgroup | Tools | Description |
|----------|-------|-------------|
| [angles/](angles/) | `token_space`, `direction_angles`, `subspace_decomposition`, `residual_trajectory` | Angles between directions in activation space |
| [dimensionality/](dimensionality/) | `feature_dimensionality`, `residual_map`, `residual_atlas`, `weight_geometry` | Effective dimensionality and variance structure |
| [navigation/](navigation/) | `decode_residual`, `residual_match`, `computation_map` | Decoding, matching, and mapping the computation flow |
| [injection/](injection/) | `inject_residual`, `branch_and_collapse`, `subspace_surgery`, `compute_subspace`, `list_subspaces`, `build_dark_table`, `list_dark_tables` | Residual stream injection and subspace manipulation |
| [kv/](kv/) | `extract_k_vector`, `extract_q_vector` | Key and Query vectors (QK addressing space) — `kv_vectors.py` |
| [head_output/](head_output/) | `extract_attention_output`, `get_token_embedding` | Head output vectors and token embeddings — `head_output.py` |
| [dla/](dla/) | `compute_dla`, `batch_dla_scan`, `prefill_to_layer`, `kv_inject_test` | Per-head DLA and copy circuit test — `head_dla.py`, `prefill_inject.py` |

## Key Concepts

**Direct Logit Attribution (DLA):** Projects a vector (attention head output, neuron output, etc.) onto a token's unembedding direction. Positive = promotes the token. For heads, raw DLA is exact: sum of all per-head DLA = layer raw DLA.

**Copy circuit hypothesis:** In factual recall, a small number of attention heads (copy heads) dominate the logit attribution for the correct answer token. Their output is effectively 1-dimensional — a scalar coefficient in the answer token's unembedding direction. This means the full KV cache content for a fact can be compressed to 12 bytes: `(token_id: u32, coefficient: f64)`.

**Subspace operations:** `compute_subspace` extracts a low-rank PCA basis from diverse prompt activations and stores it in `SubspaceRegistry`. `inject_residual` and `subspace_surgery` can then operate within that subspace while preserving the orthogonal complement.

**Dark tables:** Precomputed coordinate lookup tables (built via `build_dark_table`) allow `subspace_surgery` in `lookup` mode to inject any stored entity's subspace representation with zero extra forward passes.
