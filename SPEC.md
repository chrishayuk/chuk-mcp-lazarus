# chuk-mcp-lazarus -- Specification

**Mechanistic interpretability MCP server wrapping chuk-lazarus**
Built on `chuk-mcp-server` -- Model-agnostic -- Apple Silicon native via MLX

---

## Overview

`chuk-mcp-lazarus` exposes chuk-lazarus's introspection primitives as
MCP tools so that a reasoning model (Claude) can autonomously design
and execute mechanistic interpretability experiments on any
locally-loaded language model.

The flagship demo is **language transition probing on TranslateGemma
4B**: Claude trains linear probes to discover which layer the model
transitions from source-language to target-language representations,
then performs activation steering to redirect translations to a
different target language -- a live computational surgery
demonstration.

The server is model-agnostic. Any model supported by chuk-lazarus
(Gemma, Llama, Qwen, Granite, Jamba, Mamba, StarCoder2, GPT-2) works
with every tool.

---

## Repository Layout

```
chuk-mcp-lazarus/
├── src/
│   └── chuk_mcp_lazarus/
│       ├── __init__.py              # Package init, imports server + tools + resources
│       ├── server.py                # ChukMCPServer instance
│       ├── main.py                  # Entry point (stdio / http)
│       ├── model_state.py           # Singleton: loaded model + tokenizer
│       ├── probe_store.py           # Singleton: probe registry
│       ├── steering_store.py        # Singleton: steering vector registry
│       ├── resources.py             # MCP resources (model://info, probes, vectors)
│       ├── errors.py                # ToolError enum + error envelope helper (17 error types)
│       ├── _bootstrap.py            # Optional dependency stubs
│       ├── comparison_state.py       # Singleton: comparison model (2nd model)
│       ├── experiment_store.py      # Singleton: experiment persistence
│       ├── _serialize.py            # mx.array / np.ndarray -> JSON-safe
│       ├── _generate.py             # Shared text generation helper
│       ├── _compare.py              # Shared comparison kernels
│       ├── _extraction.py           # Shared activation extraction
│       └── tools/
│           ├── __init__.py
│           ├── model_tools.py         # load_model, get_model_info
│           ├── generation_tools.py    # generate_text, predict_next_token,
│           │                          #   tokenize, logit_lens, track_token,
│           │                          #   track_race, embedding_neighbors
│           ├── activation_tools.py    # extract_activations, compare_activations
│           ├── attention_tools.py     # attention_pattern, attention_heads
│           ├── probe_tools.py         # train_probe, evaluate_probe,
│           │                          #   scan_probe_across_layers,
│           │                          #   probe_at_inference, list_probes
│           ├── steering_tools.py      # compute_steering_vector,
│           │                          #   steer_and_generate, list_steering_vectors
│           ├── ablation_tools.py      # ablate_layers, patch_activations
│           ├── causal_tools.py        # trace_token, full_causal_trace
│           ├── residual_tools.py      # residual_decomposition, layer_clustering,
│           │                          #   logit_attribution, head_attribution, top_neurons
│           ├── attribution_tools.py   # attribution_sweep
│           ├── intervention_tools.py  # component_intervention
│           ├── neuron_tools.py        # discover_neurons, analyze_neuron, neuron_trace
│           ├── direction_tools.py     # extract_direction
│           ├── experiment_tools.py    # create_experiment, add_experiment_result,
│           │                          #   get_experiment, list_experiments
│           ├── comparison_tools.py    # load_comparison_model, compare_weights,
│           │                          #   compare_representations, compare_attention,
│           │                          #   compare_generations, unload_comparison_model
│           └── geometry/              # Geometry tools (per-tool subpackage)
│               ├── _helpers.py            # Shared enums, math, direction extraction
│               ├── token_space.py         # token_space
│               ├── direction_angles.py    # direction_angles
│               ├── subspace_decomposition.py  # subspace_decomposition
│               ├── residual_trajectory.py # residual_trajectory
│               ├── feature_dimensionality.py  # feature_dimensionality
│               ├── computation_map.py     # computation_map
│               ├── inject_residual.py     # inject_residual
│               ├── residual_match.py      # residual_match
│               ├── compute_subspace.py    # compute_subspace, list_subspaces
│               ├── residual_atlas.py      # residual_atlas
│               ├── weight_geometry.py     # weight_geometry
│               └── residual_map.py        # residual_map
├── tests/
├── pyproject.toml
├── ARCHITECTURE.md
├── ROADMAP.md
├── SPEC.md
└── README.md
```

---

## Dependencies

```toml
[project]
name = "chuk-mcp-lazarus"
version = "0.15.0"
requires-python = ">=3.11"

dependencies = [
    "chuk-mcp-server>=0.25",
    "chuk-lazarus>=0.4",
    "scikit-learn>=1.4",
    "numpy>=1.26",
]
```

---

## Server Entry Point

```python
# server.py
from chuk_mcp_server import ChukMCPServer

mcp = ChukMCPServer(
    name="chuk-mcp-lazarus",
    version="0.9.0",
    title="Lazarus Interpretability Server",
    description=(
        "Mechanistic interpretability toolkit. "
        "Load any model, extract activations, train probes, "
        "steer generation, and ablate components to reveal "
        "internal structure."
    ),
)
```

```python
# __init__.py
from .server import mcp
from .tools import *          # noqa: F403 -- triggers @mcp.tool() registration

__version__ = "0.14.0"
__all__ = ["mcp"]
```

Tools register via instance decorators on the shared `mcp` object:

```python
# tools/model_tools.py
from ..server import mcp

@mcp.tool()
async def load_model(model_id: str = "google/gemma-3-4b-it") -> dict:
    ...
```

---

## Tool Catalogue

### Group 1 -- Model Management

#### `load_model`

```python
@mcp.tool(idempotent_hint=True)
async def load_model(
    model_id: str = "google/gemma-3-4b-it",
    dtype: str = "bfloat16",
) -> dict:
    """
    Load a HuggingFace model into memory.
    Must be called before any other tool.

    Works with any model chuk-lazarus supports: Gemma, Llama, Qwen,
    Granite, Jamba, Mamba, StarCoder2, GPT-2, and more.

    Args:
        model_id: HuggingFace model ID or local path.
        dtype:    Weight dtype. One of: bfloat16, float16, float32.

    Returns:
        model_id, num_layers, hidden_dim, num_heads, architecture, status
    """
```

Uses `UnifiedPipeline.from_pretrained()`. Stores result in
`ModelState` singleton. Subsequent calls with the same `model_id`
return immediately (idempotent).

#### `get_model_info`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def get_model_info() -> dict:
    """
    Return architecture metadata for the currently-loaded model.

    Returns:
        model_id, num_layers, hidden_dim, num_attention_heads,
        num_kv_heads, vocab_size, max_position_embeddings,
        architecture, family, is_moe, num_experts
    """
```

---

### Group 2 -- Activation Extraction

#### `extract_activations`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def extract_activations(
    prompt: str,
    layers: list[int],
    token_position: int = -1,
    capture_attention: bool = False,
) -> dict:
    """
    Run a forward pass and return hidden-state activations at the
    specified layers for one token position.

    Args:
        prompt:            Input text.
        layers:            Layer indices (0-indexed).
        token_position:    Which token's activation to return.
                           -1 = last token (default).
        capture_attention: Also return attention weights (memory-heavy).

    Returns:
        prompt, token_position, token_text, num_tokens,
        activations: {layer_idx: [float, ...]}
    """
```

Wraps `ModelHooks` with `CaptureConfig`. Returns hidden states as
plain Python lists.

#### `compare_activations`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def compare_activations(
    prompts: list[str],
    layer: int,
    token_position: int = -1,
) -> dict:
    """
    Extract activations at a single layer for multiple prompts.
    Returns pairwise cosine similarities and a 2-D PCA projection.

    Args:
        prompts:        2-8 input strings.
        layer:          Layer index.
        token_position: Token position (default: last).

    Returns:
        layer, prompts, cosine_similarity_matrix, pca_2d,
        centroid_distance
    """
```

---

### Group 3 -- Probing

#### `train_probe`

```python
@mcp.tool()
async def train_probe(
    probe_name: str,
    layer: int,
    examples: list[dict],
    probe_type: str = "linear",
    token_position: int = -1,
) -> dict:
    """
    Train a probe classifier on activations at the specified layer.

    Args:
        probe_name:     Unique name for this probe.
        layer:          Layer to extract activations from.
        examples:       [{"prompt": str, "label": str}, ...]
        probe_type:     "linear" (LogisticRegression) or "mlp".
        token_position: Token position (default: last).

    Returns:
        probe_name, layer, probe_type, num_examples, classes,
        train_accuracy, val_accuracy, coefficients_norm
    """
```

#### `evaluate_probe`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def evaluate_probe(
    probe_name: str,
    examples: list[dict],
    token_position: int = -1,
) -> dict:
    """
    Evaluate a stored probe on new examples.

    Returns:
        probe_name, accuracy, per_class_accuracy, confusion_matrix,
        predictions: [{prompt, true_label, predicted_label, confidence}]
    """
```

#### `scan_probe_across_layers`

```python
@mcp.tool()
async def scan_probe_across_layers(
    probe_name_prefix: str,
    layers: list[int],
    examples: list[dict],
    probe_type: str = "linear",
    token_position: int = -1,
) -> dict:
    """
    Train and evaluate a probe at every specified layer in one call.
    Primary tool for finding the crossover layer in language
    transition experiments.

    Creates probes named "{probe_name_prefix}_L{layer}" for each layer.

    Args:
        probe_name_prefix: Base name; layer suffix appended.
        layers:            Layers to scan.
        examples:          [{"prompt": str, "label": str}, ...]
        probe_type:        "linear" or "mlp".
        token_position:    Token position (default: last).

    Returns:
        probe_name_prefix, layers_scanned,
        accuracy_by_layer: {layer: {train_accuracy, val_accuracy}},
        peak_layer, peak_val_accuracy, crossover_layer, interpretation
    """
```

Implementation caches activations: each example runs one forward pass
through all layers, then probes are trained from the cache. For a
34-layer model with 40 examples this avoids 34x redundant forward
passes.

#### `list_probes`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_probes() -> dict:
    """
    List all probes in memory.

    Returns:
        probes: [{name, layer, classes, probe_type, val_accuracy, trained_at}],
        count
    """
```

---

### Group 4 -- Steering

#### `compute_steering_vector`

```python
@mcp.tool()
async def compute_steering_vector(
    vector_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int = -1,
) -> dict:
    """
    Compute a steering vector as the mean difference between two sets
    of activations (contrastive activation addition).

    Args:
        vector_name:       Name to store this vector under.
        layer:             Layer to compute the vector at.
        positive_prompts:  Target direction (e.g. German sentences).
        negative_prompts:  Source direction (e.g. English sentences).
        token_position:    Token position (default: last).

    Returns:
        vector_name, layer, vector_norm, cosine_similarity_within_positive,
        cosine_similarity_within_negative, separability_score,
        num_positive, num_negative
    """
```

#### `steer_and_generate`

```python
@mcp.tool()
async def steer_and_generate(
    prompt: str,
    vector_name: str,
    alpha: float = 20.0,
    max_new_tokens: int = 100,
) -> dict:
    """
    Generate text with a steering vector applied to the residual stream.

    Runs both a steered and baseline generation for comparison.

    Args:
        prompt:         Input prompt.
        vector_name:    Name of a stored steering vector.
        alpha:          Steering strength (start at 10-20).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        prompt, vector_name, alpha, steered_output, baseline_output,
        steered_tokens, baseline_tokens
    """
```

#### `list_steering_vectors`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_steering_vectors() -> dict:
    """
    List all steering vectors in memory.

    Returns:
        vectors: [{name, layer, vector_norm, separability_score, computed_at}],
        count
    """
```

---

### Group 5 -- Ablation

#### `ablate_layers`

```python
@mcp.tool()
async def ablate_layers(
    prompt: str,
    layers: list[int],
    max_new_tokens: int = 50,
    ablation_type: str = "zero",
) -> dict:
    """
    Generate text with specific layers zeroed out or mean-ablated.

    Args:
        prompt:         Input text.
        layers:         Layer indices to ablate.
        max_new_tokens: Tokens to generate.
        ablation_type:  "zero" or "mean".

    Returns:
        prompt, ablated_layers, ablated_output, baseline_output,
        output_similarity, disruption_score
    """
```

#### `patch_activations`

```python
@mcp.tool()
async def patch_activations(
    source_prompt: str,
    target_prompt: str,
    layer: int,
    max_new_tokens: int = 50,
) -> dict:
    """
    Activation patching: run target_prompt but replace the hidden state
    at a layer with the hidden state from source_prompt.

    Args:
        source_prompt:  Prompt whose activations we borrow.
        target_prompt:  Prompt we generate for.
        layer:          Layer at which to patch.
        max_new_tokens: Tokens to generate.

    Returns:
        source_prompt, target_prompt, patched_layer,
        patched_output, baseline_output, disruption_score
    """
```

---

### Group 6 -- Generation & Prediction

#### `generate_text`

```python
@mcp.tool()
async def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> dict:
    """
    Generate text from the loaded model.

    Uses greedy decoding when temperature is 0, otherwise samples.
    This is the basic "what does the model say?" tool -- essential for
    seeing actual model outputs before and after interpretability work.

    Args:
        prompt:         Input text.
        max_new_tokens: Maximum tokens to generate (default: 100).
        temperature:    Sampling temperature. 0 = greedy (default).

    Returns:
        prompt, output, num_tokens, temperature, max_new_tokens
    """
```

#### `predict_next_token`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def predict_next_token(
    prompt: str,
    top_k: int = 10,
) -> dict:
    """
    Get the model's top-k next-token predictions with probabilities.

    Shows what the model thinks comes next without generating text.
    Useful for understanding model confidence and comparing how
    different prompts affect the prediction distribution.

    Args:
        prompt: Input text.
        top_k:  Number of top predictions to return (default: 10).

    Returns:
        prompt, num_input_tokens,
        predictions: [{token, token_id, probability, log_probability}]
    """
```

#### `tokenize`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def tokenize(
    text: str,
) -> dict:
    """
    Show how text is tokenized by the loaded model's tokenizer.

    Essential for understanding attention patterns (which token is being
    attended to) and debugging multi-token words. Returns token IDs,
    decoded text for each token, and positions.

    Args:
        text: Input text to tokenize.

    Returns:
        text, num_tokens,
        tokens: [{position, token_id, token_text}],
        token_ids
    """
```

#### `logit_lens`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def logit_lens(
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 5,
    token_position: int = -1,
) -> dict:
    """
    Apply the "logit lens" technique: project hidden states from each
    layer back to vocabulary space to see how the model's prediction
    evolves through its depth.

    Shows what the model would predict if computation stopped at each
    layer. Critical for understanding when specific knowledge emerges
    (e.g., "at which layer does the model start predicting the correct
    translation token?").

    Args:
        prompt:         Input text.
        layers:         Layer indices to analyze. None = sample ~10 layers.
        top_k:          Number of top predictions per layer (default: 5).
        token_position: Which token to analyze (-1 = last, default).

    Returns:
        prompt, token_position, token_text, num_layers_analyzed,
        predictions: [{layer, top_tokens, top_probabilities, top_token_ids}],
        summary: {final_prediction, emergence_layer, total_layers}
    """
```

Uses `ModelHooks.get_layer_logits()` to project each layer's hidden
state through the model's unembedding matrix. The `summary` field
reports where the final-layer prediction first emerges.

#### `track_token`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def track_token(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
) -> dict:
    """
    Track a specific token's probability and rank across all layers
    using the logit lens technique.

    Args:
        prompt: Input text.
        token:  Target token to track.
        layers: Layer indices. None = all layers.

    Returns:
        prompt, target_token, target_token_id, emergence_layer,
        peak_layer, peak_probability,
        layers: [{layer, probability, rank, is_top1}]
    """
```

#### `embedding_neighbors`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def embedding_neighbors(
    token: str,
    top_k: int = 10,
) -> dict:
    """
    Find nearest tokens in embedding space by cosine similarity.
    No forward pass needed -- operates directly on the embedding matrix.

    Args:
        token: Target token (tries bare and space-prefixed forms).
        top_k: Number of neighbors to return (1-100, default: 10).

    Returns:
        query_token, query_token_id, resolved_form, embedding_dim,
        vocab_size, top_k, self_similarity,
        neighbors: [{token, token_id, cosine_similarity}]
    """
```

---

### Group 7 -- Attention Analysis

#### `attention_pattern`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def attention_pattern(
    prompt: str,
    layers: list[int],
    token_position: int = -1,
    top_k: int = 5,
) -> dict:
    """
    Extract per-head attention weights at specified layers showing
    which tokens each head attends to.

    Args:
        prompt:         Input text.
        layers:         Layer indices.
        token_position: Token to analyze (-1 = last).
        top_k:          Top attended tokens per head (1-100).

    Returns:
        prompt, tokens, token_position, token_text,
        patterns: [{layer, num_heads, heads: [{head, top_attended}]}]
    """
```

#### `attention_heads`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def attention_heads(
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 3,
) -> dict:
    """
    Compute per-head entropy and focus for every head across layers.
    Low entropy = focused head; high entropy = diffuse head.

    Args:
        prompt: Input text.
        layers: Layer indices. None = all layers.
        top_k:  Top attended positions per head (1-100).

    Returns:
        prompt, num_layers, total_heads,
        heads: [{layer, head, entropy, max_attention, top_attended_positions}],
        summary: {most_focused_heads, most_diffuse_heads}
    """
```

---

### Group 8 -- Causal Tracing

#### `trace_token`

```python
@mcp.tool(read_only_hint=True)
async def trace_token(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
    effect_threshold: float = 0.05,
) -> dict:
    """
    Which layers are causally responsible for predicting a specific token?
    Ablates each layer and measures the effect on the target token's probability.

    Args:
        prompt:           Input text.
        token:            Target token.
        layers:           Layer indices. None = all layers.
        effect_threshold: Minimum effect to be considered critical.

    Returns:
        prompt, target_token, baseline_prob, peak_layer, peak_effect,
        critical_layers,
        layer_effects: [{layer, effect, ablated_prob}]
    """
```

#### `full_causal_trace`

```python
@mcp.tool(read_only_hint=True)
async def full_causal_trace(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
) -> dict:
    """
    Position × layer causal heatmap (Meng et al. style).
    Tests every (position, layer) combination to find the full circuit.

    Args:
        prompt:    Input text.
        token:     Target token.
        layers:    Layer indices. None = sample layers.
        positions: Token positions. None = all positions.

    Returns:
        prompt, target_token, num_layers, num_positions, tokens,
        heatmap: [{position, layer, effect}],
        peak: {position, layer, effect}
    """
```

---

### Group 9 -- Residual Stream Analysis

#### `residual_decomposition`

```python
@mcp.tool(read_only_hint=True)
async def residual_decomposition(
    prompt: str,
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Separate attention vs MLP contribution to the residual stream per layer.

    Returns:
        prompt, token_position, token_text,
        layers: [{layer, attention_norm, ffn_norm, attention_cosine_to_residual,
                  ffn_cosine_to_residual, dominant_component}],
        summary: {attention_dominant_count, ffn_dominant_count}
    """
```

#### `layer_clustering`

```python
@mcp.tool(read_only_hint=True)
async def layer_clustering(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Cluster prompts by representation similarity at each layer,
    measuring separability and convergence.

    Returns:
        prompts, token_position,
        layers: [{layer, cosine_similarity_matrix, centroid_distance,
                  cluster_separation}],
        summary: {peak_separation_layer, convergence_layer}
    """
```

#### `logit_attribution`

```python
@mcp.tool(read_only_hint=True)
async def logit_attribution(
    prompt: str,
    target_token: str | None = None,
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Direct logit attribution: per-layer component contributions
    to the predicted token's logit.

    Args:
        prompt:       Input text.
        target_token: Token to attribute. None = top predicted token.
        layers:       Layer indices. None = all layers.
        token_position: Token position (-1 = last).

    Returns:
        prompt, target_token, target_token_id, token_position, token_text,
        layers: [{layer, attention_logit, ffn_logit, total_logit}],
        summary: {peak_attention_layer, peak_ffn_layer, total_logit}
    """
```

#### `head_attribution`

```python
@mcp.tool(read_only_hint=True)
async def head_attribution(
    prompt: str,
    layer: int,
    target_token: str | None = None,
    token_position: int = -1,
) -> dict:
    """
    Per-head logit attribution at a specific layer: which attention heads
    push toward the target token.

    Args:
        prompt:       Input text.
        layer:        Layer to decompose.
        target_token: Token to attribute. None = top predicted token.
        token_position: Token position (-1 = last).

    Returns:
        prompt, layer, token_position, token_text, target_token, target_token_id,
        num_heads,
        heads: [{head, logit_contribution, fraction_of_layer, top_token}],
        layer_total_logit,
        summary: {top_positive_head, top_negative_head, concentration}
    """
```

#### `top_neurons`

```python
@mcp.tool(read_only_hint=True)
async def top_neurons(
    prompt: str,
    layer: int,
    target_token: str | None = None,
    top_k: int = 10,
    token_position: int = -1,
) -> dict:
    """
    Per-neuron MLP identification: which neurons push toward the target token.

    Args:
        prompt:       Input text.
        layer:        Layer to analyze.
        target_token: Token to attribute. None = top predicted token.
        top_k:        Number of top neurons to return (default: 10).
        token_position: Token position (-1 = last).

    Returns:
        prompt, layer, token_position, token_text, target_token, target_token_id,
        mlp_type, intermediate_size, top_k,
        top_positive: [{neuron_index, activation, logit_contribution, top_token}],
        top_negative: [{neuron_index, activation, logit_contribution, top_token}],
        total_neuron_logit,
        summary: {concentration, sparsity, top_neuron_share}
    """
```

---

### Group 10 -- Model Comparison

#### `load_comparison_model`

```python
@mcp.tool()
async def load_comparison_model(
    model_id: str,
    dtype: str = "bfloat16",
) -> dict:
    """
    Load a second model for two-model comparison analysis.
    Models must share the same architecture (num_layers, hidden_dim).

    Returns:
        model_id, family, architecture, num_layers, hidden_dim,
        num_attention_heads, parameter_count, status
    """
```

#### `compare_weights`

```python
@mcp.tool(read_only_hint=True)
async def compare_weights(
    layers: list[int] | None = None,
) -> dict:
    """
    Compare weight matrices between primary and comparison models.
    Returns per-layer, per-component Frobenius norm and cosine similarity.
    No inference needed -- cheapest comparison operation.

    Returns:
        primary_model, comparison_model, num_layers_compared,
        divergences, summary
    """
```

#### `compare_representations`

```python
@mcp.tool(read_only_hint=True)
async def compare_representations(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Compare hidden-state activations between primary and comparison
    models for the same prompts. Shows where representations diverge.

    Returns:
        primary_model, comparison_model, num_prompts,
        num_layers_compared, divergences, layer_averages
    """
```

#### `compare_attention`

```python
@mcp.tool(read_only_hint=True)
async def compare_attention(
    prompt: str,
    layers: list[int] | None = None,
) -> dict:
    """
    Compare attention patterns between primary and comparison models.
    Returns per-head Jensen-Shannon divergence and cosine similarity.

    Returns:
        primary_model, comparison_model, prompt,
        num_layers_compared, divergences, top_divergent_heads
    """
```

#### `compare_generations`

```python
@mcp.tool()
async def compare_generations(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> dict:
    """
    Generate text from both models using the same prompt and compare
    the outputs side-by-side.

    Essential for seeing *what actually changes* in model output after
    fine-tuning, before diving into internal representations.

    Args:
        prompt:         Input text.
        max_new_tokens: Maximum tokens to generate (default: 100).
        temperature:    Sampling temperature. 0 = greedy (default).

    Returns:
        prompt, primary_model, comparison_model,
        primary_output, comparison_output,
        primary_tokens, comparison_tokens
    """
```

#### `unload_comparison_model`

```python
@mcp.tool()
async def unload_comparison_model() -> dict:
    """
    Unload the comparison model and free VRAM.

    Returns:
        model_id, status
    """
```

---

### Group 11 -- Attribution

#### `attribution_sweep`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def attribution_sweep(
    prompts: list[str],
    layers: list[int] | None = None,
    position: int = -1,
    target_token: str | None = None,
    normalized: bool = True,
    labels: list[str] | None = None,
) -> dict:
    """
    Batch logit attribution across multiple prompts. Returns per-layer
    summary statistics, per-prompt summary rows, and dominant
    component analysis.

    Args:
        prompts:      2-50 input texts.
        layers:       Layer indices. None = all layers.
        position:     Token position (default: last).
        target_token: Token to attribute. None = top predicted per prompt.
        normalized:   Use calibrated norm→lm_head projection (default: True).
        labels:       Optional labels for each prompt (must match length).

    Returns:
        num_prompts, num_layers, dominant_layer, dominant_component,
        layer_summary: [{layer, mean_attention_logit, mean_ffn_logit, ...}],
        prompt_summary: [{prompt, label, embedding_logit, net_attention,
                         net_ffn, final_logit, top_prediction, probability,
                         dominant_component, top_positive_layer, top_negative_layer}],
        per_prompt: [full per-prompt attribution results]
    """
```

---

### Group 12 -- Intervention

#### `component_intervention`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def component_intervention(
    prompt: str,
    layer: int,
    component: str,
    intervention: str = "zero",
    scale_factor: float = 0.0,
    head: int | None = None,
    top_k: int = 10,
    token_position: int = -1,
) -> dict:
    """
    Surgical causal intervention: zero/scale a specific component
    (attention, FFN, or individual head) at a layer and compare
    predictions with the clean forward pass.

    Args:
        prompt:       Input text.
        layer:        Layer to intervene at.
        component:    "attention", "ffn", or "head".
        intervention: "zero" or "scale".
        scale_factor: Scale factor (0.0 for zero, any float for scale).
        head:         Head index (required when component="head").
        top_k:        Number of top predictions to compare (1-50).
        token_position: Token position (-1 = last).

    Returns:
        prompt, layer, component, intervention, scale_factor, head,
        token_position, token_text,
        original_top_k, intervened_top_k,
        kl_divergence, target_delta, original_top1, intervened_top1,
        top1_changed, summary
    """
```

---

### Group 13 -- Neuron Analysis

#### `discover_neurons`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def discover_neurons(
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    top_k: int = 10,
    token_position: int = -1,
) -> dict:
    """
    Auto-find neurons that discriminate between two prompt groups
    using Cohen's d effect size.

    Returns:
        layer, top_k, num_positive, num_negative,
        top_neurons: [{neuron_idx, cohens_d, positive_mean, negative_mean}]
    """
```

#### `analyze_neuron`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def analyze_neuron(
    layer: int,
    neuron_indices: list[int],
    prompts: list[str],
    token_position: int = -1,
    detailed: bool = False,
) -> dict:
    """
    Profile specific neurons: activation statistics across prompts.

    Returns:
        layer, num_prompts,
        neurons: [{neuron_idx, min_val, max_val, mean_val, std_val}],
        per_prompt_activations (when detailed=True)
    """
```

#### `neuron_trace`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def neuron_trace(
    prompt: str,
    layer: int,
    neuron_index: int,
    target_layers: list[int] | None = None,
    token_position: int = -1,
    top_k_heads: int = 5,
) -> dict:
    """
    Trace a neuron's output direction through downstream layers.
    Computes cosine similarity between the neuron's output vector
    and the residual stream, attention output, and FFN output at
    each target layer.

    Args:
        prompt:        Input text.
        layer:         Source layer containing the neuron.
        neuron_index:  Neuron index in the MLP intermediate space.
        target_layers: Layers to trace through. None = next 9 layers.
        token_position: Token position (-1 = last).
        top_k_heads:   Top aligned heads to return per layer (1-64).

    Returns:
        prompt, token_position, token_text,
        neuron: {layer, neuron_index, activation, output_direction_norm, top_token},
        num_trace_layers,
        trace: [{layer, residual_alignment, attention_alignment,
                 ffn_alignment, residual_projection, top_aligned_heads}],
        summary
    """
```

---

### Group 14 -- Generation (Extended)

#### `track_race`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def track_race(
    prompt: str,
    candidates: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """
    Race multiple candidate tokens across layers. Shows how each
    candidate's probability evolves and detects where the leader
    changes (crossing events).

    Args:
        prompt:         Input text.
        candidates:     2-20 candidate tokens to race.
        layers:         Layer indices. None = auto-sample ~12 layers.
        token_position: Token position (-1 = last).

    Returns:
        prompt, token_position, token_text, num_candidates,
        candidates: [{token, token_id, layers: [{layer, probability, rank, is_top1}],
                      emergence_layer, peak_layer, peak_probability}],
        crossings: [{layer, previous_leader, new_leader, new_leader_probability}],
        final_winner, final_probability
    """
```

#### `probe_at_inference`

```python
@mcp.tool(read_only_hint=True)
async def probe_at_inference(
    prompt: str,
    probe_name: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    token_position: int = -1,
) -> dict:
    """
    Run a trained probe during autoregressive generation to monitor
    the model's internal state token-by-token.

    Args:
        prompt:         Input text.
        probe_name:     Name of a stored probe.
        max_tokens:     Maximum tokens to generate (1-500).
        temperature:    Sampling temperature. 0 = greedy.
        token_position: Token position for probe extraction (-1 = last).

    Returns:
        prompt, probe_name, probe_layer, generated_text, tokens_generated,
        per_token: [{step, token, token_id, probe_prediction,
                     probe_confidence, probe_probabilities}],
        overall_majority_class, overall_mean_confidence, class_distribution
    """
```

---

### Group 15 -- Direction Extraction

#### `extract_direction`

```python
@mcp.tool()
async def extract_direction(
    direction_name: str,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    method: str = "mean_diff",
    token_position: int = -1,
) -> dict:
    """
    Extract an interpretable direction in activation space.
    Stored in SteeringVectorRegistry for use with steer_and_generate.

    Args:
        direction_name:    Name for this direction.
        layer:             Layer to extract at.
        positive_prompts:  Target direction prompts.
        negative_prompts:  Source direction prompts.
        method:            "mean_diff", "lda", "pca", or "probe".
        token_position:    Token position (-1 = last).

    Returns:
        direction_name, layer, method, vector_norm,
        separation_score, classification_accuracy
    """
```

---

### Group 16 -- Experiment Persistence

#### `create_experiment`

```python
@mcp.tool()
async def create_experiment(
    name: str,
    description: str = "",
    metadata: dict | None = None,
) -> dict:
    """Create a named experiment for result persistence."""
```

#### `add_experiment_result`

```python
@mcp.tool()
async def add_experiment_result(
    experiment_id: str,
    step_name: str,
    result: dict,
    notes: str = "",
) -> dict:
    """Add a step result to an existing experiment."""
```

#### `get_experiment`

```python
@mcp.tool(read_only_hint=True)
async def get_experiment(
    experiment_id: str,
) -> dict:
    """Retrieve an experiment and all its results."""
```

#### `list_experiments`

```python
@mcp.tool(read_only_hint=True)
async def list_experiments() -> dict:
    """List all saved experiments."""
```

---

### Group 17 -- Geometry

Angles, projections, and dimensionality analysis in the full native
activation space. All angles reported in degrees. PCA projections are
optional and flagged as lossy.

#### `token_space`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def token_space(
    prompt: str,
    layer: int,
    tokens: list[str],
    token_position: int = -1,
    include_projection: bool = False,
) -> dict:
    """Map geometric relationships between token directions and the residual
    stream at a specific layer. Works in the full native dimensionality.

    Reports angles (degrees) and projections between each token's unembedding
    vector and the residual stream, plus pairwise angles between all tokens.

    Args:
        prompt:             Input text.
        layer:              Layer to analyse.
        tokens:             Tokens to map (e.g. ["Sydney", "Canberra"]).
        token_position:     Which token position to use (-1 = last).
        include_projection: Include lossy 2D PCA projection (default: False).
    """
```

#### `direction_angles`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def direction_angles(
    prompt: str,
    layer: int,
    directions: list[dict],
    token_position: int = -1,
) -> dict:
    """Compute pairwise angles between arbitrary directions in activation space.

    Each direction is a dict with 'type' (token, neuron, residual, ffn_output,
    attention_output, head_output, steering_vector) and optional 'value'.

    Args:
        prompt:         Input text.
        layer:          Layer to analyse.
        directions:     List of direction specs.
        token_position: Token position (-1 = last).
    """
```

#### `subspace_decomposition`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def subspace_decomposition(
    prompt: str,
    layer: int,
    target: dict,
    basis_directions: list[dict],
    token_position: int = -1,
    orthogonalize: bool = True,
) -> dict:
    """Decompose a target vector into components along basis directions
    plus an orthogonal residual.

    Args:
        prompt:            Input text.
        layer:             Layer to analyse.
        target:            Target direction spec.
        basis_directions:  List of basis direction specs.
        token_position:    Token position (-1 = last).
        orthogonalize:     Apply Gram-Schmidt to basis (default: True).
    """
```

#### `residual_trajectory`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def residual_trajectory(
    prompt: str,
    reference_tokens: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict:
    """Track how the residual stream moves through activation space across
    layers, measured by angles to reference token directions.

    At each layer, reports the angle to each reference token, the rotation
    from the previous layer, and what fraction of the residual is orthogonal
    to all reference tokens.

    Args:
        prompt:           Input text.
        reference_tokens: Tokens to measure against (e.g. ["Sydney", "Canberra"]).
        layers:           Layer indices (None = all layers).
        token_position:   Token position (-1 = last).
    """
```

#### `feature_dimensionality`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def feature_dimensionality(
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    token_position: int = -1,
    max_dims: int = 50,
) -> dict:
    """Estimate the effective dimensionality of a feature at a layer.

    Extracts activations for positive/negative prompt groups and computes
    principal components of the difference. Reports how many dimensions
    are needed to capture the feature.

    1 dimension = clean direction. 50 dimensions = holographically distributed.

    Args:
        layer:             Layer to analyse.
        positive_prompts:  Prompts representing the positive class (min 2).
        negative_prompts:  Prompts representing the negative class (min 2).
        token_position:    Token position (-1 = last).
        max_dims:          Maximum dimensions to analyse (default: 50).
    """
```

#### `computation_map`

```python
@mcp.tool(read_only_hint=True)
async def computation_map(
    prompt: str,
    candidates: list[str],
    layers: list[int] | None = None,
    top_k_heads: int = 3,
    top_k_neurons: int = 5,
    token_position: int = -1,
) -> dict:
    """Map the complete prediction flow for a prompt in one call.

    Runs a single decomposition forward pass and returns per-layer:
    residual geometry (angles to candidates), component attribution
    (attention vs FFN logit deltas), logit lens race (probabilities),
    top attention heads, and top MLP neurons.

    Structured for rendering as a network flow diagram.

    Args:
        prompt:         Input text.
        candidates:     Candidate tokens to track (e.g. ["Sydney", "Canberra"]).
        layers:         Layer indices to analyse (None = auto-select).
        top_k_heads:    Top attention heads per layer (default: 3).
        top_k_neurons:  Top MLP neurons per layer (default: 5).
        token_position: Token position to analyse (-1 = last).
    """
```

#### `decode_residual`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def decode_residual(
    prompt: str,
    layers: list[int],
    top_k: int = 20,
    token_position: int = -1,
    include_mean_decode: bool = True,
) -> dict:
    """Decode the residual stream into vocabulary space at specified
    layers, showing both raw and normalised nearest neighbours.

    For each layer, returns three views of the same residual stream:
    1. Raw — dot product with unembedding vectors (before normalisation).
    2. Normalised — after applying the model's actual layer norm.
    3. Gap — rank changes between raw and normalised rankings.

    Optionally decodes the mean direction itself — what the subtracted
    common-mode signal "points at" in vocabulary space.

    Args:
        prompt:              Input text.
        layers:              Layer indices to decode at.
        top_k:               Number of top words per view (1-100).
        token_position:      Token position (-1 = last).
        include_mean_decode: Decode the mean direction (default: True).
    """
```

#### `inject_residual`

```python
@mcp.tool()
async def inject_residual(
    donor_prompt: str,
    recipient_prompt: str,
    layer: int,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    donor_position: int = -1,
    recipient_position: int = -1,
    subspace_only: bool = False,
    subspace_tokens: list[str] | None = None,
) -> dict:
    """Inject the donor's residual stream into the recipient at a
    specific layer and continue generation.

    Tests the Markov property: if downstream layers depend only on
    the current residual state, the injected output should match the
    donor's output regardless of the recipient's earlier processing.

    Args:
        donor_prompt:      Prompt whose residual stream to inject.
        recipient_prompt:  Prompt to receive the injection.
        layer:             Layer at which to inject (0 to num_layers-1).
        max_new_tokens:    Tokens to generate after injection (1-500).
        temperature:       Sampling temperature (0 = greedy).
        donor_position:    Token position in donor (-1 = last).
        recipient_position: Token position in recipient (-1 = last).
        subspace_only:     Only inject the subspace component.
        subspace_tokens:   Tokens defining the injection subspace.
    """
```

#### `residual_match`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def residual_match(
    target_prompt: str,
    candidate_prompts: list[str],
    layer: int,
    token_position: int = -1,
    subspace_tokens: list[str] | None = None,
) -> dict:
    """Find which candidate prompts produce the most similar residual
    stream to the target at a given layer.

    Useful for finding natural test cases for the Markov property:
    two prompts that arrive at similar states through different paths.

    Args:
        target_prompt:     Reference prompt.
        candidate_prompts: 1-20 prompts to compare against.
        layer:             Layer to compare at.
        token_position:    Token position (-1 = last).
        subspace_tokens:   Optional tokens defining a task-relevant
                           subspace for separate similarity reporting.
    """
```

#### `compute_subspace`

```python
@mcp.tool()
async def compute_subspace(
    subspace_name: str,
    layer: int,
    prompts: list[str],
    rank: int = 10,
    token_position: int = -1,
) -> dict:
    """Compute a PCA subspace from model activations and store it in
    the SubspaceRegistry for use with inject_residual.

    Runs each prompt through the model at the given layer, collects
    hidden-state vectors at token_position, centres them, and computes
    the top-rank principal components via SVD.

    Args:
        subspace_name:  Name to store this subspace under.
        layer:          Layer to extract activations at.
        prompts:        Varied prompts (min 3, max 500).
        rank:           Number of PCA components to retain (1-100).
        token_position: Token position (-1 = last).
    """
```

#### `list_subspaces`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def list_subspaces() -> dict:
    """List all named subspaces stored in the SubspaceRegistry.

    Returns names, layers, ranks, and variance explained for each
    stored subspace.
    """
```

#### `residual_atlas`

```python
@mcp.tool(read_only_hint=True)
async def residual_atlas(
    prompts: list[str],
    layers: list[int] | int,
    token_position: int = -1,
    max_components: int = 50,
    top_k_tokens: int = 10,
    store_subspace: str | None = None,
) -> dict:
    """Map the residual stream structure via PCA on diverse prompt activations.

    Collects hidden-state vectors from many prompts at specified layers,
    runs PCA to find the dominant directions of variation, and projects
    each principal component through the unembedding matrix to decode
    what it represents in vocabulary space.

    Args:
        prompts:          Diverse prompts to extract activations from (10-2000).
        layers:           Layer index or list of layer indices.
        token_position:   Token position (-1 = last).
        max_components:   PCA components to analyse in detail (1-200).
        top_k_tokens:     Vocabulary tokens to decode per component (1-50).
        store_subspace:   If provided, store PCA basis in SubspaceRegistry.
    """
```

#### `weight_geometry`

```python
@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def weight_geometry(
    layer: int,
    top_k_neurons: int = 100,
    top_k_vocab: int = 5,
    include_pca: bool = True,
) -> dict:
    """Map what directions every component can push toward (supply side).

    Extracts each attention head's output-projection push direction and
    the top-k MLP neuron down_proj columns, projects them through
    the unembedding matrix, and optionally runs PCA on all directions.

    No forward pass required — purely from weights.

    Args:
        layer:          Layer index to analyse.
        top_k_neurons:  Number of MLP neurons to analyse (by L2 norm, 1-500).
        top_k_vocab:    Vocabulary tokens to decode per direction (1-20).
        include_pca:    Run PCA on all push directions for supply rank.
    """
```

#### `residual_map`

```python
@mcp.tool(read_only_hint=True)
async def residual_map(
    prompts: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
    max_components: int = 100,
) -> dict:
    """Compact per-layer variance spectrum across the full model.

    Runs PCA at each layer and returns effective dimensionality and
    variance spectrum — no vocabulary projections. Use this to see
    how representation structure evolves across layers.

    Args:
        prompts:          Diverse prompts to extract activations from (10-2000).
        layers:           Layer indices (None = auto-select evenly-spaced layers).
        token_position:   Token position (-1 = last).
        max_components:   PCA components to compute (1-200).
    """
```

---

### Group 8 -- Resources

```python
@mcp.resource("model://info", mime_type="application/json")
async def model_info_resource() -> dict:
    """Current model metadata."""

@mcp.resource("probes://registry", mime_type="application/json")
async def probes_resource() -> dict:
    """All trained probes and their accuracy metrics."""

@mcp.resource("vectors://registry", mime_type="application/json")
async def vectors_resource() -> dict:
    """All computed steering vectors."""

@mcp.resource("comparisons://state", mime_type="application/json")
def comparison_state_resource() -> dict:
    """Current comparison model state."""
```

---

## State Management

Four in-memory singletons, each with a `threading.Lock`:

| Singleton | Contents | Thread-Safe |
|-----------|----------|-------------|
| `ModelState` | MLX model, tokenizer, config, family info | Yes |
| `ProbeRegistry` | `{name: (sklearn_model, ProbeMetadata)}` | Yes |
| `SteeringVectorRegistry` | `{name: (np.ndarray, VectorMetadata)}` | Yes |
| `ComparisonState` | Second MLX model for comparison | Yes |

All state is lost when the server process restarts. Persistence is a
v1.0 concern.

---

## Error Handling

Every tool returns a consistent error envelope on failure:

```python
{
    "error": True,
    "error_type": "ModelNotLoaded",
    "message": "Call load_model() first.",
    "tool": "extract_activations"
}
```

Error types are defined in `errors.py` as a `ToolError` enum (16 types):

- `ModelNotLoaded` -- no model loaded
- `LayerOutOfRange` -- requested layer exceeds model depth
- `ProbeNotFound` -- named probe does not exist
- `VectorNotFound` -- named steering vector does not exist
- `InvalidInput` -- bad parameters
- `ExtractionFailed` -- activation extraction error
- `TrainingFailed` -- probe training error
- `EvaluationFailed` -- probe evaluation error
- `GenerationFailed` -- text generation error
- `AblationFailed` -- ablation or patching error
- `ComparisonFailed` -- model comparison error
- `ComparisonIncompatible` -- models have different architecture
- `ExperimentNotFound` -- named experiment does not exist
- `ExperimentStoreError` -- experiment persistence error
- `InterventionFailed` -- component intervention error
- `LoadFailed` -- model loading error

---

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "lazarus": {
      "command": "uv",
      "args": ["run", "chuk-mcp-lazarus", "stdio"],
      "cwd": "/path/to/chuk-mcp-lazarus"
    }
  }
}
```

HTTP mode for development:

```bash
uv run chuk-mcp-lazarus http --port 8765
```

---

## Intended Claude Workflow

Given the goal: *"Discover where TranslateGemma 4B switches from
source to target language, then redirect a French translation to
German."*

```
 1. load_model("google/gemma-3-4b-it")

 2. get_model_info()
    -> num_layers = 34

 3. tokenize("Translate to French: The weather is beautiful today.")
    -> see token boundaries, multi-token words

 4. generate_text("Translate to French: The weather is beautiful today.")
    -> baseline output before any interpretability work

 5. extract_activations("Bonjour", layers=[0, 8, 16, 24, 33])
    -> sanity check: activations are reasonable

 6. compare_activations(
        prompts=["Hello", "Bonjour", "Hola"],
        layer=0
    )
    -> at layer 0: representations are language-distinct

 7. compare_activations(
        prompts=["Hello", "Bonjour", "Hola"],
        layer=33
    )
    -> at layer 33: representations converge

 8. scan_probe_across_layers(
        probe_name_prefix="source_lang",
        layers=list(range(34)),
        examples=[
            {"prompt": "Translate to French: Hello", "label": "english"},
            {"prompt": "Translate to French: Goodbye", "label": "english"},
            {"prompt": "Traduire en anglais: Bonjour", "label": "french"},
            ...  # 20+ examples per language
        ]
    )
    -> accuracy_by_layer + crossover_layer

 9. evaluate_probe(
        probe_name="source_lang_L{crossover}",
        examples=[new held-out set]
    )
    -> confirm crossover generalises

10. logit_lens(
        prompt="Translate to French: The weather is beautiful today.",
        layers=list(range(34))
    )
    -> see at which layer the translation token first appears

11. compute_steering_vector(
        vector_name="english_to_german",
        layer=crossover_layer,
        positive_prompts=["Der Hund", "Die Katze", "Das Haus", ...],
        negative_prompts=["The dog", "The cat", "The house", ...]
    )

12. steer_and_generate(
        prompt="Translate to French: The weather is beautiful today.",
        vector_name="english_to_german",
        alpha=20.0
    )
    -> steered_output: German
    -> baseline_output: French

13. ablate_layers(
        prompt="Translate to French: The weather is beautiful today.",
        layers=[crossover_layer]
    )
    -> confirm the crossover layer is causally necessary
```

---

## Version

`0.15.0` -- 52 tools, 4 resources, 13 demo scripts, 888 tests. Apache 2.0.
