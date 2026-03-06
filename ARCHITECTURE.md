# chuk-mcp-lazarus -- Architectural Principles

These principles govern every MCP server in the Lazarus family.
They mirror the conventions established in `chuk-lazarus` (the underlying
ML library) and `chuk-mcp-server` (the transport framework).

---

## 1. Async-Native

The MCP protocol is inherently asynchronous. Every tool, resource, and
lifecycle hook is an `async def`. Blocking work (MLX forward passes,
sklearn fitting) runs inside the framework's thread-pool so the event
loop is never starved.

```python
# Yes
@mcp.tool()
async def extract_activations(prompt: str, layers: list[int]) -> dict:
    ...

# No -- sync tools block the entire server
@mcp.tool()
def extract_activations(prompt: str, layers: list[int]) -> dict:
    ...
```

Corollary: never call `time.sleep()` inside a tool. Use `asyncio.sleep()`
or, more likely, just do the work.

---

## 2. Pydantic-Native -- No Magic Strings, No Dictionary Goop

Every data structure in the codebase is a Pydantic `BaseModel` with
explicit `Field` descriptions, types, and validation. This applies to:

- **Metadata**: `ModelMetadata`, `ProbeMetadata`, `VectorMetadata`
- **Tool results**: `LoadModelResult`, `ExtractionResult`, `ComparisonResult`
- **Error envelopes**: `ToolErrorResult`
- **Registry dumps**: `ProbeRegistryDump`, `VectorRegistryDump`
- **Internal state**: `_InternalState` (with `arbitrary_types_allowed`
  for MLX/HF objects)

Immutable models use `frozen=True`:

```python
class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str = Field("", description="HuggingFace model ID.")
    num_layers: int = Field(0, description="Number of transformer layers.")
    hidden_dim: int = Field(0, description="Hidden dimension size.")
```

Enums replace stringly-typed switches everywhere:

```python
class WeightDType(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"

class ProbeType(str, Enum):
    LINEAR = "linear"
    MLP = "mlp"

class ToolError(str, Enum):
    MODEL_NOT_LOADED = "ModelNotLoaded"
    LAYER_OUT_OF_RANGE = "LayerOutOfRange"
```

No `@dataclass`. No hand-built dicts for structured data. The only
place raw `dict` appears is at the **tool return boundary** where
`.model_dump()` converts a Pydantic model to JSON-safe Python
(see Principle 6).

---

## 3. Model-Agnostic

Tools accept a `model_id: str` and work with any model family that
`chuk-lazarus` supports:

| Family | Examples |
|--------|----------|
| Gemma | Gemma 3 270M -- 27B, TranslateGemma 4B/12B |
| Llama | Llama 2/3, Mistral, SmolLM2 |
| Llama4 | Llama 4 (hybrid Mamba-Transformer MoE) |
| Qwen3 | Qwen 2/3 |
| Granite | IBM Granite 3.x/4.x (hybrid Mamba-2/Transformer) |
| Jamba | AI21 Jamba (hybrid Mamba-Transformer MoE) |
| Mamba | Pure SSM models |
| StarCoder2 | Code generation |
| GPT-2 | GPT-2 and compatible |

The server never hard-codes layer access patterns. It delegates to
`chuk-lazarus` abstractions:

- `UnifiedPipeline.from_pretrained()` -- auto-detects family
- `ModelHooks` -- captures hidden states regardless of backbone structure
- `ActivationSteering` -- steers any model with a residual stream
- `CounterfactualIntervention` -- patches activations generically
- `AblationStudy` -- ablates components by `ComponentType` enum

Model-specific behaviour (Gemma's embedding scale, sliding-window
attention pattern, etc.) is handled inside `chuk-lazarus`, not in
the MCP server.

---

## 4. Clean Separation

```
server.py           Server identity and lifecycle.
                    No tool logic, no state.

model_state.py      ModelState singleton.
                    Owns the loaded model, tokenizer, config.
                    No tool logic, no MCP imports.

probe_store.py      ProbeRegistry singleton.
                    Owns trained probes and metadata.

steering_store.py   SteeringVectorRegistry singleton.
                    Owns computed steering vectors.

comparison_state.py ComparisonState singleton.
                    Holds a second model for two-model comparison.

resources.py        MCP resources (model://info, probes://registry,
                    vectors://registry, comparisons://state).
                    Read-only state views.

_serialize.py       MLX/NumPy -> JSON-safe conversion.
_generate.py        Shared text generation helper.
_compare.py         Shared comparison kernels (weight, activation,
                    attention divergence).
_extraction.py      Shared activation extraction helpers.

tools/
  model_tools.py        load_model, get_model_info
  generation_tools.py   generate_text, predict_next_token, tokenize,
                        logit_lens, track_token, embedding_neighbors
  activation_tools.py   extract_activations, compare_activations
  attention_tools.py    attention_pattern, attention_heads
  probe_tools.py        train_probe, evaluate_probe, scan_probe_across_layers,
                        list_probes
  steering_tools.py     compute_steering_vector, steer_and_generate,
                        list_steering_vectors
  ablation_tools.py     ablate_layers, patch_activations
  causal_tools.py       trace_token, full_causal_trace
  residual_tools.py     residual_decomposition, layer_clustering,
                        logit_attribution, head_attribution, top_neurons
  comparison_tools.py   load_comparison_model, compare_weights,
                        compare_representations, compare_attention,
                        compare_generations, unload_comparison_model
  geometry/             Per-tool subpackage (14 tools)
    _helpers.py           Shared enums, math, direction extraction,
                          PCA helpers (collect_activations,
                          effective_dimensionality)
    token_space.py        token_space
    direction_angles.py   direction_angles
    subspace_decomposition.py  subspace_decomposition
    residual_trajectory.py     residual_trajectory
    feature_dimensionality.py  feature_dimensionality
    decode_residual.py    decode_residual
    computation_map.py    computation_map
    inject_residual.py    inject_residual
    residual_match.py     residual_match
    compute_subspace.py   compute_subspace, list_subspaces
    residual_atlas.py     residual_atlas
    weight_geometry.py    weight_geometry
    residual_map.py       residual_map
```

Each layer has a single responsibility:

- **Server** -- wiring and transport. Knows about MCP.
- **State** -- in-memory singletons. Knows about chuk-lazarus objects.
- **Tools** -- thin translation between MCP parameters and chuk-lazarus APIs.
  Knows about both.

Tools never import each other. Shared logic lives in state modules or
a private `_util.py`.

---

## 5. Stateless Tools, Stateful Singletons

Tool functions are **pure translations**: they read parameters, delegate
to chuk-lazarus, convert the result to a JSON-safe dict, and return.
They hold no state themselves.

All mutable state lives in exactly four singletons:

| Singleton | Owns |
|-----------|------|
| `ModelState` | Loaded MLX model, tokenizer, config, family info |
| `ProbeRegistry` | Trained sklearn probes + accuracy metadata |
| `SteeringVectorRegistry` | Computed steering vectors + separability metadata |
| `ComparisonState` | Second model for two-model comparison |

Singletons are plain Python classes with `threading.Lock` for thread
safety. Their **internal state and metadata** are Pydantic models
(following Principle 2), but the singleton itself is a regular class
because it manages mutable references and lock lifecycle. Singletons
are created once at import time and shared across all tool invocations.

---

## 6. JSON-Safe Boundary

MCP tools communicate via JSON. The **tool function return** is the
serialisation boundary. Inside that boundary, use Pydantic models and
native types freely (`mx.array`, `np.ndarray`, sklearn models). At
the boundary, call `.model_dump()` to produce a JSON-safe dict:

```python
@mcp.tool()
async def extract_activations(prompt: str, layers: list[int]) -> dict:
    # ... do work, build typed result ...
    result = ExtractionResult(
        prompt=prompt,
        token_position=token_position,
        num_tokens=int(num_tokens),
        activations=activations,          # already list[float] via _serialize
    )
    return result.model_dump(exclude_none=True)
```

A shared `_serialize.py` module converts MLX/NumPy objects to plain
Python before they enter Pydantic models:

```python
def mx_to_list(arr: mx.array) -> list[float]: ...
def hidden_state_to_list(arr: mx.array, position: int = -1) -> list[float]: ...
def np_to_python(val: Any) -> int | float | str | bool | None: ...
def cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]: ...
def pca_2d(vectors: list[list[float]]) -> list[list[float]]: ...
```

The pipeline is: **MLX array -> `_serialize` helper -> Pydantic field -> `.model_dump()` -> JSON**.

---

## 7. Error Envelopes, Never Exceptions

Tools never raise exceptions that would kill the MCP session. Every
tool catches exceptions and returns a structured error dict:

```python
{
    "error": True,
    "error_type": "ModelNotLoaded",
    "message": "Call load_model() first.",
    "tool": "extract_activations"
}
```

Error types are an enum, not free-form strings:

```python
class ToolError(str, Enum):
    MODEL_NOT_LOADED = "ModelNotLoaded"
    LAYER_OUT_OF_RANGE = "LayerOutOfRange"
    PROBE_NOT_FOUND = "ProbeNotFound"
    VECTOR_NOT_FOUND = "VectorNotFound"
    INVALID_INPUT = "InvalidInput"
    EXTRACTION_FAILED = "ExtractionFailed"
    TRAINING_FAILED = "TrainingFailed"
    EVALUATION_FAILED = "EvaluationFailed"
    GENERATION_FAILED = "GenerationFailed"
    ABLATION_FAILED = "AblationFailed"
    COMPARISON_FAILED = "ComparisonFailed"
    COMPARISON_INCOMPATIBLE = "ComparisonIncompatible"
    EXPERIMENT_NOT_FOUND = "ExperimentNotFound"
    EXPERIMENT_STORE_ERROR = "ExperimentStoreError"
    INTERVENTION_FAILED = "InterventionFailed"
    GEOMETRY_FAILED = "GeometryFailed"
    LOAD_FAILED = "LoadFailed"
```

---

## 8. Zero-Overhead Optionals

Features that are not always needed (attention capture, PCA projection,
language detection) are off by default and add zero cost when unused.
Tools accept boolean flags or optional parameters rather than always
computing everything:

```python
@mcp.tool()
async def extract_activations(
    prompt: str,
    layers: list[int],
    token_position: int = -1,
    capture_attention: bool = False,    # Off by default -- saves memory
) -> dict:
    ...
```

---

## 9. Deterministic Where Possible

- Probes use explicit `random_state` seeds for reproducibility.
- Steering vectors are computed as deterministic mean-differences.
- Generation uses `temperature=0.0` by default for repeatable demos.
- State singletons track timestamps so operations can be audited.

---

## 10. Progressive Disclosure

Tools are ordered by complexity. A new user can:

1. `load_model` -- one call, reasonable defaults
2. `get_model_info` -- understand what they loaded
3. `tokenize` / `generate_text` -- see what the model produces
4. `extract_activations` -- see real numbers
5. `compare_activations` -- see structure
6. `logit_lens` / `track_token` -- watch predictions evolve layer by layer
7. `train_probe` / `scan_probe_across_layers` -- find the interesting layer
8. `attention_pattern` / `attention_heads` -- see information routing
9. `residual_decomposition` / `logit_attribution` -- what each layer contributes
10. `head_attribution` / `top_neurons` -- drill into specific components
11. `compute_steering_vector` + `steer_and_generate` -- the payoff
12. `ablate_layers` / `patch_activations` / `trace_token` -- causal analysis
13. `embedding_neighbors` -- vocabulary-level analysis

Each tool's docstring explains what it does, what to call before it,
and what to call after it. Claude can follow this progression
autonomously.
