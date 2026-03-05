# chuk-mcp-lazarus -- Roadmap

Four MCP servers, built incrementally, exposing `chuk-lazarus`
capabilities to reasoning models.

---

## Server Family Overview

| Server | Purpose | Loads Weights? | Dependencies |
|--------|---------|----------------|--------------|
| `chuk-mcp-lazarus` | Mechanistic interpretability experiments | Yes | chuk-lazarus, scikit-learn, mlx |
| `chuk-mcp-tokenizer` | Universal tokenization toolkit | No | chuk-lazarus (data module) |
| `chuk-mcp-introspect` | Model architecture inspector | No | chuk-lazarus (config parsing) |
| `chuk-mcp-moe` | MoE expert analysis | Yes | chuk-lazarus, mlx |

Claude Desktop can run the lightweight servers (tokenizer, introspect)
always-on, and spin up the heavy servers (lazarus, moe) only when
doing interpretability work.

---

## Phase 1 -- chuk-mcp-lazarus (Core)

The flagship server. Exposes probing, steering, ablation, and
activation extraction so Claude can autonomously design mechanistic
interpretability experiments on any locally-loaded model.

**Default demo target:** TranslateGemma 4B (Gemma 3 architecture, 34
layers, fits comfortably on Apple Silicon). Works with any model
chuk-lazarus supports.

### Implementation Order

#### Step 1: Skeleton + Model Management ✅
- `pyproject.toml`, package structure, entry point
- `server.py` -- ChukMCPServer instance
- `model_state.py` -- ModelState singleton wrapping UnifiedPipeline
- `tools/model_tools.py` -- `load_model`, `get_model_info`
- Verify: load TranslateGemma 4B, return architecture metadata

#### Step 2: Activation Extraction ✅
- `_serialize.py` -- mx.array to JSON-safe conversion
- `tools/activation_tools.py` -- `extract_activations`, `compare_activations`
- Uses `ModelHooks` + `CaptureConfig` from chuk-lazarus
- `compare_activations` adds cosine similarity matrix + PCA projection
- Verify: extract activations at layers 0, 16, 33 for a prompt

#### Step 3: Probing ✅
- `probe_store.py` -- ProbeRegistry singleton
- `tools/probe_tools.py` -- `train_probe`, `evaluate_probe`,
  `scan_probe_across_layers`, `list_probes`
- `scan_probe_across_layers` is the key demo tool: trains a probe at
  every layer in one call, returns accuracy curve + crossover layer
- Uses sklearn LogisticRegression / MLPClassifier
- Verify: scan all 34 layers with source-language labelled examples

#### Step 4: Steering ✅
- `steering_store.py` -- SteeringVectorRegistry singleton
- `tools/steering_tools.py` -- `compute_steering_vector`,
  `steer_and_generate`, `list_steering_vectors`
- Uses `ActivationSteering` from chuk-lazarus for contrastive
  activation addition (CAA)
- `steer_and_generate` runs both steered + baseline generation
- Verify: steer a "Translate to French" prompt toward German

#### Step 5: Ablation ✅
- `tools/ablation_tools.py` -- `ablate_layers`, `patch_activations`
- Uses `AblationStudy` and `CounterfactualIntervention` from chuk-lazarus
- Verify: ablate the crossover layer, observe disrupted translation

#### Step 6: Resources + Polish ✅
- MCP resources: `model://info`, `probes://registry`, `vectors://registry`
- Claude Desktop JSON config
- End-to-end demo script (the 10-step Claude workflow)
- Error handling audit

**Status:** Steps 1--6 complete. 13 tools + 3 resources.
End-to-end demo script runs the Claude workflow.

#### Step 6b: Model Comparison ✅
- `comparison_state.py` -- ComparisonState singleton (second model)
- `_compare.py` -- Shared comparison kernels (weight, activation, attention)
- `tools/comparison_tools.py` -- `load_comparison_model`, `compare_weights`,
  `compare_representations`, `compare_attention`, `unload_comparison_model`
- Resource: `comparisons://state`
- Demo: `examples/comparison_demo.py` (Gemma 3 4B vs TranslateGemma 4B)
- Designed for low-resource language analysis (Icelandic, Swahili, Estonian, Marathi)

**Status:** 18 tools + 4 resources. Comparison tools enable two-model analysis
to find where fine-tuning changes weights, activations, and attention patterns.

#### Step 6c: Generation & Prediction Tools ✅
- `tools/generation_tools.py` -- `generate_text`, `predict_next_token`,
  `tokenize`, `logit_lens`
- `_generate.py` -- Shared text generation helper (used by generation and steering tools)
- `compare_generations` added to `tools/comparison_tools.py`
- Language transition demo expanded to 13 steps (was 10): added tokenize,
  generate_text, logit_lens
- Comparison demo expanded to 7 steps (was 6): added compare_generations

**Status:** 23 tools + 4 resources. 27/27 smoke tests passing. Generation tools
let Claude see model I/O (tokenization, text generation, next-token predictions)
and watch predictions evolve layer-by-layer via logit lens.

#### Step 6d: Token Tracking & Attention Inspection ✅
- `tools/generation_tools.py` -- `track_token` (track specific token probability across layers)
- `tools/attention_tools.py` -- `attention_pattern`, `attention_heads`
- `track_token` uses logit lens to follow a target token's probability and rank
- `attention_pattern` extracts per-head attention weights via manual Q*K+RoPE
- `attention_heads` computes per-head entropy (focused vs diffuse attention)
- Language transition demo expanded to 14 steps (was 13): added track_token

**Status:** 26 tools + 4 resources. 32/32 smoke tests passing. Phase 1b Steps 8-9
complete. Attention tools reuse `_compute_attention_weights` from `_compare.py`.

#### Step 6e: Causal Tracing ✅
- `tools/causal_tools.py` -- `trace_token`, `full_causal_trace`
- `trace_token` ablates each layer and measures effect on target token probability
- `full_causal_trace` produces position × layer causal heatmap (Meng et al. style)
- Demo: `examples/causal_tracing_demo.py`
- Language transition demo expanded to 15 steps (was 14): added causal tracing

**Status:** 28 tools + 4 resources. 36/36 smoke tests passing.

#### Step 6f: Residual Stream Analysis ✅
- `tools/residual_tools.py` -- `residual_decomposition`, `layer_clustering`
- `residual_decomposition` manually unrolls forward pass to separate attention vs MLP
- `layer_clustering` computes cosine similarity and cluster separation across layers
- Demo: `examples/residual_stream_demo.py`

**Status:** 30 tools + 4 resources. 40/40 smoke tests passing.

#### Step 6g: Logit Attribution ✅
- `tools/residual_tools.py` -- `logit_attribution` (direct logit attribution)
- Per-layer attention + MLP contribution to predicted token's logit
- Demo: `examples/logit_attribution_demo.py`

**Status:** 31 tools + 4 resources. 42/42 smoke tests passing.

#### Step 6h: Deep Interpretability Tools ✅
- `tools/residual_tools.py` -- `head_attribution`, `top_neurons`
- `tools/generation_tools.py` -- `embedding_neighbors`
- `head_attribution` decomposes attention output through `o_proj` per head
- `top_neurons` computes per-neuron MLP contributions via SwiGLU decomposition
- `embedding_neighbors` finds nearest tokens by cosine similarity in embedding space
- Shared helpers: `_get_unembed_vector()`, `_resolve_target_token()` in residual_tools.py
- Shared extraction: `_extraction.py` consolidates duplicate activation extraction
- Demo: `examples/deep_dive_demo.py`

**Status:** 34 tools + 4 resources. 53/53 smoke tests passing. All CPU-bound work
wrapped in `asyncio.to_thread`. Input validation (top_k, max_new_tokens, empty layers)
across all tools. Shared comparison guard `_require_comparison_models()`.

#### Step 6i: Comprehensive Test Suite ✅
- `tests/conftest.py` -- Shared fixtures: MLX stub (MockMxArray with full arithmetic,
  indexing, comparison operators), chuk-lazarus stubs (ModelHooks, CaptureConfig,
  ablation/intervention modules), mock model/tokenizer/config, autouse fixtures
  for `asyncio.to_thread` sync execution and singleton reset
- 22 test files covering all 25 source files
- Tests run on any platform (no Apple Silicon / MLX required)
- Mocks all chuk-lazarus dependencies via `sys.modules` stubs
- Tests cover: error envelopes, input validation, success paths, exception handling,
  pure helper functions, `_impl` function internals, singleton lifecycle, Pydantic
  model serialization

**Status:** 595 tests passing. 96% overall coverage. Per-file coverage:

| Coverage | Files |
|----------|-------|
| 100% | `__init__`, `_bootstrap`, `_extraction`, `_serialize`, `errors`, `model_state`, `probe_store`, `resources`, `server`, `steering_store`, `tools/__init__`, `ablation_tools`, `attention_tools`, `probe_tools`, `residual_tools` |
| 90--99% | `comparison_state` (99%), `activation_tools` (98%), `generation_tools` (94%), `steering_tools` (92%), `_generate` (92%), `_compare` (91%), `causal_tools` (91%), `comparison_tools` (90%), `model_tools` (90%) |

---

## Phase 1b -- chuk-mcp-lazarus (Extended)

Deeper interpretability tools that unlock neuron-level analysis,
causal tracing, direction extraction, and residual decomposition.
These turn lazarus from a probing/steering toolkit into a full
mechanistic interpretability research platform.

**Steps 7--12 are complete.** Steps 13--14 remain but are
deprioritized in favor of Phase 1c (experiment-driven gaps).
All backends exist in chuk-lazarus.

### Step 7: Neuron Analysis ✅ (v0.10.0)

Map the neuron landscape -- discover which individual neurons encode
specific features.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `discover_neurons` | Auto-find neurons that discriminate between prompt groups (e.g. which neurons fire for French vs English) | Reimplemented from `NeuronAnalysisService` using `_extraction.py` |
| `analyze_neuron` | Profile a specific neuron: activation stats, separation score, group means across prompts | Reimplemented from `NeuronAnalysisService` using `_extraction.py` |

> **Note:** Neuron tools reimplement the core Cohen's d / activation statistics
> algorithms from `NeuronAnalysisService` using shared `_extraction.py` helpers,
> avoiding the service's own model-loading path. Pure numpy, no chuk-lazarus imports.

**Unlocks:** Automated Circuit Discovery, Computational Stratigraphy,
Fine-Tuning Delta, Metacognition Probing, Universal Circuits.

### Step 8: Logit Lens ✅ (v0.4.0--v0.5.0)

Watch the model think -- track how predictions evolve layer by layer.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `logit_lens` | Top-k predictions at each layer, find where model "decides" | `ModelHooks.get_layer_logits()` |
| `track_token` | Track a specific token's probability and rank across layers | `ModelHooks.get_layer_logits()` |

> **Note:** `logit_lens` implemented in v0.4.0. `track_token` added in v0.5.0.

### Step 9: Attention Analysis ✅

See what attends to what -- understand information routing.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `attention_pattern` | Extract full attention matrix at a layer (which tokens attend to which) | Manual Q*K + RoPE via `_compare._compute_attention_weights()` |
| `attention_heads` | Per-head entropy and focus analysis -- find which heads are doing the work | Entropy from attention weights |

### Step 10: Causal Tracing ✅ (v0.6.0)

Trace causal circuits -- identify which components are responsible
for specific predictions. Probes show correlation; causal tracing
shows causation.

> **Implemented in v0.6.0.** Both tools wrap `CounterfactualIntervention`
> from `introspection/interventions.py`.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `trace_token` | Which layers are causally responsible for predicting a specific token? | `CounterfactualIntervention.trace_token()` |
| `full_causal_trace` | Position × layer heatmap of causal importance (Meng et al. style) | `CounterfactualIntervention.full_causal_trace()` |

**Unlocks:** Automated Circuit Discovery, Knowledge Surgery,
Computational Stratigraphy, Fine-Tuning Delta, Safety Patching.

**Backend:** `CounterfactualIntervention` in `introspection/interventions.py`
(975 lines). Supports `InterventionType` enum (ZERO, MEAN, PATCH, NOISE,
STEER, SCALE) and `ComponentTarget` enum (HIDDEN, ATTENTION, MLP,
ATTENTION_HEAD, MLP_NEURON). Returns recovery rates, KL divergence,
and peak effect layers.

### Step 11: Direction Extraction ✅ (v0.10.0)

Find interpretable directions in activation space using multiple
methods beyond simple mean-difference. Enables understanding what
features the model represents and how they change with fine-tuning.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `extract_direction` | Find directions via mean-diff, LDA, PCA, or probe weights | `DirectionExtractor` with `DirectionMethod` enum |

> **Note:** Extracted directions are automatically stored in the
> `SteeringVectorRegistry` for immediate use with `steer_and_generate()`.

**Unlocks:** Computational Stratigraphy, Fine-Tuning Delta,
Metacognition Probing, Universal Circuits, Safety Patching.

**Backend:** `DirectionExtractor` in `introspection/circuit/directions.py`
(489 lines). `DirectionMethod` enum: DIFFERENCE_OF_MEANS, LDA,
PROBE_WEIGHTS, CONTRASTIVE, PCA. Returns `ExtractedDirection` with
separation score (Cohen's d), classification accuracy, mean projections.

### Step 12: Residual Stream Dynamics ✅ (v0.7.0--v0.9.0)

Understand how information flows through the residual stream.

> **v0.7.0:** `residual_decomposition` and `layer_clustering`.
> **v0.8.0:** `logit_attribution` (direct logit attribution per layer).
> **v0.9.0:** `head_attribution` (per-head), `top_neurons` (per-neuron).

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `residual_decomposition` | Separate attention vs MLP contribution to the residual stream per layer | Manual sub-component forward pass via `ModelHooks` |
| `layer_clustering` | Cluster prompts by representation similarity at each layer, measure separability | `ModelHooks` + `cosine_similarity_matrix` |
| `logit_attribution` | Direct logit attribution: per-layer attention + MLP contribution to predicted token | Manual decomposition forward pass |
| `head_attribution` | Per-head logit attribution at a specific layer | `o_proj` decomposition per head |
| `top_neurons` | Per-neuron MLP identification: which neurons push toward target token | SwiGLU decomposition via `down_proj` |

**Unlocks:** Automated Circuit Discovery, Computational Stratigraphy,
Universal Circuits.

### Step 13: Knowledge & Confidence

Probe what the model knows and how confident it is.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `measure_confidence` | Internal confidence calibration via activation space geometry | `UncertaintyAnalysis` in `introspection/models/uncertainty.py` |
| `detect_format_gate` | Find layer where model decides CoT vs direct computation | `MetacognitiveResult` strategy detection |

**Unlocks:** Metacognition Probing (does the model know what it knows?).

### Step 14: External Memory & Knowledge Surgery

Runtime knowledge editing without fine-tuning.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `inject_fact` | Add a fact to external memory for inference-time injection | `ExternalMemory.add_facts()` |
| `query_with_memory` | Generate with external memory augmentation active | `ExternalMemory.query()` |

**Unlocks:** Knowledge Surgery, Safety Patching.

**Backend:** `ExternalMemory` in `introspection/external_memory.py`
(723 lines). Circuit-guided memory externalization. Builds searchable
(query_vec, value_vec) pairs at specific layers. Matches incoming
queries and injects retrieved values into the residual stream.

---

## Phase 1c -- Experiment-Driven Priorities

Steps 13--14 were planned theoretically. Phase 1c is prioritized by
what **actually broke** when running real experiments: the
Computational Stratigraphy (full-depth layer map) and the Embedding
War (attention vs FFN attribution analysis across prompts). These
experiments exposed seven concrete tool gaps, ranked by how badly
they block the research program.

### What Happened

The stratigraphy experiment ran six methods across all 34 layers of
Gemma 3 4B. Logit lens failed completely because `ModelHooks.get_layer_logits()`
applies the wrong normalization for Gemma's architecture — one of
six measurement methods went blind. The embedding war experiment ran
`logit_attribution` prompt-by-prompt and manually tracked the
attention-vs-FFN balance in the context window, burning tokens on
bookkeeping instead of analysis. The agent found critical neurons
(605, 1633, 889) but couldn't trace how information flows between
them across layers. And when the session ended, all findings vanished.

### Steps 15--22 — Complete ✅

#### Step 15: Calibrated Logit Lens ✅

Fixed `logit_lens` and `track_token` to use calibrated `final_norm → lm_head`
projection. Works on all architectures including Gemma 3 (4-norm layers, tied embeddings).

#### Step 16: Attribution Sweep ✅

Batch logit attribution across multiple prompts with per-layer summary
statistics and per-prompt summary rows (`attribution_sweep`).

#### Step 17: Experiment Persistence ✅

Cross-session experiment memory: `create_experiment`, `add_experiment_result`,
`get_experiment`, `list_experiments`. Stores to `~/.chuk-lazarus/experiments/`.

#### Step 18: Enhanced Attribution Sweep ✅

Added `prompt_summary` field to `attribution_sweep` — per-prompt summary rows
(embedding logit, net attention, net FFN, final logit, probability, dominant component).

#### Step 19: Track Race ✅

Multi-candidate logit trajectory (`track_race`). Race 2-20 tokens across layers
in a single forward pass with crossing detection.

#### Step 20: Component Intervention ✅

Surgical causal intervention (`component_intervention`). Zero/scale attention,
FFN, or individual heads at a specific layer. Compares clean vs intervened predictions
with KL divergence.

#### Step 21: Probe at Inference ✅

Runtime probe monitoring (`probe_at_inference`). Run a trained probe during
autoregressive generation to track internal state evolution token-by-token.

#### Step 22: Neuron Trace ✅

Neuron influence tracing (`neuron_trace`). Trace a neuron's output direction
through downstream layers via cosine similarity with residual, attention, and FFN states.

**Status:** Steps 15--22 complete. **46 tools**, **772 tests**, `make check` green.

Steps 13--14 (confidence/metacognition, external memory) remain valid
but are deprioritized.

---

## Revolutionary Experiments

These are the experiments that become possible as Phase 1b/1c tools
are implemented. Each requires multi-tool composition that no single
technique can achieve. See `REVOLUTIONARY-EXPERIMENTS.md` for full
details.

### Experiment 1: Automated Circuit Discovery

**Goal:** Given any prompt + target behavior, autonomously trace
the circuit responsible — specific (position, layer, component)
tuples, verified causally.

**Why revolutionary:** No one has demonstrated fully autonomous
circuit discovery on 4B+ models. Current circuit analysis is manual
and takes weeks per circuit.

**Tool chain:** `logit_lens` → `track_token` →
`full_causal_trace` → `residual_decomposition` → `attention_pattern` →
`discover_neurons` → `neuron_trace` → `patch_activations`

**Requires:** Steps 10, 7, 12, 15, 20

### Experiment 2: Knowledge Surgery Without Fine-Tuning

**Goal:** Identify where a factual error is stored, construct a
correction vector, and inject it at inference time. Fully reversible,
no gradients.

**Why revolutionary:** Changes the paradigm from "retrain to fix"
to "patch at runtime." Direct safety implications.

**Tool chain:** `generate_text` → `track_token` → `trace_token` →
`extract_direction` → `inject_fact` → `query_with_memory`

**Requires:** Steps 10, 11, 14

### Experiment 3: Computational Stratigraphy

**Goal:** Multi-method map of what each layer computes. For every
layer: what information it contains (probes), what it predicts
(calibrated logit lens), whether it's necessary (ablation), what
direction the feature lives in (directions), how many neurons encode
it (neurons), and whether attention or MLP is responsible
(decomposition). Results persisted for cross-session analysis.

**Why revolutionary:** No complete multi-method stratigraphy exists
for any 4B+ model. Combines 6 independent measurement methods.

**Tool chain:** `scan_probe_across_layers` + `logit_lens` (calibrated) +
`ablate_layers` + `extract_direction` + `discover_neurons` +
`residual_decomposition` + `save_experiment`

**Requires:** Steps 7, 11, 12, 15, 17

### Experiment 4: The Fine-Tuning Delta Problem

**Goal:** Mechanistic answer to "what did fine-tuning change?"
Not just output differences, but which neurons, directions,
attention heads, and causal circuits changed.

**Why revolutionary:** Transforms model development from empirical
("train and benchmark") to mechanistic ("train and understand").

**Tool chain:** `compare_generations` → `compare_weights` →
`compare_representations` → `attribution_sweep` (both models) →
`compare_attributions` → `discover_neurons` (both models) →
`trace_token` (both models)

**Requires:** Steps 10, 11, 7, 16, 18

### Experiment 5: Does the Model Know What It Knows?

**Goal:** Test whether neural networks have internal representations
of their own uncertainty. If a probe can decode "this model will
get this wrong" from hidden states, the model has metacognition.

**Why revolutionary:** Directly tests self-knowledge in neural
networks. If positive, enables hallucination detection at the
representation level.

**Tool chain:** `scan_probe_across_layers` (knows/doesn't-know labels) →
`extract_direction` (certainty direction) → `bootstrap_test`
(significance) → `steer_and_generate` (along uncertainty direction)

**Requires:** Steps 11, 21 (Steps 13 optional enhancement)

### Experiment 6: Cross-Model Universal Circuits

**Goal:** Do all transformer architectures discover the same
computational primitives? Run identical stratigraphy on Gemma,
Llama, and Qwen. Normalize by relative depth and compare.

**Why revolutionary:** Tests whether there are universal laws of
neural network computation, or if each architecture finds its own
solution.

**Tool chain:** Full stratigraphy (Experiment 3) × 3 models,
then cross-model `compare_attributions` of probe curves, emergence
layers, neuron counts, and attention-vs-MLP ratios. Requires
`logit_lens` (calibrated) to work across architectures.

**Requires:** Steps 7, 11, 12, 15, 18

### Experiment 7: Runtime Safety Patching

**Goal:** Identify harmful circuits via causal tracing, extract
the harmful direction, and inject a correction at the
representation level. Not output filtering -- the model's
computation is changed so the harmful association never reaches
the prediction head.

**Why revolutionary:** Representation-level safety. The model
doesn't "know but is told not to say" -- the harmful computation
is replaced in the residual stream.

**Tool chain:** `full_causal_trace` → `extract_direction` →
`component_intervention` (surgical suppression) →
`generate_text` (verify)

**Requires:** Steps 10, 11, 19 (Step 14 for persistent patching)

### Experiment 8: The Embedding War

**Goal:** Map how the embedding layer's initial bias (encoding
surface-level token identity) gets overwritten by deeper
computation. Track the battle between attention (which builds
context) and FFN (which applies learned transformations) across
all layers. Identify where correctness emerges and whether the
attention-vs-FFN balance predicts hallucination.

**Why revolutionary:** Reveals the fundamental computational
tension in transformers: pre-trained token identity vs contextual
computation. No one has systematically mapped this war across
an entire model.

**Tool chain:** `attribution_sweep` (20+ prompts) →
`compare_attributions` → `component_intervention` (surgical tests) →
`discover_neurons` → `neuron_trace` → `bootstrap_test`

**Requires:** Steps 16, 18, 19, 7, 20, 21

### Experiment Priority Matrix (Updated)

| Tool (Step) | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | Exp 8 | Status |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| `logit_lens` calibrated (15) | ✓ | | ✓ | ✓ | ✓ | ✓ | | | ✅ |
| `attribution_sweep` (16) | | | ✓ | ✓ | | | | ✓ | ✅ |
| `create/get_experiment` (17) | ✓ | | ✓ | ✓ | ✓ | ✓ | | ✓ | ✅ |
| `track_race` (19) | | | | | | ✓ | | ✓ | ✅ |
| `component_intervention` (20) | ✓ | | | | | | ✓ | ✓ | ✅ |
| `probe_at_inference` (21) | | | | | ✓ | | | | ✅ |
| `neuron_trace` (22) | ✓ | | | ✓ | | | | ✓ | ✅ |
| `trace_token` / `full_causal_trace` (10) | ✓ | ✓ | | ✓ | | | ✓ | | ✅ |
| `extract_direction` (11) | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✅ |
| `discover_neurons` / `analyze_neuron` (7) | ✓ | | ✓ | ✓ | | ✓ | | ✓ | ✅ |
| `measure_confidence` (13) | | | | | opt | | | | |
| `inject_fact` / `query_with_memory` (14) | | ✓ | | | | | opt | | |

**Remaining:** Steps 13 → 14

---

## Phase 2 -- chuk-mcp-tokenizer

Standalone tokenization toolkit. No model weights loaded.

### Tools

| Tool | Purpose |
|------|---------|
| `tokenize` | Encode text to token IDs + token strings + byte offsets |
| `detokenize` | Decode token IDs back to text |
| `vocab_search` | Search vocabulary by substring or regex |
| `compare_tokenizers` | Side-by-side tokenization with two tokenizers |
| `tokenizer_info` | Vocab size, special tokens, tokenizer type |
| `tokenizer_fingerprint` | Hash + characteristic token set for identification |
| `count_tokens` | Fast token count for prompt budgeting |
| `tokenizer_coverage` | UNK rate, vocab utilization against a dataset |
| `tokenizer_entropy` | Token distribution entropy -- measure tokenizer efficiency |
| `tokenizer_fit_score` | Tokenizer-dataset compatibility score |
| `tokenizer_doctor` | Health check: find common pathologies |

### Why Separate
Tokenization is universally useful for any coding, prompting, or
analysis workflow. It should not require loading a multi-billion
parameter model into memory.

### Implementation
Wraps `chuk-lazarus` data module: `CustomTokenizer`, the CLI
encode/decode/compare/fingerprint commands, and HuggingFace
`AutoTokenizer` for any model's tokenizer. Tokenizer analysis
commands (coverage, entropy, fit_score, doctor) wrap the CLI
analyze and health modules.

---

## Phase 3 -- chuk-mcp-introspect

Lightweight model architecture inspector. Reads config.json from
HuggingFace, no weights loaded.

### Tools

| Tool | Purpose |
|------|---------|
| `inspect_model_config` | Parse config.json, return architecture details |
| `detect_model_family` | Identify architecture family from model ID |
| `estimate_memory` | Predict VRAM for a model at various dtypes |
| `estimate_flops` | FLOPs per forward pass |
| `count_parameters` | Total / trainable / frozen parameter counts |
| `list_supported_families` | What architectures chuk-lazarus supports |
| `compare_architectures` | Side-by-side diff of two model configs |

### Why Separate
Architecture inspection is config-only. Claude can reason about which
model to load, what fits in memory, and how models differ structurally
without touching weights.

### Implementation
Wraps `chuk-lazarus` functions: `detect_model_family`,
`count_parameters`, `estimate_flops`, `estimate_memory`,
`get_model_info` from `models_v2.introspection`.

---

## Phase 4 -- chuk-mcp-moe

MoE expert analysis. The chuk-lazarus MoE introspection module is
30+ files deep -- this is a differentiator.

### Core Tools

| Tool | Purpose |
|------|---------|
| `detect_moe_architecture` | Identify MoE type (Mixtral, Llama 4, Granite, etc.) |
| `get_moe_layers` | Locate MoE layers in the model |
| `identify_expert` | Find specialist experts for a task |
| `cluster_experts` | Group experts by specialization |
| `analyze_routing` | Router entropy, load balance, co-activation |
| `compare_routing` | Routing pattern differences across prompts |
| `ablate_expert` | Zero out an expert, measure impact |
| `find_causal_experts` | Find experts critical for a specific task |
| `expert_compression_plan` | Identify merge/prune candidates |
| `expert_vocabulary` | Token preferences per expert (MoE logit lens) |

### Extended Tools

| Tool | Purpose |
|------|---------|
| `classify_expert_functions` | Systematic ablation to classify experts as STORAGE / COMPUTATION / ROUTING / REDUNDANT |
| `find_cold_experts` | Identify rarely-activated experts and their trigger conditions |
| `expert_circuits` | Cross-layer expert circuit analysis -- which experts hand off to which |
| `expert_interference` | Measure multi-expert penalty and linearity of expert combinations |
| `predict_routing` | Predict expert routing from attention patterns |
| `generation_dynamics` | Trace routing decisions token-by-token during generation |
| `merge_experts` | Analyze and execute expert merging for model compression |
| `expert_heatmap` | Expert activation heatmap across tokens and layers |
| `full_expert_taxonomy` | Complete expert profiling: specialist, generalist, cold, redundant |
| `domain_test` | Test expert specialization on domain-specific prompts |

### Why Separate
MoE analysis requires a loaded MoE model and has a distinct user
workflow (routing analysis, expert profiling, compression planning).
Dense-model users never need these tools.

### Implementation
Wraps `chuk-lazarus` MoE module: `MoEHooks`, `identify_expert`,
`analyze_coactivation`, `compute_routing_diversity`,
`ablate_expert`, `find_causal_experts`, `create_compression_plan`,
`MoELogitLens`, `NeuronAnalysisService`, and the 25+ MoE CLI
command handlers.

---

## Future Considerations

### Virtual Expert Framework
The virtual expert system routes to external solvers (CSP, compilers,
math verifiers) via learned invocation formats:

| Tool | Purpose |
|------|---------|
| `run_virtual_expert` | Solve problem via learned routing to expert functions |
| `benchmark_virtual_expert` | Compare virtual expert vs base model accuracy |

### Training Integration
If chuk-lazarus training capabilities are exposed:

| Tool | Purpose |
|------|---------|
| `train_sft` | Supervised fine-tuning on dataset |
| `train_dpo` | Direct Preference Optimization |
| `generate_training_data` | Generate synthetic training data |

---

## Version Milestones

| Version | Scope | Status |
|---------|-------|--------|
| 0.1.0 | Phase 1 Steps 1--5 (load, extract, probe, steer, ablate) | ✅ |
| 0.2.0 | Phase 1 Step 6 (resources, polish, end-to-end demo) | ✅ |
| 0.3.0 | Phase 1 Step 6b (model comparison tools, two-model analysis) | ✅ |
| 0.4.0 | Phase 1 Step 6c (generation, prediction, tokenize, logit lens) | ✅ |
| 0.5.0 | Phase 1 Step 6d (track_token, attention_pattern, attention_heads) | ✅ |
| 0.6.0 | Phase 1 Step 6e (causal tracing: trace_token, full_causal_trace) | ✅ |
| 0.7.0 | Phase 1 Step 6f (residual stream: residual_decomposition, layer_clustering) | ✅ |
| 0.8.0 | Phase 1 Step 6g (logit_attribution: direct logit attribution) | ✅ |
| 0.9.0 | Phase 1 Step 6h (head_attribution, top_neurons, embedding_neighbors) | ✅ |
| 0.9.1 | Phase 1 Step 6i (comprehensive test suite: 595 tests, 96% coverage) | ✅ |
| 0.10.0 | Phase 1b Steps 7+11 (direction, neurons: 3 tools, 639 tests) | ✅ |
| 0.11.0 | Phase 1c Step 15 (calibrated logit lens: architecture-aware projection) | ✅ |
| 0.12.0 | Phase 1c Step 16 (attribution_sweep: batch logit attribution) | ✅ |
| 0.13.0 | Phase 1c Step 17 (experiment persistence: create/add/get/list) | ✅ |
| 0.14.0 | Phase 1d Steps 18--22 (track_race, component_intervention, probe_at_inference, neuron_trace) | ✅ |
| 0.15.0 | Phase 1b Steps 13--14 (confidence, metacognition, external memory) | |
| 0.17.0 | Phase 2 (tokenizer server) | |
| 0.18.0 | Phase 3 (introspect server) | |
| 0.19.0 | Phase 4 core (MoE: routing, ablation, identification) | |
| 0.19.x | Phase 4 extended (MoE: circuits, compression, dynamics) | |
| 1.0.0 | Cross-session research, multi-model sessions, production hardening | |

---

## Tool Count Summary

| Phase | Server | Tools |
|-------|--------|-------|
| 1+1b+1c+1d | lazarus (core + extended + experiment-driven) | 46 |
| 2 | tokenizer | 11 |
| 3 | introspect | 7 |
| 4 | moe | 20 |
| **Total** | | **~84** |

---

## Backend Coverage

### Phase 1b — chuk-lazarus backends (production-ready, MCP wrapping only)

| Tool | Backend Class | File | Lines |
|------|---------------|------|-------|
| `trace_token` | `CounterfactualIntervention` | `introspection/interventions.py` | 975 |
| `full_causal_trace` | `CounterfactualIntervention` | `introspection/interventions.py` | 975 |
| `extract_direction` | `DirectionExtractor` | `introspection/circuit/directions.py` | 489 |
| `discover_neurons` | `NeuronAnalysisService` | `introspection/steering/neuron_service.py` | 290 |
| `analyze_neuron` | `NeuronAnalysisService` | `introspection/steering/neuron_service.py` | 290 |
| `residual_decomposition` | `ModelAnalyzer` | `introspection/analyzer/core.py` | 200+ |
| `layer_clustering` | `LayerAnalyzer` | `introspection/layer_analysis.py` | 548 |
| `measure_confidence` | `UncertaintyAnalysis` | `introspection/models/uncertainty.py` | 103 |
| `detect_format_gate` | `MetacognitiveResult` | `introspection/models/uncertainty.py` | 103 |
| `inject_fact` | `ExternalMemory` | `introspection/external_memory.py` | 723 |
| `query_with_memory` | `ExternalMemory` | `introspection/external_memory.py` | 723 |

### Phase 1c/1d — implementation notes

| Tool | Backend | Status |
|------|---------|--------|
| `logit_lens` (calibrated) | `residual_tools._get_lm_projection()` + `ModelHooks._get_final_norm()` | ✅ |
| `attribution_sweep` | Loop `_logit_attribution_impl()` + prompt_summary post-processing | ✅ |
| `create/get_experiment` | `ExperimentStore` singleton, filesystem JSON | ✅ |
| `track_race` | `ModelHooks` + `_norm_project` per candidate per layer | ✅ |
| `component_intervention` | Manual forward with `_run_forward_with_intervention` | ✅ |
| `probe_at_inference` | `ModelHooks.forward()` per step + sklearn `predict_proba` | ✅ |
| `neuron_trace` | `_run_decomposition_forward` + SwiGLU + cosine similarity | ✅ |
