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

---

## Phase 1b -- chuk-mcp-lazarus (Extended)

Deeper interpretability tools that unlock neuron-level analysis,
causal tracing, direction extraction, and residual decomposition.
These turn lazarus from a probing/steering toolkit into a full
mechanistic interpretability research platform capable of running
the revolutionary experiments described below.

**Implementation is prioritized by how many revolutionary experiments
each tool unblocks.** All backends exist in chuk-lazarus.

### Step 7: Neuron Analysis

Map the neuron landscape -- discover which individual neurons encode
specific features.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `discover_neurons` | Auto-find neurons that discriminate between prompt groups (e.g. which neurons fire for French vs English) | `NeuronAnalysisService.auto_discover_neurons()` |
| `analyze_neuron` | Profile a specific neuron: activation stats, separation score, group means across prompts | `NeuronAnalysisService.analyze_neurons()` |

**Unlocks:** Automated Circuit Discovery, Computational Stratigraphy,
Fine-Tuning Delta, Metacognition Probing, Universal Circuits.

### Step 8: Logit Lens ✅

Watch the model think -- track how predictions evolve layer by layer.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `logit_lens` | Top-k predictions at each layer, find where model "decides" | `ModelHooks.get_layer_logits()` |
| `track_token` | Track a specific token's probability and rank across layers | `ModelHooks.get_layer_logits()` |

> **Note:** `logit_lens` implemented in v0.4.0. `track_token` added in v0.5.0.
> `layer_predictions` (full vocab at each layer) deferred — `logit_lens` already returns top-k per layer.

### Step 9: Attention Analysis ✅

See what attends to what -- understand information routing.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `attention_pattern` | Extract full attention matrix at a layer (which tokens attend to which) | Manual Q*K + RoPE via `_compare._compute_attention_weights()` |
| `attention_heads` | Per-head entropy and focus analysis -- find which heads are doing the work | Entropy from attention weights |

### Step 10: Causal Tracing ⭐ HIGH PRIORITY

Trace causal circuits -- identify which components are responsible
for specific predictions. **This is the single most important missing
capability.** Probes show correlation; causal tracing shows causation.

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

### Step 11: Direction Extraction ⭐ HIGH PRIORITY

Find interpretable directions in activation space using multiple
methods beyond simple mean-difference. **Second most important
capability** -- enables understanding what features the model
represents and how they change with fine-tuning.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `extract_direction` | Find directions via mean-diff, LDA, PCA, probe weights, or contrastive methods | `DirectionExtractor` with `DirectionMethod` enum |

**Unlocks:** Computational Stratigraphy, Fine-Tuning Delta,
Metacognition Probing, Universal Circuits, Safety Patching.

**Backend:** `DirectionExtractor` in `introspection/circuit/directions.py`
(489 lines). `DirectionMethod` enum: DIFFERENCE_OF_MEANS, LDA,
PROBE_WEIGHTS, CONTRASTIVE, PCA. Returns `ExtractedDirection` with
separation score (Cohen's d), classification accuracy, mean projections.

### Step 12: Residual Stream Dynamics

Understand how information flows through the residual stream.

| Tool | Purpose | Backed by |
|------|---------|-----------|
| `residual_decomposition` | Separate attention vs MLP contribution to the residual stream per layer | `ModelAnalyzer.compute_residual_decomposition()` |
| `layer_clustering` | Cluster prompts by representation similarity at each layer, measure separability | `LayerAnalyzer.analyze_representations()` |

**Unlocks:** Automated Circuit Discovery, Computational Stratigraphy,
Universal Circuits.

**Backend:** `ModelAnalyzer` in `introspection/analyzer/core.py` (async-native).
`LayerAnalyzer` in `introspection/layer_analysis.py` (548 lines).

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

## Revolutionary Experiments

These are the experiments that become possible as Phase 1b tools
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

**Tool chain:** `logit_lens` → `track_token` → `full_causal_trace` →
`residual_decomposition` → `attention_pattern` → `discover_neurons` →
`patch_activations`

**Requires:** Steps 10, 7, 12

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
(logit lens), whether it's necessary (ablation), what direction
the feature lives in (directions), how many neurons encode it
(neurons), and whether attention or MLP is responsible (decomposition).

**Why revolutionary:** No complete multi-method stratigraphy exists
for any 4B+ model. Combines 6 independent measurement methods.

**Tool chain:** `scan_probe_across_layers` + `logit_lens` +
`ablate_layers` + `extract_direction` + `discover_neurons` +
`residual_decomposition`

**Requires:** Steps 7, 11, 12

### Experiment 4: The Fine-Tuning Delta Problem

**Goal:** Mechanistic answer to "what did fine-tuning change?"
Not just output differences, but which neurons, directions,
attention heads, and causal circuits changed.

**Why revolutionary:** Transforms model development from empirical
("train and benchmark") to mechanistic ("train and understand").

**Tool chain:** `compare_generations` → `compare_weights` →
`compare_representations` → `extract_direction` (both models) →
`discover_neurons` (both models) → `trace_token` (both models)

**Requires:** Steps 10, 11, 7

### Experiment 5: Does the Model Know What It Knows?

**Goal:** Test whether neural networks have internal representations
of their own uncertainty. If a probe can decode "this model will
get this wrong" from hidden states, the model has metacognition.

**Why revolutionary:** Directly tests self-knowledge in neural
networks. If positive, enables hallucination detection at the
representation level.

**Tool chain:** `scan_probe_across_layers` (knows/doesn't-know labels) →
`extract_direction` (certainty direction) → `measure_confidence` →
`detect_format_gate` → `steer_and_generate` (along uncertainty direction)

**Requires:** Steps 11, 13

### Experiment 6: Cross-Model Universal Circuits

**Goal:** Do all transformer architectures discover the same
computational primitives? Run identical stratigraphy on Gemma,
Llama, and Qwen. Normalize by relative depth and compare.

**Why revolutionary:** Tests whether there are universal laws of
neural network computation, or if each architecture finds its own
solution.

**Tool chain:** Full stratigraphy (Experiment 3) × 3 models,
then cross-model comparison of probe curves, emergence layers,
neuron counts, and attention-vs-MLP ratios.

**Requires:** Steps 7, 11, 12

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
`inject_fact` → `query_with_memory` → `generate_text` (verify)

**Requires:** Steps 10, 11, 14

### Experiment Priority Matrix

| Tool (Step) | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | Total |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|
| `trace_token` / `full_causal_trace` (10) | ✓ | ✓ | ✓ | ✓ | | | ✓ | **5** |
| `extract_direction` (11) | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **5** |
| `discover_neurons` / `analyze_neuron` (7) | ✓ | | ✓ | ✓ | ✓ | ✓ | | **5** |
| `residual_decomposition` (12) | ✓ | | ✓ | | | ✓ | | **3** |
| `layer_clustering` (12) | | | ✓ | | | ✓ | | **2** |
| `measure_confidence` (13) | | | | | ✓ | | | **1** |
| `detect_format_gate` (13) | | | | | ✓ | | | **1** |
| `inject_fact` / `query_with_memory` (14) | | ✓ | | | | | ✓ | **2** |

**Implementation order:** Steps 10 → 11 → 7 → 12 → 14 → 13

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

| Version | Scope |
|---------|-------|
| 0.1.0 | Phase 1 Steps 1--5 (demo-ready: load, extract, probe, steer, ablate) |
| 0.2.0 | Phase 1 Step 6 (resources, polish, end-to-end demo) |
| 0.3.0 | Phase 1 Step 6b (model comparison tools, two-model analysis) |
| 0.4.0 | Phase 1 Step 6c (generation, prediction, tokenize, logit lens, compare_generations) |
| 0.5.0 | Phase 1b Steps 8--9 (track_token, attention_pattern, attention_heads) |
| 0.6.0 | Phase 1b Step 10 (causal tracing: trace_token, full_causal_trace) |
| 0.7.0 | Phase 1b Step 11 (direction extraction: extract_direction) |
| 0.8.0 | Phase 1b Steps 7, 12 (neurons, residual decomposition, layer clustering) |
| 0.9.0 | Phase 1b Steps 13--14 (confidence, metacognition, external memory) |
| 0.10.0 | Phase 2 (tokenizer server) |
| 0.11.0 | Phase 3 (introspect server) |
| 0.12.0 | Phase 4 core (MoE: routing, ablation, identification) |
| 0.12.x | Phase 4 extended (MoE: circuits, compression, dynamics) |
| 1.0.0 | State persistence, multi-model sessions, production hardening |

---

## Tool Count Summary

| Phase | Server | Tools |
|-------|--------|-------|
| 1 | lazarus (core) | 26 |
| 1b | lazarus (extended) | ~13 |
| 2 | tokenizer | 11 |
| 3 | introspect | 7 |
| 4 | moe | 20 |
| **Total** | | **~77** |

---

## chuk-lazarus Backend Coverage

All Phase 1b tools have production-ready backends in chuk-lazarus.
No new backend development is required -- only MCP wrapping.

| Phase 1b Tool | Backend Class | File | Lines |
|---------------|---------------|------|-------|
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
