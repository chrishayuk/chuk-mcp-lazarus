# Revolutionary Experiments

What would genuinely advance the field of mechanistic interpretability,
and what tools does an autonomous agent need to run those experiments?

This document distinguishes between experiments that are **incremental**
(applying known techniques) and experiments that would be **revolutionary**
(producing new understanding of how neural networks work). For each
revolutionary experiment, we identify the tools that exist today, the
tools we need to build, and the chuk-lazarus backend that supports them.

---

## The Gap: What We Have vs What We Need

### What we have (34 tools)
Comprehensive layer-level and component-level observation and intervention.
We can see what each layer produces (`logit_lens`, `extract_activations`),
how layers relate (`compare_activations`, `attention_heads`), whether
features are linearly decodable (`scan_probe_across_layers`), what happens
when we remove or modify layers (`ablate_layers`, `steer_and_generate`),
trace causal circuits (`trace_token`, `full_causal_trace`), decompose the
residual stream (`residual_decomposition`, `logit_attribution`), drill into
specific components (`head_attribution`, `top_neurons`), and analyze
vocabulary geometry (`embedding_neighbors`).

### What's missing for revolutionary work
1. **Multiple direction methods** — We only have mean-difference
   steering vectors. LDA, PCA, and probe-weight directions capture
   different aspects of the same feature.
2. **Automated neuron discovery** — `top_neurons` identifies neurons
   for a specific prediction, but we can't auto-discover neurons that
   discriminate between prompt groups (e.g. which neurons fire for
   French vs English).
3. **Confidence geometry** — We can't measure whether the model
   "knows that it knows" something.
4. **External memory** — Runtime knowledge editing without fine-tuning.

### What chuk-lazarus already provides

Some capabilities are now exposed as MCP tools; others remain unexposed:

| Capability | Status | MCP Tool / chuk-lazarus class |
|---|---|---|
| Position × layer patching | ✅ Exposed | `full_causal_trace`, `patch_activations` |
| Causal tracing | ✅ Exposed | `trace_token`, `full_causal_trace` |
| Residual decomposition | ✅ Exposed | `residual_decomposition`, `logit_attribution` |
| Per-head attribution | ✅ Exposed | `head_attribution` |
| Per-neuron identification | ✅ Exposed | `top_neurons` |
| Embedding neighbors | ✅ Exposed | `embedding_neighbors` |
| Neuron group discovery | Not yet | `NeuronAnalysisService` in `introspection/steering/neuron_service.py` |
| Direction extraction (5 methods) | Not yet | `DirectionExtractor` in `introspection/circuit/directions.py` |
| Confidence geometry | Not yet | `UncertaintyAnalysis` in `introspection/models/uncertainty.py` |
| Circuit geometry (PCA, probes) | Not yet | `GeometryAnalyzer` in `introspection/circuit/geometry.py` |
| External memory injection | Not yet | `ExternalMemory` in `introspection/external_memory.py` |

---

## Revolutionary Experiment 1: Automated Circuit Discovery

### Why it's revolutionary
Today, circuit discovery is manual. A researcher picks a behavior,
guesses which components matter, and tests one at a time. This takes
weeks per circuit. An agent with the right tools could **systematically
trace a circuit in minutes**: find the critical (position, layer,
component) tuples, verify causality, and describe the circuit in
human-readable terms.

No one has demonstrated fully autonomous circuit discovery on models
larger than GPT-2 small. Doing it on 4B+ parameter models would be
a significant result.

### The experiment
Given any prompt + target behavior (e.g., "The model predicts 'Paris'
when asked about France's capital"):

1. **Narrow the search space** — `logit_lens` + `track_token` to find
   layers where the prediction crystallizes
2. **Causal trace** — `full_causal_trace` to produce a position × layer
   heatmap showing which (position, layer) pairs are causally important
3. **Decompose** — `residual_decomposition` to separate attention vs MLP
   contribution at causal layers
4. **Identify heads** — `attention_pattern` at causal layers to find
   which heads route information
5. **Identify neurons** — `discover_neurons` at causal MLP layers to
   find which neurons store the knowledge
6. **Verify** — `patch_activations` at the discovered positions to
   confirm the circuit is sufficient
7. **Describe** — Agent produces: "Token 'France' at position 4 is
   attended to by head 18.3, which routes to MLP layer 20 neurons
   [1422, 3891, 7102], which write 'Paris' into the residual stream"

### Tools available today
`logit_lens`, `track_token`, `full_causal_trace`, `trace_token`,
`residual_decomposition`, `logit_attribution`, `head_attribution`,
`top_neurons`, `attention_pattern`, `attention_heads`, `ablate_layers`,
`patch_activations`, `extract_activations`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `discover_neurons` | Find discriminative neurons at a layer | `NeuronAnalysisService.auto_discover_neurons()` |
| `analyze_neuron` | Profile a specific neuron's behavior | `NeuronAnalysisService.analyze_neurons()` |

---

## Revolutionary Experiment 2: Knowledge Surgery Without Fine-Tuning

### Why it's revolutionary
Current model editing (ROME, MEMIT) requires gradient computation and
modifies weights permanently. chuk-lazarus has an `ExternalMemory`
system that can inject facts into the residual stream at inference
time — no fine-tuning, no gradients, fully reversible. If an agent
can autonomously: (1) identify where a fact is stored, (2) construct
a correction vector, and (3) inject it at the right layer with the
right blend — that's runtime knowledge editing.

This has direct safety implications. If a model has a factual error
or harmful association, an agent could patch it in real time.

### The experiment
1. **Find the wrong answer** — `generate_text("The capital of
   Myanmar is")` → model says "Rangoon" (outdated)
2. **Locate the knowledge** — `track_token` for "Naypyidaw" to find
   where the correct answer *would* emerge (if it does at all)
3. **Causal trace** — `trace_token` to find which layers are
   responsible for the wrong answer
4. **Extract correction direction** — `extract_direction` using
   prompts where the model gets similar facts right vs wrong
5. **Inject correction** — `inject_fact` to add the correct
   fact to external memory at the causal layer
6. **Verify** — `generate_text` again — does it now say "Naypyidaw"?
7. **Test side effects** — `generate_text` on unrelated prompts —
   did the injection break anything?

### Tools available today
`generate_text`, `track_token`, `trace_token`, `logit_lens`,
`logit_attribution`, `top_neurons`, `ablate_layers`,
`compute_steering_vector`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `extract_direction` | Find correction direction via multiple methods | `DirectionExtractor.extract_direction()` |
| `inject_fact` | Add fact to external memory for runtime injection | `ExternalMemory.add_facts()` |
| `query_with_memory` | Generate with external memory active | `ExternalMemory.query()` |

---

## Revolutionary Experiment 3: Computational Stratigraphy

### Why it's revolutionary
Geologists read Earth's history from rock layers. The same principle
applies to transformer layers — each one adds something to the
computation. "Computational stratigraphy" would produce a complete
map of **what each layer computes** for a given task, using multiple
independent measurement methods.

No one has produced a complete multi-method stratigraphy for a 4B+
model. Existing work uses one method (probes OR logit lens OR
ablation) — combining all three with neuron-level resolution and
direction extraction would be genuinely new.

### The experiment
For a given task (e.g., translation, factual recall, arithmetic):

1. **Probe stratigraphy** — `scan_probe_across_layers` with task
   labels at every layer. Produces an accuracy curve showing where
   task-relevant information becomes linearly decodable.
2. **Logit lens stratigraphy** — `logit_lens` at every layer.
   Produces a prediction evolution curve. Compare: does the probe
   say "information is here" before or after the logit lens says
   "the model is predicting it"?
3. **Causal stratigraphy** — `trace_token` or per-layer `ablate_layers`.
   Produces a necessity curve: which layers, if removed, break the
   behavior?
4. **Direction stratigraphy** — `extract_direction` at every layer
   using multiple methods (mean-diff, LDA, PCA, probe weights).
   Do all methods agree on when the feature emerges?
5. **Neuron stratigraphy** — `discover_neurons` at key layers.
   How many neurons encode the feature at each layer? Does it
   concentrate or distribute?
6. **Decomposition stratigraphy** — `residual_decomposition` at
   every layer. Is the feature written by attention or MLP at
   each stage?

The result is a multi-layer visualization: for each layer, you
know what information it contains (probes), what it predicts
(logit lens), whether it's necessary (ablation), what direction
the feature lives in (directions), how many neurons encode it
(neurons), and whether attention or MLP is responsible
(decomposition).

### Tools available today
`scan_probe_across_layers`, `logit_lens`, `track_token`,
`trace_token`, `residual_decomposition`, `logit_attribution`,
`head_attribution`, `top_neurons`, `layer_clustering`,
`ablate_layers`, `extract_activations`, `attention_heads`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `extract_direction` | Direction via diff-means, LDA, PCA, probe weights | `DirectionExtractor` |
| `discover_neurons` | Find feature-encoding neurons per layer | `NeuronAnalysisService` |

---

## Revolutionary Experiment 4: The Fine-Tuning Delta Problem

### Why it's revolutionary
When you fine-tune a model, you change weights. But *what did you
actually change about the computation*? Current evaluation compares
outputs (benchmarks). No one systematically answers: "Fine-tuning
changed neurons [X, Y, Z] at layers [18, 19, 20], which encode
[low-resource translation knowledge], routed by attention heads
[18.3, 19.7], activated by the direction [v] in activation space."

Doing this would transform model development from "train and hope"
to "train and understand."

### The experiment
Using two models (base + fine-tuned):

1. **Output delta** — `compare_generations` on diverse prompts.
   Categorize: where does the fine-tuned model improve? Where
   does it regress? Where is it unchanged?
2. **Weight delta** — `compare_weights` to find which layers and
   components changed most. Hypothesis: changes concentrate in
   specific layer bands.
3. **Representation delta** — `compare_representations` to find
   where activations diverge. Key: divergence on improved prompts
   should be higher than on unchanged prompts.
4. **Attention delta** — `compare_attention` to find rewired heads.
5. **Direction delta** — `extract_direction` on both models at the
   same layer. Do they develop different feature directions?
   Compute cosine similarity between directions — if high, fine-tuning
   preserved the geometry but changed magnitudes. If low, it
   learned a genuinely new representation.
6. **Neuron delta** — `discover_neurons` on both models. Are the
   same neurons discriminative, or did fine-tuning recruit new ones?
7. **Causal delta** — `trace_token` on both models for the same
   prompt. Did the causal circuit change?

### Tools available today
`compare_generations`, `compare_weights`, `compare_representations`,
`compare_attention`, `scan_probe_across_layers`, `logit_lens`,
`track_token`, `trace_token`, `logit_attribution`, `head_attribution`,
`top_neurons`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `extract_direction` | Compare feature directions across models | `DirectionExtractor` |
| `discover_neurons` | Compare discriminative neurons across models | `NeuronAnalysisService` |

---

## Revolutionary Experiment 5: Does the Model Know What It Knows?

### Why it's revolutionary
Calibration measures whether model confidence matches accuracy. But
that's measured externally (output probabilities vs ground truth).
The revolutionary question is: **does the model have an internal
representation of its own uncertainty?** If yes, we can detect
hallucination before it reaches the output. If no, we learn something
fundamental about what self-knowledge means for neural networks.

### The experiment
1. **Build a fact dataset** — Prompts the model gets right vs wrong.
   Include edge cases: facts the model "almost" knows (low confidence
   correct answers) and facts it confidently gets wrong (hallucinations).
2. **Probe for metacognition** — `scan_probe_across_layers` with
   labels "knows" vs "doesn't know" (based on whether the model's
   answer is correct). If a probe can decode this, the model has
   internal uncertainty representations.
3. **Locate the uncertainty signal** — `extract_direction` for the
   "certain" vs "uncertain" direction. At which layer does this
   direction emerge?
4. **Geometric analysis** — `measure_confidence` to see if correct
   and incorrect answers occupy different regions of activation space.
5. **Strategy detection** — `detect_format_gate` to find if/where
   the model decides to use chain-of-thought vs direct answers.
   Hypothesis: the model uses CoT when it's uncertain.
6. **Intervention** — `steer_and_generate` along the uncertainty
   direction. Can we make a confident answer uncertain? Can we make
   an uncertain answer confident? Does steering toward "confident"
   on a wrong answer make the hallucination worse?

### Tools available today
`scan_probe_across_layers`, `compute_steering_vector`,
`steer_and_generate`, `logit_lens`, `track_token`,
`extract_activations`, `logit_attribution`, `top_neurons`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `extract_direction` | Find the "certainty" direction in activation space | `DirectionExtractor` |
| `measure_confidence` | Geometric uncertainty analysis | `UncertaintyAnalysis` |
| `detect_format_gate` | Find where model decides CoT vs direct | `MetacognitiveResult` |
| `discover_neurons` | Find neurons encoding uncertainty | `NeuronAnalysisService` |

---

## Revolutionary Experiment 6: Cross-Model Universal Circuits

### Why it's revolutionary
Do all transformer models discover the same circuits? If a
"subject→verb agreement head" exists in Gemma, does Llama have one
at the same relative depth? If yes, there may be universal
computational primitives that emerge from the training process
regardless of architecture. This would be a fundamental discovery
about how neural networks learn.

### The experiment
1. Load Model A (Gemma 4B), run full stratigraphy (Experiment 3)
   for a set of tasks (translation, factual recall, syntax)
2. Unload, load Model B (Llama 3B), run identical stratigraphy
3. Unload, load Model C (Qwen 2.5 3B), run identical stratigraphy
4. Compare: normalize layer indices to [0, 1] (relative depth)
   - Do probe accuracy curves have the same shape?
   - Do features emerge at the same relative depth?
   - Do the same number of neurons encode each feature?
   - Is the attention-vs-MLP contribution ratio similar?
5. **Direction alignment** — `extract_direction` for the same feature
   in all three models. Are the directions similar in structure
   (even though the spaces have different dimensions)?

### Tools available today
`scan_probe_across_layers`, `logit_lens`, `track_token`,
`attention_heads`, `ablate_layers`, `extract_activations`,
`compare_activations`, `residual_decomposition`, `layer_clustering`,
`logit_attribution`, `head_attribution`, `top_neurons`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `extract_direction` | Compare feature directions across architectures | `DirectionExtractor` |
| `discover_neurons` | Compare neuron counts per feature | `NeuronAnalysisService` |

---

## Revolutionary Experiment 7: Runtime Knowledge Patching for Safety

### Why it's revolutionary
If we can identify where harmful knowledge is stored (causal tracing)
and inject corrections at runtime (external memory), we have a
**real-time safety filter that operates at the representation level**,
not the output level. This is fundamentally different from output
filtering because it changes what the model *computes*, not just what
it *says*. The model doesn't "know the harmful thing but is told not
to say it" — the harmful association is replaced in the residual stream.

### The experiment
1. **Identify harmful circuit** — Find prompts that elicit harmful
   outputs. Use `full_causal_trace` to locate the (position, layer)
   tuple responsible.
2. **Extract harmful direction** — `extract_direction` with
   harmful vs safe responses as positive/negative examples.
3. **Compute correction** — The negative of the harmful direction,
   or a safe-response direction from the same layer.
4. **Inject at runtime** — `inject_fact` to add the correction to
   external memory. When the harmful pattern activates, the
   correction is injected.
5. **Verify safety** — `generate_text` with the harmful prompt.
   Does the model produce a safe response?
6. **Verify capability** — `generate_text` with benign prompts.
   Is the model still functional? Does it still answer factual
   questions correctly?
7. **Adversarial test** — Try rephrased harmful prompts. Does the
   protection generalize?

### Tools available today
`generate_text`, `full_causal_trace`, `trace_token`, `ablate_layers`,
`compute_steering_vector`, `steer_and_generate`,
`scan_probe_across_layers`, `logit_attribution`

### Tools still needed

| Tool | Purpose | Backend |
|------|---------|---------|
| `extract_direction` | Find harmful direction | `DirectionExtractor` |
| `inject_fact` | Runtime correction injection | `ExternalMemory` |
| `query_with_memory` | Generate with corrections active | `ExternalMemory` |

---

## Implementation Priority

Ordered by how many revolutionary experiments each tool unblocks:

| Priority | Tool | Status | Experiments Unblocked |
|----------|------|--------|----------------------|
| 1 | `trace_token` | ✅ v0.6.0 | 1, 2, 3, 4, 7 |
| 2 | `full_causal_trace` | ✅ v0.6.0 | 1, 3, 7 |
| 3 | `residual_decomposition` | ✅ v0.7.0 | 1, 3, 6 |
| 4 | `layer_clustering` | ✅ v0.7.0 | 3, 6 |
| 5 | `logit_attribution` | ✅ v0.8.0 | 1, 3, 4 |
| 6 | `head_attribution` | ✅ v0.9.0 | 1, 4 |
| 7 | `top_neurons` | ✅ v0.9.0 | 1, 3, 4 |
| 8 | `extract_direction` | **Next** | 3, 4, 5, 6, 7 |
| 9 | `discover_neurons` | Planned | 1, 3, 4, 5, 6 |
| 10 | `measure_confidence` | Planned | 5 |
| 11 | `inject_fact` / `query_with_memory` | Planned | 2, 7 |
| 12 | `detect_format_gate` | Planned | 5 |
| 13 | `analyze_neuron` | Planned | 1, 3 |

**Current state:** 7 of 13 priority tools are implemented. The remaining
high-priority items are `extract_direction` (unblocks 5 experiments) and
`discover_neurons` (unblocks 5 experiments).

---

## What Makes These "Revolutionary" vs "Incremental"

**Incremental:** applying a known technique (probing, ablation,
steering) to a new model or task. Useful, but expected results.

**Revolutionary:** combining multiple techniques to answer questions
that no single technique can answer:

- Automated circuit discovery (Experiment 1) — no one has done
  this autonomously on 4B+ models
- Knowledge surgery without fine-tuning (Experiment 2) — changes
  the paradigm from "retrain" to "patch at runtime"
- Computational stratigraphy (Experiment 3) — multi-method layer
  mapping at a level of detail that doesn't exist for any model
- Understanding fine-tuning (Experiment 4) — transforms model
  development from empirical to mechanistic
- Metacognition probing (Experiment 5) — directly tests whether
  neural networks have self-knowledge
- Universal circuits (Experiment 6) — tests whether there are
  computational universals across architectures
- Runtime safety patching (Experiment 7) — representation-level
  safety that operates on what the model computes, not what it says

The common thread: **these experiments require an agent that can
form hypotheses, design multi-step experiments, and iterate based
on results.** No single tool call produces a revolutionary result.
The revolution is in the composition.
