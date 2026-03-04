# Experiment Use Cases

Experiments and discoveries an AI agent can make autonomously using
chuk-mcp-lazarus's 34 tools. Each experiment is a multi-step workflow
where the agent forms hypotheses, designs tests, and iterates based
on results.

---

## Single-Model Experiments

### 1. Knowledge Localization

**Question:** At which layer does the model "know" that Paris is the
capital of France?

**Tools:** `logit_lens` → `track_token` → `logit_attribution` → `head_attribution` → `top_neurons` → `ablate_layers`

**Workflow:**
1. `logit_lens("The capital of France is")` — see when "Paris" first
   appears in the top-k predictions at each layer
2. `track_token("The capital of France is", token="Paris")` — watch
   "Paris" probability and rank climb from layer 0 to the final layer
3. `logit_attribution` — find which layers' attention and MLP components
   contribute most to the "Paris" logit
4. `head_attribution` at the peak attention layer — which specific heads
   push toward "Paris"?
5. `top_neurons` at the peak MLP layer — which individual neurons
   store the France→Paris knowledge?
6. `ablate_layers` at the critical layer — if ablation destroys the
   answer, that layer is causal, not just correlated

**Possible discoveries:**
- Factual knowledge concentrates in specific layer bands
- Some facts emerge early (common knowledge) vs late (rare knowledge)
- MLP layers store facts, attention layers route them
- The "emergence layer" from `track_token` and the "peak probe layer"
  from `scan_probe_across_layers` may differ — knowledge can be
  present (decodable) before it reaches the prediction head

---

### 2. Language Transition Mapping

**Question:** How does the model process multilingual input? Where
does language identity live?

**Tools:** `tokenize` → `compare_activations` → `scan_probe_across_layers` → `attention_heads`

**Workflow:**
1. `tokenize` multilingual prompts to understand token boundaries
   (e.g., "Bonjour" is one token, "Guten Tag" is two)
2. `compare_activations` at layer 0 — representations should be
   language-distinct (low cosine similarity)
3. `compare_activations` at the final layer — representations should
   converge (high cosine similarity) if the model maps to a shared
   semantic space
4. `scan_probe_across_layers` with language labels — find the exact
   crossover where language identity becomes decodable
5. `attention_heads` at the crossover layer — which heads are most
   focused? Those focused heads may be translation circuitry

**Possible discoveries:**
- The "language-independent meaning" layer for different architectures
- Whether translation models have sharper crossover points
- Specific attention heads that act as "language bridge" components
- Language identity may be decodable at ALL layers (even final) but
  the representation geometry changes from "surface form" to
  "semantic role"

---

### 3. Behavioral Steering

**Question:** Can I redirect the model's behavior by adding a
direction in activation space?

**Tools:** `compute_steering_vector` → `steer_and_generate` → `track_token`

**Workflow:**
1. `compute_steering_vector` with contrastive prompts (e.g.,
   French→German, formal→informal, truthful→hallucinating)
2. `steer_and_generate` with an alpha sweep (0, 5, 10, 15, 20, 30)
   to find the sweet spot where behavior changes without degeneracy
3. `track_token` on the steered model to see how the probability
   landscape shifts — does steering change *when* the target token
   emerges, or just *how strongly*?
4. Compare `logit_lens` on baseline vs steered — at which layer does
   the steered model first diverge from baseline predictions?

**Possible discoveries:**
- Optimal steering strength varies by concept type (sentiment needs
  alpha=5, language needs alpha=20)
- Steering works by shifting emergence layers earlier or later
- Some directions are "clean" (change one thing), others are "messy"
  (corrupt everything above a threshold)
- The separability score from `compute_steering_vector` predicts
  how effective steering will be

---

### 4. Attention Circuit Discovery

**Question:** What is attention head 15.6 doing? Is it part of a
recognizable circuit?

**Tools:** `attention_pattern` → `attention_heads` → `ablate_layers` → `generate_text`

**Workflow:**
1. `attention_heads` across all layers — find the most focused heads
   (lowest entropy) and most diffuse heads (highest entropy)
2. `attention_pattern` on the focused heads — does head 15.6 always
   attend to the subject token? The previous token? A specific
   syntactic position?
3. Test across multiple prompts with `attention_pattern` — does the
   pattern generalize or is it prompt-specific?
4. `ablate_layers` at layer 15 — does subject-verb agreement break?
   Does the model lose the ability to copy from context?
5. `generate_text` before and after ablation to confirm behavioral
   impact

**Possible discoveries:**
- Induction heads (copy from context by attending to previous
  occurrences of the current token)
- Name mover heads (route entity names to the prediction position)
- Syntactic heads (attend to grammatically related positions)
- Redundant vs critical heads (some ablations do nothing, others
  destroy output)
- Head function can change with context — a head may be an induction
  head for English but a position head for code

---

### 5. Layer Contribution Mapping

**Question:** What does each layer contribute to the final output?

**Tools:** `extract_activations` → `logit_lens` → `ablate_layers`

**Workflow:**
1. `extract_activations` at layers N-1 and N — compute the delta to
   understand what layer N adds to the residual stream
2. `logit_lens` at every layer — watch predictions evolve and identify
   "decision points" where the top prediction changes
3. `ablate_layers` one layer at a time — measure disruption score to
   rank layers by importance
4. `ablate_layers` on MLP vs attention components separately — which
   component matters more at each layer?

**Possible discoveries:**
- Early layers handle syntax, middle layers handle semantics, late
  layers handle formatting/style
- Some layers are "refinement" layers (small prediction changes),
  others are "decision" layers (top prediction flips)
- Layer clusters that function as computational units
- The model may have "bypass" layers that contribute very little
  (candidates for pruning)

---

### 6. Tokenization Effects on Model Behavior

**Question:** How does tokenization affect what the model can do?

**Tools:** `tokenize` → `attention_pattern` → `predict_next_token` → `track_token`

**Workflow:**
1. `tokenize` the same concept in different phrasings — find cases
   where tokenization splits words differently
2. `attention_pattern` on multi-token words — can the model attend
   across token boundaries within a word?
3. `predict_next_token` for prompts where the answer requires
   understanding a multi-token entity
4. `track_token` on each sub-token of a multi-token target — do
   they emerge at the same layer or sequentially?

**Possible discoveries:**
- Multi-token entities have delayed emergence compared to
  single-token equivalents
- Attention patterns differ for whole-word tokens vs sub-word pieces
- The model handles some tokenization splits better than others
  (common splits vs rare splits)

---

## Two-Model Experiments

### 7. Fine-Tuning Forensics

**Question:** What did fine-tuning actually change inside the model?

**Tools:** `load_comparison_model` → `compare_generations` → `compare_weights` → `compare_representations` → `compare_attention`

**Workflow:**
1. `load_comparison_model` — load base + fine-tuned variant
2. `compare_generations` on targeted prompts — see actual output
   differences (languages where fine-tuning helps most)
3. `compare_weights` — find which layers and components (attention
   vs MLP) diverged most in weight space
4. `compare_representations` on the same prompts — find which layers
   show the most activation divergence
5. `compare_attention` — find which heads rewired their attention
   patterns

**Possible discoveries:**
- Fine-tuning changes later layers more than early layers
- Weight changes concentrate in MLP (knowledge) vs attention (routing)
- Low-resource language improvements correlate with specific layer bands
- Embedding layers may be frozen (TranslateGemma confirms this)
- Some heads completely change their attention pattern post-fine-tuning

---

### 8. Knowledge vs Routing

**Question:** Is the fine-tuned model better because it *knows* more,
or because it *routes* better?

**Tools:** `scan_probe_across_layers` (both models) → `compare_weights` → `compare_attention`

**Workflow:**
1. Load both models and run `scan_probe_across_layers` with identical
   examples on each — if probe accuracy is the same, the knowledge
   exists in both models
2. If probes are equally accurate: the fine-tuned model's advantage
   is routing (it uses the knowledge better). Confirm with
   `compare_attention` — attention patterns should differ
3. If probes differ: the fine-tuning changed what's stored. Confirm
   with `compare_weights` — MLP weights should show large divergence
4. Test across high-resource (French) and low-resource (Icelandic)
   languages — the answer may differ by language

**Possible discoveries:**
- Translation fine-tuning may not add new language knowledge for
  high-resource languages, just better routing
- Low-resource languages may require actual knowledge changes
- The knowledge-vs-routing distinction is layer-dependent: early
  layers route similarly, late layers diverge

---

### 9. Architecture Comparison

**Question:** Do different model families develop the same internal
structure?

**Tools:** `load_model` (different families) → `scan_probe_across_layers` → `logit_lens` → `attention_heads`

**Workflow:**
1. Load Model A (e.g., Gemma 4B), run probe scan and logit lens
2. Unload, load Model B (e.g., Llama 3B), run the same experiments
3. Compare: do both models develop language identity at the same
   relative depth (e.g., 60% through)?
4. Compare attention head entropy distributions — do architectures
   with more heads have more specialized heads?

**Possible discoveries:**
- Universal patterns: all transformer models develop language
  identity in the first third and factual knowledge in the middle
- Architecture-specific patterns: GQA models may have different
  head specialization profiles than MHA models
- Model size effects: larger models may develop features at
  relatively earlier layers

---

## Compositional Experiments

### 10. Full Interpretability Report

**Question:** Explain everything about how the model answers this
one prompt.

**Tools:** `tokenize` → `generate_text` → `logit_lens` → `track_token` → `attention_pattern` → `attention_heads` → `extract_activations`

**Workflow:**
1. `tokenize` — understand the input structure
2. `generate_text` — see the actual output
3. `logit_lens` — watch prediction evolution layer by layer
4. `track_token` on the output token — find the emergence layer
5. `attention_heads` — identify which heads are most active
6. `attention_pattern` on the active heads — see what they attend to
7. `extract_activations` at key layers — capture the representation
   for further analysis

This produces a complete profile: "The model tokenized the input into
8 tokens, generated 'Paris' as the answer, which first emerged at
layer 18. The most focused head (15.6) attends strongly to 'France'
at position 4, suggesting a subject→answer routing circuit."

---

### 11. Hypothesis-Driven Circuit Tracing

**Question:** Is there a circuit responsible for [specific behavior]?

**Tools:** (all tools, iteratively)

**Workflow:**
1. **Observe:** `generate_text` to identify the behavior
2. **Hypothesize:** "Head 12.3 routes the subject to the verb"
3. **Test correlation:** `attention_pattern` — does 12.3 attend
   subject→verb consistently across prompts?
4. **Test necessity:** `ablate_layers` at layer 12 — does agreement
   break?
5. **Test sufficiency:** `patch_activations` — swap the subject
   between prompts, does the verb change?
6. **Test direction:** `compute_steering_vector` on the circuit's
   activations, `steer_and_generate` to flip the behavior
7. **Iterate:** refine the hypothesis based on results

This is the full mechanistic interpretability research loop,
automated.

---

### 12. Steering Vector Quality Assessment

**Question:** How reliable is a steering vector? Does it generalize?

**Tools:** `compute_steering_vector` → `steer_and_generate` → `scan_probe_across_layers` → `compare_activations`

**Workflow:**
1. `compute_steering_vector` — note the separability score
2. `steer_and_generate` on held-out prompts (not used to compute the
   vector) — does it generalize?
3. `steer_and_generate` on adversarial prompts (designed to resist
   steering) — how robust is it?
4. `scan_probe_across_layers` with steered/unsteered labels — is the
   steering effect localized to one layer or distributed?
5. `compare_activations` between steered and unsteered runs — measure
   how much of activation space changed

**Possible discoveries:**
- Separability score predicts generalization
- Steering vectors computed at later layers generalize better
- Some concepts are "steer-resistant" (require multiple vectors)
- Steering at one layer propagates differently through the network
  depending on the concept

---

## Experiment Design Principles

When designing experiments with these tools, the agent should:

1. **Start with observation** — `generate_text`, `logit_lens`,
   `tokenize` to understand baseline behavior
2. **Form a specific hypothesis** — not "what happens if I ablate?"
   but "layer 20's MLP stores the fact that Paris is in France"
3. **Test with multiple methods** — correlation (probes) ≠ causation
   (ablation). Use both.
4. **Control for confounds** — compare across prompts, languages,
   and models. One data point is an anecdote.
5. **Iterate** — unexpected results are more interesting than
   confirmations. Follow them.

---

## Tool Composition Matrix

Which tools pair well together:

| First Tool | Then Use | To Answer |
|------------|----------|-----------|
| `logit_lens` | `track_token` | When does the model "decide" on a specific answer? |
| `logit_lens` | `ablate_layers` | Is the decision layer causal or correlational? |
| `logit_attribution` | `head_attribution` | Which specific attention heads drive the prediction? |
| `logit_attribution` | `top_neurons` | Which MLP neurons store the knowledge? |
| `head_attribution` | `attention_pattern` | What is the top head actually attending to? |
| `scan_probe_across_layers` | `ablate_layers` | Is the probed feature used or just present? |
| `attention_heads` | `attention_pattern` | What are the focused heads actually attending to? |
| `attention_pattern` | `patch_activations` | Is the attention pattern functionally important? |
| `compute_steering_vector` | `track_token` | How does steering change token emergence? |
| `compare_generations` | `compare_representations` | Where do output differences originate? |
| `compare_weights` | `compare_attention` | Did weight changes rewire attention? |
| `extract_activations` | `compare_activations` | How do representations differ across conditions? |
| `tokenize` | `attention_pattern` | Which tokens attend across word boundaries? |
| `trace_token` | `residual_decomposition` | Is the causal layer driven by attention or MLP? |
| `embedding_neighbors` | `top_neurons` | Do neurons push toward semantically related tokens? |
