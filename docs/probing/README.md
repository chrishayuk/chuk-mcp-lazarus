# Probing

Train and evaluate linear probes on model activations.  Core tool for the language-transition experiment.

---

## `train_probe`

```python
async def train_probe(
    probe_name: str,
    layer: int,
    examples: list[dict],
    probe_type: str = "linear",
    token_position: int = -1,
) -> dict
```

Train a probe classifier (LogisticRegression or MLP) on hidden states at a specified layer.

**`examples` format:** `[{"prompt": str, "label": str}, ...]`

**Returns** `{probe_name, layer, probe_type, num_examples, classes, train_accuracy, val_accuracy, coefficients_norm}`.

---

## `evaluate_probe`

```python
async def evaluate_probe(
    probe_name: str,
    examples: list[dict],
    token_position: int = -1,
) -> dict
```

Evaluate a stored probe on new examples.  Returns accuracy, per-class accuracy, and a confusion matrix.

---

## `scan_probe_across_layers`

```python
async def scan_probe_across_layers(
    probe_name_prefix: str,
    layers: list[int],
    examples: list[dict],
    probe_type: str = "linear",
    token_position: int = -1,
) -> dict
```

Train and evaluate a probe at every specified layer in a single call.  **Primary tool for finding the language transition crossover layer.**

- Caches activations: each example runs **one** forward pass through all layers, avoiding N × redundant forward passes.
- Creates probes named `"{probe_name_prefix}_L{layer}"`.
- Returns `accuracy_by_layer`, `peak_layer`, `peak_val_accuracy`, `crossover_layer`, and `interpretation`.

---

## `list_probes`

```python
async def list_probes() -> dict
```

List all probes stored in `ProbeRegistry`.

---

### Workflow: language transition experiment

```python
# 1. Build examples: 20+ per language
examples = [
    {"prompt": "Translate to French: Hello", "label": "english"},
    {"prompt": "Translate to French: Goodbye", "label": "english"},
    {"prompt": "Traduire en anglais: Bonjour", "label": "french"},
    # ... more examples
]

# 2. Scan all 34 layers at once
result = await scan_probe_across_layers(
    probe_name_prefix="lang",
    layers=list(range(34)),
    examples=examples,
)
crossover = result["crossover_layer"]  # e.g. 18

# 3. Evaluate on held-out data
held_out = [...]
eval_result = await evaluate_probe(f"lang_L{crossover}", held_out)
```

Then use `probe_at_inference` (see [Generation](../generation/)) to monitor the probe during live generation.
