# Generation and Prediction

Eight tools for text generation, token prediction, tokenization, and probing during inference.

---

## `generate_text`

```python
async def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> dict
```

Generate text from the loaded model.  Temperature 0 = greedy decoding.

---

## `predict_next_token`

```python
async def predict_next_token(
    prompt: str,
    top_k: int = 10,
) -> dict
```

Get the model's top-k next-token predictions with probabilities.  No text generated — just the distribution over the next token.

---

## `tokenize`

```python
async def tokenize(text: str) -> dict
```

Show how the model's tokenizer splits text.  Returns token IDs, decoded text per token, and positions.  Essential for correctly specifying `token_position` in other tools.

---

## `logit_lens`

```python
async def logit_lens(
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 5,
    token_position: int = -1,
) -> dict
```

Project hidden states at each layer back to vocabulary space.  Shows what the model would predict if computation stopped at each layer.  The `emergence_layer` field reports where the final prediction first appears.

---

## `track_token`

```python
async def track_token(
    prompt: str,
    token: str,
    layers: list[int] | None = None,
) -> dict
```

Track a specific token's probability and rank across all layers using the logit lens.  Returns `emergence_layer`, `peak_layer`, and per-layer `{probability, rank, is_top1}`.

---

## `track_race`

```python
async def track_race(
    prompt: str,
    candidates: list[str],
    layers: list[int] | None = None,
    token_position: int = -1,
) -> dict
```

Race 2–20 candidate tokens across layers.  Reports crossing events (where the leading candidate changes) and the final winner.  Great for understanding competition between alternatives in factual recall.

---

## `embedding_neighbors`

```python
async def embedding_neighbors(
    token: str,
    top_k: int = 10,
) -> dict
```

Find nearest tokens in static embedding space by cosine similarity.  No forward pass — operates directly on the embedding matrix.

---

## `probe_at_inference`

```python
async def probe_at_inference(
    prompt: str,
    probe_name: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    token_position: int = -1,
) -> dict
```

Run a trained probe during autoregressive generation to monitor the model's internal state token-by-token.  Requires a probe trained with `train_probe` (see [Probing](../probing/)).

**Returns** `per_token: [{step, token, probe_prediction, probe_confidence, probe_probabilities}]` and `overall_majority_class`.
