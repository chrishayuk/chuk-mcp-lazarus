# Model Management

Two tools for loading and inspecting a model.  Call `load_model` before any other tool.

---

## `load_model`

```python
async def load_model(
    model_id: str = "google/gemma-3-4b-it",
    dtype: str = "bfloat16",
) -> dict
```

Load a HuggingFace model into the `ModelState` singleton.

- Subsequent calls with the **same** `model_id` return immediately (idempotent).
- Calling with a **different** `model_id` unloads the previous model first.
- Supported architectures: Gemma, Llama, Qwen, Granite, Jamba, Mamba, StarCoder2, GPT-2.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_id` | `str` | `"google/gemma-3-4b-it"` | HuggingFace model ID or local path |
| `dtype` | `str` | `"bfloat16"` | Weight dtype: `bfloat16`, `float16`, or `float32` |

**Returns**

```json
{
  "model_id": "HuggingFaceTB/SmolLM2-135M",
  "family": "SmolLM2",
  "architecture": "LlamaForCausalLM",
  "num_layers": 30,
  "hidden_dim": 576,
  "num_attention_heads": 9,
  "num_kv_heads": 3,
  "vocab_size": 49152,
  "status": "loaded"
}
```

---

## `get_model_info`

```python
async def get_model_info() -> dict
```

Return architecture metadata for the currently-loaded model.  No forward pass.

**Returns**

```json
{
  "model_id": "HuggingFaceTB/SmolLM2-135M",
  "num_layers": 30,
  "hidden_dim": 576,
  "num_attention_heads": 9,
  "num_kv_heads": 3,
  "vocab_size": 49152,
  "max_position_embeddings": 2048,
  "architecture": "LlamaForCausalLM",
  "family": "SmolLM2",
  "is_moe": false,
  "num_experts": null
}
```
