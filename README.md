# chuk-mcp-lazarus

Mechanistic interpretability MCP server wrapping [chuk-lazarus](https://github.com/chuk-ai/chuk-lazarus).

Load any model, extract activations, train probes, steer generation,
and ablate components -- all via MCP tools that Claude (or any MCP client)
can call autonomously.

## Quick Start

```bash
# Clone and install
git clone https://github.com/chuk-ai/chuk-mcp-lazarus.git
cd chuk-mcp-lazarus
uv sync

# Run the smoke test (loads SmolLM2-135M, ~3 seconds)
uv run python examples/smoke_test.py

# Run the full 10-step language transition demo
uv run python examples/language_transition_demo.py
```

## Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

## Tools (31)

| Group | Tool | Purpose |
|-------|------|---------|
| Model | `load_model` | Load any HuggingFace model into memory |
| Model | `get_model_info` | Return architecture metadata |
| Generation | `generate_text` | Generate text from the loaded model |
| Generation | `predict_next_token` | Top-k next-token predictions with probabilities |
| Generation | `tokenize` | Show how text is tokenized |
| Generation | `logit_lens` | Layer-by-layer prediction evolution (logit lens) |
| Generation | `track_token` | Track a specific token's probability across layers |
| Activations | `extract_activations` | Hidden states at specific layers and positions |
| Activations | `compare_activations` | Cosine similarity + PCA across prompts |
| Attention | `attention_pattern` | Per-head attention weights at specified layers |
| Attention | `attention_heads` | Per-head entropy and focus analysis |
| Probing | `train_probe` | Train a classifier on activations |
| Probing | `evaluate_probe` | Evaluate on held-out data |
| Probing | `scan_probe_across_layers` | Find the crossover layer |
| Probing | `list_probes` | List all trained probes |
| Steering | `compute_steering_vector` | Contrastive activation addition |
| Steering | `steer_and_generate` | Generate with steering applied |
| Steering | `list_steering_vectors` | List all computed vectors |
| Ablation | `ablate_layers` | Zero out layers, measure disruption |
| Ablation | `patch_activations` | Swap activations between prompts |
| Causal | `trace_token` | Which layers are causally necessary for a prediction |
| Causal | `full_causal_trace` | Position × layer causal heatmap (Meng et al. style) |
| Residual | `residual_decomposition` | Attention vs MLP contribution per layer |
| Residual | `layer_clustering` | Representation similarity and cluster separation across layers |
| Residual | `logit_attribution` | Direct logit attribution: per-layer component contributions to predicted token |
| Comparison | `load_comparison_model` | Load a second model for side-by-side analysis |
| Comparison | `compare_weights` | Frobenius norm + cosine sim per layer per component |
| Comparison | `compare_representations` | Per-layer activation divergence across prompts |
| Comparison | `compare_attention` | Per-head JS divergence in attention patterns |
| Comparison | `compare_generations` | Side-by-side text output from both models |
| Comparison | `unload_comparison_model` | Free VRAM from comparison model |

## Resources (4)

| URI | Description |
|-----|-------------|
| `model://info` | Current model metadata |
| `probes://registry` | All trained probes and accuracy metrics |
| `vectors://registry` | All computed steering vectors |
| `comparisons://state` | Comparison model state |

## Supported Models

Works with any model [chuk-lazarus](https://github.com/chuk-ai/chuk-lazarus) supports:

- **Gemma** -- Gemma 3 (270M--27B), TranslateGemma 4B/12B
- **Llama** -- Llama 2/3, Mistral, SmolLM2
- **Qwen** -- Qwen 2/3
- **Granite** -- IBM Granite 3.x/4.x (hybrid Mamba-2/Transformer)
- **Jamba** -- AI21 Jamba (hybrid Mamba-Transformer MoE)
- **Mamba** -- Pure SSM models
- **StarCoder2** -- Code generation
- **GPT-2** -- GPT-2 and compatible

Default demo target: **TranslateGemma 4B** (34 layers, fits on Apple Silicon).
Smoke tests use **SmolLM2-135M** for speed.

## The Demo: Language Transition Probing

The flagship experiment follows a 15-step workflow:

1. **Load model** -- `load_model("google/gemma-3-4b-it")`
2. **Inspect architecture** -- `get_model_info()` reveals 34 layers
3. **Tokenize** -- see how the prompt breaks into tokens
4. **Generate text** -- see baseline model output
5. **Sanity-check activations** -- verify activations are non-trivial
6. **Compare at early layer** -- language representations are distinct
7. **Compare at late layer** -- representations converge
8. **Logit lens** -- see how predictions evolve through layers
9. **Track token** -- watch a specific token's probability rise across layers
10. **Scan probes across layers** -- find where language identity becomes decodable
11. **Evaluate best probe** -- confirm on held-out data
12. **Compute steering vector** -- French-to-German direction
13. **Steer generation** -- redirect a French translation to German
14. **Alpha sweep** -- iterate with different steering strengths
15. **Causal tracing** -- prove which layers are necessary for the prediction

Run it: `uv run python examples/language_transition_demo.py`

## The Demo: Model Comparison

Compare a base model against its fine-tuned variant. First see actual
output differences with `compare_generations`, then find where
fine-tuning changed weights, activations, and attention patterns.
Designed for Gemma 3 4B vs TranslateGemma 4B using low-resource
languages (Icelandic, Swahili, Estonian, Marathi) where TranslateGemma
shows 25-30% improvement.

Run it: `uv run python examples/comparison_demo.py`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the 10 design principles.

Key points:
- **Async-native** -- all tools are `async def`
- **Pydantic-native** -- every data structure is a typed `BaseModel`
- **Model-agnostic** -- works with 9+ model families
- **Error envelopes** -- tools never raise; always return structured errors
- **JSON-safe boundary** -- MLX arrays converted at the tool return

## Project Structure

```
src/chuk_mcp_lazarus/
├── server.py            # ChukMCPServer instance
├── main.py              # Entry point (stdio / http)
├── model_state.py       # ModelState singleton
├── probe_store.py       # ProbeRegistry singleton
├── steering_store.py    # SteeringVectorRegistry singleton
├── resources.py         # MCP resources
├── errors.py            # Error types + envelope helper
├── _bootstrap.py        # Optional dependency stubs
├── comparison_state.py  # ComparisonState singleton (2nd model)
├── _serialize.py        # MLX/NumPy -> JSON-safe
├── _generate.py         # Shared text generation
├── _compare.py          # Shared comparison kernels
└── tools/
    ├── model_tools.py       # load_model, get_model_info
    ├── generation_tools.py  # generate_text, predict_next_token, tokenize, logit_lens, track_token
    ├── activation_tools.py  # extract_activations, compare_activations
    ├── attention_tools.py   # attention_pattern, attention_heads
    ├── probe_tools.py       # train_probe, evaluate_probe, scan, list
    ├── steering_tools.py    # compute_vector, steer, list
    ├── ablation_tools.py    # ablate_layers, patch_activations
    ├── causal_tools.py      # trace_token, full_causal_trace
    ├── residual_tools.py    # residual_decomposition, layer_clustering
    └── comparison_tools.py  # compare_weights, representations, attention, generations
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run smoke tests
uv run python examples/smoke_test.py

# Run with a different model
uv run python examples/smoke_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# HTTP mode for development
uv run chuk-mcp-lazarus http --port 8765
```

## Requirements

- Python >= 3.11
- Apple Silicon Mac (for MLX)
- [chuk-lazarus](https://github.com/chuk-ai/chuk-lazarus) >= 0.4
- [chuk-mcp-server](https://github.com/chuk-ai/chuk-mcp-server) >= 0.25

## License

Apache 2.0
