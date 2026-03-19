# chuk-mcp-lazarus Documentation

Mechanistic interpretability MCP server — 72 tools, 4 resources, 7 singletons.

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| [Model](model/) | `load_model`, `get_model_info` | Load and inspect models |
| [Generation](generation/) | `generate_text`, `predict_next_token`, `tokenize`, `logit_lens`, `track_token`, `track_race`, `embedding_neighbors`, `probe_at_inference` | Text generation and prediction |
| [Activation](activation/) | `extract_activations`, `compare_activations` | Raw activation extraction |
| [Attention](attention/) | `attention_pattern`, `attention_heads` | Attention weight analysis |
| [Probing](probing/) | `train_probe`, `evaluate_probe`, `scan_probe_across_layers`, `list_probes` | Linear probes for concept detection |
| [Steering](steering/) | `compute_steering_vector`, `steer_and_generate`, `list_steering_vectors` | Activation steering |
| [Ablation](ablation/) | `ablate_layers`, `patch_activations` | Causal ablation and patching |
| [Causal](causal/) | `trace_token`, `full_causal_trace` | Causal tracing (Meng et al.) |
| [Residual](residual/) | `residual_decomposition`, `layer_clustering`, `logit_attribution`, `head_attribution`, `top_neurons` | Residual stream decomposition |
| [Attribution](attribution/) | `attribution_sweep` | Batch logit attribution |
| [Intervention](intervention/) | `component_intervention` | Surgical component interventions |
| [Neuron](neuron/) | `discover_neurons`, `analyze_neuron`, `neuron_trace` | Neuron-level analysis |
| [Direction](direction/) | `extract_direction` | Direction extraction in activation space |
| [Comparison](comparison/) | `load_comparison_model`, `compare_weights`, `compare_representations`, `compare_attention`, `compare_generations`, `unload_comparison_model` | Two-model comparison |
| [Experiment](experiment/) | `create_experiment`, `add_experiment_result`, `get_experiment`, `list_experiments` | Experiment persistence |
| [Geometry](geometry/) | 26 tools | Geometric analysis: angles, subspaces, DLA, copy circuits |

## Quick Start

```bash
# HTTP mode (for development / demo scripts)
uv run chuk-mcp-lazarus http --port 8765

# Stdio mode (for Claude Desktop / MCP clients)
uv run chuk-mcp-lazarus stdio
```

Every tool follows the same pattern:
1. Call `load_model` first
2. Run any tool — all return dicts, never raise
3. Errors return `{"error": True, "error_type": "...", "message": "..."}`

## Demo Scripts

| Script | What it demonstrates |
|--------|---------------------|
| `examples/smoke_test.py` | 55 end-to-end validation tests |
| `examples/attention_demo.py` | Attention patterns and entropy |
| `examples/causal_tracing_demo.py` | Causal traces and heatmaps |
| `examples/logit_attribution_demo.py` | DLA, head attribution, top neurons |
| `examples/residual_stream_demo.py` | Residual decomposition and clustering |
| `examples/ablation_demo.py` | Layer ablation and activation patching |
| `examples/language_transition_demo.py` | Probe-based language crossover detection |
| `examples/deep_dive_demo.py` | Full mechanistic interpretability pipeline |
| `examples/track_race_demo.py` | Token racing across layers |
| `examples/intervention_demo.py` | Surgical component interventions |
| `examples/attribution_sweep_demo.py` | Batch attribution across prompts |
| `examples/experiment_demo.py` | Experiment creation and persistence |
| `examples/comparison_demo.py` | Two-model weight and representation comparison |
| `examples/geometry_demo.py` | Geometry tools: angles, subspaces, trajectories |
| `examples/copy_circuit_demo.py` | Copy circuit hypothesis: DLA + injection test |
| `examples/neuron_demo.py` | Neuron discovery, analysis, and tracing |
| `examples/direction_demo.py` | Direction extraction + probe-at-inference |
