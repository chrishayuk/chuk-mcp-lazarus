#!/usr/bin/env python3
"""
Causal tracing demo: observation vs intervention.

This script demonstrates the difference between observational tools
(track_token: "where does a prediction appear?") and interventional
tools (trace_token: "which layers are necessary for the prediction?").

Steps:
1. Load model and inspect architecture
2. predict_next_token to discover what the model actually predicts
3. track_token (observational) -- watch probability emerge layer by layer
4. trace_token (interventional) -- ablate each layer, measure causal impact
5. Compare observation vs intervention side by side
6. full_causal_trace -- position x layer heatmap (Meng et al. style)
7. Repeat for a second prompt to compare circuits

Note on tokenization: SentencePiece tokenizers (Gemma, Llama) encode
" Paris" (with space) and "Paris" (without) as different tokens.
The model predicts " Paris" as the next token since it follows a space.
This demo uses predict_next_token to discover the exact token, avoiding
tokenization mismatches.

Usage:
    # Quick run (SmolLM2-135M -- validates tooling, effects near zero):
    python examples/causal_tracing_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/causal_tracing_demo.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pp(label: str, result: dict) -> None:
    """Pretty-print a tool result."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    compact = {}
    for k, v in result.items():
        if isinstance(v, list) and all(isinstance(row, list) for row in v):
            compact[k] = f"[{len(v)}x{len(v[0]) if v else 0} grid]"
        elif isinstance(v, list) and len(v) > 12:
            compact[k] = f"[{len(v)} items]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


def bar_chart(value: float, width: int = 30, char: str = "█") -> str:
    """Return an ASCII bar scaled to width."""
    n = int(abs(value) * width)
    return char * max(n, 0)


def heatmap_char(value: float) -> str:
    """Map a 0-1 value to a shade character."""
    shades = " ░▒▓█"
    idx = min(int(value * len(shades)), len(shades) - 1)
    return shades[idx]


def sample_layers(all_layers: list[int], max_layers: int = 12) -> list[int]:
    """Pick an evenly-spaced subset of layers (always include first + last)."""
    if len(all_layers) <= max_layers:
        return all_layers
    step = max(1, (len(all_layers) - 1) // (max_layers - 1))
    sampled = list(range(all_layers[0], all_layers[-1], step))
    if all_layers[-1] not in sampled:
        sampled.append(all_layers[-1])
    return sampled


def trace_progress(done: int, total: int, layer_idx: int) -> None:
    """Progress callback for trace_token."""
    print(f"    ablating layer {layer_idx:>3}  ({done}/{total})", flush=True)


def heatmap_progress(pos_done: int, total_pos: int, n_layers: int) -> None:
    """Progress callback for full_causal_trace."""
    print(
        f"    position {pos_done}/{total_pos} done  "
        f"({pos_done * n_layers}/{total_pos * n_layers} passes)",
        flush=True,
    )


async def run_experiment(
    label: str,
    prompt: str,
    all_layers: list[int],
    predict_next_token,
    track_token,
    ci,
    show_full_trace: bool = True,
) -> None:
    """Run a complete causal tracing experiment on one prompt."""

    # Sample layers for causal tracing (expensive: 1 forward pass per layer)
    causal_layers = sample_layers(all_layers, max_layers=12)
    # Full trace is positions × layers forward passes — use fewer layers
    heatmap_layers = sample_layers(all_layers, max_layers=6)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Prompt: {prompt!r}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Step A: Discover what the model actually predicts
    # ------------------------------------------------------------------
    result = await predict_next_token(prompt=prompt, top_k=5)
    pp("predict_next_token", result)

    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return

    top = result["predictions"][0]
    target = top["token"]  # e.g. " Paris" (with space prefix)
    target_prob = top["probability"]

    print(f"\n  Model's top prediction: {target!r} (p={target_prob:.4f})")
    print("  Top 5:")
    for p in result["predictions"]:
        print(f"    {p['token']!r:>15}  p={p['probability']:.4f}")

    if target_prob < 0.01:
        print(f"\n  WARNING: top prediction probability is very low ({target_prob:.4f}).")
        print("  Causal effects may be near zero.  Try a larger model.")

    # ------------------------------------------------------------------
    # Step B: track_token (observational)
    # ------------------------------------------------------------------
    result = await track_token(prompt=prompt, token=target, layers=all_layers)
    pp("track_token (observation)", result)

    track_data = {}
    if not result.get("error"):
        print(f"\n  Target: {result['target_token']!r} (id={result['target_token_id']})")
        print(f"  Emergence layer: {result['emergence_layer']}")
        print(f"  Peak layer: {result['peak_layer']} (p={result['peak_probability']:.4f})")
        print("\n  Probability curve (observational -- logit lens):")
        for entry in result["layers"]:
            marker = " << top-1" if entry["is_top1"] else ""
            prob_bar = bar_chart(entry["probability"])
            print(
                f"    Layer {entry['layer']:>3}: p={entry['probability']:.6f} "
                f"rank={entry['rank']:>5} {prob_bar}{marker}"
            )
            track_data[entry["layer"]] = entry["probability"]

    # ------------------------------------------------------------------
    # Step C: trace_token (interventional) -- with progress
    # ------------------------------------------------------------------
    print(
        f"\n  Running trace_token: {len(causal_layers)} ablation passes "
        f"(layers {causal_layers[0]}..{causal_layers[-1]})..."
    )
    t_trace = time.time()
    trace_result = await asyncio.to_thread(
        ci.trace_token,
        prompt,
        target,
        causal_layers,
        0.01,  # effect_threshold
        trace_progress,
    )
    elapsed = time.time() - t_trace
    print(f"  Done in {elapsed:.1f}s")

    # Convert backend result to dict for display
    result = {
        "prompt": trace_result.prompt,
        "target_token": trace_result.target_token,
        "target_token_id": trace_result.target_token_id,
        "baseline_prob": trace_result.baseline_prob,
        "layer_effects": [
            {"layer": layer, "effect": effect} for layer, effect in trace_result.layer_effects
        ],
        "critical_layers": list(trace_result.critical_layers),
        "peak_layer": trace_result.peak_layer,
        "peak_effect": trace_result.peak_effect,
        "effect_threshold": 0.01,
    }
    pp("trace_token (intervention)", result)

    trace_data = {}
    print(f"\n  Baseline prob: {result['baseline_prob']:.6f}")
    print(f"  Peak layer: {result['peak_layer']} (effect={result['peak_effect']:.6f})")
    print(f"  Critical layers: {result['critical_layers']}")
    print("\n  Causal effect curve (interventional -- ablation):")
    for entry in result["layer_effects"]:
        is_critical = entry["layer"] in result["critical_layers"]
        marker = " << critical" if is_critical else ""
        effect_bar = bar_chart(entry["effect"])
        print(f"    Layer {entry['layer']:>3}: effect={entry['effect']:+.6f} {effect_bar}{marker}")
        trace_data[entry["layer"]] = entry["effect"]

    # ------------------------------------------------------------------
    # Step D: Compare observation vs intervention
    # ------------------------------------------------------------------
    if track_data and trace_data:
        print(f"\n{'─' * 60}")
        print("  Observation vs Intervention comparison")
        print(f"{'─' * 60}")

        # Find maxima for scaling bars
        max_prob = max(track_data.values()) or 1.0
        max_effect = max(abs(v) for v in trace_data.values()) or 1.0

        # Show only the layers we tested causally
        print(
            f"\n  {'Layer':>5}  {'Prob (obs)':>14}  {'Effect (int)':>14}  "
            f"{'Observation':>12}  {'Intervention':>12}"
        )
        print(f"  {'─' * 5}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 12}")
        for layer in causal_layers:
            prob = track_data.get(layer, 0.0)
            effect = trace_data.get(layer, 0.0)
            obs_bar = bar_chart(prob / max_prob, width=12)
            int_bar = bar_chart(effect / max_effect if max_effect else 0, width=12)
            print(f"  {layer:>5}  {prob:>14.6f}  {effect:>+14.6f}  {obs_bar:<12}  {int_bar:<12}")

        print("\n  Key insight: layers where the prediction first *appears*")
        print("  (observation) may differ from layers that *cause* it")
        print("  (intervention).")

    # ------------------------------------------------------------------
    # Step E: full_causal_trace (position x layer heatmap) -- with progress
    # ------------------------------------------------------------------
    if not show_full_trace:
        return

    print(f"\n  Running full_causal_trace: {len(heatmap_layers)} layers x N positions...")
    t_full = time.time()
    full_result = await asyncio.to_thread(
        ci.full_causal_trace,
        prompt,
        target,
        None,  # corrupt_prompt
        heatmap_layers,
        heatmap_progress,
    )
    print(f"  Done in {time.time() - t_full:.1f}s")

    # Convert backend result to dict for display
    result = {
        "prompt": full_result.prompt,
        "target_token": full_result.target_token,
        "tokens": list(full_result.tokens),
        "effects": [list(row) for row in full_result.effects],
        "critical_positions": list(full_result.critical_positions),
        "critical_layers": list(full_result.critical_layers),
        "num_positions": len(full_result.tokens),
        "num_layers_tested": len(heatmap_layers),
    }
    pp("full_causal_trace (heatmap)", result)

    tokens = result["tokens"]
    effects = result["effects"]
    n_pos = result["num_positions"]
    n_lay = result["num_layers_tested"]

    print(f"\n  Grid: {n_pos} positions x {n_lay} layers")
    print(f"  Tokens: {tokens}")
    print(f"  Critical positions: {result['critical_positions']}")
    print(f"  Critical layers: {result['critical_layers']}")

    # Find max effect for normalization
    grid_max = (
        max(
            (abs(effects[p][lay]) for p in range(n_pos) for lay in range(n_lay)),
            default=1.0,
        )
        or 1.0
    )

    # ASCII heatmap
    print("\n  Position x Layer heatmap (recovery rate):")
    print("  Shade: ' '=0  ░=low  ▒=mid  ▓=high  █=max")
    print()

    # Header: layer numbers
    header = "  " + " " * 14
    for lay in range(n_lay):
        if lay % 5 == 0:
            header += f"{lay:<3}"
        else:
            header += "   "
    print(header)

    # Rows: one per token position
    for p in range(n_pos):
        tok_label = f"{tokens[p][:12]:<12}" if p < len(tokens) else f"pos {p:<8}"
        row = f"  {tok_label}  "
        for lay in range(n_lay):
            normalized = abs(effects[p][lay]) / grid_max
            row += f" {heatmap_char(normalized)} "
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.generation_tools import (
        predict_next_token,
        track_token,
    )
    from chuk_mcp_lazarus.model_state import ModelState
    from chuk_lazarus.introspection.interventions import CounterfactualIntervention

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  CAUSAL TRACING DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    num_layers = result["num_layers"]
    print(f"  Loaded: {result['family']} ({num_layers} layers, {result['hidden_dim']} hidden dim)")

    all_layers = list(range(num_layers))

    # Create CounterfactualIntervention for causal tools (with progress)
    state = ModelState.get()
    ci = CounterfactualIntervention(model=state.model, tokenizer=state.tokenizer)

    # ==================================================================
    # Experiment A: factual recall
    # ==================================================================
    await run_experiment(
        label="EXPERIMENT A: factual recall",
        prompt="The capital of France is",
        all_layers=all_layers,
        predict_next_token=predict_next_token,
        track_token=track_token,
        ci=ci,
        show_full_trace=True,
    )

    # ==================================================================
    # Experiment B: common phrase completion
    # ==================================================================
    await run_experiment(
        label="EXPERIMENT B: common phrase completion",
        prompt="The cat sat on the",
        all_layers=all_layers,
        predict_next_token=predict_next_token,
        track_token=track_token,
        ci=ci,
        show_full_trace=False,  # Skip heatmap to save time
    )

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  CAUSAL TRACING DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - track_token shows WHERE a prediction appears (observational)")
    print("  - trace_token shows WHICH layers are necessary (interventional)")
    print("  - full_causal_trace reveals which TOKEN POSITIONS carry")
    print("    the critical information at each layer")
    print("  - Different facts may use different circuits")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal tracing demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
