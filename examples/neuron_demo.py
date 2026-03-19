#!/usr/bin/env python3
"""
Neuron analysis demo: discover, profile, and trace individual MLP neurons.

Steps:
1. Load model
2. discover_neurons: find neurons that discriminate capital cities vs common nouns
3. analyze_neuron: profile top neurons across multiple prompts (detailed=True)
4. neuron_trace: trace a neuron's output direction through downstream layers
5. Repeat discovery at multiple layers to find where neurons are most specific

Usage:
    python examples/neuron_demo.py
    python examples/neuron_demo.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


def pp(label: str, result: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    compact = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 8:
            compact[k] = f"[{len(v)} items]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.neuron_tools import (
        discover_neurons,
        analyze_neuron,
        neuron_trace,
    )

    t0 = time.time()

    print(f"\n{'=' * 60}")
    print("  NEURON ANALYSIS DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1
    num_layers = result["num_layers"]
    mid_layer = num_layers // 2
    print(f"  Loaded: {result['family']} ({num_layers} layers)")

    positive_prompts = [
        "Paris is the capital",
        "Berlin is the capital",
        "Tokyo is the capital",
        "Rome is the capital",
        "London is the capital",
    ]
    negative_prompts = [
        "The cat sat on the mat",
        "Water boils at 100 degrees",
        "She opened the red door slowly",
        "The train arrived at midnight",
        "Blue is a common color",
    ]

    # ------------------------------------------------------------------
    # Step 2: discover_neurons at multiple layers
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 2: discover_neurons (capital cities vs common sentences)")
    print(f"{'=' * 60}")

    best_layer = mid_layer
    best_neuron = None
    best_d = 0.0

    check_layers = [
        num_layers // 4,
        num_layers // 2,
        3 * num_layers // 4,
    ]

    for layer in check_layers:
        disc = await discover_neurons(
            layer=layer,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            top_k=5,
        )
        if disc.get("error"):
            print(f"  Layer {layer}: FAILED — {disc.get('message')}")
            continue

        neurons = disc.get("top_neurons", [])
        if neurons:
            top = neurons[0]
            print(
                f"\n  Layer {layer}: top neuron={top['neuron_idx']}, "
                f"Cohen's d={top['cohens_d']:.3f}, "
                f"+mean={top['positive_mean']:.3f}, "
                f"-mean={top['negative_mean']:.3f}"
            )
            if top["cohens_d"] > best_d:
                best_d = top["cohens_d"]
                best_layer = layer
                best_neuron = top["neuron_idx"]

        pp(f"discover_neurons (layer={layer})", disc)

    if best_neuron is None:
        print("  No neurons found — try a larger model.")
        return 1

    print(
        f"\n  Best discriminating neuron: layer={best_layer}, neuron={best_neuron}, d={best_d:.3f}"
    )

    # ------------------------------------------------------------------
    # Step 3: analyze_neuron (detailed activation profile)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 3: analyze_neuron (layer={best_layer}, neuron={best_neuron})")
    print(f"{'=' * 60}")

    profile_prompts = positive_prompts + negative_prompts
    profile = await analyze_neuron(
        layer=best_layer,
        neuron_indices=[best_neuron],
        prompts=profile_prompts,
        detailed=True,
    )
    pp("analyze_neuron", profile)

    if not profile.get("error"):
        neurons = profile.get("neurons", [])
        if neurons:
            n = neurons[0]
            print(f"\n  Neuron {n['neuron_idx']} at layer {best_layer}:")
            print(
                f"    min={n['min_val']:+.4f}  max={n['max_val']:+.4f}  "
                f"mean={n['mean_val']:+.4f}  std={n['std_val']:.4f}"
            )

        per_prompt = profile.get("per_prompt_activations", [])
        if per_prompt:
            print("\n  Per-prompt activations:")
            for pp_entry in per_prompt:
                act = pp_entry.get("activations", [])
                val = act[0] if act else 0.0
                prompt_text = pp_entry.get("prompt", "?")[:45]
                print(f"    {val:>+8.4f}  {prompt_text!r}")

    # ------------------------------------------------------------------
    # Step 4: neuron_trace — where does this neuron's direction flow?
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 4: neuron_trace (layer={best_layer}, neuron={best_neuron})")
    print(f"{'=' * 60}")

    trace = await neuron_trace(
        prompt="Paris is the capital of France",
        layer=best_layer,
        neuron_index=best_neuron,
        top_k_heads=3,
    )
    pp("neuron_trace", trace)

    if not trace.get("error"):
        neuron_info = trace.get("neuron", {})
        print(f"\n  Neuron at layer {best_layer}:")
        print(
            f"    activation={neuron_info.get('activation', '?'):.4f}  "
            f"output_direction_norm={neuron_info.get('output_direction_norm', '?'):.4f}  "
            f"top_token={neuron_info.get('top_token', '?')!r}"
        )

        print("\n  Trace through downstream layers:")
        print(
            f"  {'Layer':>5}  {'Resid':>7}  {'Attn':>7}  {'FFN':>7}  {'Proj':>8}  Top aligned heads"
        )
        print(f"  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 8}  {'─' * 30}")
        for entry in trace.get("trace", []):
            top_heads = entry.get("top_aligned_heads", [])[:2]
            heads_str = "  ".join(f"L{h['layer']}H{h['head']}={h['cosine']:.3f}" for h in top_heads)
            print(
                f"  {entry['layer']:>5}  "
                f"{entry.get('residual_alignment', 0):>+7.3f}  "
                f"{entry.get('attention_alignment', 0):>+7.3f}  "
                f"{entry.get('ffn_alignment', 0):>+7.3f}  "
                f"{entry.get('residual_projection', 0):>+8.4f}  "
                f"{heads_str}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  NEURON ANALYSIS DEMO COMPLETE")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - discover_neurons uses Cohen's d to find discriminating neurons quickly")
    print("  - analyze_neuron profiles activation distributions per prompt")
    print("  - neuron_trace reveals how a neuron's output direction propagates")
    print("    downstream and which heads it aligns with")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuron analysis demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
