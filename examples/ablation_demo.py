#!/usr/bin/env python3
"""
Ablation demo: testing causal necessity by breaking and patching layers.

This script demonstrates two complementary interventional tools:

1. ablate_layers -- zero out layer weights and compare output to baseline.
   High disruption proves the layer is causally necessary.

2. patch_activations -- swap a hidden state from one prompt into another's
   forward pass.  If the output shifts toward the source prompt's behavior,
   that layer carries the distinguishing information.

Together these answer: "Which layers MUST work for this prediction?" and
"Where does the model encode the difference between two prompts?"

Usage:
    # Quick run (SmolLM2-135M -- validates tooling):
    python examples/ablation_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/ablation_demo.py --model google/gemma-3-4b-it
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
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    compact = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 12:
            compact[k] = f"[{len(v)} items]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


def bar_chart(value: float, width: int = 30, char: str = "#") -> str:
    """Return an ASCII bar scaled to width."""
    n = int(abs(value) * width)
    return char * max(n, 0)


def sample_layers(num_layers: int, count: int = 6) -> list[int]:
    """Pick evenly-spaced layers (always include first and last)."""
    if num_layers <= count:
        return list(range(num_layers))
    step = max(1, (num_layers - 1) // (count - 1))
    layers = list(range(0, num_layers, step))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.generation_tools import generate_text
    from chuk_mcp_lazarus.tools.ablation_tools import (
        ablate_layers,
        patch_activations,
    )

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  ABLATION DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    num_layers = result["num_layers"]
    print(f"  Loaded: {result['family']} ({num_layers} layers, {result['hidden_dim']} hidden dim)")

    prompt = "The capital of France is"

    # ==================================================================
    # Step 2: Baseline generation
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 1: Baseline generation")
    print(f"{'=' * 60}")

    gen_result = await generate_text(prompt=prompt, max_new_tokens=20, temperature=0.0)
    if gen_result.get("error"):
        print(f"  FAILED: {gen_result.get('message')}")
        return 1

    print(f"  Prompt:   {prompt!r}")
    print(f"  Output:   {gen_result['output']!r}")
    print(f"  Tokens:   {gen_result['num_tokens']}")

    # ==================================================================
    # Step 3: Ablate individual layers and compare disruption
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 2: Layer ablation scan (ablate_layers)")
    print(f"{'=' * 60}")
    print("\n  Ablating one layer at a time to find causally important layers.")
    print("  High disruption = the layer is necessary for normal output.\n")

    test_layers = sample_layers(num_layers, count=8)
    ablation_results = []

    for layer in test_layers:
        result = await ablate_layers(
            prompt=prompt,
            layers=[layer],
            max_new_tokens=20,
            ablation_type="zero",
            component="both",
        )

        if result.get("error"):
            print(f"    Layer {layer:>3}: ERROR - {result.get('message')}")
            continue

        disruption = result["disruption_score"]
        ablation_results.append(
            {
                "layer": layer,
                "disruption": disruption,
                "similarity": result["output_similarity"],
                "ablated_output": result["ablated_output"][:60],
            }
        )

        bar = bar_chart(disruption)
        print(f"    Layer {layer:>3}: disruption={disruption:.4f}  {bar}")

    if ablation_results:
        # Find most and least critical
        most_critical = max(ablation_results, key=lambda x: x["disruption"])
        least_critical = min(ablation_results, key=lambda x: x["disruption"])

        print(
            f"\n  Most critical layer:  {most_critical['layer']} "
            f"(disruption={most_critical['disruption']:.4f})"
        )
        print(f"    Ablated output: {most_critical['ablated_output']!r}...")
        print(
            f"\n  Least critical layer: {least_critical['layer']} "
            f"(disruption={least_critical['disruption']:.4f})"
        )
        print(f"    Ablated output: {least_critical['ablated_output']!r}...")

    # ==================================================================
    # Step 4: Multi-layer ablation
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 3: Multi-layer ablation")
    print(f"{'=' * 60}")
    print("\n  Ablating groups of layers to test cumulative effects.\n")

    # Early, middle, late layer groups
    early = list(range(0, num_layers // 3))
    middle = list(range(num_layers // 3, 2 * num_layers // 3))
    late = list(range(2 * num_layers // 3, num_layers))

    for label, layers in [("Early", early), ("Middle", middle), ("Late", late)]:
        result = await ablate_layers(
            prompt=prompt,
            layers=layers,
            max_new_tokens=20,
            component="mlp",
        )
        if result.get("error"):
            print(f"    {label} layers ({layers[0]}-{layers[-1]}): ERROR")
            continue

        print(
            f"    {label:>6} layers ({layers[0]:>2}-{layers[-1]:>2}): "
            f"disruption={result['disruption_score']:.4f}  "
            f"output={result['ablated_output'][:50]!r}..."
        )

    # ==================================================================
    # Step 5: Activation patching between two prompts
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 4: Activation patching (patch_activations)")
    print(f"{'=' * 60}")

    source = "The capital of France is"
    target = "The capital of Germany is"

    print(f"\n  Source prompt: {source!r}")
    print(f"  Target prompt: {target!r}")
    print("\n  Patching source activations into target's forward pass.")
    print("  High recovery = that layer carries the France/Germany distinction.\n")

    patch_layers = sample_layers(num_layers, count=8)
    patch_results = []

    for layer in patch_layers:
        result = await patch_activations(
            source_prompt=source,
            target_prompt=target,
            layer=layer,
            max_new_tokens=15,
        )

        if result.get("error"):
            print(f"    Layer {layer:>3}: ERROR - {result.get('message')}")
            continue

        recovery = result["recovery_rate"]
        effect = result["effect_size"]
        patch_results.append(
            {
                "layer": layer,
                "recovery": recovery,
                "effect": effect,
                "patched_output": result["patched_output"][:50],
            }
        )

        direction = "-> source" if effect > 0 else "-> target"
        bar = bar_chart(abs(effect))
        print(
            f"    Layer {layer:>3}: recovery={recovery:.4f}  "
            f"effect={effect:+.4f}  {bar}  {direction}"
        )

    if patch_results:
        best_patch = max(patch_results, key=lambda x: x["recovery"])
        print(
            f"\n  Best patching layer: {best_patch['layer']} "
            f"(recovery={best_patch['recovery']:.4f})"
        )
        print(f"    Patched output: {best_patch['patched_output']!r}...")

        print("\n  Baselines:")
        # Show what each prompt generates
        source_gen = await generate_text(prompt=source, max_new_tokens=15, temperature=0.0)
        target_gen = await generate_text(prompt=target, max_new_tokens=15, temperature=0.0)
        if not source_gen.get("error"):
            print(f"    Source ({source!r}): {source_gen['output']!r}")
        if not target_gen.get("error"):
            print(f"    Target ({target!r}): {target_gen['output']!r}")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  ABLATION DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - ablate_layers tests causal NECESSITY: zeroing a layer and")
    print("    measuring how much the output changes.")
    print("  - patch_activations tests causal SUFFICIENCY: swapping a")
    print("    hidden state from one prompt into another to see if it")
    print("    shifts the output toward the source prompt's behavior.")
    print("  - Together they reveal the model's causal circuit for")
    print("    specific predictions and distinctions.")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
