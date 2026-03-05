#!/usr/bin/env python3
"""
Component intervention demo: surgical causal interventions on model internals.

Demonstrates:
1. component_intervention — zero/scale attention, FFN, or individual heads
2. Comparing clean vs intervened predictions
3. KL divergence as a measure of component importance

Steps:
1. Load model
2. Zero attention at an early layer → measure prediction change
3. Zero FFN at the same layer → compare impact
4. Zero attention at the last layer → measure late-stage impact
5. Scale attention by 2x at mid layer → amplification experiment

Usage:
    # Quick run (SmolLM2-135M):
    python examples/intervention_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/intervention_demo.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


def pp(label: str, result: dict) -> None:
    """Pretty-print a tool result as indented JSON."""
    print(f"\n  [{label}]")
    print("  " + json.dumps(result, indent=2, default=str).replace("\n", "\n  "))


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.intervention_tools import component_intervention

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"{'=' * 60}")
    print("  COMPONENT INTERVENTION DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    load_result = await load_model(model_id=model_id)
    if load_result.get("error"):
        print(f"  FAILED: {load_result.get('message')}")
        return 1

    num_layers = load_result["num_layers"]
    prompt = "The capital of France is"
    early = 1
    mid = num_layers // 2
    late = num_layers - 1

    print(f"  Loaded: {num_layers} layers")
    print(f"  Prompt: '{prompt}'")
    print(f"  Intervention layers: early={early}, mid={mid}, late={late}")

    # ==================================================================
    # Step 2: Zero attention at early layer
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: ZERO ATTENTION @ LAYER {early}")
    print(f"{'=' * 60}")

    result_attn = await component_intervention(
        prompt=prompt,
        layer=early,
        component="attention",
        intervention="zero",
    )
    if result_attn.get("error"):
        print(f"  FAILED: {result_attn.get('message')}")
        return 1

    _display_intervention(result_attn)

    # ==================================================================
    # Step 3: Zero FFN at early layer
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  STEP 3: ZERO FFN @ LAYER {early}")
    print(f"{'=' * 60}")

    result_ffn = await component_intervention(
        prompt=prompt,
        layer=early,
        component="ffn",
        intervention="zero",
    )
    if result_ffn.get("error"):
        print(f"  FAILED: {result_ffn.get('message')}")
        return 1

    _display_intervention(result_ffn)

    # ==================================================================
    # Step 4: Zero attention at last layer
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  STEP 4: ZERO ATTENTION @ LAYER {late}")
    print(f"{'=' * 60}")

    result_late = await component_intervention(
        prompt=prompt,
        layer=late,
        component="attention",
        intervention="zero",
    )
    if result_late.get("error"):
        print(f"  FAILED: {result_late.get('message')}")
        return 1

    _display_intervention(result_late)

    # ==================================================================
    # Step 5: Scale attention 2x at mid layer
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  STEP 5: SCALE ATTENTION 2x @ LAYER {mid}")
    print(f"{'=' * 60}")

    result_scale = await component_intervention(
        prompt=prompt,
        layer=mid,
        component="attention",
        intervention="scale",
        scale_factor=2.0,
    )
    if result_scale.get("error"):
        print(f"  FAILED: {result_scale.get('message')}")
        return 1

    _display_intervention(result_scale)

    # ==================================================================
    # Comparison
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  COMPARISON: COMPONENT IMPORTANCE BY KL DIVERGENCE")
    print(f"{'=' * 60}")

    experiments = [
        (f"Zero attn @ {early}", result_attn),
        (f"Zero FFN @ {early}", result_ffn),
        (f"Zero attn @ {late}", result_late),
        (f"Scale 2x attn @ {mid}", result_scale),
    ]

    print(f"\n  {'Experiment':>25}  {'KL Div':>8}  {'Top1 Δ':>8}  {'Changed?':>8}")
    print(f"  {'─' * 25}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for name, r in experiments:
        changed = "YES" if r["top1_changed"] else "no"
        print(f"  {name:>25}  {r['kl_divergence']:>8.4f}  {r['target_delta']:>+8.4f}  {changed:>8}")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  COMPONENT INTERVENTION DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print("  Tools demonstrated: component_intervention")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    return 0


def _display_intervention(result: dict) -> None:
    """Display a single intervention result."""
    print(
        f"\n  Layer {result['layer']}, component={result['component']}, "
        f"intervention={result['intervention']}"
    )
    if result.get("scale_factor") is not None:
        print(f"  Scale factor: {result['scale_factor']}")

    print(f"\n  Original top-1: {result['original_top1']}")
    print(f"  Intervened top-1: {result['intervened_top1']}")
    print(f"  Top-1 changed: {result['top1_changed']}")
    print(f"  KL divergence: {result['kl_divergence']:.6f}")
    print(f"  Target delta: {result['target_delta']:+.6f}")

    # Show top-5 comparison
    orig = result["original_top_k"][:5]
    intv = result["intervened_top_k"][:5]

    print(f"\n  {'Original Top-5':>30}    {'Intervened Top-5':>30}")
    print(f"  {'─' * 30}    {'─' * 30}")
    for i in range(min(len(orig), len(intv))):
        o = orig[i]
        v = intv[i]
        print(
            f"  {o['token']:>15} p={o['probability']:.4f}    "
            f"{v['token']:>15} p={v['probability']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component intervention demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model to load",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
