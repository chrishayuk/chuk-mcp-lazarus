#!/usr/bin/env python3
"""
Direction extraction and probe-at-inference demo.

This script demonstrates:
1. extract_direction — find interpretable directions in activation space
   using multiple methods (mean_diff, LDA, PCA, probe)
2. steer_and_generate — apply the extracted direction at inference time
3. probe_at_inference — monitor the direction live during generation

Steps:
1. Load model
2. Extract a "formal vs informal" direction using mean_diff
3. Extract the same direction using LDA and PCA for comparison
4. Compare direction angles using direction_angles
5. Steer generation with the formal direction
6. Train a probe on the same examples
7. Monitor the probe live during generation with probe_at_inference

Usage:
    python examples/direction_demo.py
    python examples/direction_demo.py --model google/gemma-3-4b-it
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
    from chuk_mcp_lazarus.tools.direction_tools import extract_direction
    from chuk_mcp_lazarus.tools.steering_tools import steer_and_generate, list_steering_vectors
    from chuk_mcp_lazarus.tools.probe_tools import train_probe, probe_at_inference
    from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

    t0 = time.time()

    print(f"\n{'=' * 60}")
    print("  DIRECTION EXTRACTION + PROBE-AT-INFERENCE DEMO")
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

    # Use mid-model layer (often good for semantic features)
    direction_layer = mid_layer

    formal_prompts = [
        "I would like to formally request your assistance with this matter.",
        "Please accept my sincerest apologies for the inconvenience.",
        "We are pleased to inform you that your application has been approved.",
        "Kindly direct your correspondence to the appropriate department.",
        "The aforementioned policy shall take effect immediately.",
    ]
    informal_prompts = [
        "Hey, can you help me out with this?",
        "Sorry about that, my bad!",
        "Great news — you're in!",
        "Just send it to whoever handles this stuff.",
        "This kicks in right away.",
    ]

    # ------------------------------------------------------------------
    # Step 2: extract_direction (mean_diff)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: extract_direction (mean_diff) at layer {direction_layer}")
    print(f"{'=' * 60}")

    dir_mean = await extract_direction(
        direction_name="formal_mean_diff",
        layer=direction_layer,
        positive_prompts=formal_prompts,
        negative_prompts=informal_prompts,
        method="mean_diff",
    )
    pp("extract_direction (mean_diff)", dir_mean)

    if not dir_mean.get("error"):
        print("\n  mean_diff direction:")
        print(f"    vector_norm={dir_mean['vector_norm']:.4f}")
        print(f"    separation_score={dir_mean['separation_score']:.4f}")
        print(f"    classification_accuracy={dir_mean['classification_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Step 3: extract_direction (PCA and LDA for comparison)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 3: extract_direction (pca and lda) for comparison")
    print(f"{'=' * 60}")

    dir_pca = await extract_direction(
        direction_name="formal_pca",
        layer=direction_layer,
        positive_prompts=formal_prompts,
        negative_prompts=informal_prompts,
        method="pca",
    )
    pp("extract_direction (pca)", dir_pca)

    dir_lda = await extract_direction(
        direction_name="formal_lda",
        layer=direction_layer,
        positive_prompts=formal_prompts,
        negative_prompts=informal_prompts,
        method="lda",
    )
    pp("extract_direction (lda)", dir_lda)

    # ------------------------------------------------------------------
    # Step 4: direction_angles — compare the three extracted directions
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 4: direction_angles (comparing mean_diff, pca, lda)")
    print(f"{'=' * 60}")

    test_prompt = formal_prompts[0]
    da = await direction_angles(
        prompt=test_prompt,
        layer=direction_layer,
        directions=[
            {"type": "steering_vector", "value": "formal_mean_diff"},
            {"type": "steering_vector", "value": "formal_pca"},
            {"type": "steering_vector", "value": "formal_lda"},
            {"type": "residual"},
        ],
        token_position=-1,
    )
    pp("direction_angles", da)

    if not da.get("error"):
        print("\n  Pairwise angles between directions:")
        for pair in da.get("pairwise_angles", []):
            print(f"    {pair['a']:>25} vs {pair['b']:>25}: {pair['angle']:.1f}°")

    # ------------------------------------------------------------------
    # Step 5: steer_and_generate
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 5: steer_and_generate")
    print("  (steering with formal_mean_diff direction, alpha=15)")
    print(f"{'=' * 60}")

    if not dir_mean.get("error"):
        steer = await steer_and_generate(
            prompt="Hey can you explain how transformers work?",
            vector_name="formal_mean_diff",
            alpha=15.0,
            max_new_tokens=80,
        )
        pp("steer_and_generate", steer)

        if not steer.get("error"):
            print(f"\n  Baseline:  {steer['baseline_output']!r}")
            print(f"  Steered:   {steer['steered_output']!r}")

    # ------------------------------------------------------------------
    # Step 6: Train a probe on the same examples
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 6: train_probe (formality classifier at layer {direction_layer})")
    print(f"{'=' * 60}")

    examples = [{"prompt": p, "label": "formal"} for p in formal_prompts] + [
        {"prompt": p, "label": "informal"} for p in informal_prompts
    ]

    probe_result = await train_probe(
        probe_name="formality",
        layer=direction_layer,
        examples=examples,
        probe_type="linear",
    )
    pp("train_probe", probe_result)

    if not probe_result.get("error"):
        print(f"\n  Probe trained: val_accuracy={probe_result['val_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Step 7: probe_at_inference — monitor formality during generation
    # ------------------------------------------------------------------
    if not probe_result.get("error"):
        print(f"\n{'=' * 60}")
        print("  STEP 7: probe_at_inference (monitor formality live)")
        print(f"{'=' * 60}")

        test_text = "The weather today is really"
        pai = await probe_at_inference(
            prompt=test_text,
            probe_name="formality",
            max_tokens=30,
            temperature=0.0,
        )
        pp("probe_at_inference", pai)

        if not pai.get("error"):
            print(f"\n  Prompt: {test_text!r}")
            print(f"  Generated: {pai.get('generated_text', '?')!r}")
            print(f"  Overall majority class: {pai.get('overall_majority_class', '?')}")
            print(f"  Overall mean confidence: {pai.get('overall_mean_confidence', 0):.4f}")
            print("\n  Token-by-token probe predictions:")
            print(f"  {'Step':>4}  {'Token':>12}  {'Prediction':>10}  {'Confidence':>10}")
            print(f"  {'─' * 4}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
            for entry in pai.get("per_token", [])[:15]:
                print(
                    f"  {entry['step']:>4}  "
                    f"{entry['token']!r:>12}  "
                    f"{entry['probe_prediction']:>10}  "
                    f"{entry['probe_confidence']:>10.4f}"
                )

    # ------------------------------------------------------------------
    # List all stored steering vectors
    # ------------------------------------------------------------------
    vectors = await list_steering_vectors()
    if not vectors.get("error"):
        print(f"\n  Stored steering vectors ({vectors['count']} total):")
        for v in vectors.get("vectors", []):
            print(
                f"    {v['name']}: layer={v['layer']} norm={v['vector_norm']:.4f} sep={v.get('separability_score', '?'):.4f}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  DIRECTION DEMO COMPLETE")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - extract_direction finds semantic directions via mean_diff, LDA, PCA, probe")
    print("  - All methods store in SteeringVectorRegistry → usable with steer_and_generate")
    print("  - direction_angles compares extracted directions: small angle = similar direction")
    print("  - probe_at_inference monitors a concept's activation token-by-token during generation")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction extraction + probe-at-inference demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
