#!/usr/bin/env python3
"""
Deep dive demo: from prediction to individual heads and neurons.

This script demonstrates the full interpretability pipeline:
1. Establish prediction (generate_text + predict_next_token)
2. Layer-level attribution (logit_attribution → find peak layers)
3. Head attribution at peak attention layer
4. Neuron analysis at peak FFN layer
5. Cross-reference: attention_pattern at peak layer
6. Embedding neighbors for the target token

Usage:
    # Quick run (SmolLM2-135M -- validates tooling):
    python examples/deep_dive_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/deep_dive_demo.py --model google/gemma-3-4b-it
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
        if isinstance(v, list) and len(v) > 12:
            compact[k] = f"[{len(v)} items]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


def sample_layers(all_layers: list[int], max_layers: int = 12) -> list[int]:
    """Pick an evenly-spaced subset of layers (always include first + last)."""
    if len(all_layers) <= max_layers:
        return all_layers
    step = max(1, (len(all_layers) - 1) // (max_layers - 1))
    sampled = list(range(all_layers[0], all_layers[-1], step))
    if all_layers[-1] not in sampled:
        sampled.append(all_layers[-1])
    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.generation_tools import (
        generate_text,
        predict_next_token,
        embedding_neighbors,
    )
    from chuk_mcp_lazarus.tools.residual_tools import (
        logit_attribution,
        head_attribution,
        top_neurons,
    )
    from chuk_mcp_lazarus.tools.attention_tools import attention_pattern

    t0 = time.time()
    prompt = "The capital of France is"

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  DEEP DIVE DEMO")
    print(f"  Model: {model_id}")
    print(f"  Prompt: {prompt!r}")
    print(f"{'=' * 60}")

    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    num_layers = result["num_layers"]
    print(f"  Loaded: {result['family']} ({num_layers} layers, {result['hidden_dim']} hidden dim)")

    all_layers = list(range(num_layers))
    attr_layers = sample_layers(all_layers, max_layers=12)

    # ==================================================================
    # Step 2: Establish prediction
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 1: What does the model predict?")
    print(f"{'=' * 60}")

    gen_result = await generate_text(prompt=prompt, max_new_tokens=10, temperature=0.0)
    print(f"  Generation: {prompt}{gen_result['output']}")

    pred_result = await predict_next_token(prompt=prompt, top_k=5)
    print("\n  Top-5 next-token predictions:")
    for p in pred_result["predictions"]:
        bar_len = max(1, int(p["probability"] * 40))
        bar = "█" * bar_len
        print(f"    {p['token']:>12} ({p['probability']:>6.2%})  {bar}")

    target = pred_result["predictions"][0]["token"].strip()
    print(f"\n  → Target token for analysis: {target!r}")

    # ==================================================================
    # Step 3: Layer-level attribution
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 2: Which layers matter? (logit_attribution)")
    print(f"{'=' * 60}")

    attr_result = await logit_attribution(
        prompt=prompt,
        layers=attr_layers,
        position=-1,
        target_token=target,
    )
    pp("logit_attribution", attr_result)

    if attr_result.get("error"):
        print(f"  FAILED: {attr_result.get('message')}")
        return 1

    # Find peak attention and FFN layers
    best_attn_layer = max(attr_result["layers"], key=lambda e: e["attention_logit"])
    best_ffn_layer = max(attr_result["layers"], key=lambda e: e["ffn_logit"])

    print(
        f"\n  Model logit for {target!r}: {attr_result['model_logit']:.2f} "
        f"(prob={attr_result['model_probability']:.4f})"
    )
    print("\n  Per-layer breakdown:")

    for entry in attr_result["layers"]:
        total = entry["total_logit"]
        marker = ""
        if entry["layer"] == best_attn_layer["layer"]:
            marker += " ← PEAK ATTN"
        if entry["layer"] == best_ffn_layer["layer"]:
            marker += " ← PEAK FFN"
        print(
            f"    Layer {entry['layer']:>3}: "
            f"attn={entry['attention_logit']:>+7.2f}  "
            f"ffn={entry['ffn_logit']:>+7.2f}  "
            f"total={total:>+7.2f}{marker}"
        )

    print(
        f"\n  → Peak attention layer: {best_attn_layer['layer']} "
        f"(logit={best_attn_layer['attention_logit']:+.2f})"
    )
    print(
        f"  → Peak FFN layer: {best_ffn_layer['layer']} (logit={best_ffn_layer['ffn_logit']:+.2f})"
    )

    # ==================================================================
    # Step 4: Head attribution at peak attention layer
    # ==================================================================
    peak_attn = best_attn_layer["layer"]

    print(f"\n{'=' * 60}")
    print(f"  STEP 3: Which heads at layer {peak_attn}? (head_attribution)")
    print(f"{'=' * 60}")

    head_result = await head_attribution(
        prompt=prompt,
        layer=peak_attn,
        target_token=target,
    )
    pp("head_attribution", head_result)

    if head_result.get("error"):
        print(f"  FAILED: {head_result.get('message')}")
    else:
        print(f"\n  Layer {peak_attn}: {head_result['num_heads']} heads")
        print(f"  Layer total logit (raw DLA): {head_result['layer_total_logit']:+.4f}")

        # Sort heads by absolute contribution
        sorted_heads = sorted(
            head_result["heads"],
            key=lambda h: abs(h["logit_contribution"]),
            reverse=True,
        )

        print(f"\n  {'Head':>6}  {'Logit':>10}  {'Share':>8}  {'Points→':>10}")
        print(f"  {'─' * 6}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
        for h in sorted_heads[:10]:
            print(
                f"  {h['head']:>6}  {h['logit_contribution']:>+10.4f}  "
                f"{h['fraction_of_layer']:>8.1%}  "
                f"{h['top_token']!r:>10}"
            )

        s = head_result["summary"]
        print(
            f"\n  Top positive head: {s['top_positive_head']} "
            f"(logit={s['top_positive_logit']:+.4f})"
        )
        print(
            f"  Top negative head: {s['top_negative_head']} (logit={s['top_negative_logit']:+.4f})"
        )
        print(f"  Concentration (top-3 share): {s['concentration']:.1%}")

    # ==================================================================
    # Step 5: Neuron analysis at peak FFN layer
    # ==================================================================
    peak_ffn = best_ffn_layer["layer"]

    print(f"\n{'=' * 60}")
    print(f"  STEP 4: Which neurons at layer {peak_ffn}? (top_neurons)")
    print(f"{'=' * 60}")

    neuron_result = await top_neurons(
        prompt=prompt,
        layer=peak_ffn,
        target_token=target,
        top_k=10,
    )
    pp("top_neurons", neuron_result)

    if neuron_result.get("error"):
        print(f"  FAILED: {neuron_result.get('message')}")
    else:
        print(
            f"\n  MLP type: {neuron_result['mlp_type']} "
            f"({neuron_result['intermediate_size']} neurons)"
        )
        print(f"  Total neuron logit (raw DLA): {neuron_result['total_neuron_logit']:+.4f}")

        print("\n  Top PROMOTING neurons:")
        print(f"  {'Neuron':>8}  {'Activation':>12}  {'Logit':>10}  {'Points→':>10}")
        print(f"  {'─' * 8}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
        for n in neuron_result["top_positive"]:
            print(
                f"  {n['neuron_index']:>8}  {n['activation']:>12.4f}  "
                f"{n['logit_contribution']:>+10.4f}  "
                f"{n['top_token']!r:>10}"
            )

        print("\n  Top SUPPRESSING neurons:")
        print(f"  {'Neuron':>8}  {'Activation':>12}  {'Logit':>10}  {'Points→':>10}")
        print(f"  {'─' * 8}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
        for n in neuron_result["top_negative"]:
            print(
                f"  {n['neuron_index']:>8}  {n['activation']:>12.4f}  "
                f"{n['logit_contribution']:>+10.4f}  "
                f"{n['top_token']!r:>10}"
            )

        s = neuron_result["summary"]
        print(f"\n  Sparsity: {s['sparsity']:.1%} of neurons are active")
        print(f"  Top-10 concentration: {s['concentration_top10']:.4f}")

    # ==================================================================
    # Step 6: Attention pattern at peak layer
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  STEP 5: What does layer {peak_attn} attend to? (attention_pattern)")
    print(f"{'=' * 60}")

    attn_result = await attention_pattern(
        prompt=prompt,
        layers=[peak_attn],
        token_position=-1,
        top_k=5,
    )

    if attn_result.get("error"):
        print(f"  FAILED: {attn_result.get('message')}")
    else:
        for layer_entry in attn_result.get("patterns", []):
            print(f"\n  Layer {layer_entry['layer']}:")
            for head_entry in layer_entry.get("heads", []):
                head_idx = head_entry["head"]
                top_attn = head_entry.get("top_attended", [])
                tokens_str = ", ".join(f"{a['token']!r}({a['weight']:.3f})" for a in top_attn[:3])
                print(f"    Head {head_idx:>2}: {tokens_str}")

    # ==================================================================
    # Step 7: Embedding neighbors for target token
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  STEP 6: What's near {target!r} in embedding space?")
    print(f"{'=' * 60}")

    embed_result = await embedding_neighbors(token=target, top_k=10)

    if embed_result.get("error"):
        print(f"  FAILED: {embed_result.get('message')}")
    else:
        print(
            f"  Query: {embed_result['query_token']!r} → "
            f"{embed_result['resolved_form']!r} "
            f"(id={embed_result['query_token_id']})"
        )
        print(f"  Self similarity: {embed_result['self_similarity']:.4f}")
        print(f"\n  {'Rank':>4}  {'Token':>12}  {'Similarity':>10}")
        print(f"  {'─' * 4}  {'─' * 12}  {'─' * 10}")
        for i, n in enumerate(embed_result["neighbors"], 1):
            print(f"  {i:>4}  {n['token']!r:>12}  {n['cosine_similarity']:>10.4f}")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  DEEP DIVE COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print(f"  Prompt: {prompt!r} → {target!r}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Pipeline summary:")
    print(f"  1. predict_next_token → model predicts {target!r}")
    print(
        f"  2. logit_attribution  → peak attention at layer {peak_attn}, "
        f"peak FFN at layer {peak_ffn}"
    )
    if not head_result.get("error"):
        s = head_result["summary"]
        print(
            f"  3. head_attribution   → head {s['top_positive_head']} "
            f"at layer {peak_attn} contributes most"
        )
    if not neuron_result.get("error"):
        s = neuron_result["summary"]
        print(
            f"  4. top_neurons        → {s['sparsity']:.1%} of neurons active, "
            f"top-10 concentration={s['concentration_top10']:.4f}"
        )
    print(f"  5. attention_pattern  → shows what layer {peak_attn}'s heads attend to")
    print(f"  6. embedding_neighbors → tokens near {target!r} in static embedding space")

    print("\n  This is the full drill-down: from 'what does the model predict?'")
    print("  down to individual attention heads and MLP neurons responsible")
    print("  for pushing that prediction.")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep dive demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
