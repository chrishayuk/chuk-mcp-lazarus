#!/usr/bin/env python3
"""
Attention analysis demo: patterns and head entropy.

This script demonstrates two attention analysis tools:

1. attention_pattern -- extract full attention weight matrices showing
   which tokens each head attends to at a given position.  Answers:
   "What is the last token looking at?"

2. attention_heads -- compute entropy and focus for every head across
   layers.  Low entropy = focused head (looks at specific tokens).
   High entropy = diffuse head (looks everywhere equally).  Answers:
   "Which heads are specialized vs generic?"

Together these reveal the model's information routing: which heads are
"doing work" (focused on specific tokens) vs attending uniformly, and
exactly what each focused head is looking at.

Usage:
    # Quick run (SmolLM2-135M -- validates tooling):
    python examples/attention_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/attention_demo.py --model google/gemma-3-4b-it
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


def bar_chart(value: float, width: int = 20, char: str = "#") -> str:
    """Return an ASCII bar scaled to width."""
    n = int(abs(value) * width)
    return char * max(n, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.generation_tools import predict_next_token
    from chuk_mcp_lazarus.tools.attention_tools import (
        attention_pattern,
        attention_heads,
    )

    t0 = time.time()
    prompt = "The capital of France is"

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  ATTENTION ANALYSIS DEMO")
    print(f"  Model: {model_id}")
    print(f"  Prompt: {prompt!r}")
    print(f"{'=' * 60}")

    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    num_layers = result["num_layers"]
    num_heads = result["num_attention_heads"]
    print(
        f"  Loaded: {result['family']} ({num_layers} layers, "
        f"{num_heads} heads, {result['hidden_dim']} hidden dim)"
    )

    # Quick prediction to set context
    pred = await predict_next_token(prompt=prompt, top_k=3)
    if not pred.get("error"):
        print("\n  Model predicts: ", end="")
        for p in pred["predictions"]:
            print(f"{p['token']!r}({p['probability']:.2%}) ", end="")
        print()

    # Pick representative layers: early, middle, late
    early = 0
    mid = num_layers // 2
    late = num_layers - 1
    demo_layers = [early, mid, late]

    # ==================================================================
    # Step 2: Attention patterns -- what does each head look at?
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 1: Attention patterns (attention_pattern)")
    print(f"  Layers: {demo_layers}")
    print(f"{'=' * 60}")
    print("\n  Showing which tokens the last token attends to, per head.")
    print("  This reveals information routing: where is the model looking?\n")

    result = await attention_pattern(
        prompt=prompt,
        layers=demo_layers,
        token_position=-1,
        top_k=5,
    )

    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    tokens = result["tokens"]
    print(f"  Tokens: {tokens}")
    print(f"  Analyzing position: {result['token_position']} ({result['token_text']!r})\n")

    for layer_entry in result["patterns"]:
        layer_idx = layer_entry["layer"]
        label = "EARLY" if layer_idx == early else "MIDDLE" if layer_idx == mid else "LATE"
        print(f"  Layer {layer_idx} ({label}) -- {layer_entry['num_heads']} heads:")
        print(f"  {'Head':>6}  {'Top-3 attended tokens (weight)':>50}")
        print(f"  {'----':>6}  {'-' * 50}")

        for head in layer_entry["heads"]:
            top3 = head["top_attended"][:3]
            attended_str = "  ".join(f"{a['token']!r}({a['weight']:.3f})" for a in top3)
            print(f"  {head['head']:>6}  {attended_str}")
        print()

    # ==================================================================
    # Step 3: Head entropy -- focused vs diffuse heads
    # ==================================================================
    print(f"{'=' * 60}")
    print("  STEP 2: Head entropy analysis (attention_heads)")
    print(f"  Layers: {demo_layers}")
    print(f"{'=' * 60}")
    print("\n  Entropy measures how spread out attention is:")
    print("  Low entropy  = focused (head looks at specific tokens)")
    print("  High entropy = diffuse (head looks everywhere equally)\n")

    result = await attention_heads(
        prompt=prompt,
        layers=demo_layers,
        top_k=3,
    )

    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    # Group heads by layer
    heads_by_layer: dict[int, list] = {}
    for h in result["heads"]:
        heads_by_layer.setdefault(h["layer"], []).append(h)

    for layer_idx in demo_layers:
        heads = heads_by_layer.get(layer_idx, [])
        label = "EARLY" if layer_idx == early else "MIDDLE" if layer_idx == mid else "LATE"
        print(f"  Layer {layer_idx} ({label}):")
        print(
            f"  {'Head':>6}  {'Entropy':>8}  {'Max Attn':>9}  {'Focus':>20}  {'Top attended':>30}"
        )
        print(
            f"  {'----':>6}  {'-------':>8}  {'--------':>9}  {'-----':>20}  {'------------':>30}"
        )

        for h in sorted(heads, key=lambda x: x["entropy"]):
            focus_bar = bar_chart(1.0 - h["entropy"], width=15)
            top_str = ", ".join(
                f"{a['token']!r}({a['weight']:.2f})" for a in h["top_attended_positions"][:2]
            )
            print(
                f"  {h['head']:>6}  {h['entropy']:>8.4f}  "
                f"{h['max_attention']:>9.4f}  "
                f"{focus_bar:>20}  {top_str:>30}"
            )
        print()

    # Summary
    summary = result["summary"]
    print("  Most FOCUSED heads (lowest entropy -- specialized):")
    for h in summary["most_focused_heads"][:5]:
        print(f"    Layer {h['layer']:>3}, Head {h['head']:>2}: entropy={h['entropy']:.4f}")

    print("\n  Most DIFFUSE heads (highest entropy -- generic):")
    for h in summary["most_diffuse_heads"][:5]:
        print(f"    Layer {h['layer']:>3}, Head {h['head']:>2}: entropy={h['entropy']:.4f}")

    # ==================================================================
    # Step 4: Full-model scan -- entropy evolution across layers
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 3: Entropy evolution across ALL layers")
    print(f"{'=' * 60}")
    print(f"\n  Scanning all {num_layers} layers to see how attention focus")
    print("  changes from early to late layers.\n")

    all_result = await attention_heads(
        prompt=prompt,
        layers=list(range(num_layers)),
        top_k=1,
    )

    if all_result.get("error"):
        print(f"  FAILED: {all_result.get('message')}")
    else:
        # Compute per-layer average entropy
        layer_entropies: dict[int, list[float]] = {}
        for h in all_result["heads"]:
            layer_entropies.setdefault(h["layer"], []).append(h["entropy"])

        print(f"  {'Layer':>5}  {'Avg Entropy':>12}  {'Min':>8}  {'Max':>8}  {'Focus':>25}")
        print(f"  {'-----':>5}  {'----------':>12}  {'---':>8}  {'---':>8}  {'-----':>25}")

        for layer in sorted(layer_entropies.keys()):
            vals = layer_entropies[layer]
            avg_ent = sum(vals) / len(vals)
            min_ent = min(vals)
            max_ent = max(vals)
            focus = bar_chart(1.0 - avg_ent, width=20)
            print(f"  {layer:>5}  {avg_ent:>12.4f}  {min_ent:>8.4f}  {max_ent:>8.4f}  {focus}")

        # Find the most focused individual head overall
        all_heads_sorted = sorted(all_result["heads"], key=lambda h: h["entropy"])
        most_focused = all_heads_sorted[0]
        print(
            f"\n  Most focused head overall: "
            f"Layer {most_focused['layer']}, Head {most_focused['head']} "
            f"(entropy={most_focused['entropy']:.4f}, "
            f"max_attn={most_focused['max_attention']:.4f})"
        )
        print("    Attends to: ", end="")
        for a in most_focused["top_attended_positions"]:
            print(f"{a['token']!r}({a['weight']:.3f}) ", end="")
        print()

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  ATTENTION ANALYSIS COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers, {num_heads} heads)")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - attention_pattern reveals what each head looks at --")
    print("    critical for understanding information routing.")
    print("  - attention_heads computes entropy to identify which heads")
    print("    are specialized (low entropy) vs generic (high entropy).")
    print("  - Early layers tend to have more diffuse attention (learning")
    print("    general context), while late layers often have focused")
    print("    heads that pinpoint specific information.")
    print("  - Combine with head_attribution to find which focused heads")
    print("    actually matter for specific predictions.")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention analysis demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
