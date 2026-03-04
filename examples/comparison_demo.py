#!/usr/bin/env python3
"""
Model comparison demo: Gemma 3 4B vs TranslateGemma 4B.

Compares base and fine-tuned models using low-resource languages where
TranslateGemma shows the biggest improvements (Icelandic, Swahili,
Estonian, Marathi) with French as a high-resource control.

Key insight from the TranslateGemma paper (arXiv:2601.09012):
- Embeddings were frozen during SFT, so early layers should be similar.
- Later layers should diverge, especially for low-resource languages.
- 4B MetricX gains: Icelandic 19.22->15.54, Swahili 14.05->10.65.

Steps:
1. Load primary model (gemma-3-4b-it)
2. Load comparison model (translategemma-4b-it)
3. Compare generations -- see actual output differences
4. Compare weights -- find which layers diverged most
5. Compare representations on low-resource translation prompts
6. Compare attention patterns
7. Unload comparison model

Usage:
    python examples/comparison_demo.py

    # Use different models:
    python examples/comparison_demo.py --base google/gemma-3-4b-it --ft google/translategemma-4b-it

    # Quick mode with fewer layers:
    python examples/comparison_demo.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


def pp(label: str, result: dict) -> None:
    """Pretty-print a tool result."""
    print(f"\n{'─' * 60}")
    print(f"  Step: {label}")
    print(f"{'─' * 60}")
    compact = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 20:
            compact[k] = f"[{len(v)} items]"
        elif isinstance(v, dict) and len(str(v)) > 200:
            compact[k] = f"{{... {len(v)} keys}}"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


async def main(base_id: str, ft_id: str, quick: bool = False) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model, get_model_info
    from chuk_mcp_lazarus.tools.comparison_tools import (
        load_comparison_model,
        compare_weights,
        compare_representations,
        compare_attention,
        compare_generations,
        unload_comparison_model,
    )

    t0 = time.time()

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON DEMO")
    print(f"  Base:       {base_id}")
    print(f"  Fine-tuned: {ft_id}")
    print("=" * 60)

    # ==================================================================
    # Step 1: Load primary model
    # ==================================================================
    result = await load_model(model_id=base_id)
    pp("1. Load primary model", result)
    if result.get("error"):
        print(f"\nFailed to load primary model: {result['message']}")
        return 1

    info = await get_model_info()
    num_layers = info["num_layers"]
    hidden_dim = info["hidden_dim"]
    print(f"\n  Architecture: {info['architecture']}, {num_layers} layers, {hidden_dim} hidden")
    print(f"  Parameters: {info['parameter_count']:,}")

    # ==================================================================
    # Step 2: Load comparison model
    # ==================================================================
    result = await load_comparison_model(model_id=ft_id)
    pp("2. Load comparison model", result)
    if result.get("error"):
        print(f"\nFailed to load comparison model: {result['message']}")
        return 1

    print(f"\n  Parameters: {result['parameter_count']:,}")
    param_diff = result["parameter_count"] - info["parameter_count"]
    print(f"  Parameter difference: {param_diff:+,}")

    # ==================================================================
    # Step 3: Compare generations -- see actual output differences
    # ==================================================================
    # Languages chosen for biggest MetricX improvements (4B model):
    #   Icelandic: 19.22 -> 15.54
    #   Swahili:   14.05 -> 10.65
    #   Estonian:  14.78 -> 11.03
    #   Marathi:    5.64 ->  4.30
    #   French:     3.76 ->  2.92 (high-resource control)
    prompts = [
        "Translate to English: Veðrið er fallegt í dag",  # Icelandic
        "Translate to English: Hali ya hewa ni nzuri leo",  # Swahili
        "Translate to English: Täna on ilus ilm",  # Estonian
        "Translate to English: आज हवामान छान आहे",  # Marathi
        "Translate to English: Il fait beau aujourd'hui",  # French (control)
    ]
    prompt_labels = ["Icelandic", "Swahili", "Estonian", "Marathi", "French"]

    # Compare generations for each language
    gen_prompts = prompts[:3] if quick else prompts  # Quick: first 3 only
    gen_labels = prompt_labels[:3] if quick else prompt_labels

    print(f"\n{'─' * 60}")
    print("  Step: 3. Compare generations (side-by-side)")
    print(f"{'─' * 60}")
    for prompt, label in zip(gen_prompts, gen_labels):
        result = await compare_generations(prompt=prompt, max_new_tokens=50)
        if not result.get("error"):
            match = "SAME" if result["outputs_match"] else "DIFF"
            print(f"\n  [{match}] {label}:")
            print(f"    Prompt:  {prompt}")
            print(f"    Base:    {result['primary_output'][:80]}")
            print(f"    Ftuned:  {result['comparison_output'][:80]}")
        else:
            print(f"\n  [ERR] {label}: {result.get('message', '')[:60]}")

    # ==================================================================
    # Step 4: Compare weights
    # ==================================================================
    if quick:
        # Sample layers: first, 25%, 50%, 75%, last
        weight_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    else:
        weight_layers = None  # All layers

    result = await compare_weights(layers=weight_layers)
    pp("4. Compare weights", result)
    if not result.get("error"):
        print(f"\n  Components compared: {result['summary']['total_components']}")
        print("\n  Top divergent layers (by avg Frobenius norm):")
        for entry in result["summary"]["top_divergent_layers"]:
            bar = "█" * int(entry["avg_frobenius"] * 200)
            print(f"    Layer {entry['layer']:>3}: {entry['avg_frobenius']:.6f} {bar}")

        # Show per-component breakdown for the most divergent layer
        top_layer = result["summary"]["top_divergent_layers"][0]["layer"]
        print(f"\n  Component breakdown for layer {top_layer}:")
        layer_divs = [d for d in result["divergences"] if d["layer"] == top_layer]
        for d in sorted(layer_divs, key=lambda x: x["frobenius_norm_diff"], reverse=True):
            print(
                f"    {d['component']:<12}: frob={d['frobenius_norm_diff']:.6f}  "
                f"cos={d['cosine_similarity']:.6f}"
            )
    else:
        print(f"\n  Weight comparison failed: {result.get('message')}")

    # ==================================================================
    # Step 5: Compare representations on low-resource prompts
    # ==================================================================
    if quick:
        repr_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    else:
        # Every 4th layer
        step = max(1, num_layers // 8)
        repr_layers = list(range(0, num_layers, step))
        if (num_layers - 1) not in repr_layers:
            repr_layers.append(num_layers - 1)

    result = await compare_representations(prompts=prompts, layers=repr_layers)
    pp("5. Compare representations (low-resource languages)", result)
    if not result.get("error"):
        print("\n  Layer-by-layer divergence (1 - cosine_similarity):")
        print(f"  {'Layer':>5}", end="")
        for label in prompt_labels:
            print(f"  {label[:8]:>10}", end="")
        print()
        print("  " + "-" * (7 + 12 * len(prompt_labels)))

        for layer_avg in result["layer_averages"]:
            layer = layer_avg["layer"]
            print(f"  {layer:>5}", end="")

            # Find per-prompt divergence for this layer
            for prompt in prompts:
                match = [
                    d
                    for d in result["divergences"]
                    if d["layer"] == layer and d["prompt"] == prompt
                ]
                if match:
                    div = 1.0 - match[0]["cosine_similarity"]
                    print(f"  {div:>10.4f}", end="")
                else:
                    print(f"  {'n/a':>10}", end="")
            print()

        # Summary: which language has highest average divergence?
        print("\n  Average divergence by language:")
        for i, (prompt, label) in enumerate(zip(prompts, prompt_labels)):
            divs = [d for d in result["divergences"] if d["prompt"] == prompt]
            if divs:
                avg_div = 1.0 - sum(d["cosine_similarity"] for d in divs) / len(divs)
                bar = "█" * int(avg_div * 100)
                print(f"    {label:<12}: {avg_div:.4f} {bar}")
    else:
        print(f"\n  Representation comparison failed: {result.get('message')}")

    # ==================================================================
    # Step 6: Compare attention patterns
    # ==================================================================
    # Use the Icelandic prompt (highest expected divergence)
    attn_prompt = prompts[0]  # Icelandic

    if quick:
        attn_layers = [0, num_layers // 2, num_layers - 1]
    else:
        # Sample 6 layers across the model
        attn_layers = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 2,
            num_layers - 1,
        ]
        attn_layers = sorted(set(attn_layers))

    result = await compare_attention(prompt=attn_prompt, layers=attn_layers)
    pp("6. Compare attention (Icelandic prompt)", result)
    if not result.get("error"):
        print("\n  Top 10 most divergent attention heads:")
        print(f"  {'Layer':>5} {'Head':>5} {'JS Div':>10} {'Cos Sim':>10}")
        print("  " + "-" * 35)
        for head in result["top_divergent_heads"][:10]:
            marker = " ***" if head["js_divergence"] > 0.1 else ""
            print(
                f"  {head['layer']:>5} {head['head']:>5} "
                f"{head['js_divergence']:>10.6f} {head['cosine_similarity']:>10.4f}{marker}"
            )

        # Per-layer summary
        print("\n  Per-layer average JS divergence:")
        for layer in attn_layers:
            layer_divs = [d for d in result["divergences"] if d["layer"] == layer]
            if layer_divs:
                avg_js = sum(d["js_divergence"] for d in layer_divs) / len(layer_divs)
                bar = "█" * int(avg_js * 200)
                print(f"    Layer {layer:>3}: {avg_js:.6f} {bar}")
    else:
        print(f"\n  Attention comparison failed: {result.get('message')}")

    # ==================================================================
    # Step 7: Unload comparison model
    # ==================================================================
    result = await unload_comparison_model()
    pp("7. Unload comparison model", result)

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISON DEMO COMPLETE")
    print(f"  Base:       {base_id}")
    print(f"  Fine-tuned: {ft_id}")
    print(f"  Completed all 7 steps in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison demo")
    parser.add_argument(
        "--base",
        default="google/gemma-3-4b-it",
        help="Base model ID (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--ft",
        default="google/translategemma-4b-it",
        help="Fine-tuned model ID (default: google/translategemma-4b-it)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: compare fewer layers",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.base, args.ft, args.quick)))
