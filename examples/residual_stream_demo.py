#!/usr/bin/env python3
"""
Residual stream demo: decomposition, clustering, and logit attribution.

This script demonstrates the residual stream tools:
- residual_decomposition: which component dominates at each layer?
- layer_clustering: where do different prompts converge/diverge?
- logit_attribution: where does the model's knowledge come from?

Steps:
1. Load model and inspect architecture
2. residual_decomposition on a factual recall prompt
3. residual_decomposition on a phrase completion prompt
4. Compare the two decomposition profiles
5. layer_clustering: multilingual convergence
6. layer_clustering with labels: language family separation
7. logit_attribution: which layers/components predict the target token?

Usage:
    # Quick run (SmolLM2-135M -- validates tooling):
    python examples/residual_stream_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/residual_stream_demo.py --model google/gemma-3-4b-it
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


def dual_bar(attn_frac: float, ffn_frac: float, width: int = 40) -> str:
    """Render an attention/FFN split as a dual bar: ◀attn | ffn▶."""
    attn_w = int(attn_frac * width)
    ffn_w = width - attn_w
    return f"{'◁' * attn_w}{'▷' * ffn_w}"


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
# Experiments
# ---------------------------------------------------------------------------

async def run_decomposition(
    label: str,
    prompt: str,
    layers: list[int],
    residual_decomposition,
) -> dict | None:
    """Run residual_decomposition and display results."""

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Prompt: {prompt!r}")
    print(f"{'=' * 60}")

    result = await residual_decomposition(
        prompt=prompt, layers=layers, position=-1,
    )
    pp("residual_decomposition", result)

    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return None

    print(f"\n  Token analyzed: {result['token_text']!r} "
          f"(position {result['token_position']})")
    print(f"\n  {'Layer':>5}  {'Total':>10}  {'Attn':>8}  {'FFN':>8}  "
          f"{'Dominant':>10}  {'◁ attention │ ffn ▷'}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 8}  {'─' * 8}  "
          f"{'─' * 10}  {'─' * 40}")

    for entry in result["layers"]:
        bar = dual_bar(entry["attention_fraction"], entry["ffn_fraction"])
        print(f"  {entry['layer']:>5}  {entry['total_norm']:>10.2f}  "
              f"{entry['attention_fraction']:>8.1%}  {entry['ffn_fraction']:>8.1%}  "
              f"{entry['dominant_component']:>10}  {bar}")

    s = result["summary"]
    print(f"\n  Summary:")
    print(f"    Attention-dominant layers: {s['attention_dominant_count']}")
    print(f"    FFN-dominant layers:       {s['ffn_dominant_count']}")
    print(f"    Peak layer: {s['peak_layer']} "
          f"(norm={s['peak_total_norm']:.2f}, {s['peak_component']})")

    return result


async def run_comparison(
    decomp_a: dict,
    decomp_b: dict,
    label_a: str,
    label_b: str,
) -> None:
    """Compare two decomposition profiles side by side."""

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'=' * 60}")

    layers_a = {e["layer"]: e for e in decomp_a["layers"]}
    layers_b = {e["layer"]: e for e in decomp_b["layers"]}

    common_layers = sorted(set(layers_a.keys()) & set(layers_b.keys()))

    print(f"\n  {'Layer':>5}  {'Attn A':>8}  {'Attn B':>8}  "
          f"{'FFN A':>8}  {'FFN B':>8}  {'Δ Attn':>8}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 8}  "
          f"{'─' * 8}  {'─' * 8}  {'─' * 8}")

    for layer in common_layers:
        a = layers_a[layer]
        b = layers_b[layer]
        delta = a["attention_fraction"] - b["attention_fraction"]
        print(f"  {layer:>5}  {a['attention_fraction']:>8.1%}  "
              f"{b['attention_fraction']:>8.1%}  "
              f"{a['ffn_fraction']:>8.1%}  {b['ffn_fraction']:>8.1%}  "
              f"{delta:>+8.1%}")

    print(f"\n  Positive Δ = more attention-heavy in {label_a}")
    print(f"  Negative Δ = more attention-heavy in {label_b}")


async def run_clustering(
    label: str,
    prompts: list[str],
    layers: list[int],
    layer_clustering,
    labels: list[str] | None = None,
) -> dict | None:
    """Run layer_clustering and display results."""

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    for i, p in enumerate(prompts):
        tag = f" [{labels[i]}]" if labels else ""
        print(f"    {i + 1}. {p!r}{tag}")

    result = await layer_clustering(
        prompts=prompts, layers=layers, labels=labels, position=-1,
    )
    pp("layer_clustering", result)

    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return None

    print(f"\n  Layers analyzed: {result['num_layers_analyzed']}")

    for entry in result["layers"]:
        sim = entry["similarity_matrix"]
        n = len(sim)
        print(f"\n  Layer {entry['layer']}  (mean similarity: {entry['mean_similarity']:.4f})")

        # Header
        header = "          "
        for j in range(n):
            p_label = prompts[j][:8]
            header += f"  {p_label:>8}"
        print(f"  {header}")

        # Rows
        for i in range(n):
            row_label = prompts[i][:8]
            row = f"  {row_label:>8}"
            for j in range(n):
                val = sim[i][j]
                row += f"  {val:>8.4f}"
            print(row)

        if entry.get("separation_score") is not None:
            print(f"    Separation score: {entry['separation_score']:.4f}")

    s = result["summary"]
    print(f"\n  Summary:")
    print(f"    Most similar layer:  {s['most_similar_layer']} "
          f"(sim={s['most_similar_value']:.4f})")
    print(f"    Least similar layer: {s['least_similar_layer']} "
          f"(sim={s['least_similar_value']:.4f})")

    if "best_separation_layer" in s:
        print(f"    Best separation:     layer {s['best_separation_layer']} "
              f"(score={s['best_separation_score']:.4f})")

    if "separation_trend" in s:
        print(f"\n  Separation trend:")
        for entry in s["separation_trend"]:
            bar_len = max(0, int(entry["separation"] * 40))
            bar = "█" * bar_len
            print(f"    Layer {entry['layer']:>3}: {entry['separation']:>+.4f}  {bar}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.residual_tools import (
        residual_decomposition,
        layer_clustering,
        logit_attribution,
    )

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  RESIDUAL STREAM DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    num_layers = result["num_layers"]
    print(f"  Loaded: {result['family']} ({num_layers} layers, "
          f"{result['hidden_dim']} hidden dim)")

    all_layers = list(range(num_layers))
    decomp_layers = sample_layers(all_layers, max_layers=12)
    cluster_layers = sample_layers(all_layers, max_layers=6)

    # ==================================================================
    # Step 2: Decomposition — factual recall
    # ==================================================================
    decomp_a = await run_decomposition(
        label="DECOMPOSITION A: factual recall",
        prompt="The capital of France is",
        layers=decomp_layers,
        residual_decomposition=residual_decomposition,
    )

    # ==================================================================
    # Step 3: Decomposition — phrase completion
    # ==================================================================
    decomp_b = await run_decomposition(
        label="DECOMPOSITION B: phrase completion",
        prompt="The cat sat on the",
        layers=decomp_layers,
        residual_decomposition=residual_decomposition,
    )

    # ==================================================================
    # Step 4: Compare the two profiles
    # ==================================================================
    if decomp_a and decomp_b:
        await run_comparison(
            decomp_a, decomp_b,
            label_a="factual recall",
            label_b="phrase completion",
        )

    # ==================================================================
    # Step 5: Clustering — multilingual convergence
    # ==================================================================
    await run_clustering(
        label="CLUSTERING A: multilingual convergence",
        prompts=[
            "The capital of France is",
            "La capitale de la France est",
            "Die Hauptstadt von Frankreich ist",
            "La capital de Francia es",
        ],
        layers=cluster_layers,
        layer_clustering=layer_clustering,
    )

    # ==================================================================
    # Step 6: Clustering with labels — language family separation
    # ==================================================================
    await run_clustering(
        label="CLUSTERING B: language family separation",
        prompts=[
            "The cat sat on the mat",
            "The dog ran through the park",
            "Le chat est assis sur le tapis",
            "Le chien a couru dans le parc",
        ],
        layers=cluster_layers,
        layer_clustering=layer_clustering,
        labels=["english", "english", "french", "french"],
    )

    # ==================================================================
    # Step 7: Logit attribution — where does the prediction come from?
    # ==================================================================
    prompt_attr = "The capital of France is"

    print(f"\n{'=' * 60}")
    print(f"  LOGIT ATTRIBUTION")
    print(f"  Prompt: {prompt_attr!r}")
    print(f"{'=' * 60}")

    result = await logit_attribution(
        prompt=prompt_attr, layers=decomp_layers, position=-1,
        target_token="Paris",
    )
    pp("logit_attribution", result)

    if not result.get("error"):
        print(f"\n  Target: {result['target_token']!r} "
              f"(id={result['target_token_id']})")
        print(f"  Model logit: {result['model_logit']:.2f}  "
              f"probability: {result['model_probability']:.4f}")
        print(f"  Embedding contribution: {result['embedding_logit']:+.2f}")

        print(f"\n  {'Layer':>5}  {'Attn':>10}  {'FFN':>10}  "
              f"{'Total':>10}  {'Cumul':>10}  {'AttnPred':>12}  {'FFNPred':>12}")
        print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}  "
              f"{'─' * 10}  {'─' * 10}  {'─' * 12}  {'─' * 12}")

        for entry in result["layers"]:
            print(f"  {entry['layer']:>5}  "
                  f"{entry['attention_logit']:>+10.2f}  "
                  f"{entry['ffn_logit']:>+10.2f}  "
                  f"{entry['total_logit']:>+10.2f}  "
                  f"{entry['cumulative_logit']:>+10.2f}  "
                  f"{entry['attention_top_token']!r:>12}  "
                  f"{entry['ffn_top_token']!r:>12}")

        s = result["summary"]
        print(f"\n  Summary:")
        print(f"    Top positive layer: {s['top_positive_layer']} "
              f"(logit={s['top_positive_logit']:+.2f})")
        print(f"    Top negative layer: {s['top_negative_layer']} "
              f"(logit={s['top_negative_logit']:+.2f})")
        print(f"    Total attention: {s['total_attention_logit']:+.2f}  "
              f"Total FFN: {s['total_ffn_logit']:+.2f}  "
              f"→ {s['dominant_component']} dominant")
        print(f"    Attribution sum: {result['attribution_sum']:.2f} "
              f"(model logit: {result['model_logit']:.2f})")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  RESIDUAL STREAM DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print(f"\n  Takeaways:")
    print(f"  - residual_decomposition shows WHETHER attention or MLP")
    print(f"    dominates at each layer for a given prompt")
    print(f"  - Different prompts may have different attn/MLP profiles")
    print(f"    (factual recall vs pattern completion)")
    print(f"  - layer_clustering shows WHERE prompts converge/diverge")
    print(f"    in representation space across layers")
    print(f"  - With labels, separation_score reveals the layer where")
    print(f"    categories (e.g. languages) are most distinguishable")
    print(f"  - logit_attribution shows WHERE the model's knowledge")
    print(f"    comes from: which layers/components push toward the")
    print(f"    target token, with logit lens predictions at each step")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual stream demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
