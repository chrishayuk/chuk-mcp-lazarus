#!/usr/bin/env python3
"""
Logit attribution demo: where does the model's knowledge come from?

This script demonstrates the logit_attribution tool which uses
Direct Logit Attribution (DLA) to trace how each layer's attention
and FFN components contribute to the predicted token's logit.

Steps:
1. Load model
2. Attribute "The capital of France is" → Paris (factual recall)
3. Attribute with a different target token (Lyon vs Paris)
4. Attribute "The cat sat on the" (phrase completion, auto-detect)
5. Compare attribution profiles for factual vs phrase completion
6. Show the knowledge localization pipeline: decomposition + attribution

Usage:
    # Quick run (SmolLM2-135M -- validates tooling):
    python examples/logit_attribution_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/logit_attribution_demo.py --model google/gemma-3-4b-it
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


def logit_bar(value: float, max_abs: float, width: int = 30) -> str:
    """Render a logit contribution as a centered bar: negative ◁ | ▷ positive."""
    if max_abs == 0:
        return " " * width
    half = width // 2
    scale = min(abs(value) / max_abs, 1.0)
    bar_len = max(1, int(scale * half))
    if value >= 0:
        return " " * half + "│" + "▷" * bar_len + " " * (half - bar_len)
    else:
        padding = half - bar_len
        return " " * padding + "◁" * bar_len + "│" + " " * half


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


async def run_attribution(
    label: str,
    prompt: str,
    layers: list[int],
    logit_attribution,
    target_token: str | None = None,
) -> dict | None:
    """Run logit_attribution and display results with ASCII visualization."""

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Prompt: {prompt!r}")
    if target_token:
        print(f"  Target: {target_token!r} (user-specified)")
    print(f"{'=' * 60}")

    kwargs = dict(prompt=prompt, layers=layers, position=-1)
    if target_token is not None:
        kwargs["target_token"] = target_token

    result = await logit_attribution(**kwargs)
    pp("logit_attribution", result)

    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return None

    print(f"\n  Predicted: {result['target_token']!r} (prob={result['model_probability']:.4f})")
    print(f"  Model logit (with norm): {result['model_logit']:.2f}")
    print(f"  Attribution sum (DLA):   {result['attribution_sum']:.2f}")
    print(f"  Embedding contribution:  {result['embedding_logit']:+.2f}")

    # Find max absolute logit for bar scaling
    all_logits = [result["embedding_logit"]]
    for entry in result["layers"]:
        all_logits.extend([entry["attention_logit"], entry["ffn_logit"]])
    max_abs = max(abs(v) for v in all_logits) if all_logits else 1.0

    # Table header
    print(
        f"\n  {'Layer':>5}  {'Attn':>8}  {'FFN':>8}  "
        f"{'Total':>8}  {'Cumul':>8}  "
        f"{'◁ negative │ positive ▷':^30}  "
        f"{'AttnPred':>8}  {'FFNPred':>8}"
    )
    print(
        f"  {'─' * 5}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 30}  {'─' * 8}  {'─' * 8}"
    )

    # Embedding row
    embed_bar = logit_bar(result["embedding_logit"], max_abs)
    print(
        f"  {'embed':>5}  {'':>8}  {'':>8}  "
        f"{result['embedding_logit']:>+8.2f}  "
        f"{result['embedding_logit']:>+8.2f}  "
        f"{embed_bar}  "
        f"{'':>8}  {'':>8}"
    )

    # Per-layer rows
    for entry in result["layers"]:
        total_bar = logit_bar(entry["total_logit"], max_abs)
        attn_tok = entry["attention_top_token"][:8]
        ffn_tok = entry["ffn_top_token"][:8]
        print(
            f"  {entry['layer']:>5}  "
            f"{entry['attention_logit']:>+8.2f}  "
            f"{entry['ffn_logit']:>+8.2f}  "
            f"{entry['total_logit']:>+8.2f}  "
            f"{entry['cumulative_logit']:>+8.2f}  "
            f"{total_bar}  "
            f"{attn_tok:>8}  "
            f"{ffn_tok:>8}"
        )

    s = result["summary"]
    print("\n  Summary:")
    print(
        f"    Top positive: layer {s['top_positive_layer']} (logit={s['top_positive_logit']:+.2f})"
    )
    print(
        f"    Top negative: layer {s['top_negative_layer']} (logit={s['top_negative_logit']:+.2f})"
    )
    print(f"    Attention total: {s['total_attention_logit']:+.2f}")
    print(f"    FFN total:       {s['total_ffn_logit']:+.2f}")
    print(f"    Dominant: {s['dominant_component']}")

    return result


async def run_comparison(
    result_a: dict,
    result_b: dict,
    label_a: str,
    label_b: str,
) -> None:
    """Compare attribution profiles side by side."""

    print(f"\n{'=' * 60}")
    print("  ATTRIBUTION COMPARISON")
    print(f"  A: {label_a} → {result_a['target_token']!r}")
    print(f"  B: {label_b} → {result_b['target_token']!r}")
    print(f"{'=' * 60}")

    layers_a = {e["layer"]: e for e in result_a["layers"]}
    layers_b = {e["layer"]: e for e in result_b["layers"]}
    common = sorted(set(layers_a.keys()) & set(layers_b.keys()))

    print(
        f"\n  {'Layer':>5}  {'Total A':>10}  {'Total B':>10}  "
        f"{'Δ Total':>10}  {'FFN A':>10}  {'FFN B':>10}"
    )
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    for layer in common:
        a = layers_a[layer]
        b = layers_b[layer]
        delta = a["total_logit"] - b["total_logit"]
        print(
            f"  {layer:>5}  {a['total_logit']:>+10.2f}  "
            f"{b['total_logit']:>+10.2f}  "
            f"{delta:>+10.2f}  "
            f"{a['ffn_logit']:>+10.2f}  "
            f"{b['ffn_logit']:>+10.2f}"
        )

    sa = result_a["summary"]
    sb = result_b["summary"]
    print(
        f"\n  A attribution sum: {result_a['attribution_sum']:+.2f}  "
        f"model logit: {result_a['model_logit']:+.2f}"
    )
    print(
        f"  B attribution sum: {result_b['attribution_sum']:+.2f}  "
        f"model logit: {result_b['model_logit']:+.2f}"
    )
    print(f"  A dominant: {sa['dominant_component']}  B dominant: {sb['dominant_component']}")


async def run_pipeline(
    prompt: str,
    layers: list[int],
    residual_decomposition,
    logit_attribution,
    target_token: str | None = None,
) -> None:
    """Show the decomposition → attribution pipeline."""

    print(f"\n{'=' * 60}")
    print("  KNOWLEDGE LOCALIZATION PIPELINE")
    print(f"  Prompt: {prompt!r}")
    print(f"{'=' * 60}")

    # Step A: Decomposition (norm-based)
    decomp = await residual_decomposition(
        prompt=prompt,
        layers=layers,
        position=-1,
    )
    if decomp.get("error"):
        print(f"  Decomposition FAILED: {decomp.get('message')}")
        return

    # Step B: Attribution (logit-based)
    kwargs = dict(prompt=prompt, layers=layers, position=-1)
    if target_token is not None:
        kwargs["target_token"] = target_token
    attr = await logit_attribution(**kwargs)
    if attr.get("error"):
        print(f"  Attribution FAILED: {attr.get('message')}")
        return

    print(f"\n  Target: {attr['target_token']!r} (prob={attr['model_probability']:.4f})")

    # Combined view
    decomp_by_layer = {e["layer"]: e for e in decomp["layers"]}
    attr_by_layer = {e["layer"]: e for e in attr["layers"]}
    common = sorted(set(decomp_by_layer.keys()) & set(attr_by_layer.keys()))

    print(
        f"\n  {'Layer':>5}  {'Norm':>8}  {'Dominant':>10}  "
        f"{'Attn→logit':>11}  {'FFN→logit':>10}  "
        f"{'Predicts':>10}"
    )
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 10}  {'─' * 11}  {'─' * 10}  {'─' * 10}")

    for layer in common:
        d = decomp_by_layer[layer]
        a = attr_by_layer[layer]
        ffn_tok = a["ffn_top_token"][:10]
        print(
            f"  {layer:>5}  {d['total_norm']:>8.1f}  "
            f"{d['dominant_component']:>10}  "
            f"{a['attention_logit']:>+11.2f}  "
            f"{a['ffn_logit']:>+10.2f}  "
            f"{ffn_tok:>10}"
        )

    print("\n  Reading the table:")
    print("  - 'Norm' shows HOW MUCH each layer contributes (magnitude)")
    print("  - 'Dominant' shows WHETHER attention or FFN dominates (by norm)")
    print("  - 'Attn/FFN→logit' shows WHAT each component does to the")
    print("    target token (positive = pushes toward, negative = away)")
    print("  - 'Predicts' shows the logit lens prediction after FFN")
    print("    (what the model would output if computation stopped here)")
    print("\n  A layer can have a large norm (high activity) but push AWAY")
    print("  from the prediction -- norm and logit attribution diverge!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.residual_tools import (
        residual_decomposition,
        logit_attribution,
    )

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  LOGIT ATTRIBUTION DEMO")
    print(f"  Model: {model_id}")
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
    # Step 2: Attribution — factual recall (target='Paris')
    # ==================================================================
    attr_a = await run_attribution(
        label="ATTRIBUTION A: factual recall (target='Paris')",
        prompt="The capital of France is",
        layers=attr_layers,
        logit_attribution=logit_attribution,
        target_token="Paris",
    )

    # ==================================================================
    # Step 3: Attribution — same prompt, wrong answer (target='Lyon')
    # ==================================================================
    attr_b = await run_attribution(
        label="ATTRIBUTION B: factual recall (target='Lyon')",
        prompt="The capital of France is",
        layers=attr_layers,
        logit_attribution=logit_attribution,
        target_token="Lyon",
    )

    # ==================================================================
    # Step 4: Attribution — phrase completion (auto-detect target)
    # ==================================================================
    await run_attribution(
        label="ATTRIBUTION C: phrase completion (auto target)",
        prompt="The cat sat on the",
        layers=attr_layers,
        logit_attribution=logit_attribution,
    )

    # ==================================================================
    # Step 5: Compare Paris vs Lyon on the same prompt
    # ==================================================================
    if attr_a and attr_b:
        await run_comparison(
            attr_a,
            attr_b,
            label_a="correct (Paris)",
            label_b="wrong (Lyon)",
        )

    # ==================================================================
    # Step 6: Knowledge localization pipeline (norm + logit combined)
    # ==================================================================
    await run_pipeline(
        prompt="The capital of France is",
        layers=attr_layers,
        residual_decomposition=residual_decomposition,
        logit_attribution=logit_attribution,
        target_token="Paris",
    )

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  LOGIT ATTRIBUTION DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - logit_attribution reveals WHERE knowledge comes from:")
    print("    which layers/components push toward the predicted token")
    print("  - Comparing correct vs wrong answers (Paris vs Lyon) shows")
    print("    which layers discriminate -- the same layers that boost")
    print("    the correct answer often suppress the wrong one")
    print("  - Logit lens predictions show the model's evolving 'guess':")
    print("    watch how the prediction changes layer by layer")
    print("  - Attribution sum matches model logit (normalized mode)")
    print("    so contributions are on the same scale as actual logits")
    print("  - The pipeline view (decomposition + attribution) reveals")
    print("    the difference between 'high activity' and 'helpful")
    print("    activity' -- a layer can be busy but counterproductive")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit attribution demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
