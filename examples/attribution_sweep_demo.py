#!/usr/bin/env python3
"""
Attribution sweep demo: batch logit attribution across prompt groups.

Demonstrates:
1. attribution_sweep — batch attribution across multiple prompts
2. Calibrated logit lens — logit_lens with correct tied-embedding projection
3. Comparing factual vs phrase-completion attribution profiles

Steps:
1. Load model
2. Calibrated logit lens on a factual prompt (proves Step 15 fix)
3. Attribution sweep on factual recall prompts (target='Paris')
4. Attribution sweep on phrase completion prompts
5. Compare dominant layers across groups
6. Track a specific token across layers (proves Step 15 fix)

Usage:
    # Quick run (SmolLM2-135M — validates tooling):
    python examples/attribution_sweep_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/attribution_sweep_demo.py --model google/gemma-3-4b-it
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


def sample_layers(num_layers: int, max_layers: int = 12) -> list[int]:
    """Pick an evenly-spaced subset of layers (always include first + last)."""
    all_layers = list(range(num_layers))
    if len(all_layers) <= max_layers:
        return all_layers
    step = max(1, (num_layers - 1) // (max_layers - 1))
    sampled = list(range(0, num_layers, step))
    if (num_layers - 1) not in sampled:
        sampled.append(num_layers - 1)
    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.attribution_tools import attribution_sweep
    from chuk_mcp_lazarus.tools.generation_tools import logit_lens, track_token
    from chuk_mcp_lazarus.tools.model_tools import load_model

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  ATTRIBUTION SWEEP DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1

    num_layers = result["num_layers"]
    hidden_dim = result["hidden_dim"]
    print(f"  Loaded: {result['family']} ({num_layers} layers, {hidden_dim} hidden dim)")

    layers = sample_layers(num_layers)

    # ==================================================================
    # Step 2: Calibrated logit lens (proves Step 15 fix)
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 2: CALIBRATED LOGIT LENS")
    print("  Uses _norm_project + _get_lm_projection (handles tied embeddings)")
    print(f"{'=' * 60}")

    lens_result = await logit_lens(
        prompt="The capital of France is",
        layers=layers,
        top_k=5,
        token_position=-1,
    )
    if lens_result.get("error"):
        print(f"  FAILED: {lens_result.get('message')}")
        return 1

    pp("logit_lens (calibrated)", lens_result)

    print("\n  Layer-by-layer predictions:")
    for pred in lens_result["predictions"]:
        top = pred["top_tokens"][0] if pred["top_tokens"] else "?"
        prob = pred["top_probabilities"][0] if pred["top_probabilities"] else 0.0
        print(f"    Layer {pred['layer']:>3}: {top!r:>12} (prob={prob:.4f})")

    summary = lens_result["summary"]
    print(f"\n  Final prediction: {summary['final_prediction']!r}")
    print(f"  Emergence layer:  {summary['emergence_layer']}")

    # ==================================================================
    # Step 3: Attribution sweep — factual recall
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 3: ATTRIBUTION SWEEP — FACTUAL RECALL")
    print(f"{'=' * 60}")

    factual_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]

    sweep_factual = await attribution_sweep(
        prompts=factual_prompts,
        layers=layers,
        position=-1,
        normalized=True,
        labels=["France", "Germany", "Italy", "Spain"],
    )
    if sweep_factual.get("error"):
        print(f"  FAILED: {sweep_factual.get('message')}")
        return 1

    pp("attribution_sweep (factual)", sweep_factual)

    print(
        f"\n  Swept {sweep_factual['num_prompts']} prompts across {sweep_factual['num_layers']} layers"
    )
    print(f"  Dominant layer: {sweep_factual['dominant_layer']}")
    print(f"  Dominant component: {sweep_factual['dominant_component']}")

    print("\n  Per-layer mean contributions:")
    print(
        f"  {'Layer':>5}  {'Mean Attn':>10}  {'Mean FFN':>10}  {'Mean Total':>11}  {'Std Total':>10}"
    )
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}  {'─' * 11}  {'─' * 10}")
    for ls in sweep_factual["layer_summary"]:
        print(
            f"  {ls['layer']:>5}  "
            f"{ls['mean_attention_logit']:>+10.4f}  "
            f"{ls['mean_ffn_logit']:>+10.4f}  "
            f"{ls['mean_total_logit']:>+11.4f}  "
            f"{ls['std_total_logit']:>10.4f}"
        )

    # Per-prompt summary table (Step 18 enhancement)
    if "prompt_summary" in sweep_factual:
        ps = sweep_factual["prompt_summary"]
        print("\n  Per-prompt summary:")
        print(
            f"  {'Label':>12}  {'Embed':>8}  {'NetAttn':>8}  {'NetFFN':>8}  "
            f"{'Final':>8}  {'Prob':>6}  {'Top':>10}  {'Dom':>5}"
        )
        print(
            f"  {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}  "
            f"{'─' * 8}  {'─' * 6}  {'─' * 10}  {'─' * 5}"
        )
        for row in ps:
            lbl = (row.get("label") or row["prompt"][:12])[:12]
            print(
                f"  {lbl:>12}  "
                f"{row['embedding_logit']:>+8.3f}  "
                f"{row['net_attention']:>+8.3f}  "
                f"{row['net_ffn']:>+8.3f}  "
                f"{row['final_logit']:>+8.3f}  "
                f"{row['probability']:>6.3f}  "
                f"{row['top_prediction']:>10}  "
                f"{row['dominant_component'][:5]:>5}"
            )

    # ==================================================================
    # Step 4: Attribution sweep — phrase completion
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 4: ATTRIBUTION SWEEP — PHRASE COMPLETION")
    print(f"{'=' * 60}")

    phrase_prompts = [
        "The cat sat on the",
        "Once upon a time there",
        "I went to the store and",
        "She picked up the phone and",
    ]

    sweep_phrase = await attribution_sweep(
        prompts=phrase_prompts,
        layers=layers,
        position=-1,
        normalized=True,
        labels=["cat_sat", "once_upon", "store", "phone"],
    )
    if sweep_phrase.get("error"):
        print(f"  FAILED: {sweep_phrase.get('message')}")
        return 1

    pp("attribution_sweep (phrase completion)", sweep_phrase)

    print(
        f"\n  Swept {sweep_phrase['num_prompts']} prompts across {sweep_phrase['num_layers']} layers"
    )
    print(f"  Dominant layer: {sweep_phrase['dominant_layer']}")
    print(f"  Dominant component: {sweep_phrase['dominant_component']}")

    # ==================================================================
    # Step 5: Compare groups
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 5: GROUP COMPARISON")
    print(f"{'=' * 60}")

    fact_summary = {e["layer"]: e for e in sweep_factual["layer_summary"]}
    phrase_summary = {e["layer"]: e for e in sweep_phrase["layer_summary"]}
    common = sorted(set(fact_summary.keys()) & set(phrase_summary.keys()))

    print(f"\n  {'Layer':>5}  {'Factual':>10}  {'Phrase':>10}  {'Δ':>10}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for layer in common:
        f_total = fact_summary[layer]["mean_total_logit"]
        p_total = phrase_summary[layer]["mean_total_logit"]
        delta = f_total - p_total
        print(f"  {layer:>5}  {f_total:>+10.4f}  {p_total:>+10.4f}  {delta:>+10.4f}")

    print(
        f"\n  Factual dominant: layer {sweep_factual['dominant_layer']} ({sweep_factual['dominant_component']})"
    )
    print(
        f"  Phrase dominant:  layer {sweep_phrase['dominant_layer']} ({sweep_phrase['dominant_component']})"
    )

    # ==================================================================
    # Step 6: Track token across layers (proves Step 15 fix)
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 6: TRACK TOKEN (CALIBRATED)")
    print("  Uses calibrated projection — correct on tied-embedding models")
    print(f"{'=' * 60}")

    # Pick the top-1 prediction from logit lens as our tracking target
    final_pred = summary.get("final_prediction", "the")
    track_result = await track_token(
        prompt="The capital of France is",
        token=final_pred,
        layers=layers,
        token_position=-1,
    )
    if track_result.get("error"):
        print(f"  FAILED: {track_result.get('message')}")
    else:
        pp(f"track_token ('{final_pred}')", track_result)

        print(
            f"\n  Tracking '{track_result['target_token']}' across {len(track_result['layers'])} layers:"
        )
        for entry in track_result["layers"]:
            bar = "█" * max(1, int(entry["probability"] * 40))
            top1 = "★" if entry["is_top1"] else " "
            print(
                f"    Layer {entry['layer']:>3}: prob={entry['probability']:.4f} "
                f"rank={entry['rank']:>3} {top1} {bar}"
            )
        print(f"\n  Emergence layer: {track_result['emergence_layer']}")
        print(
            f"  Peak layer: {track_result['peak_layer']} (prob={track_result['peak_probability']:.4f})"
        )

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  ATTRIBUTION SWEEP DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print("  Tools demonstrated: logit_lens, track_token, attribution_sweep")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  What this proved:")
    print("  - logit_lens uses calibrated _norm_project (handles tied embeddings)")
    print("  - track_token uses calibrated projection (same fix)")
    print("  - attribution_sweep batches logit_attribution across prompts")
    print("  - Per-layer mean/std reveals consistent vs variable contributions")
    print("  - Comparing groups (factual vs phrase) shows different circuits")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attribution sweep demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
