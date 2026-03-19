#!/usr/bin/env python3
"""
Subspace & injection demo: PCA subspaces, residual surgery, and dark tables.

This script demonstrates the advanced geometry tools that work with
learned subspaces and residual injection:

Steps:
1.  Load model
2.  compute_subspace: learn a PCA subspace from diverse prompts
3.  list_subspaces: see stored subspaces
4.  subspace_decomposition: decompose a residual into subspace components
5.  residual_atlas: demand-side PCA with vocab-decoded principal components
6.  weight_geometry: supply-side head/neuron push directions
7.  computation_map: full prediction flow in one call
8.  inject_residual: inject donor residual into recipient (Markov test)
9.  residual_match: find prompts with most similar residual streams
10. branch_and_collapse: non-collapsing superposition test
11. build_dark_table: precompute dark coordinate lookup table
12. list_dark_tables: see stored tables
13. subspace_surgery: all-position subspace replacement

Usage:
    python examples/subspace_demo.py
    python examples/subspace_demo.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


def pp(label: str, result: dict) -> None:
    """Pretty-print a tool result (truncate long lists)."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    compact = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 8:
            compact[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            compact[k] = {
                kk: (f"[{len(vv)} items]" if isinstance(vv, list) and len(vv) > 8 else vv)
                for kk, vv in v.items()
            }
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


def check(label: str, result: dict) -> bool:
    """Check for error and print result. Returns True on success."""
    if result.get("error"):
        print(f"  {label}: FAILED — {result.get('message', '?')}")
        return False
    pp(label, result)
    return True


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.geometry.compute_subspace import (
        compute_subspace,
        list_subspaces,
    )
    from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import (
        subspace_decomposition,
    )
    from chuk_mcp_lazarus.tools.geometry.residual_atlas import residual_atlas
    from chuk_mcp_lazarus.tools.geometry.weight_geometry import weight_geometry
    from chuk_mcp_lazarus.tools.geometry.computation_map import computation_map
    from chuk_mcp_lazarus.tools.geometry.inject_residual import inject_residual
    from chuk_mcp_lazarus.tools.geometry.residual_match import residual_match
    from chuk_mcp_lazarus.tools.geometry.branch_and_collapse import (
        branch_and_collapse,
    )
    from chuk_mcp_lazarus.tools.geometry.build_dark_table import (
        build_dark_table,
        list_dark_tables,
    )
    from chuk_mcp_lazarus.tools.geometry.subspace_surgery import subspace_surgery

    t0 = time.time()
    passed = 0
    failed = 0

    print(f"\n{'=' * 60}")
    print("  SUBSPACE & INJECTION DEMO")
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
    print(f"  Loaded: {result['family']} ({num_layers} layers, {result['hidden_dim']} hidden dim)")

    # Prompts for subspace learning
    prompts_diverse = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Brazil is",
        "Cats are known for",
        "Dogs are known for",
        "The color of the sky is",
        "Water freezes at a temperature of",
        "The largest ocean on Earth is",
        "Birds can typically be seen",
    ]
    prompt = "The capital of France is"

    # ------------------------------------------------------------------
    # Step 2: compute_subspace — learn a PCA subspace
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 2: compute_subspace")
    print(f"{'=' * 60}")
    r = await compute_subspace(
        subspace_name="demo_subspace",
        prompts=prompts_diverse,
        layer=mid_layer,
        rank=min(5, len(prompts_diverse) - 1),
    )
    if check("compute_subspace", r):
        passed += 1
        rank = r.get("rank", "?")
        print(f"\n  Learned subspace 'demo_subspace' at layer {mid_layer}, rank={rank}")
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 3: list_subspaces
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 3: list_subspaces")
    print(f"{'=' * 60}")
    r = await list_subspaces()
    if check("list_subspaces", r):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 4: subspace_decomposition
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 4: subspace_decomposition")
    print(f"{'=' * 60}")
    r = await subspace_decomposition(
        prompt=prompt,
        layer=mid_layer,
        target={"type": "residual", "value": None},
        basis_directions=[
            {"type": "token", "value": "Paris"},
            {"type": "token", "value": "London"},
        ],
    )
    if check("subspace_decomposition", r):
        passed += 1
        for comp in r.get("components", [])[:3]:
            print(f"    {comp.get('label', '?'):>20}: {comp.get('angle', 0):.1f}°")
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 5: residual_atlas — demand-side PCA
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 5: residual_atlas")
    print(f"{'=' * 60}")
    r = await residual_atlas(
        prompts=prompts_diverse,
        layers=[0, mid_layer, num_layers - 1],
        store_subspace="demo_atlas",
        max_components=min(10, len(prompts_diverse) - 1),
    )
    if check("residual_atlas", r):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 6: weight_geometry — supply-side geometry
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 6: weight_geometry")
    print(f"{'=' * 60}")
    r = await weight_geometry(
        layer=mid_layer,
        top_k_vocab=5,
        include_pca=False,
    )
    if check("weight_geometry", r):
        passed += 1
        for head in r.get("heads", [])[:3]:
            print(f"    Head {head['head']}: top_token={head.get('top_token', '?')}")
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 7: computation_map — full prediction flow
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 7: computation_map")
    print(f"{'=' * 60}")
    r = await computation_map(
        prompt=prompt,
        candidates=["Paris", "London", "Berlin"],
        layers=[0, mid_layer, num_layers - 1],
    )
    if check("computation_map", r):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 8: inject_residual — Markov property test
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 8: inject_residual")
    print(f"{'=' * 60}")
    r = await inject_residual(
        donor_prompt="The capital of France is",
        recipient_prompt="The capital of Germany is",
        layer=mid_layer,
    )
    if check("inject_residual", r):
        passed += 1
        metrics = r.get("comparison_metrics", {})
        print(f"\n  Injected matches donor: {metrics.get('injected_matches_donor')}")
        print(f"  Donor-Injected KL: {metrics.get('donor_injected_kl', '?')}")
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 9: residual_match — find similar residual streams
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 9: residual_match")
    print(f"{'=' * 60}")
    r = await residual_match(
        target_prompt=prompt,
        candidate_prompts=[
            "The capital of Germany is",
            "Cats are known for",
            "The capital of Japan is",
        ],
        layer=mid_layer,
    )
    if check("residual_match", r):
        passed += 1
        for m in r.get("matches", [])[:3]:
            print(
                f"    {m.get('candidate', '?')[:40]:>40}: cos={m.get('cosine_similarity', 0):.3f}"
            )
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 10: branch_and_collapse — superposition test
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 10: branch_and_collapse")
    print(f"{'=' * 60}")
    r = await branch_and_collapse(
        donor_prompt="The capital of France is",
        branch_prompts=[
            "The largest city in Europe is",
            "I like to eat",
        ],
        layer=mid_layer,
    )
    if check("branch_and_collapse", r):
        passed += 1
        collapsed = r.get("collapsed", {})
        print(f"\n  Winner: {collapsed.get('winner_prompt', '?')}")
        print(f"  Top prediction: {collapsed.get('top_prediction', '?')}")
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 11: build_dark_table — precompute lookup table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 11: build_dark_table")
    print(f"{'=' * 60}")
    r = await build_dark_table(
        table_name="demo_dark",
        subspace_name="demo_subspace",
        layer=mid_layer,
        entries={
            "France": "The capital of France is",
            "Germany": "The capital of Germany is",
            "Japan": "The capital of Japan is",
        },
    )
    if check("build_dark_table", r):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 12: list_dark_tables
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 12: list_dark_tables")
    print(f"{'=' * 60}")
    r = await list_dark_tables()
    if check("list_dark_tables", r):
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Step 13: subspace_surgery — all-position subspace swap
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 13: subspace_surgery")
    print(f"{'=' * 60}")
    r = await subspace_surgery(
        recipient_prompt="The capital of Germany is",
        layer=mid_layer,
        subspace_name="demo_subspace",
        mode="donor",
        donor_prompt="The capital of France is",
    )
    if check("subspace_surgery", r):
        passed += 1
        print(f"\n  Donor top: {r.get('donor_output', {}).get('top_prediction', '?')}")
        print(f"  Recipient top: {r.get('recipient_output', {}).get('top_prediction', '?')}")
        print(f"  After surgery: {r.get('surgery_output', {}).get('top_prediction', '?')}")
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    total = passed + failed
    print(f"\n{'=' * 60}")
    print("  SUBSPACE DEMO COMPLETE")
    print(f"  {passed}/{total} steps passed  ({elapsed:.1f}s)")
    print(f"{'=' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subspace & injection demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
