#!/usr/bin/env python3
"""
Track race demo: multi-candidate logit trajectory across layers.

Demonstrates:
1. track_race — race N candidates through the residual stream
2. Crossing detection — where the leader changes
3. Emergence and peak layers per candidate

Steps:
1. Load model
2. Race "Canberra" vs "Sydney" vs "Melbourne" for Australia's capital
3. Race "Paris" vs "London" vs "Berlin" for France's capital
4. Display trajectory tables and crossing events

Usage:
    # Quick run (SmolLM2-135M):
    python examples/track_race_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/track_race_demo.py --model google/gemma-3-4b-it
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
    from chuk_mcp_lazarus.tools.generation_tools import track_race

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"{'=' * 60}")
    print("  TRACK RACE DEMO")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    load_result = await load_model(model_id=model_id)
    if load_result.get("error"):
        print(f"  FAILED: {load_result.get('message')}")
        return 1

    num_layers = load_result["num_layers"]
    layers = list(range(0, num_layers, max(1, num_layers // 12)))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)

    print(f"  Loaded: {num_layers} layers, tracking across {len(layers)} checkpoints")

    # ==================================================================
    # Step 2: Australia capital race
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  RACE 1: AUSTRALIA'S CAPITAL")
    print("  Prompt: 'The capital of Australia is'")
    print("  Candidates: Canberra, Sydney, Melbourne")
    print(f"{'=' * 60}")

    race1 = await track_race(
        prompt="The capital of Australia is",
        candidates=["Canberra", "Sydney", "Melbourne"],
        layers=layers,
    )
    if race1.get("error"):
        print(f"  FAILED: {race1.get('message')}")
        return 1

    _display_race(race1)

    # ==================================================================
    # Step 3: France capital race
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  RACE 2: FRANCE'S CAPITAL")
    print("  Prompt: 'The capital of France is'")
    print("  Candidates: Paris, London, Berlin")
    print(f"{'=' * 60}")

    race2 = await track_race(
        prompt="The capital of France is",
        candidates=["Paris", "London", "Berlin"],
        layers=layers,
    )
    if race2.get("error"):
        print(f"  FAILED: {race2.get('message')}")
        return 1

    _display_race(race2)

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  TRACK RACE DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print("  Tools demonstrated: track_race")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    return 0


def _display_race(race: dict) -> None:
    """Display race results with trajectory table and crossings."""
    print(f"\n  Final winner: {race['final_winner']} (p={race['final_probability']:.4f})")
    print(f"  Candidates tracked: {race['num_candidates']}")

    # Trajectory table
    candidates = race["candidates"]
    layers_in_race = candidates[0]["layers"] if candidates else []

    # Header
    names = [c["token"] for c in candidates]
    header = f"  {'Layer':>5}"
    for name in names:
        header += f"  {name:>12}"
    print(f"\n{header}")
    print(f"  {'─' * 5}" + f"  {'─' * 12}" * len(names))

    # Rows
    for li, _ in enumerate(layers_in_race):
        row = f"  {layers_in_race[li]['layer']:>5}"
        for ci, cand in enumerate(candidates):
            entry = cand["layers"][li]
            marker = "★" if entry["is_top1"] else " "
            row += f"  {marker}{entry['probability']:>8.4f} r{entry['rank']:<2}"
        print(row)

    # Per-candidate summary
    print("\n  Per-candidate summary:")
    print(f"  {'Token':>12}  {'Emergence':>9}  {'Peak':>5}  {'PeakProb':>8}")
    print(f"  {'─' * 12}  {'─' * 9}  {'─' * 5}  {'─' * 8}")
    for c in candidates:
        emerg = str(c["emergence_layer"]) if c["emergence_layer"] is not None else "—"
        print(
            f"  {c['token']:>12}  {emerg:>9}  {c['peak_layer']:>5}  {c['peak_probability']:>8.4f}"
        )

    # Crossings
    crossings = race.get("crossings", [])
    if crossings:
        print(f"\n  Lead changes ({len(crossings)}):")
        for cx in crossings:
            print(
                f"    Layer {cx['layer']:>3}: "
                f"{cx['previous_leader']} → {cx['new_leader']} "
                f"(p={cx['new_leader_probability']:.4f})"
            )
    else:
        print("\n  No lead changes — winner led throughout.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track race demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model to load",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
