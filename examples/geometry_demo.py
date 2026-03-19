#!/usr/bin/env python3
"""
Geometry demo: angles, trajectories, and dimensionality in activation space.

This script demonstrates the core geometry tools that work in the full
native activation space (no lossy 2D projections).

Steps:
1. Load model
2. token_space: map "Paris" and "London" directions vs the residual stream
3. direction_angles: pairwise angles between token, residual, and attention directions
4. residual_trajectory: track how the residual rotates toward "Paris" across layers
5. feature_dimensionality: is "capital cities" a 1-dimensional or distributed feature?
6. residual_map: variance spectrum across the full model
7. decode_residual: what does the residual stream "say" at each layer?

Usage:
    python examples/geometry_demo.py
    python examples/geometry_demo.py --model google/gemma-3-4b-it
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


def sample_layers(num_layers: int, max_layers: int = 10) -> list[int]:
    """Sample evenly-spaced layers including first and last."""
    if num_layers <= max_layers:
        return list(range(num_layers))
    step = max(1, (num_layers - 1) // (max_layers - 1))
    layers = list(range(0, num_layers, step))
    if num_layers - 1 not in layers:
        layers.append(num_layers - 1)
    return layers[:max_layers]


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.geometry.token_space import token_space
    from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles
    from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory
    from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality
    from chuk_mcp_lazarus.tools.geometry.residual_map import residual_map
    from chuk_mcp_lazarus.tools.geometry.decode_residual import decode_residual

    t0 = time.time()

    print(f"\n{'=' * 60}")
    print("  GEOMETRY DEMO")
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

    prompt = "The capital of France is"

    # ------------------------------------------------------------------
    # Step 2: token_space — angles between token directions and residual
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 2: token_space (geometry at mid-model layer)")
    print(f"{'=' * 60}")

    ts = await token_space(
        prompt=prompt,
        layer=mid_layer,
        tokens=["Paris", "London", "Berlin"],
        token_position=-1,
    )
    pp("token_space", ts)

    if not ts.get("error"):
        print(f"\n  Residual stream at layer {mid_layer}:")
        for item in ts.get("tokens", []):
            print(
                f"    {item['token']:>10}: angle={item['angle_to_residual']:.1f}°  "
                f"projection={item['projection_on_residual']:+.3f}"
            )
        print("\n  Pairwise token angles:")
        for pair in ts.get("pairwise_angles", [])[:3]:
            print(f"    {pair['token_a']:>8} vs {pair['token_b']:>8}: {pair['angle']:.1f}°")

    # ------------------------------------------------------------------
    # Step 3: direction_angles — pairwise angles between directions
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 3: direction_angles")
    print(f"{'=' * 60}")

    da = await direction_angles(
        prompt=prompt,
        layer=mid_layer,
        directions=[
            {"type": "token", "value": "Paris"},
            {"type": "token", "value": "London"},
            {"type": "residual"},
            {"type": "attention_output"},
        ],
        token_position=-1,
    )
    pp("direction_angles", da)

    if not da.get("error"):
        print(f"\n  Pairwise angles at layer {mid_layer}:")
        for pair in da.get("pairwise_angles", []):
            print(f"    {pair['a']:>20} vs {pair['b']:>20}: {pair['angle']:.1f}°")

    # ------------------------------------------------------------------
    # Step 4: residual_trajectory — track rotation across layers
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 4: residual_trajectory")
    print(f"{'=' * 60}")

    layers = sample_layers(num_layers, max_layers=12)
    traj = await residual_trajectory(
        prompt=prompt,
        reference_tokens=["Paris", "London"],
        layers=layers,
        token_position=-1,
    )
    pp("residual_trajectory", traj)

    if not traj.get("error"):
        print("\n  Residual rotation toward 'Paris' and 'London' by layer:")
        print(f"  {'Layer':>5}  {'→Paris°':>8}  {'→London°':>9}  {'Rotation°':>9}  {'Ortho%':>7}")
        print(f"  {'─' * 5}  {'─' * 8}  {'─' * 9}  {'─' * 9}  {'─' * 7}")
        for entry in traj.get("layers", []):
            angles = entry.get("angles_to_references", {})
            paris_a = angles.get("Paris", angles.get(" Paris", "?"))
            london_a = angles.get("London", angles.get(" London", "?"))
            rot = entry.get("rotation_from_previous", 0.0)
            orth = entry.get("orthogonal_fraction", 0.0)
            print(
                f"  {entry['layer']:>5}  "
                f"{paris_a if isinstance(paris_a, str) else f'{paris_a:.1f}':>8}  "
                f"{london_a if isinstance(london_a, str) else f'{london_a:.1f}':>9}  "
                f"{rot:>+9.1f}  "
                f"{orth * 100:>6.1f}%"
            )

    # ------------------------------------------------------------------
    # Step 5: feature_dimensionality — is "capital city" 1D or distributed?
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 5: feature_dimensionality (capital cities vs common nouns)")
    print(f"{'=' * 60}")

    fd = await feature_dimensionality(
        layer=mid_layer,
        positive_prompts=[
            "The capital of France is Paris",
            "The capital of Germany is Berlin",
            "The capital of Japan is Tokyo",
            "The capital of Italy is Rome",
        ],
        negative_prompts=[
            "The color of grass is green",
            "The sound of music is pleasant",
            "The taste of coffee is bitter",
            "The weight of a feather is light",
        ],
        max_dims=20,
    )
    pp("feature_dimensionality", fd)

    if not fd.get("error"):
        print(f"\n  Effective dimensionality at layer {mid_layer}:")
        print(f"    dims_for_90pct_variance: {fd.get('dims_for_90pct_variance', '?')}")
        print(f"    dims_for_99pct_variance: {fd.get('dims_for_99pct_variance', '?')}")
        print(f"    interpretation: {fd.get('interpretation', '?')}")

    # ------------------------------------------------------------------
    # Step 6: residual_map — variance spectrum across model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 6: residual_map (compact per-layer variance spectrum)")
    print(f"{'=' * 60}")

    rmap = await residual_map(
        prompts=[
            "The capital of France is Paris",
            "The sky is blue today",
            "A neural network learns from data",
            "The cat sat on the mat",
            "Scientists discovered a new species",
            "The market opened lower this morning",
            "To be or not to be that is the question",
            "Water freezes at zero degrees Celsius",
        ],
        layers=sample_layers(num_layers, 8),
        max_components=20,
    )
    pp("residual_map", rmap)

    if not rmap.get("error"):
        print("\n  Per-layer effective dimensionality:")
        print(f"  {'Layer':>5}  {'EffDim':>7}  {'Top1 Var%':>9}  {'Top5 Var%':>9}")
        print(f"  {'─' * 5}  {'─' * 7}  {'─' * 9}  {'─' * 9}")
        for entry in rmap.get("layers", []):
            ed = entry.get("effective_dimensionality", "?")
            spectrum = entry.get("variance_spectrum", [])
            top1 = spectrum[0] * 100 if spectrum else 0
            top5 = sum(spectrum[:5]) * 100 if len(spectrum) >= 5 else 0
            print(f"  {entry['layer']:>5}  {ed:>7}  {top1:>8.1f}%  {top5:>8.1f}%")

    # ------------------------------------------------------------------
    # Step 7: decode_residual — vocabulary decode at multiple layers
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 7: decode_residual (logit lens with raw vs normalised)")
    print(f"{'=' * 60}")

    decode_layers = [0, mid_layer // 2, mid_layer, num_layers - 1]
    dr = await decode_residual(
        prompt=prompt,
        layers=decode_layers,
        top_k=5,
        token_position=-1,
    )
    pp("decode_residual", dr)

    if not dr.get("error"):
        print("\n  Residual stream vocabulary decoding:")
        for entry in dr.get("layers", []):
            raw_top = entry.get("raw_top_tokens", [])[:3]
            norm_top = entry.get("normalised_top_tokens", [])[:3]
            raw_str = ", ".join(f'"{t["token"]}"' for t in raw_top)
            norm_str = ", ".join(f'"{t["token"]}"' for t in norm_top)
            print(f"  Layer {entry['layer']:>2}: raw=[{raw_str}] | norm=[{norm_str}]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  GEOMETRY DEMO COMPLETE")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Takeaways:")
    print("  - token_space: angles reveal how aligned the residual is with candidate tokens")
    print("  - residual_trajectory: watch the residual rotate toward 'Paris' as layers deepen")
    print("  - feature_dimensionality: 1-2 dims → clean direction; 50+ dims → distributed")
    print("  - residual_map: effective dimensionality increases in middle layers")
    print("  - decode_residual: raw vs normalised gap reveals structured mean direction")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometry tools demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
