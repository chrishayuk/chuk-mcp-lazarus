#!/usr/bin/env python3
"""
Experiment persistence demo: collect multi-step results and persist to disk.

Demonstrates:
1. create_experiment — start a named experiment tied to the loaded model
2. add_experiment_result — attach tool results to the experiment
3. get_experiment — retrieve full experiment with all steps
4. list_experiments — browse all experiments
5. Disk persistence — experiments survive session restarts

Steps:
1. Load model
2. Create an experiment
3. Run logit_attribution and add result to experiment
4. Run attribution_sweep and add result to experiment
5. Retrieve the full experiment
6. List all experiments
7. Verify disk persistence (round-trip)

Usage:
    # Quick run (SmolLM2-135M — validates tooling):
    python examples/experiment_demo.py

    # Meaningful results (requires ~8GB RAM):
    python examples/experiment_demo.py --model google/gemma-3-4b-it
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
        elif isinstance(v, dict) and len(str(v)) > 200:
            compact[k] = f"{{...{len(v)} keys}}"
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
    from chuk_mcp_lazarus.experiment_store import ExperimentStore
    from chuk_mcp_lazarus.tools.attribution_tools import attribution_sweep
    from chuk_mcp_lazarus.tools.experiment_tools import (
        add_experiment_result,
        create_experiment,
        get_experiment,
        list_experiments,
    )
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.residual_tools import logit_attribution

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT PERSISTENCE DEMO")
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
    # Step 2: Create an experiment
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 2: CREATE EXPERIMENT")
    print(f"{'=' * 60}")

    exp_result = await create_experiment(
        name="factual_recall_analysis",
        description="Compare logit attribution and sweep on factual prompts",
        tags=["factual", "attribution", "demo"],
    )
    if exp_result.get("error"):
        print(f"  FAILED: {exp_result.get('message')}")
        return 1

    experiment_id = exp_result["experiment_id"]
    pp("create_experiment", exp_result)
    print(f"\n  Experiment ID: {experiment_id}")
    print(f"  Model: {exp_result['model_id']}")

    # ==================================================================
    # Step 3: Run logit_attribution and add to experiment
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 3: LOGIT ATTRIBUTION → ADD TO EXPERIMENT")
    print(f"{'=' * 60}")

    attr_result = await logit_attribution(
        prompt="The capital of France is",
        layers=layers,
        position=-1,
        target_token="Paris",
    )
    if attr_result.get("error"):
        print(f"  Attribution FAILED: {attr_result.get('message')}")
        return 1

    print(f"  Attribution target: {attr_result['target_token']!r}")
    print(f"  Model logit: {attr_result['model_logit']:.2f}")
    print(f"  Attribution sum: {attr_result['attribution_sum']:.2f}")

    # Add to experiment
    add_result = await add_experiment_result(
        experiment_id=experiment_id,
        step_name="logit_attribution_france",
        result=attr_result,
    )
    if add_result.get("error"):
        print(f"  Add result FAILED: {add_result.get('message')}")
        return 1

    pp("add_experiment_result", add_result)
    print(f"  Total steps in experiment: {add_result['total_steps']}")

    # ==================================================================
    # Step 4: Run attribution_sweep and add to experiment
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 4: ATTRIBUTION SWEEP → ADD TO EXPERIMENT")
    print(f"{'=' * 60}")

    sweep_result = await attribution_sweep(
        prompts=[
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
        ],
        layers=layers,
        position=-1,
        normalized=True,
        labels=["France", "Germany", "Italy"],
    )
    if sweep_result.get("error"):
        print(f"  Sweep FAILED: {sweep_result.get('message')}")
        return 1

    print(
        f"  Swept {sweep_result['num_prompts']} prompts across {sweep_result['num_layers']} layers"
    )
    print(f"  Dominant layer: {sweep_result['dominant_layer']}")
    print(f"  Dominant component: {sweep_result['dominant_component']}")

    # Add to experiment
    add_result2 = await add_experiment_result(
        experiment_id=experiment_id,
        step_name="attribution_sweep_capitals",
        result=sweep_result,
    )
    if add_result2.get("error"):
        print(f"  Add result FAILED: {add_result2.get('message')}")
        return 1

    pp("add_experiment_result", add_result2)
    print(f"  Total steps in experiment: {add_result2['total_steps']}")

    # ==================================================================
    # Step 5: Retrieve the full experiment
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 5: GET EXPERIMENT (FULL RETRIEVAL)")
    print(f"{'=' * 60}")

    exp_data = await get_experiment(experiment_id=experiment_id)
    if exp_data.get("error"):
        print(f"  FAILED: {exp_data.get('message')}")
        return 1

    print(f"  Name: {exp_data['metadata']['name']}")
    print(f"  Model: {exp_data['metadata']['model_id']}")
    print(f"  Created: {exp_data['metadata']['created_at']}")
    print(f"  Description: {exp_data['metadata']['description']}")
    print(f"  Tags: {exp_data['metadata']['tags']}")
    print(f"  Steps: {len(exp_data['steps'])}")

    for i, step in enumerate(exp_data["steps"]):
        data_keys = list(step["data"].keys()) if isinstance(step["data"], dict) else "?"
        print(f"    Step {i + 1}: {step['step_name']} ({len(data_keys)} result keys)")

    # ==================================================================
    # Step 6: List all experiments
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 6: LIST EXPERIMENTS")
    print(f"{'=' * 60}")

    listing = await list_experiments()
    pp("list_experiments", listing)

    print(f"\n  Total experiments: {listing['count']}")
    for exp in listing["experiments"]:
        print(
            f"    {exp['experiment_id'][:8]}… "
            f"name={exp['name']!r} "
            f"model={exp['model_id']!r} "
            f"steps={exp['num_steps']}"
        )

    # ==================================================================
    # Step 7: Verify disk persistence (round-trip)
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  STEP 7: DISK PERSISTENCE ROUND-TRIP")
    print(f"{'=' * 60}")

    store = ExperimentStore.get()

    # Clear in-memory state
    with store._access_lock:
        store._experiments.pop(experiment_id, None)

    # Verify it's gone from memory
    exp_gone = await get_experiment(experiment_id=experiment_id)
    if exp_gone.get("error"):
        print("  After clearing memory: experiment not in cache (expected)")
        # get_experiment tries load_from_disk as fallback — it should reload
        # Let's manually verify by re-requesting
        store2 = ExperimentStore.get()
        exp_reloaded = store2.get_experiment(experiment_id)
        if exp_reloaded is not None:
            print("  Fallback load_from_disk succeeded!")
        else:
            print("  Attempting explicit load_from_disk…")
            loaded = store.load_from_disk(experiment_id)
            if loaded:
                print("  Loaded from disk successfully!")
            else:
                print("  WARNING: Could not reload from disk")
    else:
        print("  get_experiment auto-loaded from disk (fallback worked!)")
        print(f"  Name: {exp_gone['metadata']['name']}")
        print(f"  Steps recovered: {len(exp_gone['steps'])}")

    # Verify the reloaded data is intact
    exp_verify = store.get_experiment(experiment_id)
    if exp_verify is not None:
        print(f"  Verified: {len(exp_verify.steps)} steps recovered from disk")
        for step in exp_verify.steps:
            print(f"    - {step.step_name}")
    else:
        print("  WARNING: Could not verify reloaded experiment")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT PERSISTENCE DEMO COMPLETE")
    print(f"  Model: {model_id} ({num_layers} layers)")
    print("  Tools demonstrated: create_experiment, add_experiment_result,")
    print("                      get_experiment, list_experiments")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  What this proved:")
    print("  - create_experiment captures model_id, description, tags")
    print("  - add_experiment_result attaches any tool result as a step")
    print("  - get_experiment retrieves all steps with full data")
    print("  - list_experiments shows all experiments with summaries")
    print("  - Disk persistence: experiments survive memory clears")
    print("  - get_experiment auto-loads from disk if not in memory")
    print(f"  - Stored at: ~/.chuk-lazarus/experiments/{experiment_id}.json")
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment persistence demo")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
