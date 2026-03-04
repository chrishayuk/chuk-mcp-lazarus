#!/usr/bin/env python3
"""
Language transition demo: the flagship chuk-mcp-lazarus experiment.

This script follows a 15-step Claude workflow (with two bonus registry
inspection steps 11b and 12b):
1.   Load model
2.   Inspect architecture
3.   Tokenize a multilingual prompt to see token structure
4.   Generate text to see baseline model output
5.   Sanity-check activations
6.   Compare multilingual representations at early layer
7.   Compare multilingual representations at late layer
8.   Apply logit lens to see prediction evolution
9.   Track a specific token's probability across layers
10.  Scan probes across all layers to find the crossover
11.  Evaluate the best probe on held-out data
11b. List trained probes (registry inspection)
12.  Compute a steering vector at the crossover layer
12b. List steering vectors (registry inspection)
13.  Steer generation: push output toward a different language
14.  Iterate with different alpha values
15.  Causal tracing: prove which layers are necessary

Usage:
    python examples/language_transition_demo.py

    # Use a different model:
    python examples/language_transition_demo.py --model google/gemma-3-4b-it
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
        if isinstance(v, dict) and any(isinstance(vv, list) for vv in v.values()):
            compact[k] = {
                kk: f"[{len(vv)} floats]" if isinstance(vv, list) and len(vv) > 5 else vv
                for kk, vv in v.items()
            }
        elif isinstance(v, list) and len(v) > 10:
            compact[k] = f"[{len(v)} items]"
        elif isinstance(v, list) and all(isinstance(row, list) for row in v):
            # Matrix — show shape
            compact[k] = f"[{len(v)}x{len(v[0]) if v else 0} matrix]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


def print_similarity_matrix(prompts: list[str], matrix: list[list[float]]) -> None:
    """Print a labelled similarity matrix."""
    # Header
    max_label = max(len(p[:12]) for p in prompts)
    header = " " * (max_label + 2)
    for p in prompts:
        header += f"{p[:8]:>10}"
    print(header)
    # Rows
    for i, p in enumerate(prompts):
        row = f"  {p[:max_label]:<{max_label}}"
        for j in range(len(prompts)):
            row += f"{matrix[i][j]:10.4f}"
        print(row)


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import get_model_info, load_model
    from chuk_mcp_lazarus.tools.activation_tools import (
        compare_activations,
        extract_activations,
    )
    from chuk_mcp_lazarus.tools.probe_tools import (
        evaluate_probe,
        list_probes,
        scan_probe_across_layers,
    )
    from chuk_mcp_lazarus.tools.steering_tools import (
        compute_steering_vector,
        list_steering_vectors,
        steer_and_generate,
    )
    from chuk_mcp_lazarus.tools.generation_tools import (
        generate_text,
        tokenize,
        logit_lens,
        track_token,
    )
    from chuk_mcp_lazarus.tools.causal_tools import (
        trace_token as causal_trace_token,
    )

    t0 = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    print("\n" + "=" * 60)
    print("  LANGUAGE TRANSITION DEMO")
    print("=" * 60)

    result = await load_model(model_id=model_id)
    pp("1. load_model", result)
    if result.get("error"):
        print(f"\nFailed to load model: {result['message']}")
        return 1

    # ==================================================================
    # Step 2: Inspect architecture
    # ==================================================================
    info = await get_model_info()
    pp("2. get_model_info", info)
    num_layers = info["num_layers"]
    hidden_dim = info["hidden_dim"]
    print(f"\n  Architecture: {info['architecture']}")
    print(f"  Family: {info['family']}")
    print(f"  Layers: {num_layers}, Hidden: {hidden_dim}")
    print(f"  Parameters: {info['parameter_count']:,}")

    # ==================================================================
    # Step 3: Tokenize a multilingual prompt
    # ==================================================================
    result = await tokenize(text="Translate to French: The weather is beautiful today.")
    pp("3. tokenize", result)
    if not result.get("error"):
        print(f"\n  {result['num_tokens']} tokens:")
        for tok in result["tokens"]:
            print(f"    [{tok['position']:>2}] id={tok['token_id']:<8} {tok['token_text']!r}")

    # ==================================================================
    # Step 4: Generate text to see baseline output
    # ==================================================================
    gen_prompt = "Translate to French: The weather is beautiful today."
    result = await generate_text(prompt=gen_prompt, max_new_tokens=30)
    pp("4. generate_text (baseline)", result)
    if not result.get("error"):
        print(f"\n  Prompt: {gen_prompt}")
        print(f"  Output: {result['output']}")
        print(f"  Tokens: {result['num_tokens']}")

    # ==================================================================
    # Step 5: Sanity-check activations
    # ==================================================================
    sample_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    result = await extract_activations(
        prompt="Bonjour",
        layers=sample_layers,
        token_position=-1,
    )
    pp("5. extract_activations (sanity check)", result)
    if not result.get("error"):
        print(f"\n  Token at position -1: '{result['token_text']}'")
        for layer_key, vec in result["activations"].items():
            mn, mx_val = min(vec), max(vec)
            mean = sum(vec) / len(vec)
            print(
                f"  Layer {layer_key:>3}: dim={len(vec)}, "
                f"range=[{mn:.3f}, {mx_val:.3f}], mean={mean:.4f}"
            )

    # ==================================================================
    # Step 6: Compare multilingual representations at early layer
    # ==================================================================
    greetings = ["Hello", "Bonjour", "Hola", "Guten Tag"]
    result = await compare_activations(
        prompts=greetings,
        layer=0,
        token_position=-1,
    )
    pp("6. compare_activations (layer 0 -- should be language-distinct)", result)
    if not result.get("error"):
        print("\n  Cosine similarity at layer 0:")
        print_similarity_matrix(greetings, result["cosine_similarity_matrix"])
        print(f"\n  Centroid distance: {result['centroid_distance']:.4f}")
        print("  (Higher = more distinct representations)")

    # ==================================================================
    # Step 7: Compare multilingual representations at late layer
    # ==================================================================
    result = await compare_activations(
        prompts=greetings,
        layer=num_layers - 1,
        token_position=-1,
    )
    pp(f"7. compare_activations (layer {num_layers - 1} -- should converge)", result)
    if not result.get("error"):
        print(f"\n  Cosine similarity at layer {num_layers - 1}:")
        print_similarity_matrix(greetings, result["cosine_similarity_matrix"])
        print(f"\n  Centroid distance: {result['centroid_distance']:.4f}")
        print("  (Lower = more converged representations)")

    # ==================================================================
    # Step 8: Logit lens -- see how prediction evolves through layers
    # ==================================================================
    result = await logit_lens(
        prompt="Translate to French: The weather is beautiful today.",
        top_k=3,
    )
    pp("8. logit_lens (prediction evolution)", result)
    if not result.get("error"):
        print(f"\n  Analyzing last token prediction across {result['num_layers_analyzed']} layers:")
        for pred in result["predictions"]:
            top3 = ", ".join(
                f"{t!r}:{p:.3f}"
                for t, p in zip(pred["top_tokens"][:3], pred["top_probabilities"][:3])
            )
            print(f"    Layer {pred['layer']:>3}: [{top3}]")
        print(f"\n  Final prediction: {result['summary']['final_prediction']!r}")
        print(f"  First emerged at layer: {result['summary']['emergence_layer']}")

    # ==================================================================
    # Step 9: Track a specific token's probability across layers
    # ==================================================================
    # If logit lens found a final prediction, track it; otherwise track a
    # common translation-related token
    track_target = "Le"  # French article -- expect emergence at translation layers
    result = await track_token(
        prompt="Translate to French: The weather is beautiful today.",
        token=track_target,
    )
    pp(f"9. track_token (tracking {track_target!r})", result)
    if not result.get("error"):
        print(f"\n  Target: {result['target_token']!r} (id={result['target_token_id']})")
        print(f"  Emergence layer: {result['emergence_layer']}")
        print(f"  Peak: layer {result['peak_layer']} (p={result['peak_probability']:.4f})")
        print("\n  Probability curve:")
        for entry in result["layers"]:
            bar = "█" * int(entry["probability"] * 50)
            marker = " <-- top1" if entry["is_top1"] else ""
            print(
                f"    Layer {entry['layer']:>3}: p={entry['probability']:.4f} rank={entry['rank']:>5} {bar}{marker}"
            )

    # ==================================================================
    # Step 10: Scan probes across layers to find crossover
    # ==================================================================
    # Training examples: English and French sentences
    train_examples = [
        {"prompt": "The cat sat on the mat", "label": "english"},
        {"prompt": "A dog runs in the park", "label": "english"},
        {"prompt": "Birds fly in the sky", "label": "english"},
        {"prompt": "The sun shines brightly", "label": "english"},
        {"prompt": "Water flows down the river", "label": "english"},
        {"prompt": "Le chat dort sur le lit", "label": "french"},
        {"prompt": "Un chien court dans le parc", "label": "french"},
        {"prompt": "Les oiseaux volent dans le ciel", "label": "french"},
        {"prompt": "Le soleil brille fort", "label": "french"},
        {"prompt": "L'eau coule dans la riviere", "label": "french"},
    ]

    # Scan every 4th layer (or all if model is small)
    step = max(1, num_layers // 8)
    scan_layers = list(range(0, num_layers, step))
    if (num_layers - 1) not in scan_layers:
        scan_layers.append(num_layers - 1)

    result = await scan_probe_across_layers(
        probe_name_prefix="lang_probe",
        layers=scan_layers,
        examples=train_examples,
    )
    pp("10. scan_probe_across_layers", result)
    if not result.get("error"):
        print(f"\n  Layers scanned: {len(result['layers_scanned'])}")
        print(
            f"  Peak layer: {result['peak_layer']} (val_accuracy={result['peak_val_accuracy']:.2%})"
        )
        print(f"  Crossover layer: {result.get('crossover_layer', 'none')}")
        print("\n  Accuracy by layer:")
        for entry in result["accuracy_by_layer"]:
            bar = "█" * int(entry["val_accuracy"] * 30)
            print(f"    Layer {entry['layer']:>3}: {entry['val_accuracy']:.2%} {bar}")
        print(f"\n  Interpretation: {result['interpretation']}")

        crossover = result.get("crossover_layer") or result["peak_layer"]
    else:
        print(f"\n  Scan failed: {result.get('message')}")
        crossover = num_layers // 2

    # ==================================================================
    # Step 11: Evaluate the best probe on held-out data
    # ==================================================================
    eval_examples = [
        {"prompt": "The moon is bright tonight", "label": "english"},
        {"prompt": "Children play in the garden", "label": "english"},
        {"prompt": "La lune brille ce soir", "label": "french"},
        {"prompt": "Les enfants jouent dans le jardin", "label": "french"},
    ]

    probe_name = f"lang_probe_L{crossover}"
    result = await evaluate_probe(
        probe_name=probe_name,
        examples=eval_examples,
    )
    pp("11. evaluate_probe (held-out data)", result)
    if not result.get("error"):
        print(f"\n  Probe: {probe_name}")
        print(f"  Held-out accuracy: {result['accuracy']:.2%}")
        for pred in result["predictions"]:
            mark = "+" if pred["correct"] else "x"
            conf = f" ({pred['confidence']:.0%})" if "confidence" in pred else ""
            print(
                f"    [{mark}] {pred['prompt'][:40]:<40} "
                f"true={pred['true_label']:<8} pred={pred['predicted_label']}{conf}"
            )

    # ==================================================================
    # Step 11b: List trained probes
    # ==================================================================
    result = await list_probes()
    pp("11b. list_probes (registry inspection)", result)
    if not result.get("error"):
        print(f"\n  Registered probes: {result['count']}")
        for p in result["probes"]:
            print(
                f"    {p['name']}: layer={p['layer']}, "
                f"classes={p['classes']}, accuracy={p.get('val_accuracy', 'n/a')}"
            )

    # ==================================================================
    # Step 12: Compute steering vector at the crossover layer
    # ==================================================================
    # Direction: French -> German (we want to steer French output toward German)
    result = await compute_steering_vector(
        vector_name="fr_to_de",
        layer=crossover,
        positive_prompts=[
            "Der Hund lauft im Park",
            "Die Katze schlaft auf dem Bett",
            "Die Vogel fliegen am Himmel",
            "Die Sonne scheint hell",
        ],
        negative_prompts=[
            "Le chien court dans le parc",
            "Le chat dort sur le lit",
            "Les oiseaux volent dans le ciel",
            "Le soleil brille fort",
        ],
    )
    pp("12. compute_steering_vector (French -> German)", result)
    if not result.get("error"):
        print(f"\n  Vector: fr_to_de at layer {crossover}")
        print(f"  Norm: {result['vector_norm']:.2f}")
        print(f"  Separability: {result['separability_score']:.4f}")
        print(
            f"  Within-positive (DE) similarity: {result['cosine_similarity_within_positive']:.4f}"
        )
        print(
            f"  Within-negative (FR) similarity: {result['cosine_similarity_within_negative']:.4f}"
        )

    # ==================================================================
    # Step 12b: List steering vectors
    # ==================================================================
    result = await list_steering_vectors()
    pp("12b. list_steering_vectors (registry inspection)", result)
    if not result.get("error"):
        print(f"\n  Registered vectors: {result['count']}")
        for v in result["vectors"]:
            print(f"    {v['name']}: layer={v['layer']}, norm={v.get('vector_norm', 'n/a')}")

    # ==================================================================
    # Step 13: Steer generation
    # ==================================================================
    steer_prompt = "Translate to French: The weather is beautiful today."
    result = await steer_and_generate(
        prompt=steer_prompt,
        vector_name="fr_to_de",
        alpha=15.0,
        max_new_tokens=50,
    )
    pp("13. steer_and_generate (alpha=15.0)", result)
    if not result.get("error"):
        print(f"\n  Prompt: {steer_prompt}")
        print(f"  Baseline (should be French): {result['baseline_output'][:80]}")
        print(f"  Steered  (should shift to German): {result['steered_output'][:80]}")

    # ==================================================================
    # Step 14: Iterate with different alpha values
    # ==================================================================
    print(f"\n{'─' * 60}")
    print("  Step: 14. Alpha sweep")
    print(f"{'─' * 60}")
    print(f"\n  Prompt: {steer_prompt}")
    print(f"  Vector: fr_to_de at layer {crossover}")
    print()

    for alpha in [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]:
        if alpha == 0.0:
            # Baseline -- just show what the model normally produces
            result = await steer_and_generate(
                prompt=steer_prompt,
                vector_name="fr_to_de",
                alpha=0.001,  # Near-zero steering
                max_new_tokens=30,
            )
        else:
            result = await steer_and_generate(
                prompt=steer_prompt,
                vector_name="fr_to_de",
                alpha=alpha,
                max_new_tokens=30,
            )
        if not result.get("error"):
            output = result["steered_output"] if alpha > 0 else result["baseline_output"]
            print(f"  alpha={alpha:>5.1f}: {output[:70]}")
        else:
            print(f"  alpha={alpha:>5.1f}: [error] {result.get('message', '')[:60]}")

    # ==================================================================
    # Step 15: Causal tracing -- prove which layers are necessary
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("  Step 15: Causal tracing")
    print(f"{'=' * 60}")

    result = await causal_trace_token(
        prompt="Translate to French: The weather is beautiful today.",
        token=track_target,
        layers=scan_layers,
        effect_threshold=0.05,
    )
    pp("15. trace_token (causal tracing)", result)
    if not result.get("error"):
        print(f"\n  Target: {result['target_token']!r}")
        print(f"  Baseline prob: {result['baseline_prob']:.4f}")
        print(f"  Peak layer: {result['peak_layer']} (effect={result['peak_effect']:.4f})")
        print(f"  Critical layers: {result['critical_layers']}")
        print("\n  Causal effect curve (ablation impact):")
        for entry in result["layer_effects"]:
            bar_len = int(abs(entry["effect"]) * 40)
            bar = "+" * bar_len if entry["effect"] > 0 else "-" * bar_len
            marker = " *" if entry["layer"] in result["critical_layers"] else ""
            print(f"    Layer {entry['layer']:>3}: {entry['effect']:+.4f} {bar}{marker}")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  LANGUAGE TRANSITION DEMO COMPLETE")
    print(f"  {len(scan_layers)} layers scanned, crossover at layer {crossover}")
    print(f"  Completed all 15 steps in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language transition demo")
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        help="HuggingFace model ID (default: google/gemma-3-4b-it)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
