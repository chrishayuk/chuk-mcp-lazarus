#!/usr/bin/env python3
"""
Smoke test: prove every implemented tool works end-to-end.

Loads a small model (SmolLM2-135M by default), calls each tool,
and validates the Pydantic-backed return shapes. Runs in under
a minute on any Apple Silicon Mac.

Usage:
    # From repo root, with venv activated:
    python examples/smoke_test.py

    # Override model (any HF model chuk-lazarus supports):
    python examples/smoke_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
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

PASS = "PASS"
FAIL = "FAIL"


def pp(label: str, result: dict) -> None:
    """Pretty-print a tool result."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    # Truncate large fields for readability
    compact = {}
    for k, v in result.items():
        if isinstance(v, dict) and any(isinstance(vv, list) for vv in v.values()):
            compact[k] = {kk: f"[{len(vv)} floats]" if isinstance(vv, list) and len(vv) > 5 else vv
                          for kk, vv in v.items()}
        elif isinstance(v, list) and len(v) > 10:
            compact[k] = f"[{len(v)} items]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


def check(label: str, result: dict, required_keys: list[str]) -> bool:
    """Validate a tool result has expected keys and no error."""
    if result.get("error"):
        print(f"  [{FAIL}] {label}: {result.get('message', 'unknown error')}")
        return False
    missing = [k for k in required_keys if k not in result]
    if missing:
        print(f"  [{FAIL}] {label}: missing keys {missing}")
        return False
    print(f"  [{PASS}] {label}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(model_id: str) -> int:
    """Run all smoke tests. Returns 0 on success, 1 on failure."""

    # Import tools (triggers MCP registration)
    from chuk_mcp_lazarus.tools.model_tools import get_model_info, load_model
    from chuk_mcp_lazarus.tools.activation_tools import (
        compare_activations,
        extract_activations,
    )
    from chuk_mcp_lazarus.tools.probe_tools import (
        evaluate_probe,
        list_probes,
        scan_probe_across_layers,
        train_probe,
    )
    from chuk_mcp_lazarus.tools.steering_tools import (
        compute_steering_vector,
        list_steering_vectors,
        steer_and_generate,
    )
    from chuk_mcp_lazarus.tools.ablation_tools import (
        ablate_layers,
        patch_activations,
    )
    from chuk_mcp_lazarus.tools.generation_tools import (
        generate_text,
        predict_next_token,
        tokenize,
        logit_lens,
        track_token,
    )
    from chuk_mcp_lazarus.tools.attention_tools import (
        attention_pattern,
        attention_heads,
    )
    from chuk_mcp_lazarus.tools.causal_tools import (
        trace_token as causal_trace_token,
        full_causal_trace,
    )
    from chuk_mcp_lazarus.tools.residual_tools import (
        residual_decomposition,
        layer_clustering,
        logit_attribution,
    )

    results: list[bool] = []
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. load_model
    # ------------------------------------------------------------------
    print(f"\nLoading model: {model_id}")
    result = await load_model(model_id=model_id, dtype="bfloat16")
    pp("load_model", result)
    ok = check("load_model", result, ["model_id", "family", "num_layers", "hidden_dim", "status"])
    results.append(ok)

    if not ok:
        print("\nModel failed to load. Cannot continue.")
        return 1

    num_layers = result["num_layers"]
    print(f"  Model has {num_layers} layers, hidden_dim={result['hidden_dim']}")

    # ------------------------------------------------------------------
    # 2. get_model_info
    # ------------------------------------------------------------------
    result = await get_model_info()
    pp("get_model_info", result)
    ok = check("get_model_info", result, [
        "model_id", "family", "architecture", "num_layers", "hidden_dim",
        "num_attention_heads", "num_kv_heads", "vocab_size", "parameter_count",
    ])
    results.append(ok)

    # ------------------------------------------------------------------
    # 3. generate_text
    # ------------------------------------------------------------------
    result = await generate_text(
        prompt="The capital of France is",
        max_new_tokens=15,
        temperature=0.0,
    )
    pp("generate_text", result)
    ok = check("generate_text", result, [
        "prompt", "output", "num_tokens", "temperature",
    ])
    if ok:
        print(f"  Output: {result['output'][:80]!r}")
        print(f"  Tokens generated: {result['num_tokens']}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 4. generate_text (error case: model not loaded would be caught,
    #    but we test invalid max_new_tokens)
    # ------------------------------------------------------------------
    result = await generate_text(prompt="test", max_new_tokens=0)
    ok = result.get("error") is True and result.get("error_type") == "InvalidInput"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] generate_text error envelope (InvalidInput)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 5. predict_next_token
    # ------------------------------------------------------------------
    result = await predict_next_token(
        prompt="The capital of France is",
        top_k=5,
    )
    pp("predict_next_token", result)
    ok = check("predict_next_token", result, [
        "prompt", "num_input_tokens", "predictions",
    ])
    if ok:
        print(f"  Input tokens: {result['num_input_tokens']}")
        for pred in result["predictions"][:3]:
            print(f"    {pred['token']!r}: p={pred['probability']:.4f} "
                  f"(log_p={pred['log_probability']:.2f})")
        # Top prediction probability should be > 0
        if result["predictions"][0]["probability"] <= 0:
            print(f"  [{FAIL}] Top prediction probability should be > 0")
            ok = False
    results.append(ok)

    # ------------------------------------------------------------------
    # 6. tokenize
    # ------------------------------------------------------------------
    result = await tokenize(text="The capital of France is Paris.")
    pp("tokenize", result)
    ok = check("tokenize", result, [
        "text", "num_tokens", "tokens", "token_ids",
    ])
    if ok:
        print(f"  Tokens: {result['num_tokens']}")
        for tok in result["tokens"][:5]:
            print(f"    [{tok['position']}] id={tok['token_id']} {tok['token_text']!r}")
        if result["num_tokens"] < 2:
            print(f"  [{FAIL}] Expected at least 2 tokens")
            ok = False
    results.append(ok)

    # ------------------------------------------------------------------
    # 7. logit_lens
    # ------------------------------------------------------------------
    test_layers = [0, num_layers // 2, num_layers - 1]
    result = await logit_lens(
        prompt="The capital of France is",
        layers=test_layers,
        top_k=3,
    )
    pp("logit_lens", result)
    ok = check("logit_lens", result, [
        "prompt", "token_position", "num_layers_analyzed",
        "predictions", "summary",
    ])
    if ok:
        print(f"  Layers analyzed: {result['num_layers_analyzed']}")
        print(f"  Final prediction: {result['summary']['final_prediction']!r}")
        print(f"  Emergence layer: {result['summary']['emergence_layer']}")
        for pred in result["predictions"]:
            top3 = ", ".join(f"{t!r}:{p:.3f}" for t, p in
                            zip(pred["top_tokens"][:3], pred["top_probabilities"][:3]))
            print(f"    Layer {pred['layer']:>3}: [{top3}]")
    results.append(ok)

    # ------------------------------------------------------------------
    # 8. logit_lens (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await logit_lens(prompt="test", layers=[num_layers + 100])
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] logit_lens error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 9. track_token
    # ------------------------------------------------------------------
    result = await track_token(
        prompt="The capital of France is",
        token="Paris",
        layers=test_layers,
    )
    pp("track_token", result)
    ok = check("track_token", result, [
        "prompt", "target_token", "target_token_id", "token_position",
        "layers", "peak_layer", "peak_probability",
    ])
    if ok:
        print(f"  Target: {result['target_token']!r} (id={result['target_token_id']})")
        print(f"  Emergence layer: {result['emergence_layer']}")
        print(f"  Peak layer: {result['peak_layer']} (p={result['peak_probability']:.4f})")
        for entry in result["layers"]:
            marker = " <-- top1" if entry["is_top1"] else ""
            print(f"    Layer {entry['layer']:>3}: p={entry['probability']:.6f} rank={entry['rank']}{marker}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 10. track_token (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await track_token(prompt="test", token="hello", layers=[num_layers + 100])
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] track_token error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 11. attention_pattern
    # ------------------------------------------------------------------
    result = await attention_pattern(
        prompt="Hello world",
        layers=test_layers,
        token_position=-1,
        top_k=3,
    )
    pp("attention_pattern", result)
    ok = check("attention_pattern", result, [
        "prompt", "token_position", "token_text", "tokens",
        "num_layers_analyzed", "patterns",
    ])
    if ok:
        print(f"  Layers analyzed: {result['num_layers_analyzed']}")
        print(f"  Tokens: {result['tokens']}")
        for pat in result["patterns"]:
            print(f"  Layer {pat['layer']}: {pat['num_heads']} heads")
            # Show first head's top attended
            if pat["heads"]:
                h0 = pat["heads"][0]
                top = ", ".join(f"{a['token']!r}:{a['weight']:.3f}" for a in h0["top_attended"][:3])
                print(f"    Head 0 attends to: [{top}]")
    results.append(ok)

    # ------------------------------------------------------------------
    # 12. attention_heads
    # ------------------------------------------------------------------
    result = await attention_heads(
        prompt="Hello world",
        layers=test_layers,
        top_k=3,
    )
    pp("attention_heads", result)
    ok = check("attention_heads", result, [
        "prompt", "tokens", "num_heads_analyzed", "heads", "summary",
    ])
    if ok:
        print(f"  Heads analyzed: {result['num_heads_analyzed']}")
        for h in result["heads"][:3]:
            print(f"    Layer {h['layer']} Head {h['head']}: "
                  f"entropy={h['entropy']:.4f} max_attn={h['max_attention']:.4f}")
        print(f"  Most focused: {result['summary']['most_focused_heads'][:2]}")
        print(f"  Most diffuse: {result['summary']['most_diffuse_heads'][:2]}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 13. attention_heads (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await attention_heads(prompt="test", layers=[num_layers + 100])
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] attention_heads error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 14. extract_activations
    # ------------------------------------------------------------------
    result = await extract_activations(
        prompt="The capital of France is",
        layers=test_layers,
        token_position=-1,
    )
    pp("extract_activations", result)
    ok = check("extract_activations", result, [
        "prompt", "token_position", "token_text", "num_tokens", "activations",
    ])
    if ok:
        # Verify we got activations for each requested layer
        for layer in test_layers:
            key = str(layer)
            if key not in result["activations"]:
                print(f"  [{FAIL}] missing activation for layer {layer}")
                ok = False
            else:
                vec = result["activations"][key]
                print(f"  Layer {layer}: {len(vec)}-dim vector, "
                      f"range [{min(vec):.3f}, {max(vec):.3f}]")
    results.append(ok)

    # ------------------------------------------------------------------
    # 15. extract_activations (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await extract_activations(
        prompt="Hello",
        layers=[num_layers + 100],
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] extract_activations error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 16. compare_activations
    # ------------------------------------------------------------------
    mid_layer = num_layers // 2
    result = await compare_activations(
        prompts=["Hello", "Bonjour", "Hola"],
        layer=mid_layer,
        token_position=-1,
    )
    pp("compare_activations", result)
    ok = check("compare_activations", result, [
        "layer", "prompts", "cosine_similarity_matrix", "pca_2d", "centroid_distance",
    ])
    if ok:
        sim = result["cosine_similarity_matrix"]
        print(f"  Similarity matrix: {len(sim)}x{len(sim[0])}")
        print(f"  Self-similarity [0][0]: {sim[0][0]:.4f} (should be ~1.0)")
        print(f"  PCA points: {len(result['pca_2d'])}")
        print(f"  Centroid distance: {result['centroid_distance']:.4f}")
        if abs(sim[0][0] - 1.0) > 0.01:
            print(f"  [{FAIL}] Self-similarity not ~1.0")
            ok = False
    results.append(ok)

    # ------------------------------------------------------------------
    # 17. compare_activations (error case: too few prompts)
    # ------------------------------------------------------------------
    result = await compare_activations(prompts=["solo"], layer=0)
    ok = result.get("error") is True and result.get("error_type") == "InvalidInput"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] compare_activations error envelope (InvalidInput)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 18. train_probe
    # ------------------------------------------------------------------
    probe_examples = [
        {"prompt": "The cat sat on the mat", "label": "english"},
        {"prompt": "A dog runs in the park", "label": "english"},
        {"prompt": "Birds fly in the sky", "label": "english"},
        {"prompt": "Le chat dort sur le lit", "label": "french"},
        {"prompt": "Un chien court dans le parc", "label": "french"},
        {"prompt": "Les oiseaux volent dans le ciel", "label": "french"},
    ]
    result = await train_probe(
        probe_name="lang_test",
        layer=mid_layer,
        examples=probe_examples,
        probe_type="linear",
    )
    pp("train_probe", result)
    ok = check("train_probe", result, [
        "probe_name", "layer", "probe_type", "num_examples", "classes",
        "train_accuracy", "val_accuracy",
    ])
    if ok:
        print(f"  Classes: {result['classes']}")
        print(f"  Train accuracy: {result['train_accuracy']:.4f}")
        print(f"  Val accuracy: {result['val_accuracy']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 19. train_probe (error case: too few examples)
    # ------------------------------------------------------------------
    result = await train_probe(
        probe_name="bad_probe",
        layer=0,
        examples=[{"prompt": "hi", "label": "a"}],
    )
    ok = result.get("error") is True and result.get("error_type") == "InvalidInput"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] train_probe error envelope (InvalidInput)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 20. evaluate_probe
    # ------------------------------------------------------------------
    eval_examples = [
        {"prompt": "The sun is bright today", "label": "english"},
        {"prompt": "Le soleil brille aujourd'hui", "label": "french"},
    ]
    result = await evaluate_probe(
        probe_name="lang_test",
        examples=eval_examples,
    )
    pp("evaluate_probe", result)
    ok = check("evaluate_probe", result, [
        "probe_name", "layer", "accuracy", "per_class_accuracy",
        "confusion_matrix", "predictions",
    ])
    if ok:
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Predictions: {len(result['predictions'])} items")
    results.append(ok)

    # ------------------------------------------------------------------
    # 21. evaluate_probe (error case: probe not found)
    # ------------------------------------------------------------------
    result = await evaluate_probe(
        probe_name="nonexistent_probe",
        examples=[{"prompt": "test", "label": "x"}],
    )
    ok = result.get("error") is True and result.get("error_type") == "ProbeNotFound"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] evaluate_probe error envelope (ProbeNotFound)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 22. scan_probe_across_layers
    # ------------------------------------------------------------------
    scan_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    result = await scan_probe_across_layers(
        probe_name_prefix="lang_scan",
        layers=scan_layers,
        examples=probe_examples,
    )
    pp("scan_probe_across_layers", result)
    ok = check("scan_probe_across_layers", result, [
        "probe_name_prefix", "layers_scanned", "accuracy_by_layer",
        "peak_layer", "peak_val_accuracy", "interpretation",
    ])
    if ok:
        print(f"  Layers scanned: {result['layers_scanned']}")
        print(f"  Peak layer: {result['peak_layer']} (val_acc={result['peak_val_accuracy']:.4f})")
        print(f"  Crossover: {result.get('crossover_layer', 'none')}")
        for entry in result["accuracy_by_layer"]:
            print(f"    Layer {entry['layer']}: train={entry['train_accuracy']:.4f} val={entry['val_accuracy']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 23. list_probes
    # ------------------------------------------------------------------
    result = await list_probes()
    pp("list_probes", result)
    ok = check("list_probes", result, ["probes", "count"])
    if ok:
        print(f"  Probes in registry: {result['count']}")
        # Should have: lang_test + 5 scan probes = 6
        if result["count"] < 6:
            print(f"  [{FAIL}] Expected at least 6 probes, got {result['count']}")
            ok = False
        else:
            for p in result["probes"][:3]:
                print(f"    {p['name']}: layer={p['layer']} val_acc={p['val_accuracy']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 24. compute_steering_vector
    # ------------------------------------------------------------------
    result = await compute_steering_vector(
        vector_name="en_to_fr",
        layer=mid_layer,
        positive_prompts=[
            "Le chat dort sur le lit",
            "Un chien court dans le parc",
            "Les oiseaux volent dans le ciel",
        ],
        negative_prompts=[
            "The cat sleeps on the bed",
            "A dog runs in the park",
            "Birds fly in the sky",
        ],
    )
    pp("compute_steering_vector", result)
    ok = check("compute_steering_vector", result, [
        "vector_name", "layer", "vector_norm", "separability_score",
        "num_positive", "num_negative",
    ])
    if ok:
        print(f"  Vector norm: {result['vector_norm']:.4f}")
        print(f"  Separability: {result['separability_score']:.4f}")
        print(f"  Within-positive sim: {result['cosine_similarity_within_positive']:.4f}")
        print(f"  Within-negative sim: {result['cosine_similarity_within_negative']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 25. compute_steering_vector (error case: too few prompts)
    # ------------------------------------------------------------------
    result = await compute_steering_vector(
        vector_name="bad_vec",
        layer=0,
        positive_prompts=["solo"],
        negative_prompts=["also solo"],
    )
    ok = result.get("error") is True and result.get("error_type") == "InvalidInput"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] compute_steering_vector error envelope (InvalidInput)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 26. steer_and_generate
    # ------------------------------------------------------------------
    result = await steer_and_generate(
        prompt="The weather is",
        vector_name="en_to_fr",
        alpha=10.0,
        max_new_tokens=20,
    )
    pp("steer_and_generate", result)
    ok = check("steer_and_generate", result, [
        "prompt", "vector_name", "alpha", "layer",
        "steered_output", "baseline_output",
        "steered_tokens", "baseline_tokens",
    ])
    if ok:
        print(f"  Baseline: {result['baseline_output'][:80]!r}")
        print(f"  Steered:  {result['steered_output'][:80]!r}")
        print(f"  Baseline tokens: {result['baseline_tokens']}")
        print(f"  Steered tokens:  {result['steered_tokens']}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 27. steer_and_generate (error case: vector not found)
    # ------------------------------------------------------------------
    result = await steer_and_generate(
        prompt="test",
        vector_name="nonexistent_vector",
    )
    ok = result.get("error") is True and result.get("error_type") == "VectorNotFound"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] steer_and_generate error envelope (VectorNotFound)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 28. list_steering_vectors
    # ------------------------------------------------------------------
    result = await list_steering_vectors()
    pp("list_steering_vectors", result)
    ok = check("list_steering_vectors", result, ["vectors", "count"])
    if ok:
        print(f"  Vectors in registry: {result['count']}")
        if result["count"] < 1:
            print(f"  [{FAIL}] Expected at least 1 vector")
            ok = False
        else:
            for v in result["vectors"]:
                print(f"    {v['name']}: layer={v['layer']} norm={v['vector_norm']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 29. ablate_layers
    # ------------------------------------------------------------------
    result = await ablate_layers(
        prompt="The capital of France is",
        layers=[mid_layer],
        max_new_tokens=15,
        component="mlp",
    )
    pp("ablate_layers", result)
    ok = check("ablate_layers", result, [
        "prompt", "ablated_layers", "ablation_type", "component",
        "ablated_output", "baseline_output",
        "output_similarity", "disruption_score",
    ])
    if ok:
        print(f"  Baseline: {result['baseline_output'][:60]!r}")
        print(f"  Ablated:  {result['ablated_output'][:60]!r}")
        print(f"  Similarity: {result['output_similarity']:.4f}")
        print(f"  Disruption: {result['disruption_score']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 30. ablate_layers (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await ablate_layers(
        prompt="test",
        layers=[num_layers + 100],
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] ablate_layers error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 31. patch_activations
    # ------------------------------------------------------------------
    result = await patch_activations(
        source_prompt="The capital of France is",
        target_prompt="The capital of Germany is",
        layer=mid_layer,
        max_new_tokens=15,
    )
    pp("patch_activations", result)
    ok = check("patch_activations", result, [
        "source_prompt", "target_prompt", "patched_layer",
        "patched_output", "baseline_output", "source_output",
        "recovery_rate", "effect_size",
    ])
    if ok:
        print(f"  Source output:  {result['source_output'][:60]!r}")
        print(f"  Baseline (target): {result['baseline_output'][:60]!r}")
        print(f"  Patched:  {result['patched_output'][:60]!r}")
        print(f"  Recovery: {result['recovery_rate']:.4f}")
        print(f"  Effect:   {result['effect_size']:.4f}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 32. patch_activations (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await patch_activations(
        source_prompt="hello",
        target_prompt="world",
        layer=num_layers + 100,
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] patch_activations error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 33. trace_token (causal tracing)
    # ------------------------------------------------------------------
    result = await causal_trace_token(
        prompt="The capital of France is",
        token="Paris",
        layers=test_layers,
        effect_threshold=0.05,
    )
    pp("trace_token (causal)", result)
    ok = check("trace_token (causal)", result, [
        "prompt", "target_token", "target_token_id", "baseline_prob",
        "layer_effects", "critical_layers", "peak_layer", "peak_effect",
        "effect_threshold",
    ])
    if ok:
        print(f"  Target: {result['target_token']!r} (id={result['target_token_id']})")
        print(f"  Baseline prob: {result['baseline_prob']:.4f}")
        print(f"  Peak layer: {result['peak_layer']} (effect={result['peak_effect']:.4f})")
        print(f"  Critical layers: {result['critical_layers']}")
        for entry in result["layer_effects"]:
            bar_len = int(abs(entry["effect"]) * 40)
            bar = "+" * bar_len if entry["effect"] > 0 else "-" * bar_len
            print(f"    Layer {entry['layer']:>3}: effect={entry['effect']:+.4f} {bar}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 34. trace_token (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await causal_trace_token(
        prompt="test",
        token="hello",
        layers=[num_layers + 100],
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] trace_token error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 35. full_causal_trace
    # ------------------------------------------------------------------
    result = await full_causal_trace(
        prompt="The capital of France is",
        token="Paris",
        layers=test_layers,
    )
    pp("full_causal_trace", result)
    ok = check("full_causal_trace", result, [
        "prompt", "target_token", "tokens", "effects",
        "critical_positions", "critical_layers",
        "num_positions", "num_layers_tested",
    ])
    if ok:
        print(f"  Tokens: {result['tokens']}")
        print(f"  Grid: {result['num_positions']} positions x {result['num_layers_tested']} layers")
        print(f"  Critical positions: {result['critical_positions']}")
        print(f"  Critical layers: {result['critical_layers']}")
        if len(result["effects"]) != result["num_positions"]:
            print(f"  [{FAIL}] effects rows ({len(result['effects'])}) != num_positions ({result['num_positions']})")
            ok = False
        elif result["effects"] and len(result["effects"][0]) != result["num_layers_tested"]:
            print(f"  [{FAIL}] effects cols ({len(result['effects'][0])}) != num_layers_tested ({result['num_layers_tested']})")
            ok = False
    results.append(ok)

    # ------------------------------------------------------------------
    # 36. full_causal_trace (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await full_causal_trace(
        prompt="test",
        token="hello",
        layers=[num_layers + 100],
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] full_causal_trace error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 37. residual_decomposition
    # ------------------------------------------------------------------
    result = await residual_decomposition(
        prompt="The capital of France is",
        layers=test_layers,
        position=-1,
    )
    pp("residual_decomposition", result)
    ok = check("residual_decomposition", result, [
        "prompt", "token_position", "token_text", "num_tokens",
        "layers", "summary",
    ])
    if ok:
        print(f"  Layers analyzed: {len(result['layers'])}")
        for entry in result["layers"]:
            print(f"    Layer {entry['layer']:>3}: total={entry['total_norm']:.4f} "
                  f"attn={entry['attention_fraction']:.3f} ffn={entry['ffn_fraction']:.3f} "
                  f"[{entry['dominant_component']}]")
        # Verify fractions sum to ~1.0
        for entry in result["layers"]:
            frac_sum = entry["attention_fraction"] + entry["ffn_fraction"]
            if abs(frac_sum - 1.0) > 0.01:
                print(f"  [FAIL] Layer {entry['layer']} fractions sum to {frac_sum:.4f}, expected ~1.0")
                ok = False
        print(f"  Peak: layer {result['summary']['peak_layer']} "
              f"({result['summary']['peak_component']})")
    results.append(ok)

    # ------------------------------------------------------------------
    # 38. residual_decomposition (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await residual_decomposition(
        prompt="test",
        layers=[num_layers + 100],
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] residual_decomposition error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 39. layer_clustering
    # ------------------------------------------------------------------
    result = await layer_clustering(
        prompts=["The capital of France is", "La capital de Francia es"],
        layers=test_layers,
        position=-1,
    )
    pp("layer_clustering", result)
    ok = check("layer_clustering", result, [
        "prompts", "token_position", "num_layers_analyzed",
        "layers", "summary",
    ])
    if ok:
        print(f"  Layers analyzed: {result['num_layers_analyzed']}")
        for entry in result["layers"]:
            sim = entry["similarity_matrix"]
            print(f"    Layer {entry['layer']:>3}: mean_sim={entry['mean_similarity']:.4f} "
                  f"matrix={len(sim)}x{len(sim[0])}")
            # Verify matrix is square and right size
            if len(sim) != 2 or len(sim[0]) != 2:
                print(f"  [FAIL] Expected 2x2 matrix, got {len(sim)}x{len(sim[0])}")
                ok = False
            # Self-similarity should be ~1.0
            if abs(sim[0][0] - 1.0) > 0.01:
                print(f"  [FAIL] Self-similarity {sim[0][0]:.4f} not ~1.0")
                ok = False
        print(f"  Most similar layer: {result['summary']['most_similar_layer']}")
    results.append(ok)

    # ------------------------------------------------------------------
    # 40. layer_clustering (error case: too few prompts)
    # ------------------------------------------------------------------
    result = await layer_clustering(prompts=["only one"])
    ok = result.get("error") is True and result.get("error_type") == "InvalidInput"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] layer_clustering error envelope (InvalidInput)")
    results.append(ok)

    # ------------------------------------------------------------------
    # 41. logit_attribution
    # ------------------------------------------------------------------
    result = await logit_attribution(
        prompt="The capital of France is",
        layers=test_layers,
        position=-1,
    )
    pp("logit_attribution", result)
    ok = check("logit_attribution", result, [
        "prompt", "token_position", "token_text", "target_token",
        "target_token_id", "model_logit", "model_probability",
        "embedding_logit", "layers", "attribution_sum", "summary",
    ])
    if ok:
        print(f"  Target: {result['target_token']!r} (id={result['target_token_id']})")
        print(f"  Model logit: {result['model_logit']:.4f}  prob: {result['model_probability']:.4f}")
        print(f"  Embedding logit: {result['embedding_logit']:.4f}")
        print(f"  Attribution sum: {result['attribution_sum']:.4f}")
        for entry in result["layers"]:
            print(f"    Layer {entry['layer']:>3}: attn={entry['attention_logit']:>+8.3f} "
                  f"ffn={entry['ffn_logit']:>+8.3f} total={entry['total_logit']:>+8.3f} "
                  f"cum={entry['cumulative_logit']:>+8.3f} "
                  f"attn→{entry['attention_top_token']!r} ffn→{entry['ffn_top_token']!r}")
        # Verify probability is in [0, 1]
        if not (0.0 <= result["model_probability"] <= 1.0):
            print(f"  [{FAIL}] model_probability {result['model_probability']:.4f} not in [0, 1]")
            ok = False
        # Verify attribution_sum is finite
        import math
        if not math.isfinite(result["attribution_sum"]):
            print(f"  [{FAIL}] attribution_sum is not finite")
            ok = False
    results.append(ok)

    # ------------------------------------------------------------------
    # 42. logit_attribution (error case: layer out of range)
    # ------------------------------------------------------------------
    result = await logit_attribution(
        prompt="test",
        layers=[num_layers + 100],
    )
    ok = result.get("error") is True and result.get("error_type") == "LayerOutOfRange"
    print(f"\n  [{'PASS' if ok else 'FAIL'}] logit_attribution error envelope (LayerOutOfRange)")
    results.append(ok)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    passed = sum(results)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if passed < total:
        failed = [i for i, r in enumerate(results) if not r]
        print(f"  Failed test indices: {failed}")
        return 1

    print("  All tests passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for chuk-mcp-lazarus tools")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID to test with (default: SmolLM2-135M)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
