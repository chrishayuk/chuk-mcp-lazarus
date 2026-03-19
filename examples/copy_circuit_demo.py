#!/usr/bin/env python3
"""
Copy circuit demo: per-head DLA and 12-byte injection test.

This script demonstrates the copy circuit hypothesis:
  - A small number of attention heads (copy heads) dominate factual recall.
  - Each copy head's output is effectively 1-dimensional.
  - The full KV cache for a fact compresses to 12 bytes: (token_id, coefficient).

Steps:
1. Load model
2. batch_dla_scan: build the layers × heads DLA heatmap for a factual prompt
3. compute_dla: drill into the top copy head
4. extract_attention_output: inspect the copy head's output vector
5. get_token_embedding: compare input vs unembedding vectors
6. prefill_to_layer: verify the residual is orthogonal before the copy head fires
7. kv_inject_test: replace full KV attention with a 12-byte signal; measure KL divergence
8. extract_k_vector / extract_q_vector: examine QK addressing space

Usage:
    python examples/copy_circuit_demo.py
    python examples/copy_circuit_demo.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


def pp(label: str, result: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    compact = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 8:
            compact[k] = f"[{len(v)} items]"
        else:
            compact[k] = v
    print(json.dumps(compact, indent=2, default=str))


async def main(model_id: str) -> int:
    from chuk_mcp_lazarus.tools.model_tools import load_model
    from chuk_mcp_lazarus.tools.geometry.head_dla import batch_dla_scan, compute_dla
    from chuk_mcp_lazarus.tools.geometry.head_output import (
        extract_attention_output,
        get_token_embedding,
    )
    from chuk_mcp_lazarus.tools.geometry.kv_vectors import extract_k_vector, extract_q_vector
    from chuk_mcp_lazarus.tools.geometry.prefill_inject import prefill_to_layer, kv_inject_test

    t0 = time.time()

    print(f"\n{'=' * 60}")
    print("  COPY CIRCUIT DEMO")
    print(f"  Model: {model_id}")
    print("  Hypothesis: factual content = 12 bytes (token_id + coefficient)")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    result = await load_model(model_id=model_id, dtype="bfloat16")
    if result.get("error"):
        print(f"  FAILED: {result.get('message')}")
        return 1
    num_layers = result["num_layers"]
    num_heads = result["num_attention_heads"]
    num_kv_heads = result.get("num_kv_heads") or num_heads
    print(f"  Loaded: {result['family']} ({num_layers}L, {num_heads}H, {num_kv_heads}KVH)")

    prompt = "The capital of France is"
    target = "Paris"

    # ------------------------------------------------------------------
    # Step 2: batch_dla_scan — DLA heatmap
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: batch_dla_scan (all layers × heads for '{target}')")
    print(f"{'=' * 60}")

    scan = await batch_dla_scan(
        prompt=prompt,
        target_token=target,
        top_k_cells=5,
    )
    pp("batch_dla_scan", scan)

    if scan.get("error"):
        print(f"  batch_dla_scan FAILED: {scan.get('message')}")
        return 1

    print("\n  Top copy head cells (ranked by |DLA|):")
    for cell in scan.get("hot_cells", []):
        print(
            f"    Rank {cell['abs_rank']}: layer={cell['layer']:>3} head={cell['head']:>3}  "
            f"DLA={cell['dla']:>+8.4f}  top_token={cell['top_token']!r}"
        )

    # Identify the top copy head
    hot_cells = scan.get("hot_cells", [])
    if not hot_cells:
        print("  No hot cells found — try a larger model.")
        return 1

    copy_cell = hot_cells[0]
    copy_layer = copy_cell["layer"]
    copy_head = copy_cell["head"]
    print(f"\n  Top copy head: layer={copy_layer}, head={copy_head}, DLA={copy_cell['dla']:+.4f}")

    # ------------------------------------------------------------------
    # Step 3: compute_dla — single-cell drill-in
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 3: compute_dla (layer={copy_layer}, head={copy_head})")
    print(f"{'=' * 60}")

    dla_result = await compute_dla(
        prompt=prompt,
        layer=copy_layer,
        head=copy_head,
        target_token=target,
    )
    pp("compute_dla", dla_result)

    # ------------------------------------------------------------------
    # Step 4: extract_attention_output — is the output 1-dimensional?
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 4: extract_attention_output (is_one_dimensional test)")
    print(f"{'=' * 60}")

    attn_out = await extract_attention_output(
        prompt=prompt,
        layer=copy_layer,
        head=copy_head,
        top_k_tokens=5,
    )
    pp("extract_attention_output", attn_out)

    if not attn_out.get("error"):
        dim = attn_out.get("dimensionality", {})
        print("\n  Content dimensionality:")
        print(f"    dims_for_95pct:    {dim.get('dims_for_95pct', '?')}")
        print(f"    dims_for_99pct:    {dim.get('dims_for_99pct', '?')}")
        print(f"    is_one_dimensional: {dim.get('is_one_dimensional', '?')}")

        proj = attn_out.get("top_projections", [])
        if proj:
            top = proj[0]
            print(
                f"    Top content: token={top['token']!r}  "
                f"coefficient={top['coefficient']:+.4f}  fraction={top['fraction']:.3f}"
            )
        coefficient = proj[0]["coefficient"] if proj else 1.0

    # ------------------------------------------------------------------
    # Step 5: get_token_embedding
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 5: get_token_embedding ('{target}')")
    print(f"{'=' * 60}")

    emb = await get_token_embedding(token=target)
    pp("get_token_embedding", emb)

    if not emb.get("error"):
        print(f"\n  '{emb['token']}' (id={emb['token_id']}):")
        print(f"    embeddings_tied: {emb['embeddings_tied']}")
        print(f"    unembedding_norm: {emb['unembedding_norm']:.4f}")
        if emb.get("cosine_similarity") is not None:
            print(f"    input vs unembedding cosine: {emb['cosine_similarity']:.4f}")

    # ------------------------------------------------------------------
    # Step 6: prefill_to_layer — is the residual orthogonal before copy head?
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  STEP 6: prefill_to_layer (residual at layer {copy_layer})")
    print(f"  (Should NOT contain '{target}' yet — copy head hasn't fired)")
    print(f"{'=' * 60}")

    prefill = await prefill_to_layer(
        prompt=prompt,
        layer=copy_layer,
        top_k_tokens=5,
    )
    pp("prefill_to_layer", prefill)

    if not prefill.get("error"):
        print(f"\n  Top raw logit predictions at layer {copy_layer}:")
        for tok in prefill.get("top_raw_logits", []):
            print(f"    {tok['token']!r:>10}: p={tok['probability']:.4f}")

    # ------------------------------------------------------------------
    # Step 7: kv_inject_test — 12-byte compression test
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 7: kv_inject_test (12-byte compression)")
    print(f"  inject_layer={copy_layer + 1}  token='{target}'  coefficient={coefficient:.4f}")
    print(f"{'=' * 60}")

    inject = await kv_inject_test(
        prompt=prompt,
        token=target,
        coefficient=coefficient,
        inject_layer=min(copy_layer + 1, num_layers - 1),
        top_k=8,
    )
    pp("kv_inject_test", inject)

    if not inject.get("error"):
        summary = inject.get("summary", {})
        kl = summary.get("kl_divergence", inject.get("kl_divergence", "?"))
        interp = summary.get("interpretation", "?")
        print(f"\n  KL divergence (full || injected): {kl}")
        print(f"  Interpretation: {interp}")
        if isinstance(kl, (int, float)) and kl < 0.001:
            print("\n  COPY CIRCUIT CONFIRMED:")
            print(f"  The (token_id={inject.get('token_id')}, coefficient={coefficient:.4f})")
            print("  pair is sufficient to reproduce full KV attention output.")
        else:
            print("\n  Compression not perfect — try larger model or adjust coefficient.")

    # ------------------------------------------------------------------
    # Step 8: extract_k_vector and extract_q_vector
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  STEP 8: extract_k_vector / extract_q_vector")
    print(f"  (Addressing space for copy head at layer {copy_layer})")
    print(f"{'=' * 60}")

    # For GQA models, kv_head = copy_head // (num_heads // num_kv_heads)
    kv_head = copy_head // max(1, num_heads // num_kv_heads)

    k_result = await extract_k_vector(
        prompt=prompt,
        layer=copy_layer,
        kv_head=kv_head,
        position=-1,
    )
    pp("extract_k_vector", k_result)

    q_result = await extract_q_vector(
        prompt=prompt,
        layer=copy_layer,
        head=copy_head,
        position=-1,
    )
    pp("extract_q_vector", q_result)

    if not k_result.get("error") and not q_result.get("error"):
        import numpy as np

        k_vec = np.array(k_result["k_vector"], dtype=np.float32)
        q_vec = np.array(q_result["q_vector"], dtype=np.float32)
        k_norm = k_result["k_norm"]
        q_norm = q_result["q_norm"]
        if k_norm > 1e-8 and q_norm > 1e-8:
            dot = float(np.dot(k_vec, q_vec))
            cos_sim = dot / (k_norm * q_norm)
            print(f"\n  Q·K dot product: {dot:.4f}")
            print(f"  Q·K cosine similarity: {cos_sim:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("  COPY CIRCUIT DEMO COMPLETE")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print("\n  Pipeline summary:")
    print(f"  1. batch_dla_scan     → identified copy head: L{copy_layer}H{copy_head}")
    print(
        f"  2. extract_attention_output → "
        f"{'1-dimensional' if not attn_out.get('error') and attn_out.get('dimensionality', {}).get('is_one_dimensional') else 'multi-dimensional'}"
    )
    print(
        f"  3. kv_inject_test     → "
        f"KL={inject.get('summary', {}).get('kl_divergence', inject.get('kl_divergence', '?'))}"
    )
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy circuit demo (DLA + injection test)")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID (default: SmolLM2-135M for speed)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.model)))
