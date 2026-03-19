"""
kv_vectors — Key and Query vector extraction in QK addressing space.

Tools:
    extract_k_vector — K vector for a specific KV head after k_norm + RoPE.
    extract_q_vector — Q vector for a specific query head after q_norm + RoPE.

These tools work in head_dim space (not hidden_dim).  Use them alongside
extract_attention_output and batch_dla_scan to analyse the QK addressing
mechanism of copy heads.

Implementation lives in head_dla; this module re-exports for clean module
organisation so that tool source files mirror the docs/ folder structure.
"""

from .head_dla import (  # noqa: F401
    KVectorResult,
    QVectorResult,
    _extract_k_vector_impl,
    _extract_q_vector_impl,
    extract_k_vector,
    extract_q_vector,
)
