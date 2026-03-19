"""
head_output — Attention head output vector extraction and token embeddings.

Tools:
    extract_attention_output — Head output vector [hidden_dim] + vocab projection.
    get_token_embedding      — Token input embedding and unembedding vectors.

extract_attention_output reveals whether a head's content is 1-dimensional
(supports the 12-byte copy circuit claim).  get_token_embedding returns the
unembedding direction used for DLA projection.

Implementation lives in head_dla; this module re-exports for clean module
organisation so that tool source files mirror the docs/ folder structure.
"""

from .head_dla import (  # noqa: F401
    AttentionOutputResult,
    EmbeddingResult,
    _extract_attention_output_impl,
    _get_token_embedding_impl,
    extract_attention_output,
    get_token_embedding,
)
