"""Backward-compat shim — real code in tools/causal/tools.py."""

from .causal.tools import *  # noqa: F401,F403
from .causal.tools import (  # noqa: F401
    _ablate_layers_impl,
    _patch_activations_impl,
    _word_overlap_similarity,
)
