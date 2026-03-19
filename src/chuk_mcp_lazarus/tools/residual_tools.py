"""Backward-compat shim — real code in tools/residual/tools.py."""

from .residual.tools import *  # noqa: F401,F403
from .residual.tools import (  # noqa: F401
    _head_attribution_impl,
    _top_neurons_impl,
)
from .._residual_helpers import (  # noqa: F401
    _compute_clustering_scores,
    _extract_position,
    _get_embed_weight,
    _get_lm_projection,
    _get_unembed_vector,
    _has_four_norms,
    _has_sublayers,
    _l2_norm,
    _norm_project,
    _project_to_logits,
    _resolve_target_token,
    _run_decomposition_forward,
    _token_text,
    _tokenize,
)
