"""Backward-compat shim — real code in tools/steering/tools.py."""

from .steering.tools import *  # noqa: F401,F403
from .steering.tools import (  # noqa: F401
    _compute_steering_vector_impl,
    _mean_pairwise_similarity,
    _mean_vector,
    _steer_and_generate_impl,
)
