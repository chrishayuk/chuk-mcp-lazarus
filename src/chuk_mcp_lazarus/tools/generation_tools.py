"""Backward-compat shim — real code in tools/generation/tools.py."""

from .generation.tools import *  # noqa: F401,F403
from .generation.tools import (  # noqa: F401
    _embedding_neighbors_impl,
    _logit_lens_impl,
    _predict_next,
    _track_race_impl,
    _track_token_impl,
)
