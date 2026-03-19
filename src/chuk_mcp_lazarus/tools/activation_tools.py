"""Backward-compat shim — real code in tools/activation/tools.py."""

from .activation.tools import *  # noqa: F401,F403
from .activation.tools import (  # noqa: F401
    _compare_activations_impl,
    _extract_activations_impl,
    _run_hooks,
    _token_text,
)
