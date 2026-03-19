"""Backward-compat shim — real code in tools/attention/tools.py."""

from .attention.tools import *  # noqa: F401,F403
from .attention.tools import (  # noqa: F401
    _attention_heads_impl,
    _attention_pattern_impl,
)
