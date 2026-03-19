"""Backward-compat shim — real code in tools/attribution/tools.py."""

from .attribution.tools import *  # noqa: F401,F403
from .attribution.tools import _attribution_sweep_impl  # noqa: F401
