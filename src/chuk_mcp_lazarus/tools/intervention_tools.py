"""Backward-compat shim — real code in tools/intervention/tools.py."""

from .intervention.tools import *  # noqa: F401,F403
from .intervention.tools import (  # noqa: F401
    _component_intervention_impl,
    _intervene_head,
    _run_forward_with_intervention,
)
