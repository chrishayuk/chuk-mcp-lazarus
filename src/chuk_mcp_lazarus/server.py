"""
Lazarus Interpretability Server -- ChukMCPServer instance.

This module creates the shared server object that all tool modules
import and register against via @mcp.tool() decorators.
"""

from importlib.metadata import version as _pkg_version

from chuk_mcp_server import ChukMCPServer

mcp = ChukMCPServer(
    name="chuk-mcp-lazarus",
    version=_pkg_version("chuk-mcp-lazarus"),
    title="Lazarus Interpretability Server",
    description=(
        "Mechanistic interpretability toolkit. "
        "Load any model, extract activations, train probes, "
        "steer generation, and ablate components to reveal "
        "internal structure."
    ),
)
