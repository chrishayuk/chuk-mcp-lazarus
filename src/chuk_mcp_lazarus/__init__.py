"""
chuk-mcp-lazarus -- Mechanistic interpretability MCP server.

Importing this package registers all tools and resources on the
shared ChukMCPServer instance.
"""

from ._bootstrap import ensure_optional_stubs as _ensure_optional_stubs  # noqa: F401

from .server import mcp  # noqa: F401
from .tools import *  # noqa: F401, F403
from . import resources  # noqa: F401

__version__ = "0.8.0"
__all__ = ["mcp"]
