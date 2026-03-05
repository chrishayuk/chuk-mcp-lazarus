"""
chuk-mcp-lazarus -- Mechanistic interpretability MCP server.

Importing this package registers all tools and resources on the
shared ChukMCPServer instance.
"""

from importlib.metadata import version as _pkg_version

from ._bootstrap import ensure_optional_stubs as _ensure_optional_stubs  # noqa: F401

from .server import mcp  # noqa: F401
from .tools import *  # noqa: F401, F403
from . import resources  # noqa: F401

__version__ = _pkg_version("chuk-mcp-lazarus")
__all__ = ["mcp"]
