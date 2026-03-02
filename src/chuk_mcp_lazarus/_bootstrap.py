"""
Bootstrap for optional chuk-lazarus dependencies.

chuk-lazarus v0.4 has a hard import on chuk-virtual-expert which
may not be installed. This module installs a lightweight stub so
that the import chain succeeds. Only the virtual-expert features
are unavailable; everything else (inference, introspection, hooks)
works normally.

This module must be imported before any chuk_lazarus imports.
"""

from __future__ import annotations

import sys
import types


def ensure_optional_stubs() -> None:
    """Install stubs for optional dependencies that may be missing."""
    if "chuk_virtual_expert" not in sys.modules:
        try:
            import chuk_virtual_expert  # noqa: F401
        except ImportError:
            stub = types.ModuleType("chuk_virtual_expert")
            stub.__version__ = "0.0.0-stub"
            # Create minimal stand-in classes
            for name in (
                "VirtualExpert",
                "VirtualExpertResult",
                "VirtualExpertAction",
                "VirtualExpertPlugin",
                "VirtualExpertRegistry",
            ):
                setattr(stub, name, type(name, (), {}))
            sys.modules["chuk_virtual_expert"] = stub


# Run on import
ensure_optional_stubs()
