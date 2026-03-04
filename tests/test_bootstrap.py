"""Tests for _bootstrap.py — optional dependency stubs.

Tests work whether chuk-virtual-expert is really installed or not.
"""

import sys
import types
from importlib.abc import MetaPathFinder

from chuk_mcp_lazarus._bootstrap import ensure_optional_stubs

_STUB_CLASSES = (
    "VirtualExpert",
    "VirtualExpertResult",
    "VirtualExpertAction",
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
)


class _BlockVirtualExpert(MetaPathFinder):
    """Meta-path finder that blocks importing chuk_virtual_expert."""

    def find_spec(self, fullname, path, target=None):  # type: ignore[override]
        if fullname == "chuk_virtual_expert":
            raise ImportError("blocked for test")
        return None


def _run_with_blocked_import(func):
    """Remove chuk_virtual_expert from sys.modules and block re-import, then call func."""
    saved = sys.modules.pop("chuk_virtual_expert", None)
    blocker = _BlockVirtualExpert()
    sys.meta_path.insert(0, blocker)
    try:
        return func()
    finally:
        sys.meta_path.remove(blocker)
        if saved is not None:
            sys.modules["chuk_virtual_expert"] = saved
        elif "chuk_virtual_expert" in sys.modules:
            del sys.modules["chuk_virtual_expert"]


class TestEnsureOptionalStubs:
    """ensure_optional_stubs installs stubs when chuk_virtual_expert is missing."""

    def test_stub_installed_when_missing(self) -> None:
        """When the real package is unavailable, a stub is created."""

        def check():
            ensure_optional_stubs()
            mod = sys.modules["chuk_virtual_expert"]
            assert isinstance(mod, types.ModuleType)
            assert mod.__version__ == "0.0.0-stub"  # type: ignore[attr-defined]

        _run_with_blocked_import(check)

    def test_stub_has_classes(self) -> None:
        """The stub provides all expected stand-in classes."""

        def check():
            ensure_optional_stubs()
            mod = sys.modules["chuk_virtual_expert"]
            for name in _STUB_CLASSES:
                assert hasattr(mod, name)
                assert isinstance(getattr(mod, name), type)

        _run_with_blocked_import(check)

    def test_not_replaced_when_present(self) -> None:
        """If the module is already in sys.modules, it is not replaced."""
        sentinel = types.ModuleType("chuk_virtual_expert")
        sentinel.marker = "real"  # type: ignore[attr-defined]
        saved = sys.modules.get("chuk_virtual_expert")
        sys.modules["chuk_virtual_expert"] = sentinel
        try:
            ensure_optional_stubs()
            assert sys.modules["chuk_virtual_expert"].marker == "real"  # type: ignore[attr-defined]
        finally:
            if saved is not None:
                sys.modules["chuk_virtual_expert"] = saved
            else:
                del sys.modules["chuk_virtual_expert"]
