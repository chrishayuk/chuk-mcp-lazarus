"""Tests for _bootstrap.py — optional dependency stubs."""

import sys
import types
from unittest.mock import patch

from chuk_mcp_lazarus._bootstrap import ensure_optional_stubs


class TestEnsureOptionalStubs:
    """ensure_optional_stubs installs stubs for missing chuk_virtual_expert."""

    def test_stub_installed_when_missing(self) -> None:
        # Remove existing stub
        saved = sys.modules.pop("chuk_virtual_expert", None)
        try:
            ensure_optional_stubs()
            mod = sys.modules["chuk_virtual_expert"]
            assert isinstance(mod, types.ModuleType)
            assert mod.__version__ == "0.0.0-stub"
        finally:
            if saved is not None:
                sys.modules["chuk_virtual_expert"] = saved

    def test_stub_has_classes(self) -> None:
        saved = sys.modules.pop("chuk_virtual_expert", None)
        try:
            ensure_optional_stubs()
            mod = sys.modules["chuk_virtual_expert"]
            for name in (
                "VirtualExpert",
                "VirtualExpertResult",
                "VirtualExpertAction",
                "VirtualExpertPlugin",
                "VirtualExpertRegistry",
            ):
                assert hasattr(mod, name)
                assert isinstance(getattr(mod, name), type)
        finally:
            if saved is not None:
                sys.modules["chuk_virtual_expert"] = saved

    def test_not_installed_when_present(self) -> None:
        # Put a real module in place
        real = types.ModuleType("chuk_virtual_expert")
        real.marker = "real"  # type: ignore[attr-defined]
        sys.modules["chuk_virtual_expert"] = real
        try:
            ensure_optional_stubs()
            assert sys.modules["chuk_virtual_expert"].marker == "real"  # type: ignore[attr-defined]
        finally:
            sys.modules["chuk_virtual_expert"] = real
