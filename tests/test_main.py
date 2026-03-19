"""Tests for main.py entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.main import main


class TestMainStdio:
    """Test stdio mode argument parsing and launch."""

    def test_stdio_mode(self) -> None:
        """stdio mode calls mcp.run(stdio=True)."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "stdio"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            main()
            mock_mcp.run.assert_called_once_with(stdio=True, debug=False)

    def test_stdio_debug(self) -> None:
        """stdio --debug passes debug=True."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "stdio", "--debug"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            main()
            mock_mcp.run.assert_called_once_with(stdio=True, debug=True)


class TestMainHttp:
    """Test HTTP mode argument parsing and launch."""

    def test_http_mode_explicit(self) -> None:
        """http mode calls mcp.run with host/port."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "http"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            mock_mcp.get_tools.return_value = []
            mock_mcp.get_resources.return_value = []
            main()
            mock_mcp.run.assert_called_once_with(
                host="localhost", port=8765, debug=False, stdio=False
            )

    def test_http_custom_port(self) -> None:
        """http --port overrides default."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "http", "--port", "9999"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            mock_mcp.get_tools.return_value = []
            mock_mcp.get_resources.return_value = []
            main()
            mock_mcp.run.assert_called_once_with(
                host="localhost", port=9999, debug=False, stdio=False
            )

    def test_http_custom_host(self) -> None:
        """http --host overrides default."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "http", "--host", "0.0.0.0"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            mock_mcp.get_tools.return_value = []
            mock_mcp.get_resources.return_value = []
            main()
            mock_mcp.run.assert_called_once_with(
                host="0.0.0.0", port=8765, debug=False, stdio=False
            )

    def test_http_debug(self) -> None:
        """http --debug passes debug=True."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "http", "--debug"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            mock_mcp.get_tools.return_value = []
            mock_mcp.get_resources.return_value = []
            main()
            mock_mcp.run.assert_called_once_with(
                host="localhost", port=8765, debug=True, stdio=False
            )


class TestMainDefault:
    """Test default mode (no subcommand)."""

    def test_no_mode_defaults_to_http(self) -> None:
        """No subcommand defaults to http with default host/port."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            mock_mcp.get_tools.return_value = []
            mock_mcp.get_resources.return_value = []
            main()
            mock_mcp.run.assert_called_once_with(
                host="localhost", port=8765, debug=False, stdio=False
            )


class TestMainKeyboardInterrupt:
    """Test graceful shutdown on KeyboardInterrupt."""

    def test_http_keyboard_interrupt(self, capsys: pytest.CaptureFixture[str]) -> None:
        """KeyboardInterrupt in HTTP mode prints shutdown message."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "http"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run.side_effect = KeyboardInterrupt()
            mock_mcp.get_tools.return_value = []
            mock_mcp.get_resources.return_value = []
            main()
            captured = capsys.readouterr()
            assert "shutting down" in captured.out

    def test_stdio_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt in stdio mode does not print."""
        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "stdio"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run.side_effect = KeyboardInterrupt()
            # Should not raise
            main()


class TestMainToolListing:
    """Test that HTTP mode prints tool and resource listings."""

    def test_lists_tools_and_resources(self, capsys: pytest.CaptureFixture[str]) -> None:
        """HTTP mode prints registered tools and resources."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_resource = MagicMock()
        mock_resource.uri = "test://resource"
        mock_resource.description = "A test resource"

        with (
            patch("sys.argv", ["chuk-mcp-lazarus", "http"]),
            patch("chuk_mcp_lazarus.main.mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            mock_mcp.get_tools.return_value = [mock_tool]
            mock_mcp.get_resources.return_value = [mock_resource]
            main()

        captured = capsys.readouterr()
        assert "test_tool" in captured.out
        assert "A test tool" in captured.out
        assert "test://resource" in captured.out
        assert "A test resource" in captured.out
