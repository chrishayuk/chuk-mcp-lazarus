"""Tests for server.py — ChukMCPServer instance."""

from chuk_mcp_lazarus.server import mcp


class TestServer:
    """MCP server instance is configured correctly."""

    def test_exists(self) -> None:
        assert mcp is not None

    def test_name(self) -> None:
        assert mcp.name == "chuk-mcp-lazarus"

    def test_version(self) -> None:
        assert mcp.version == "0.9.0"
