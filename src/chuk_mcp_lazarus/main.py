"""
Lazarus Interpretability Server -- entry point.

Supports stdio mode (for MCP clients) and HTTP mode (for development).
"""

import argparse
import logging
import sys

from .server import mcp


def main() -> None:
    """Run the Lazarus interpretability server."""
    parser = argparse.ArgumentParser(
        prog="chuk-mcp-lazarus",
        description="Mechanistic interpretability MCP server wrapping chuk-lazarus",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Transport mode")

    stdio_parser = subparsers.add_parser("stdio", help="Run in stdio mode for MCP clients")
    stdio_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    http_parser = subparsers.add_parser("http", help="Run in HTTP mode (default)")
    http_parser.add_argument("--host", default="localhost", help="Host to bind to")
    http_parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    http_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    if args.mode is None:
        args.mode = "http"
        args.host = "localhost"
        args.port = 8765
        args.debug = False

    if args.mode == "stdio":
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        registered_tools = mcp.get_tools()
        registered_resources = mcp.get_resources()

        print(f"Lazarus Interpretability Server v0.5.0", file=sys.stdout)
        print("=" * 50, file=sys.stdout)
        print(f"Tools ({len(registered_tools)}):", file=sys.stdout)
        for tool in registered_tools:
            print(f"  - {tool.name}: {tool.description}", file=sys.stdout)
        print(f"Resources ({len(registered_resources)}):", file=sys.stdout)
        for resource in registered_resources:
            print(f"  - {resource.uri}: {resource.description}", file=sys.stdout)
        print("=" * 50, file=sys.stdout)

    try:
        if args.mode == "stdio":
            mcp.run(stdio=True, debug=args.debug)
        else:
            mcp.run(host=args.host, port=args.port, debug=args.debug, stdio=False)
    except KeyboardInterrupt:
        if args.mode == "http":
            print("\nLazarus server shutting down.", file=sys.stdout)


if __name__ == "__main__":
    main()
