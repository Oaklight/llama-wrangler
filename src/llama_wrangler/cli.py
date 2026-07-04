"""CLI entry point for llama-wrangler."""

import argparse
import sys

from llama_wrangler import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llama-wrangler",
        description="Lightweight web admin panel for llama.cpp server management",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the admin panel (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the admin panel (default: 7860)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: ~/.config/llama-wrangler/config.json)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    from llama_wrangler.config import load_config
    from llama_wrangler.server import create_app

    config, config_path = load_config(args.config)

    print(f"llama-wrangler v{__version__}")
    print(f"Config: {config_path}")
    print(f"Models dir: {config.models_dir}")
    print(f"Starting on http://{args.host}:{args.port}")

    app = create_app(config, config_path)

    try:
        app.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
