#!/usr/bin/env python3
"""Command-line entry point for the Video Game Sales Dashboard."""

import argparse
import logging
import multiprocessing

from src.app import create_app


def parse_arguments() -> argparse.Namespace:
    """Parse dashboard command-line options."""
    parser = argparse.ArgumentParser(description="Video Game Sales Dashboard")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    parser.add_argument("--port", type=int, default=8050, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--workers", type=int, default=1, help="Worker process count")
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        help="Maximum cache memory in MB (default: no explicit limit)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=20,
        help="Maximum number of cached filter results",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--disable-custom-templates",
        action="store_true",
        help="Disable custom Plotly templates",
    )
    parser.add_argument(
        "--simple-charts",
        action="store_true",
        help="Use simplified chart configurations",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Run with deterministic built-in sample data instead of the CSV dataset",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate and normalize command-line arguments."""
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    if args.workers < 1:
        raise ValueError("Number of workers must be at least 1")
    if args.cache_size < 1:
        raise ValueError("Cache size must be at least 1")
    if args.memory_limit is not None and args.memory_limit < 1:
        raise ValueError("Memory limit must be at least 1 MB")

    args.workers = min(args.workers, multiprocessing.cpu_count())


def main() -> None:
    """Create and run the dashboard."""
    args = parse_arguments()
    validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = create_app(
        memory_limit_mb=args.memory_limit,
        cache_size=args.cache_size,
        disable_custom_templates=args.disable_custom_templates,
        simple_charts=args.simple_charts,
        use_sample_data=args.sample_data,
    )
    app.run_server(
        debug=args.debug,
        port=args.port,
        host=args.host,
        use_reloader=args.debug,
        processes=args.workers,
    )


if __name__ == "__main__":
    main()
