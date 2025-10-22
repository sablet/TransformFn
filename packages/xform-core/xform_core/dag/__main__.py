"""Unified DAG CLI entry point - handles all apps dynamically.

Usage:
    uv run python -m xform_core.dag apps/algo-trade validate configs/pipeline.yaml
    uv run python -m xform_core.dag apps/algo-trade run configs/pipeline.yaml
    uv run python -m xform_core.dag apps/algo-trade discover
"""

import sys
import argparse
from pathlib import Path
from importlib import import_module

from xform_core.dag.cli import (
    validate_command,
    run_command,
    discover_command,
    generate_config_command,
)


def main() -> int:
    """Main CLI entry point with automatic app detection."""
    parser = argparse.ArgumentParser(
        description="DAG pipeline execution (unified across all apps)"
    )
    parser.add_argument(
        "app_path",
        help="App directory path (e.g., apps/algo-trade)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate pipeline configuration",
    )
    validate_parser.add_argument("config", help="Path to config YAML")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline from configuration")
    run_parser.add_argument("config", help="Path to config YAML")

    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover available transforms",
    )
    discover_parser.add_argument("--input-type", help="Input type to search for")
    discover_parser.add_argument("--output-type", help="Output type to search for")

    # Generate-config command
    generate_parser = subparsers.add_parser(
        "generate-config",
        help="Generate sample config from skeleton",
    )
    generate_parser.add_argument(
        "--skeleton",
        help="Skeleton FQN (e.g., algo_trade_dag.skeleton.phase1_skeleton)",
    )
    generate_parser.add_argument(
        "--all", action="store_true", help="Generate for all skeletons"
    )
    generate_parser.add_argument("--output", help="Output file path")
    generate_parser.add_argument("--output-dir", help="Output directory (for --all)")
    generate_parser.add_argument(
        "--show-alternatives",
        action="store_true",
        help="Include alternative transforms as comments",
    )

    args = parser.parse_args()

    # Resolve and import app skeleton module to register skeletons
    app_module = _resolve_app_module(args.app_path)
    try:
        import_module(f"{app_module}.skeleton")
    except ImportError as e:
        print(f"✗ Failed to import {app_module}.skeleton: {e}")
        print(f"  Ensure {args.app_path}/{app_module}/skeleton.py exists")
        return 1

    # Execute command
    if args.command == "validate":
        return validate_command(args.config)
    elif args.command == "run":
        return run_command(args.config)
    elif args.command == "discover":
        return discover_command(
            getattr(args, "input_type", None),
            getattr(args, "output_type", None),
        )
    elif args.command == "generate-config":
        return generate_config_command(
            skeleton=getattr(args, "skeleton", None),
            generate_all=getattr(args, "all", False),
            output=getattr(args, "output", None),
            output_dir=getattr(args, "output_dir", None),
            show_alternatives=getattr(args, "show_alternatives", False),
        )

    return 1


def _resolve_app_module(app_path: str) -> str:
    """Convert app path to module name.

    Examples:
        apps/algo-trade -> algo_trade_dag
        apps/pipeline-app -> pipeline_app
    """
    path = Path(app_path)
    app_name = path.name.replace("-", "_")
    return f"{app_name}_dag"


if __name__ == "__main__":
    sys.exit(main())
