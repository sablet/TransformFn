"""CLI command implementations (used by __main__.py)."""

from __future__ import annotations

from typing import Any

from xform_core.dag.config import PipelineConfig
from xform_core.dag.executor import DAGExecutor
from xform_core.dag.resolver import TransformResolver
from xform_core.dag.skeleton import get_skeleton
from xform_core.dag.transform_registry import get_registry
from xform_core.dag.validator import ConfigurationValidator


def validate_command(config_path: str) -> int:
    """Validate configuration without execution.

    Returns:
        0 if valid, 1 if invalid
    """
    try:
        config = PipelineConfig(config_path)
        print(f"✓ Configuration {config_path} is valid")
        return 0
    except ValueError as e:
        print(f"✗ Configuration validation failed:\n{e}")
        return 1


def run_command(config_path: str, initial_inputs: dict[str, Any] | None = None) -> int:
    """Run pipeline from configuration.

    Returns:
        0 if successful, 1 if failed
    """
    try:
        config_obj = PipelineConfig(config_path)

        skeleton_fqn = config_obj.raw["pipeline"]["skeleton"]
        skeleton = get_skeleton(skeleton_fqn)

        registry = get_registry()
        resolver = TransformResolver(registry)
        validator = ConfigurationValidator(registry, skeleton)
        executor = DAGExecutor(skeleton, resolver, validator)

        print(f"Running pipeline: {skeleton.name}")
        result = executor.execute(config_obj.raw, initial_inputs or {})

        print("✓ Pipeline completed successfully")
        print(f"Results: {result}")
        return 0

    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def discover_command(
    input_type: str | None = None, output_type: str | None = None
) -> int:
    """Discover available transforms.

    If input_type and output_type are provided, searches for transforms
    matching those type names. Otherwise, lists all registered transforms.

    Returns:
        0 if successful
    """
    from inspect import signature, Parameter

    registry = get_registry()

    if input_type and output_type:
        # Type-based discovery: find transforms matching type names
        print(f"Searching transforms: {input_type} -> {output_type}")
        print()

        matches = []
        for fqn, sig in registry.list_all():
            # Match by type name (simple string matching)
            input_names = tuple(t.__name__ for t in sig.input_types)
            output_name = sig.output_type.__name__

            # Check if type names match
            if input_type in input_names and output_name == output_type:
                matches.append((fqn, sig))

        if not matches:
            print(f"No transforms found for {input_type} -> {output_type}")
            return 0

        print(f"Found {len(matches)} transform(s):")
        print()

        for fqn, sig in matches:
            print(f"  {fqn}")
            input_types_str = ", ".join(t.__name__ for t in sig.input_types)
            print(f"    Input:  ({input_types_str})")
            print(f"    Output: {sig.output_type.__name__}")

            # Show parameters
            if sig.params:
                print("    Parameters:")
                for param_name, param_info in sig.params.items():
                    print(f"      - {param_name}: {param_info}")
            else:
                # Try to extract parameters from function signature
                try:
                    func = registry.get_transform(fqn)
                    func_sig = signature(func)
                    params = [
                        name
                        for name, param in func_sig.parameters.items()
                        if param.kind
                        in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                        and name != list(func_sig.parameters.keys())[0]  # Skip first arg
                    ]
                    if params:
                        print("    Parameters:")
                        for param_name in params:
                            param_obj = func_sig.parameters[param_name]
                            default_str = (
                                f" (default: {param_obj.default})"
                                if param_obj.default is not Parameter.empty
                                else ""
                            )
                            type_str = (
                                f"{param_obj.annotation.__name__}"
                                if param_obj.annotation is not Parameter.empty
                                and hasattr(param_obj.annotation, "__name__")
                                else "Any"
                            )
                            print(f"      - {param_name}: {type_str}{default_str}")
                except Exception:
                    pass

            print()

    else:
        # List all registered transforms
        all_transforms = registry.list_all()

        if not all_transforms:
            print("No transforms registered in the registry.")
            return 0

        print(f"All registered transforms ({len(all_transforms)}):")
        print()

        # Group by module for better readability
        by_module: dict[str, list[tuple[str, Any]]] = {}
        for fqn, sig in all_transforms:
            module = ".".join(fqn.split(".")[:-1])
            if module not in by_module:
                by_module[module] = []
            by_module[module].append((fqn, sig))

        for module in sorted(by_module.keys()):
            print(f"  {module}:")
            for fqn, sig in by_module[module]:
                func_name = fqn.split(".")[-1]
                input_types_str = ", ".join(t.__name__ for t in sig.input_types)
                output_type_str = sig.output_type.__name__
                print(f"    - {func_name}")
                print(f"        {input_types_str} -> {output_type_str}")
            print()

    return 0


def generate_config_command(
    skeleton: str | None = None,
    generate_all: bool = False,
    output: str | None = None,
    output_dir: str | None = None,
    show_alternatives: bool = False,
) -> int:
    """Generate sample configuration from skeleton.

    Process:
    1. Get skeleton by name from registry
    2. For each step:
        - Select transform: default_transform or first candidate from registry
        - Extract parameters from function signature
        - Get default values for parameters
    3. Generate YAML with:
        - Pipeline metadata (name, skeleton reference)
        - Steps with transforms and parameters
        - Comments with alternatives (if show_alternatives)

    Returns:
        0 if successful, 1 if failed
    """
    from inspect import signature, Parameter
    from pathlib import Path

    from xform_core.dag.skeleton import get_skeleton, _SKELETON_REGISTRY

    registry = get_registry()

    if generate_all:
        # Generate for all registered skeletons
        if not output_dir:
            print("✗ Error: --output-dir required when using --all")
            return 1

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for skeleton_fqn in _SKELETON_REGISTRY.keys():
            skeleton_obj = get_skeleton(skeleton_fqn)
            config_path = output_path / f"{skeleton_obj.name}.yaml"
            _generate_single_config(
                skeleton_obj, skeleton_fqn, registry, str(config_path), show_alternatives
            )

        print(f"✓ Generated {len(_SKELETON_REGISTRY)} config files in {output_dir}")
        return 0

    if not skeleton:
        print("✗ Error: --skeleton or --all required")
        return 1

    # Generate single skeleton
    try:
        skeleton_obj = get_skeleton(skeleton)
    except ValueError as e:
        print(f"✗ Error: {e}")
        return 1

    if not output:
        print("✗ Error: --output required when using --skeleton")
        return 1

    _generate_single_config(skeleton_obj, skeleton, registry, output, show_alternatives)
    print(f"✓ Generated config: {output}")
    return 0


def _generate_single_config(
    skeleton: Any,
    skeleton_fqn: str,
    registry: Any,
    output_path: str,
    show_alternatives: bool,
) -> None:
    """Generate a single config file from skeleton."""
    from inspect import signature, Parameter
    from pathlib import Path

    lines = []
    lines.append("pipeline:")
    lines.append(f'  name: "{skeleton.name}"')
    lines.append(f'  skeleton: "{skeleton_fqn}"')
    lines.append("")
    lines.append(f"{skeleton.name}:")
    lines.append("  steps:")

    for step in skeleton.steps:
        lines.append(f"    {step.name}:")

        # Select transform
        transform_fqn = step.default_transform
        if not transform_fqn:
            # Try to find first candidate
            candidates = registry.find_transforms(step.input_types, step.output_type)
            if candidates:
                transform_fqn = candidates[0]

        if transform_fqn:
            lines.append(f'      transform: "{transform_fqn}"')

            # Show alternatives if requested
            if show_alternatives:
                candidates = registry.find_transforms(step.input_types, step.output_type)
                alternatives = [c for c in candidates if c != transform_fqn]
                if alternatives:
                    lines.append("      # Alternatives:")
                    for alt in alternatives:
                        lines.append(f"      #   - {alt}")

            # Extract parameters from function signature
            try:
                func = registry.get_transform(transform_fqn)
                sig = signature(func)

                # Get parameter list (exclude first positional arg - input data)
                param_list = list(sig.parameters.items())
                if param_list:
                    param_list = param_list[1:]  # Skip first arg

                params_with_defaults = []
                for name, param in param_list:
                    if param.kind in (
                        Parameter.KEYWORD_ONLY,
                        Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        if param.default is not Parameter.empty:
                            # Has default value
                            default_val = param.default
                            if isinstance(default_val, str):
                                params_with_defaults.append(
                                    f"        {name}: \"{default_val}\""
                                )
                            elif isinstance(default_val, bool):
                                params_with_defaults.append(
                                    f"        {name}: {str(default_val).lower()}"
                                )
                            else:
                                params_with_defaults.append(f"        {name}: {default_val}")

                if params_with_defaults:
                    lines.append("      params:")
                    lines.extend(params_with_defaults)

            except Exception:
                # If signature extraction fails, skip params
                pass
        else:
            lines.append('      # transform: "TODO: No default transform available"')

        lines.append("")

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))


__all__ = [
    "validate_command",
    "run_command",
    "discover_command",
    "generate_config_command",
]
