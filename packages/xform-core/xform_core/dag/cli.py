"""CLI command implementations (used by __main__.py)."""

from __future__ import annotations

import logging
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

from xform_core.dag.config import PipelineConfig
from xform_core.dag.executor import DAGExecutor
from xform_core.dag.resolver import TransformResolver
from xform_core.dag.skeleton import get_skeleton
from xform_core.dag.transform_registry import (
    TransformRegistry,
    TransformSignature,
    get_registry,
)
from xform_core.dag.validator import ConfigurationValidator


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep


def validate_command(config_path: str) -> int:
    """Validate configuration without execution.

    Returns:
        0 if valid, 1 if invalid
    """
    try:
        PipelineConfig(config_path)
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
    """Discover available transforms."""
    registry = get_registry()

    if input_type and output_type:
        _discover_by_signature(registry, input_type, output_type)
        return 0

    _list_all_transforms(registry)
    return 0


def _discover_by_signature(
    registry: TransformRegistry, input_type: str, output_type: str
) -> None:
    print(f"Searching transforms: {input_type} -> {output_type}")
    print()

    matches = [
        (fqn, sig)
        for fqn, sig in registry.list_all()
        if input_type in tuple(t.__name__ for t in sig.input_types)
        and sig.output_type.__name__ == output_type
    ]

    if not matches:
        print(f"No transforms found for {input_type} -> {output_type}")
        return

    print(f"Found {len(matches)} transform(s):")
    print()

    for fqn, sig in matches:
        _print_transform_details(
            fqn,
            sig,
            registry,
            heading_prefix="  ",
            indent="    ",
        )


def _list_all_transforms(registry: TransformRegistry) -> None:
    all_transforms = registry.list_all()
    if not all_transforms:
        print("No transforms registered in the registry.")
        return

    print(f"All registered transforms ({len(all_transforms)}):")
    print()

    by_module: dict[str, list[tuple[str, TransformSignature]]] = {}
    for fqn, sig in all_transforms:
        module = ".".join(fqn.split(".")[:-1])
        by_module.setdefault(module, []).append((fqn, sig))

    for module in sorted(by_module):
        print(f"  {module}:")
        for fqn, sig in by_module[module]:
            _print_transform_details(
                fqn,
                sig,
                registry,
                heading_prefix="    - ",
                indent="        ",
            )
        print()


def _print_transform_details(
    fqn: str,
    signature_info: TransformSignature,
    registry: TransformRegistry,
    *,
    heading_prefix: str,
    indent: str,
) -> None:
    print(f"{heading_prefix}{fqn}")
    input_types_str = ", ".join(t.__name__ for t in signature_info.input_types)
    print(f"{indent}Input:  ({input_types_str})")
    print(f"{indent}Output: {signature_info.output_type.__name__}")

    param_lines = _format_parameters(fqn, signature_info, registry)
    if param_lines:
        print(f"{indent}Parameters:")
        for line in param_lines:
            print(f"{indent}  - {line}")
    print()


def _format_parameters(
    fqn: str,
    signature_info: TransformSignature,
    registry: TransformRegistry,
) -> list[str]:
    if signature_info.params:
        return [f"{name}: {info}" for name, info in signature_info.params.items()]
    return _extract_parameters_from_signature(fqn, registry)


def _extract_parameters_from_signature(
    fqn: str, registry: TransformRegistry
) -> list[str]:
    try:
        func = registry.get_transform(fqn)
        func_sig = signature(func)
    except Exception as exc:  # pragma: no cover - introspection failure
        logger.debug("Failed to introspect parameters for %s: %s", fqn, exc)
        return []

    parameters = list(func_sig.parameters.items())
    if not parameters:
        return []

    first_param_name = parameters[0][0]
    lines: list[str] = []

    for name, param in parameters:
        if name == first_param_name:
            continue
        if param.kind not in (
            Parameter.KEYWORD_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue

        default_str = (
            f" (default: {param.default})"
            if param.default is not Parameter.empty
            else ""
        )
        annotation = param.annotation
        type_name = "Any"
        if annotation is not Parameter.empty:
            type_name = getattr(annotation, "__name__", str(annotation))

        lines.append(f"{name}: {type_name}{default_str}")

    return lines


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
                skeleton_obj,
                skeleton_fqn,
                registry,
                str(config_path),
                show_alternatives,
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
    skeleton: "PipelineSkeleton",
    skeleton_fqn: str,
    registry: TransformRegistry,
    output_path: str,
    show_alternatives: bool,
) -> None:
    """Generate a single config file from skeleton."""
    from pathlib import Path

    lines = []
    lines.append("pipeline:")
    lines.append(f'  name: "{skeleton.name}"')
    lines.append(f'  skeleton: "{skeleton_fqn}"')
    lines.append("")
    lines.append(f"{skeleton.name}:")
    lines.append("  steps:")

    for step in skeleton.steps:
        lines.extend(_collect_step_lines(step, registry, show_alternatives))

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))


def _collect_step_lines(
    step: "PipelineStep",
    registry: TransformRegistry,
    show_alternatives: bool,
) -> list[str]:
    lines = [f"    {step.name}:"]
    transform_fqn = _select_transform(step, registry)

    if transform_fqn is None:
        lines.append('      # transform: "TODO: No default transform available"')
        lines.append("")
        return lines

    lines.append(f'      transform: "{transform_fqn}"')

    if show_alternatives:
        lines.extend(_collect_alternative_lines(step, registry, transform_fqn))

    param_lines = _collect_param_lines(transform_fqn, registry)
    if param_lines:
        lines.append("      params:")
        lines.extend(param_lines)

    lines.append("")
    return lines


def _select_transform(step: "PipelineStep", registry: TransformRegistry) -> str | None:
    if step.default_transform:
        return step.default_transform

    candidates = registry.find_transforms(step.input_types, step.output_type)
    return candidates[0] if candidates else None


def _collect_alternative_lines(
    step: "PipelineStep",
    registry: TransformRegistry,
    selected: str,
) -> list[str]:
    candidates = registry.find_transforms(step.input_types, step.output_type)
    alternatives = [candidate for candidate in candidates if candidate != selected]
    if not alternatives:
        return []

    lines = ["      # Alternatives:"]
    lines.extend(f"      #   - {alt}" for alt in alternatives)
    return lines


def _collect_param_lines(transform_fqn: str, registry: TransformRegistry) -> list[str]:
    try:
        func = registry.get_transform(transform_fqn)
        func_sig = signature(func)
    except Exception as exc:  # pragma: no cover - introspection failure
        logger.debug("Failed to extract parameters for %s: %s", transform_fqn, exc)
        return []

    parameters = list(func_sig.parameters.items())[1:]  # Skip first arg (input data)
    lines: list[str] = []

    for name, param in parameters:
        if param.kind not in (
            Parameter.KEYWORD_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if param.default is Parameter.empty:
            continue

        formatted_default = _format_default_value(param.default)
        lines.append(f"        {name}: {formatted_default}")

    return lines


def _format_default_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value) if isinstance(value, (dict, list, tuple, set)) else str(value)


__all__ = [
    "validate_command",
    "run_command",
    "discover_command",
    "generate_config_command",
]
