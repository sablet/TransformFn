"""Configuration Validator - Fail-fast validation before execution."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, Signature, signature
from typing import Any, Callable, get_type_hints

from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep
from xform_core.dag.transform_registry import TransformRegistry


@dataclass(slots=True, frozen=True)
class ValidationError:
    """Single validation error."""

    phase: str
    step: str
    error_type: str
    message: str
    suggestion: str | None = None


@dataclass(slots=True, frozen=True)
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: tuple[ValidationError, ...]
    warnings: tuple[ValidationError, ...]

    def __str__(self) -> str:
        """Format validation result for display."""
        if self.is_valid:
            return "✓ Configuration is valid"

        lines = ["✗ Configuration validation failed:\n"]

        for err in self.errors:
            lines.append(f"  [ERROR] {err.phase}.{err.step}: {err.error_type}")
            lines.append(f"    {err.message}")
            if err.suggestion:
                lines.append(f"    Suggestion: {err.suggestion}")
            lines.append("")

        for warn in self.warnings:
            lines.append(f"  [WARNING] {warn.phase}.{warn.step}: {warn.error_type}")
            lines.append(f"    {warn.message}")
            lines.append("")

        return "\n".join(lines)


class ConfigurationValidator:
    """Validate pipeline configuration before execution."""

    def __init__(
        self,
        registry: TransformRegistry,
        skeleton: PipelineSkeleton,
    ):
        self.registry = registry
        self.skeleton = skeleton

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate entire configuration.

        Checks performed:
        1. Skeleton step coverage (all required steps present)
        2. Transform FQN existence
        3. Type signature compatibility
        4. Parameter schema validation
        5. Required parameter completeness

        Returns:
            ValidationResult with all errors and warnings
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Extract phase name from skeleton
        phase_name = self.skeleton.name

        # Check 1: Skeleton step coverage
        config_steps = set(config.get("steps", {}).keys())
        required_steps = {step.name for step in self.skeleton.steps if step.required}
        missing_steps = required_steps - config_steps

        for step_name in missing_steps:
            step = next(s for s in self.skeleton.steps if s.name == step_name)
            if step.default_transform is None:
                errors.append(
                    ValidationError(
                        phase=phase_name,
                        step=step_name,
                        error_type="MISSING_REQUIRED_STEP",
                        message=(
                            f"Required step '{step_name}' not found in configuration"
                        ),
                        suggestion="Add step configuration with transform selection",
                    )
                )

        # Check each configured step
        for step_name, step_config in config.get("steps", {}).items():
            # Find step in skeleton
            step_match = next(
                (s for s in self.skeleton.steps if s.name == step_name),
                None,
            )

            if step_match is None:
                warnings.append(
                    ValidationError(
                        phase=phase_name,
                        step=step_name,
                        error_type="UNKNOWN_STEP",
                        message=f"Step '{step_name}' not defined in skeleton",
                        suggestion="Remove this step or check skeleton definition",
                    )
                )
                continue

            step = step_match

            # Get transform FQN
            transform_fqn = step_config.get("transform", step.default_transform)

            if transform_fqn is None:
                errors.append(
                    ValidationError(
                        phase=phase_name,
                        step=step_name,
                        error_type="NO_TRANSFORM_SPECIFIED",
                        message="No transform specified and no default available",
                        suggestion=self._suggest_transforms(step),
                    )
                )
                continue

            # Check 2: Transform existence
            if not self.registry.has_transform(transform_fqn):
                errors.append(
                    ValidationError(
                        phase=phase_name,
                        step=step_name,
                        error_type="TRANSFORM_NOT_FOUND",
                        message=f"Transform '{transform_fqn}' not found in registry",
                        suggestion=self._suggest_transforms(step),
                    )
                )
                continue

            # Check 3: Type signature compatibility
            if not self.registry.validate_signature(
                transform_fqn,
                step.input_types,
                step.output_type,
            ):
                actual_sig = self.registry.get_signature(transform_fqn)
                expected_desc = f"{step.input_types} -> {step.output_type}"
                actual_desc = f"{actual_sig.input_types} -> {actual_sig.output_type}"
                message = (
                    "Transform signature mismatch:\n"
                    f"  Expected: {expected_desc}\n"
                    f"  Actual: {actual_desc}"
                )
                errors.append(
                    ValidationError(
                        phase=phase_name,
                        step=step_name,
                        error_type="TYPE_SIGNATURE_MISMATCH",
                        message=message,
                        suggestion=self._suggest_transforms(step),
                    )
                )
                continue

            # Check 4 & 5: Parameter validation
            params = step_config.get("params", {})
            param_errors = self._validate_parameters(
                phase_name,
                step_name,
                transform_fqn,
                params,
            )
            errors.extend(param_errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _validate_parameters(
        self,
        phase: str,
        step: str,
        transform_fqn: str,
        params: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate parameters against transform signature."""
        func = self.registry.get_transform(transform_fqn)
        sig = signature(func)

        transform_sig = self.registry.get_signature(transform_fqn)
        param_list = _slice_parameter_list(sig, len(transform_sig.input_types))

        errors: list[ValidationError] = []

        valid_params = _collect_valid_param_names(param_list)
        errors.extend(
            _build_unknown_param_errors(
                params,
                valid_params,
                phase,
                step,
                transform_fqn,
            )
        )

        errors.extend(
            _build_missing_param_errors(
                params,
                param_list,
                phase,
                step,
            )
        )

        type_hints = _resolve_type_hints(func)
        errors.extend(
            _build_type_mismatch_errors(
                params,
                param_list,
                type_hints,
                phase,
                step,
            )
        )

        return errors

    def _suggest_transforms(self, step: PipelineStep) -> str:
        """Suggest available transforms for a step."""
        candidates = self.registry.find_transforms(
            step.input_types,
            step.output_type,
        )

        if not candidates:
            return (
                "No compatible transforms found for "
                f"{step.input_types} -> {step.output_type}"
            )

        return "Available transforms:\n    " + "\n    ".join(
            f"- {candidate}" for candidate in candidates
        )


def _slice_parameter_list(
    sig: Signature, num_inputs: int
) -> list[tuple[str, Parameter]]:
    parameters = list(sig.parameters.items())
    if num_inputs <= 0:
        return parameters
    return parameters[num_inputs:]


def _collect_valid_param_names(param_list: list[tuple[str, Parameter]]) -> set[str]:
    return {
        name
        for name, param in param_list
        if param.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    }


def _build_unknown_param_errors(
    params: dict[str, Any],
    valid_params: set[str],
    phase: str,
    step: str,
    transform_fqn: str,
) -> list[ValidationError]:
    unknown_params = set(params.keys()) - valid_params
    return [
        ValidationError(
            phase=phase,
            step=step,
            error_type="UNKNOWN_PARAMETER",
            message=f"Parameter '{param_name}' not accepted by {transform_fqn}",
            suggestion=f"Valid parameters: {', '.join(sorted(valid_params))}",
        )
        for param_name in sorted(unknown_params)
    ]


def _build_missing_param_errors(
    params: dict[str, Any],
    param_list: list[tuple[str, Parameter]],
    phase: str,
    step: str,
) -> list[ValidationError]:
    required_params = {
        name
        for name, param in param_list
        if param.default is Parameter.empty
        and param.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    }

    missing = required_params - set(params.keys())
    return [
        ValidationError(
            phase=phase,
            step=step,
            error_type="MISSING_REQUIRED_PARAMETER",
            message=f"Required parameter '{param_name}' not provided",
            suggestion=f"Add '{param_name}' to params section",
        )
        for param_name in sorted(missing)
    ]


def _resolve_type_hints(func: Callable[..., object]) -> dict[str, Any]:
    try:
        return get_type_hints(func)
    except Exception:
        return {}


def _build_type_mismatch_errors(
    params: dict[str, Any],
    param_list: list[tuple[str, Parameter]],
    type_hints: dict[str, Any],
    phase: str,
    step: str,
) -> list[ValidationError]:
    param_map = dict(param_list)
    errors: list[ValidationError] = []

    for param_name, param_value in params.items():
        error = _validate_type_for_param(
            param_name,
            param_value,
            param_map,
            type_hints,
            phase,
            step,
        )
        if error is not None:
            errors.append(error)

    return errors


def _validate_type_for_param(
    param_name: str,
    param_value: object,
    param_map: dict[str, Parameter],
    type_hints: dict[str, Any],
    phase: str,
    step: str,
) -> ValidationError | None:
    parameter = param_map.get(param_name)
    if parameter is None:
        return None

    expected_type = type_hints.get(param_name)
    if expected_type is None:
        annotation = parameter.annotation
        if annotation is Parameter.empty:
            return None
        expected_type = annotation

    if hasattr(expected_type, "__origin__"):
        return None

    if not isinstance(expected_type, type):
        return None

    try:
        matches_type = isinstance(param_value, expected_type)
    except TypeError:
        matches_type = False

    if matches_type:
        return None

    type_name = getattr(expected_type, "__name__", str(expected_type))
    return ValidationError(
        phase=phase,
        step=step,
        error_type="PARAMETER_TYPE_MISMATCH",
        message=(
            f"Parameter '{param_name}' type mismatch:\n"
            f"  Expected: {type_name}\n"
            f"  Got: {type(param_value).__name__}"
        ),
        suggestion=f"Convert value to {type_name}",
    )


__all__ = [
    "ValidationError",
    "ValidationResult",
    "ConfigurationValidator",
]
