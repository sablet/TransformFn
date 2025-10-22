"""DAG Executor - Execute pipeline with dynamic transform selection."""

from __future__ import annotations

from typing import Any, Type

from xform_core.dag.resolver import TransformResolver
from xform_core.dag.skeleton import PipelineSkeleton
from xform_core.dag.validator import ConfigurationValidator, ValidationResult


class DAGExecutor:
    """Execute pipeline with dynamic transform selection.

    CRITICAL: Configuration must be validated before execution.
    """

    def __init__(
        self,
        skeleton: PipelineSkeleton,
        resolver: TransformResolver,
        validator: ConfigurationValidator,
    ):
        self.skeleton = skeleton
        self.resolver = resolver
        self.validator = validator

    def execute(
        self,
        config: dict[str, Any],
        initial_inputs: dict[str, Any],
        *,
        skip_validation: bool = False,
    ) -> dict[str, Any]:
        """Execute pipeline with configuration.

        Parameters:
            config: Step configurations with transform selections
            initial_inputs: Initial data for pipeline
            skip_validation: Skip validation (NOT RECOMMENDED, for testing only)

        Returns:
            Dictionary of step outputs

        Raises:
            ValueError: If configuration is invalid
        """
        # CRITICAL: Validate configuration before execution
        if not skip_validation:
            validation_result = self.validator.validate(config)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Configuration validation failed:\n{validation_result}"
                )

        outputs = {}
        context = {**initial_inputs}

        for step in self.skeleton.steps:
            # Get step configuration
            step_config = config.get("steps", {}).get(step.name, {})

            # Skip optional steps if not configured
            if not step_config and not step.required:
                continue

            # Resolve transform function and parameters
            func, params = self.resolver.resolve_step(step, step_config)

            # Collect inputs from context
            inputs = []
            for input_type in step.input_types:
                input_data = self._find_input_by_type(context, input_type)
                if input_data is None:
                    raise RuntimeError(
                        f"Required input type {input_type} not found in context "
                        f"for step {step.name}"
                    )
                inputs.append(input_data)

            # Execute transform
            print(f"Executing step: {step.name} with {func.__name__}")
            output = func(*inputs, **params)

            # Store output in context
            outputs[step.name] = output
            context[step.output_type.__name__] = output

        return outputs

    def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate configuration without executing pipeline.

        This should be called explicitly when loading configuration.
        """
        return self.validator.validate(config)

    def _find_input_by_type(
        self,
        context: dict[str, Any],
        target_type: Type[Any],
    ) -> Any | None:
        """Find data in context matching target type.

        Type matching strategy:
        1. Exact type name match
        2. Instance type check
        3. Structural compatibility (duck typing)
        """
        # Strategy 1: Exact type name match
        type_name = target_type.__name__
        if type_name in context:
            return context[type_name]

        # Strategy 2: Instance type check
        for value in context.values():
            if isinstance(value, target_type):
                return value

        # Strategy 3: Check for TypedDict or structural types
        # (Implementation depends on runtime type checking library)
        # For now, return None if not found
        return None


__all__ = ["DAGExecutor"]
