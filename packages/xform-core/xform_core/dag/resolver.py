"""Transform Resolver - Resolve transform functions from configuration."""

from __future__ import annotations

from typing import Any, Callable

from xform_core.dag.skeleton import PipelineStep
from xform_core.dag.transform_registry import TransformRegistry


class TransformResolver:
    """Resolve transform functions from configuration.

    Note: This assumes configuration has already been validated.
    Use ConfigurationValidator before calling resolver methods.
    """

    def __init__(self, registry: TransformRegistry):
        self.registry = registry

    def resolve_step(
        self,
        step: PipelineStep,
        config: dict[str, Any],
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Resolve transform function and parameters for a step.

        Returns:
            Tuple of (function, params)

        Raises:
            ValueError: If no transform specified
            TypeError: If type signature mismatch (should be caught by validator)
        """
        # Get transform FQN from config or use default
        transform_fqn = config.get("transform", step.default_transform)
        if transform_fqn is None:
            raise ValueError(f"No transform specified for step: {step.name}")

        # Get actual function (validation already done)
        func = self.registry.get_transform(transform_fqn)

        # Get parameters
        params = config.get("params", {})

        return func, params


__all__ = ["TransformResolver"]
