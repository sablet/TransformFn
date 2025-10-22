"""DAG Skeleton definition - type-driven pipeline structure."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Type


@dataclass(slots=True, frozen=True)
class PipelineStep:
    """Single step in pipeline skeleton."""

    name: str
    input_types: tuple[Type[Any], ...]
    output_type: Type[Any]
    default_transform: str | None = None
    required: bool = True


@dataclass(slots=True)
class PipelineSkeleton:
    """Pipeline structure definition (reusable across apps)."""

    name: str
    steps: list[PipelineStep]

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate that config provides all required steps.

        Args:
            config: Configuration dict with 'steps' key

        Returns:
            True if all required steps are configured or have defaults,
            False otherwise
        """
        required_steps = {step.name for step in self.steps if step.required}
        config_steps = set(config.get("steps", {}).keys())

        missing_steps = required_steps - config_steps
        for step_name in missing_steps:
            step = next(s for s in self.steps if s.name == step_name)
            # Missing step is OK only if it has a default transform
            if step.default_transform is None:
                return False

        return True


# Skeleton registry for dynamic lookup
_SKELETON_REGISTRY: dict[str, PipelineSkeleton] = {}
_registry_lock = RLock()


def register_skeleton(fqn: str, skeleton: PipelineSkeleton) -> None:
    """Register a skeleton for dynamic lookup.

    Args:
        fqn: Fully qualified name for the skeleton
        skeleton: The skeleton instance to register

    Raises:
        ValueError: If FQN is empty or already registered
    """
    if not fqn:
        raise ValueError("skeleton FQN must be a non-empty string")

    with _registry_lock:
        if fqn in _SKELETON_REGISTRY:
            raise ValueError(
                f"skeleton '{fqn}' already registered; "
                "use explicit removal before re-registration"
            )
        _SKELETON_REGISTRY[fqn] = skeleton


def get_skeleton(fqn: str) -> PipelineSkeleton:
    """Get skeleton by fully qualified name.

    Args:
        fqn: Fully qualified name of the skeleton

    Returns:
        The skeleton instance

    Raises:
        ValueError: If skeleton not found
    """
    with _registry_lock:
        if fqn not in _SKELETON_REGISTRY:
            raise ValueError(f"Skeleton '{fqn}' not found in registry")
        return _SKELETON_REGISTRY[fqn]


def clear_registry() -> None:
    """Clear all registered skeletons (for testing)."""
    with _registry_lock:
        _SKELETON_REGISTRY.clear()


__all__ = [
    "PipelineStep",
    "PipelineSkeleton",
    "register_skeleton",
    "get_skeleton",
    "clear_registry",
]
