"""Type metadata registration infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from .metadata import Check, ExampleValue
from .registry import register_check, register_example
from .type_registry import register_struct_type

T = TypeVar("T")


@dataclass
class RegisteredType(Generic[T]):
    """Type wrapper that carries example values and check functions.

    This class provides a declarative way to register types with their
    associated examples and checks. It supports method chaining for
    fluent API style. ``type_`` must point to a concrete schema class
    (e.g. TypedDict/dataclass/Enum/Pydantic model) or a repository-defined
    alias that conveys payload structure; registering undeclared third-party
    types (e.g. `pandas.DataFrame`) will emit a warning and later trigger
    TR010 during transform normalization.

    Example:
        RegisteredType(HLOCVSpec) \\
            .with_example(HLOCVSpec(n=32, seed=42), "default_spec") \\
            .with_check(check_hlocv_spec_valid) \\
            .register()

        # Or with list-based initialization
        RegisteredType(
            type_=FeatureMap,
            examples=[make_example({...}, "default")],
            checks=[check_feature_map],
        ).register()
    """

    type_: type[T] | str  # Actual type or FQN for compatibility
    examples: list[ExampleValue[T]] = field(default_factory=list)
    checks: list[Callable[[T], None]] = field(default_factory=list)

    def with_example(self, value: T, description: str = "") -> RegisteredType[T]:
        """Add an example value (chainable).

        Args:
            value: The example value to register
            description: Human-readable description of this example

        Returns:
            Self for method chaining
        """
        self.examples.append(ExampleValue(value, description))
        return self

    def with_check(self, check_func: Callable[[T], None]) -> RegisteredType[T]:
        """Add a check function (chainable).

        Args:
            check_func: Validation function that raises on failure

        Returns:
            Self for method chaining
        """
        self.checks.append(check_func)
        return self

    def register(self) -> None:
        """Register all examples and checks to the global registry."""
        register_struct_type(self.type_)
        key = self._get_type_key()

        # Register examples
        for example in self.examples:
            register_example(key, example)  # type: ignore[arg-type]

        # Register checks
        for check_func in self.checks:
            check_fqn = f"{check_func.__module__}.{check_func.__name__}"
            register_check(key, Check(check_fqn))

    def _get_type_key(self) -> str:
        """Get the fully-qualified name for the type."""
        if isinstance(self.type_, str):
            return self.type_
        return f"{self.type_.__module__}.{self.type_.__qualname__}"

    def __repr__(self) -> str:
        key = self._get_type_key()
        return (
            f"RegisteredType({key}, "
            f"examples={len(self.examples)}, checks={len(self.checks)})"
        )


def make_example(value: T, description: str = "") -> ExampleValue[T]:
    """Helper to create ExampleValue instances.

    Args:
        value: The example value
        description: Human-readable description

    Returns:
        ExampleValue instance
    """
    return ExampleValue(value=value, description=description)


__all__ = ["RegisteredType", "make_example"]
