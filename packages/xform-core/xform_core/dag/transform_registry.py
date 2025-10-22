"""Enhanced Transform Registry with type-based indexing for DAG."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Type


@dataclass(slots=True, frozen=True)
class TransformSignature:
    """Type signature of a transform function."""

    input_types: tuple[Type[Any], ...]
    output_type: Type[Any]
    params: dict[str, Any]


class TransformRegistry:
    """Registry for transform functions with type-based indexing.

    This registry extends the base xform_core.transform_registry to support
    DAG dynamic transform selection by enabling type-based queries.
    """

    def __init__(self) -> None:
        self._by_fqn: dict[str, tuple[Callable[..., Any], TransformSignature]] = {}
        self._lock = RLock()

    def register(
        self,
        fqn: str,
        func: Callable[..., Any],
        signature: TransformSignature,
    ) -> None:
        """Register a transform with its type signature.

        Args:
            fqn: Fully qualified name of the transform function
            func: The actual callable function
            signature: Type signature including input/output types and params

        Raises:
            ValueError: If FQN is empty or already registered
        """
        if not fqn:
            raise ValueError("transform FQN must be a non-empty string")
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")

        with self._lock:
            if fqn in self._by_fqn:
                raise ValueError(
                    f"transform '{fqn}' already registered; "
                    "use explicit removal before re-registration"
                )
            self._by_fqn[fqn] = (func, signature)

    def find_transforms(
        self,
        input_types: tuple[Type[Any], ...],
        output_type: Type[Any],
    ) -> list[str]:
        """Find transforms matching the type signature.

        Args:
            input_types: Required input types (tuple of types)
            output_type: Required output type

        Returns:
            List of FQNs matching the signature
        """
        with self._lock:
            matches: list[str] = []
            for fqn, (_, sig) in self._by_fqn.items():
                if self._signature_matches(sig, input_types, output_type):
                    matches.append(fqn)
            return matches

    def get_transform(self, fqn: str) -> Callable[..., Any]:
        """Resolve FQN to actual function.

        Args:
            fqn: Fully qualified name of the transform

        Returns:
            The actual callable function

        Raises:
            KeyError: If FQN not found in registry
        """
        with self._lock:
            if fqn not in self._by_fqn:
                raise KeyError(f"transform not found: {fqn}")
            return self._by_fqn[fqn][0]

    def validate_signature(
        self,
        fqn: str,
        input_types: tuple[Type[Any], ...],
        output_type: Type[Any],
    ) -> bool:
        """Validate that transform signature matches expected types.

        Args:
            fqn: Fully qualified name of the transform
            input_types: Expected input types
            output_type: Expected output type

        Returns:
            True if signature matches, False otherwise
        """
        with self._lock:
            if fqn not in self._by_fqn:
                return False
            _, sig = self._by_fqn[fqn]
            return self._signature_matches(sig, input_types, output_type)

    def has_transform(self, fqn: str) -> bool:
        """Check if transform exists in registry.

        Args:
            fqn: Fully qualified name of the transform

        Returns:
            True if transform is registered, False otherwise
        """
        with self._lock:
            return fqn in self._by_fqn

    def get_signature(self, fqn: str) -> TransformSignature:
        """Get signature for a registered transform.

        Args:
            fqn: Fully qualified name of the transform

        Returns:
            The transform signature

        Raises:
            KeyError: If FQN not found in registry
        """
        with self._lock:
            if fqn not in self._by_fqn:
                raise KeyError(f"transform not found: {fqn}")
            return self._by_fqn[fqn][1]

    def list_all(self) -> list[tuple[str, TransformSignature]]:
        """List all registered transforms with their signatures.

        Returns:
            List of tuples (fqn, signature) for all registered transforms
        """
        with self._lock:
            return [(fqn, sig) for fqn, (_, sig) in self._by_fqn.items()]

    def clear(self) -> None:
        """Clear all registered transforms."""
        with self._lock:
            self._by_fqn.clear()

    @staticmethod
    def _signature_matches(
        sig: TransformSignature,
        input_types: tuple[Type[Any], ...],
        output_type: Type[Any],
    ) -> bool:
        """Check if signature matches required types.

        Uses structural matching:
        - Input types must match exactly (same types in same order)
        - Output type must match exactly
        """
        if len(sig.input_types) != len(input_types):
            return False

        # Check input types match
        for sig_input, expected_input in zip(
            sig.input_types,
            input_types,
            strict=False,
        ):
            if not _types_compatible(sig_input, expected_input):
                return False

        # Check output type matches
        return _types_compatible(sig.output_type, output_type)


def _types_compatible(type_a: Type[Any], type_b: Type[Any]) -> bool:
    """Check if two types are compatible.

    For now, uses simple equality check.
    Future enhancement: handle subclasses, TypedDict structural compatibility, etc.
    """
    # Simple equality check
    if type_a == type_b:
        return True

    # Check if both are the same type name (handles TypeAlias)
    if hasattr(type_a, "__name__") and hasattr(type_b, "__name__"):
        return type_a.__name__ == type_b.__name__

    return False


# Global singleton instance
_registry: TransformRegistry | None = None
_registry_lock = RLock()


def get_registry() -> TransformRegistry:
    """Get the global transform registry instance."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = TransformRegistry()
        return _registry


__all__ = [
    "TransformSignature",
    "TransformRegistry",
    "get_registry",
]
