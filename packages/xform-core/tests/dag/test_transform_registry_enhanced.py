"""Tests for enhanced DAG transform registry."""

from __future__ import annotations

import pytest
from typing import Any

from xform_core.dag.transform_registry import (
    TransformSignature,
    TransformRegistry,
)


# Simple test types
class InputTypeA:
    """Test input type A."""

    pass


class InputTypeB:
    """Test input type B."""

    pass


class OutputTypeX:
    """Test output type X."""

    pass


class OutputTypeY:
    """Test output type Y."""

    pass


# Test transform functions
def transform_a_to_x(input_a: InputTypeA) -> OutputTypeX:
    """Transform A to X."""
    return OutputTypeX()


def transform_b_to_y(input_b: InputTypeB, *, param1: str = "default") -> OutputTypeY:
    """Transform B to Y with parameter."""
    return OutputTypeY()


def transform_a_to_y(input_a: InputTypeA) -> OutputTypeY:
    """Transform A to Y (alternative)."""
    return OutputTypeY()


@pytest.fixture
def registry() -> TransformRegistry:
    """Provide a fresh registry for each test."""
    reg = TransformRegistry()
    return reg


def test_TR_N_01_register_find_get_validate_integration(
    registry: TransformRegistry,
) -> None:
    """TR-N-01: Register → Find → Get → Validate (integrated normal case)."""
    # Register transform
    fqn = "test.transform_a_to_x"
    sig = TransformSignature(
        input_types=(InputTypeA,),
        output_type=OutputTypeX,
        params={},
    )
    registry.register(fqn, transform_a_to_x, sig)

    # Find by type signature
    matches = registry.find_transforms(
        input_types=(InputTypeA,),
        output_type=OutputTypeX,
    )
    assert fqn in matches
    assert len(matches) == 1

    # Get actual function
    func = registry.get_transform(fqn)
    assert func is transform_a_to_x

    # Validate signature
    assert registry.validate_signature(fqn, (InputTypeA,), OutputTypeX)
    assert not registry.validate_signature(fqn, (InputTypeB,), OutputTypeX)
    assert not registry.validate_signature(fqn, (InputTypeA,), OutputTypeY)

    # Check existence
    assert registry.has_transform(fqn)
    assert not registry.has_transform("nonexistent.transform")


def test_TR_E_01_get_with_nonexistent_fqn(registry: TransformRegistry) -> None:
    """TR-E-01: Get with non-existent FQN."""
    with pytest.raises(KeyError, match="transform not found"):
        registry.get_transform("nonexistent.transform")


def test_TR_E_02_validate_signature_with_type_mismatch(
    registry: TransformRegistry,
) -> None:
    """TR-E-02: Validate signature with type mismatch."""
    fqn = "test.transform_a_to_x"
    sig = TransformSignature(
        input_types=(InputTypeA,),
        output_type=OutputTypeX,
        params={},
    )
    registry.register(fqn, transform_a_to_x, sig)

    # Type mismatch should return False (not raise exception)
    assert not registry.validate_signature(fqn, (InputTypeB,), OutputTypeX)
    assert not registry.validate_signature(fqn, (InputTypeA,), OutputTypeY)
    assert not registry.validate_signature(fqn, (InputTypeA, InputTypeB), OutputTypeX)


def test_registry_find_multiple_matches(registry: TransformRegistry) -> None:
    """Test finding multiple transforms with same signature."""
    # Register two transforms with different implementations but same signature
    sig1 = TransformSignature(
        input_types=(InputTypeA,),
        output_type=OutputTypeY,
        params={},
    )
    sig2 = TransformSignature(
        input_types=(InputTypeA,),
        output_type=OutputTypeY,
        params={"param1": "default"},
    )

    registry.register("test.transform_a_to_y_v1", transform_a_to_y, sig1)
    registry.register("test.transform_a_to_y_v2", transform_a_to_y, sig2)

    # Both should match
    matches = registry.find_transforms(
        input_types=(InputTypeA,),
        output_type=OutputTypeY,
    )
    assert len(matches) == 2
    assert "test.transform_a_to_y_v1" in matches
    assert "test.transform_a_to_y_v2" in matches


def test_registry_duplicate_registration(registry: TransformRegistry) -> None:
    """Test that duplicate registration raises error."""
    fqn = "test.transform_a_to_x"
    sig = TransformSignature(
        input_types=(InputTypeA,),
        output_type=OutputTypeX,
        params={},
    )

    registry.register(fqn, transform_a_to_x, sig)

    # Second registration should fail
    with pytest.raises(ValueError, match="already registered"):
        registry.register(fqn, transform_a_to_x, sig)


def test_registry_get_signature(registry: TransformRegistry) -> None:
    """Test retrieving signature from registry."""
    fqn = "test.transform_b_to_y"
    sig = TransformSignature(
        input_types=(InputTypeB,),
        output_type=OutputTypeY,
        params={"param1": "default"},
    )
    registry.register(fqn, transform_b_to_y, sig)

    retrieved_sig = registry.get_signature(fqn)
    assert retrieved_sig.input_types == (InputTypeB,)
    assert retrieved_sig.output_type == OutputTypeY
    assert retrieved_sig.params == {"param1": "default"}


def test_registry_get_signature_nonexistent(registry: TransformRegistry) -> None:
    """Test get_signature with non-existent FQN."""
    with pytest.raises(KeyError, match="transform not found"):
        registry.get_signature("nonexistent.transform")


def test_registry_clear(registry: TransformRegistry) -> None:
    """Test clearing the registry."""
    fqn = "test.transform_a_to_x"
    sig = TransformSignature(
        input_types=(InputTypeA,),
        output_type=OutputTypeX,
        params={},
    )
    registry.register(fqn, transform_a_to_x, sig)

    assert registry.has_transform(fqn)

    registry.clear()

    assert not registry.has_transform(fqn)
    with pytest.raises(KeyError):
        registry.get_transform(fqn)
