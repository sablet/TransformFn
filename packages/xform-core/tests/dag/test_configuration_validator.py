"""Tests for Configuration Validator (CRITICAL component)."""

from __future__ import annotations

import pytest

from xform_core.dag.skeleton import PipelineStep, PipelineSkeleton
from xform_core.dag.transform_registry import TransformRegistry, TransformSignature
from xform_core.dag.validator import ConfigurationValidator, ValidationError


# Test types
class InputA:
    pass


class OutputB:
    pass


class OutputC:
    pass


# Test transforms
def transform_a_to_b(input_a: InputA, *, param1: str) -> OutputB:
    """Transform with required parameter."""
    return OutputB()


def transform_a_to_b_optional(input_a: InputA, *, param1: str = "default") -> OutputB:
    """Transform with optional parameter."""
    return OutputB()


def transform_a_to_c(input_a: InputA) -> OutputC:
    """Transform with no parameters."""
    return OutputC()


@pytest.fixture
def registry() -> TransformRegistry:
    """Provide registry with test transforms."""
    reg = TransformRegistry()
    reg.register(
        "test.transform_a_to_b",
        transform_a_to_b,
        TransformSignature(
            input_types=(InputA,),
            output_type=OutputB,
            params={"param1": str},
        ),
    )
    reg.register(
        "test.transform_a_to_b_optional",
        transform_a_to_b_optional,
        TransformSignature(
            input_types=(InputA,),
            output_type=OutputB,
            params={"param1": "default"},
        ),
    )
    reg.register(
        "test.transform_a_to_c",
        transform_a_to_c,
        TransformSignature(
            input_types=(InputA,),
            output_type=OutputC,
            params={},
        ),
    )
    return reg


@pytest.fixture
def skeleton() -> PipelineSkeleton:
    """Provide test skeleton."""
    return PipelineSkeleton(
        name="test_phase",
        steps=[
            PipelineStep(
                name="step1",
                input_types=(InputA,),
                output_type=OutputB,
                default_transform="test.transform_a_to_b_optional",
                required=True,
            ),
            PipelineStep(
                name="step2",
                input_types=(
                    InputA,
                ),  # Changed from OutputB to InputA to match transform_a_to_c
                output_type=OutputC,
                default_transform=None,
                required=True,
            ),
        ],
    )


def test_CV_N_01_complete_valid_configuration(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-N-01: Complete valid configuration."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b_optional",
                "params": {"param1": "custom"},
            },
            "step2": {
                "transform": "test.transform_a_to_c",
                "params": {},
            },
        }
    }

    result = validator.validate(config)
    assert result.is_valid
    assert len(result.errors) == 0
    assert len(result.warnings) == 0


def test_CV_N_02_config_using_default_transforms(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-N-02: Config using default transforms."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            # step1 uses default
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert result.is_valid


def test_CV_N_03_config_with_all_optional_parameters(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-N-03: Config with all optional parameters."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b_optional",
                "params": {},  # No params, all optional
            },
            "step2": {
                "transform": "test.transform_a_to_c",
                "params": {},
            },
        }
    }

    result = validator.validate(config)
    assert result.is_valid


def test_CV_N_04_config_with_correct_type_annotations(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-N-04: Config with correct type annotations."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b_optional",
                "params": {"param1": "string_value"},  # Correct type
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert result.is_valid


def test_CV_E_01_missing_required_step(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-01: MISSING_REQUIRED_STEP."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {"transform": "test.transform_a_to_b_optional"},
            # step2 missing and has no default
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "MISSING_REQUIRED_STEP"
    assert "step2" in result.errors[0].message


def test_CV_E_02_no_transform_specified(registry: TransformRegistry) -> None:
    """CV-E-02: NO_TRANSFORM_SPECIFIED."""
    skeleton = PipelineSkeleton(
        name="test_phase",
        steps=[
            PipelineStep(
                name="step1",
                input_types=(InputA,),
                output_type=OutputB,
                default_transform=None,  # No default
                required=True,
            ),
        ],
    )
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                # No transform specified and no default
                "params": {},
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "NO_TRANSFORM_SPECIFIED"


def test_CV_E_03_transform_not_found(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-03: TRANSFORM_NOT_FOUND."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.nonexistent_transform",
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert any(e.error_type == "TRANSFORM_NOT_FOUND" for e in result.errors)
    error = next(e for e in result.errors if e.error_type == "TRANSFORM_NOT_FOUND")
    assert "nonexistent_transform" in error.message


def test_CV_E_04_type_signature_mismatch_input(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-04: TYPE_SIGNATURE_MISMATCH (input)."""
    # Register transform with wrong input type
    registry.register(
        "test.wrong_input",
        lambda x: OutputB(),
        TransformSignature(
            input_types=(OutputC,),  # Wrong input type
            output_type=OutputB,
            params={},
        ),
    )

    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.wrong_input",  # Wrong signature
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert any(e.error_type == "TYPE_SIGNATURE_MISMATCH" for e in result.errors)


def test_CV_E_05_type_signature_mismatch_output(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-05: TYPE_SIGNATURE_MISMATCH (output)."""
    # Register transform with wrong output type
    registry.register(
        "test.wrong_output",
        lambda x: OutputC(),
        TransformSignature(
            input_types=(InputA,),
            output_type=OutputC,  # Wrong output type for step1
            params={},
        ),
    )

    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.wrong_output",  # Wrong output type
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert any(e.error_type == "TYPE_SIGNATURE_MISMATCH" for e in result.errors)


def test_CV_E_06_unknown_parameter(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-06: UNKNOWN_PARAMETER."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b_optional",
                "params": {
                    "param1": "valid",
                    "unknown_param": "invalid",  # Unknown parameter
                },
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert any(e.error_type == "UNKNOWN_PARAMETER" for e in result.errors)
    error = next(e for e in result.errors if e.error_type == "UNKNOWN_PARAMETER")
    assert "unknown_param" in error.message


def test_CV_E_07_missing_required_parameter(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-07: MISSING_REQUIRED_PARAMETER."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b",  # Has required param1
                "params": {},  # Missing param1
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert any(e.error_type == "MISSING_REQUIRED_PARAMETER" for e in result.errors)
    error = next(
        e for e in result.errors if e.error_type == "MISSING_REQUIRED_PARAMETER"
    )
    assert "param1" in error.message


def test_CV_E_08_parameter_type_mismatch(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-E-08: PARAMETER_TYPE_MISMATCH."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b_optional",
                "params": {
                    "param1": 123,  # Wrong type (int instead of str)
                },
            },
            "step2": {
                "transform": "test.transform_a_to_c",
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert any(e.error_type == "PARAMETER_TYPE_MISMATCH" for e in result.errors)
    error = next(e for e in result.errors if e.error_type == "PARAMETER_TYPE_MISMATCH")
    assert "param1" in error.message


def test_CV_W_01_unknown_step(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-W-01: UNKNOWN_STEP (warning)."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {"transform": "test.transform_a_to_b_optional"},
            "step2": {"transform": "test.transform_a_to_c"},
            "unknown_step": {"transform": "test.transform_a_to_c"},  # Unknown step
        }
    }

    result = validator.validate(config)
    assert result.is_valid  # Valid despite warning
    assert len(result.warnings) == 1
    assert result.warnings[0].error_type == "UNKNOWN_STEP"


def test_CV_M_01_multiple_errors(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-M-01: Multiple errors (2+ types)."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.nonexistent",  # TRANSFORM_NOT_FOUND
                "params": {
                    "unknown": "value"
                },  # UNKNOWN_PARAMETER (won't be checked due to first error)
            },
            # step2 missing - MISSING_REQUIRED_STEP
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert len(result.errors) >= 2
    error_types = {e.error_type for e in result.errors}
    assert "TRANSFORM_NOT_FOUND" in error_types
    assert "MISSING_REQUIRED_STEP" in error_types


def test_CV_M_02_errors_and_warnings(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-M-02: Errors and warnings."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.nonexistent",  # ERROR
            },
            "step2": {"transform": "test.transform_a_to_c"},
            "unknown_step": {"transform": "test.transform_a_to_c"},  # WARNING
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert len(result.errors) > 0
    assert len(result.warnings) > 0


def test_CV_S_01_transform_not_found_includes_suggestions(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-S-01: TRANSFORM_NOT_FOUND includes suggestions."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {"transform": "test.nonexistent"},
            "step2": {"transform": "test.transform_a_to_c"},
        }
    }

    result = validator.validate(config)
    error = next(e for e in result.errors if e.error_type == "TRANSFORM_NOT_FOUND")
    assert error.suggestion is not None
    assert (
        "Available transforms" in error.suggestion
        or "No compatible" in error.suggestion
    )


def test_CV_S_02_unknown_parameter_includes_valid_params(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-S-02: UNKNOWN_PARAMETER includes valid params."""
    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {
                "transform": "test.transform_a_to_b_optional",
                "params": {"unknown": "value"},
            },
            "step2": {"transform": "test.transform_a_to_c"},
        }
    }

    result = validator.validate(config)
    error = next(e for e in result.errors if e.error_type == "UNKNOWN_PARAMETER")
    assert error.suggestion is not None
    assert "Valid parameters" in error.suggestion


def test_CV_S_03_type_signature_mismatch_includes_alternatives(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """CV-S-03: TYPE_SIGNATURE_MISMATCH includes alternatives."""
    registry.register(
        "test.wrong_type",
        lambda x: OutputC(),
        TransformSignature(
            input_types=(InputA,),
            output_type=OutputC,
            params={},
        ),
    )

    validator = ConfigurationValidator(registry, skeleton)

    config = {
        "steps": {
            "step1": {"transform": "test.wrong_type"},
            "step2": {"transform": "test.transform_a_to_c"},
        }
    }

    result = validator.validate(config)
    error = next(e for e in result.errors if e.error_type == "TYPE_SIGNATURE_MISMATCH")
    assert error.suggestion is not None
    assert (
        "Available transforms" in error.suggestion
        or "No compatible" in error.suggestion
    )


# ========================================
# Multi-Input Validation Tests
# ========================================


class FeaturesType:
    """Test class representing features dataset."""

    pass


class TargetType:
    """Test class representing target dataset."""

    pass


class MetricsType:
    """Test class representing computed metrics."""

    pass


def transform_multi_input(features: FeaturesType, target: TargetType) -> MetricsType:
    """Transform that takes two data inputs (features + target)."""
    return MetricsType()


def transform_multi_input_with_params(
    features: FeaturesType, target: TargetType, *, weight: float = 1.0
) -> MetricsType:
    """Transform that takes two data inputs + parameter."""
    return MetricsType()


def transform_wrong_input_count(features: FeaturesType) -> MetricsType:
    """Transform with only one input (should mismatch)."""
    return MetricsType()


@pytest.fixture
def multi_input_registry() -> TransformRegistry:
    """Registry with multi-input transforms."""
    reg = TransformRegistry()
    reg.register(
        "test.multi_input",
        transform_multi_input,
        TransformSignature(
            input_types=(FeaturesType, TargetType),
            output_type=MetricsType,
            params={},
        ),
    )
    reg.register(
        "test.multi_input_with_params",
        transform_multi_input_with_params,
        TransformSignature(
            input_types=(FeaturesType, TargetType),
            output_type=MetricsType,
            params={"weight": 1.0},
        ),
    )
    reg.register(
        "test.wrong_input_count",
        transform_wrong_input_count,
        TransformSignature(
            input_types=(FeaturesType,),
            output_type=MetricsType,
            params={},
        ),
    )
    return reg


@pytest.fixture
def multi_input_skeleton() -> PipelineSkeleton:
    """Skeleton with multi-input step."""
    return PipelineSkeleton(
        name="multi_input_phase",
        steps=[
            PipelineStep(
                name="compute_metrics",
                input_types=(FeaturesType, TargetType),
                output_type=MetricsType,
                default_transform="test.multi_input",
                required=True,
            ),
        ],
    )


def test_CV_N_06_validate_multi_input_transform(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """CV-N-06: Validate transform with multiple inputs (features + target).

    This tests validation of transforms that require multiple datasets,
    such as ML training functions that need both features and target.
    """
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)

    config = {
        "steps": {
            "compute_metrics": {"transform": "test.multi_input"},
        }
    }

    result = validator.validate(config)
    assert result.is_valid
    assert len(result.errors) == 0


def test_CV_N_07_validate_multi_input_transform_with_params(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """CV-N-07: Validate multi-input transform with parameters."""
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)

    config = {
        "steps": {
            "compute_metrics": {
                "transform": "test.multi_input_with_params",
                "params": {"weight": 2.0},
            },
        }
    }

    result = validator.validate(config)
    assert result.is_valid
    assert len(result.errors) == 0


def test_CV_E_06_validate_multi_input_type_mismatch(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """CV-E-06: Detect type signature mismatch in multi-input transform.

    This tests validation error when transform expects different number of inputs
    than skeleton specifies (e.g., transform expects 1 input but skeleton expects 2).
    """
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)

    config = {
        "steps": {
            "compute_metrics": {
                "transform": "test.wrong_input_count",  # Only takes 1 input
            },
        }
    }

    result = validator.validate(config)
    assert not result.is_valid
    assert len(result.errors) > 0

    # Should have TYPE_SIGNATURE_MISMATCH error
    error_types = [e.error_type for e in result.errors]
    assert "TYPE_SIGNATURE_MISMATCH" in error_types
