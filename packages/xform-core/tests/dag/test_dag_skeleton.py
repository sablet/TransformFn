"""Tests for DAG Skeleton Definition."""

from __future__ import annotations

import pytest

from xform_core.dag.skeleton import (
    PipelineStep,
    PipelineSkeleton,
    register_skeleton,
    get_skeleton,
    clear_registry,
)


# Simple test types
class TestInputType:
    """Test input type."""

    pass


class TestOutputType:
    """Test output type."""

    pass


@pytest.fixture(autouse=True)
def clean_skeleton_registry():
    """Clear skeleton registry before each test."""
    clear_registry()
    yield
    clear_registry()


def test_SK_N_01_create_register_get_validate_integration() -> None:
    """SK-N-01: Create → Register → Get → Validate config (integrated normal case)."""
    # Create skeleton
    skeleton = PipelineSkeleton(
        name="test_pipeline",
        steps=[
            PipelineStep(
                name="step1",
                input_types=(TestInputType,),
                output_type=TestOutputType,
                default_transform="test.transform1",
                required=True,
            ),
            PipelineStep(
                name="step2",
                input_types=(TestOutputType,),
                output_type=TestOutputType,
                default_transform="test.transform2",
                required=True,
            ),
        ],
    )

    # Register skeleton
    fqn = "test.skeleton.test_pipeline"
    register_skeleton(fqn, skeleton)

    # Get skeleton
    retrieved = get_skeleton(fqn)
    assert retrieved is skeleton
    assert retrieved.name == "test_pipeline"
    assert len(retrieved.steps) == 2

    # Validate config - complete config
    valid_config = {
        "steps": {
            "step1": {"transform": "test.transform1"},
            "step2": {"transform": "test.transform2"},
        }
    }
    assert skeleton.validate_config(valid_config)

    # Validate config - partial config (OK if default transform exists)
    partial_config = {
        "steps": {
            "step1": {"transform": "test.transform1"},
            # step2 missing but has default
        }
    }
    assert skeleton.validate_config(partial_config)

    # Validate config - missing required step without default
    skeleton_no_default = PipelineSkeleton(
        name="test_pipeline_no_default",
        steps=[
            PipelineStep(
                name="step1",
                input_types=(TestInputType,),
                output_type=TestOutputType,
                default_transform=None,  # No default
                required=True,
            ),
        ],
    )
    missing_config = {"steps": {}}  # Empty config
    assert not skeleton_no_default.validate_config(missing_config)


def test_SK_E_01_get_skeleton_with_nonexistent_fqn() -> None:
    """SK-E-01: Get skeleton with non-existent FQN."""
    with pytest.raises(ValueError, match="not found in registry"):
        get_skeleton("nonexistent.skeleton")


def test_skeleton_duplicate_registration() -> None:
    """Test that duplicate registration raises error."""
    skeleton = PipelineSkeleton(
        name="test_pipeline",
        steps=[
            PipelineStep(
                name="step1",
                input_types=(TestInputType,),
                output_type=TestOutputType,
                required=True,
            ),
        ],
    )

    fqn = "test.skeleton.duplicate"
    register_skeleton(fqn, skeleton)

    # Second registration should fail
    with pytest.raises(ValueError, match="already registered"):
        register_skeleton(fqn, skeleton)


def test_skeleton_optional_steps() -> None:
    """Test skeleton with optional steps."""
    skeleton = PipelineSkeleton(
        name="test_pipeline_optional",
        steps=[
            PipelineStep(
                name="required_step",
                input_types=(TestInputType,),
                output_type=TestOutputType,
                required=True,
            ),
            PipelineStep(
                name="optional_step",
                input_types=(TestOutputType,),
                output_type=TestOutputType,
                required=False,
            ),
        ],
    )

    # Config without optional step should be valid
    config_without_optional = {
        "steps": {
            "required_step": {"transform": "test.transform"},
            # optional_step missing
        }
    }
    assert skeleton.validate_config(config_without_optional)

    # Config with optional step should also be valid
    config_with_optional = {
        "steps": {
            "required_step": {"transform": "test.transform"},
            "optional_step": {"transform": "test.transform2"},
        }
    }
    assert skeleton.validate_config(config_with_optional)
