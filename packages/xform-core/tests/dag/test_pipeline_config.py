"""Tests for PipelineConfig - Configuration loading with validation."""

from pathlib import Path

import pytest
import yaml

from xform_core.dag.config import PipelineConfig
from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep


def test_CFG_N_01_load_valid_yaml_auto_validate_get_phase_config(
    tmp_path: Path, test_skeleton_registered: tuple
) -> None:
    """CFG-N-01: Load valid YAML → auto-validate → get phase config.

    Expected:
    - YAML loads successfully
    - Auto-validation passes
    - get_phase_config returns correct data
    """
    fqn, skeleton = test_skeleton_registered

    # Create valid config
    config_path = tmp_path / "valid_config.yaml"
    config_data = {
        "pipeline": {
            "name": "test_pipeline",
            "skeleton": fqn,
        },
        "test_pipeline": {
            "steps": {
                "step1": {
                    "transform": "pipeline_app.transforms.compute_feature_map",
                    "params": {},
                },
            },
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Load config - should auto-validate
    config = PipelineConfig(str(config_path))

    # Verify validation passed
    assert config.validation_result.is_valid, "Config should be valid"
    assert len(config.validation_result.errors) == 0, "Should have no errors"

    # Get phase config
    phase_config = config.get_phase_config("test_pipeline")
    assert phase_config is not None, "Phase config should exist"
    assert "steps" in phase_config, "Phase config should have steps"
    assert "step1" in phase_config["steps"], "Should have step1"


def test_CFG_E_01_load_config_with_validation_errors_fail_fast(
    tmp_path: Path, test_skeleton_registered: tuple
) -> None:
    """CFG-E-01: Load config with validation errors (Fail Fast).

    Expected:
    - ValueError raised immediately on load
    - Error message contains validation details
    """
    fqn, skeleton = test_skeleton_registered

    # Create invalid config (non-existent transform FQN)
    config_path = tmp_path / "invalid_config.yaml"
    config_data = {
        "pipeline": {
            "name": "test_pipeline",
            "skeleton": fqn,
        },
        "test_pipeline": {
            "steps": {
                "generate_bars": {
                    "transform": "nonexistent.transform.function",  # Invalid FQN
                    "params": {},
                },
            },
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Load config - should fail fast
    with pytest.raises(ValueError) as exc_info:
        PipelineConfig(str(config_path))

    # Verify error message
    error_message = str(exc_info.value)
    assert "Configuration validation failed" in error_message, (
        "Should mention validation failure"
    )


def test_CFG_E_02_load_nonexistent_config_file(tmp_path: Path) -> None:
    """CFG-E-02: Load non-existent config file.

    Expected:
    - FileNotFoundError or similar
    """
    nonexistent_path = tmp_path / "nonexistent.yaml"

    with pytest.raises((FileNotFoundError, IOError)):
        PipelineConfig(str(nonexistent_path))


def test_CFG_N_02_config_missing_skeleton_field() -> None:
    """CFG-N-02: Config missing pipeline.skeleton field.

    Expected:
    - ValueError raised with clear message
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"pipeline": {"name": "test"}}, f)
        config_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            PipelineConfig(config_path)

        assert "skeleton" in str(exc_info.value).lower(), (
            "Error should mention missing skeleton"
        )
    finally:
        Path(config_path).unlink()


def test_CFG_N_03_config_with_nonexistent_skeleton() -> None:
    """CFG-N-03: Config references non-existent skeleton.

    Expected:
    - ValueError raised when getting skeleton
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "pipeline": {
                    "name": "test",
                    "skeleton": "nonexistent_skeleton_fqn",
                }
            },
            f,
        )
        config_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            PipelineConfig(config_path)

        assert "not found" in str(exc_info.value).lower(), (
            "Error should mention skeleton not found"
        )
    finally:
        Path(config_path).unlink()
