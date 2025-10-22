"""Pytest fixtures for DAG tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture(scope="session", autouse=True)
def import_pipeline_app() -> None:
    """Auto-import pipeline-app to register transforms (session scope)."""
    # Add pipeline-app and algo-trade to sys.path
    apps_dir = Path(__file__).parent.parent.parent.parent.parent / "apps"
    pipeline_app_path = apps_dir / "pipeline-app"
    algo_trade_path = apps_dir / "algo-trade"

    if pipeline_app_path.exists():
        sys.path.insert(0, str(pipeline_app_path))
    if algo_trade_path.exists():
        sys.path.insert(0, str(algo_trade_path))

    # Import to trigger @transform registration
    try:
        import pipeline_app.transforms  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Could not import pipeline_app.transforms: {e}")


@pytest.fixture(scope="session")
def test_registry():
    """Provide a registry populated with pipeline-app transforms."""
    from xform_core.dag.transform_registry import get_registry

    registry = get_registry()
    # Registry is already populated by auto-import
    return registry


@pytest.fixture
def test_skeleton_registered():
    """Register test skeleton and return its FQN."""
    from xform_core.dag.skeleton import (
        register_skeleton,
        PipelineSkeleton,
        PipelineStep,
    )
    from xform_core.dag.transform_registry import get_registry
    import pandas as pd

    # Create simple test skeleton programmatically
    # Use only pipeline_dtype types to avoid cross-module dependencies
    from pipeline_dtype import HLOCVSpec, FeatureMap

    skeleton = PipelineSkeleton(
        name="test_pipeline",
        steps=[
            PipelineStep(
                name="generate_bars",
                input_types=(HLOCVSpec,),
                output_type=pd.DataFrame,
                default_transform="pipeline_app.transforms.generate_price_bars",
            ),
            PipelineStep(
                name="compute_features",
                input_types=(pd.DataFrame,),
                output_type=FeatureMap,
                default_transform="pipeline_app.transforms.compute_feature_map",
            ),
        ],
    )

    fqn = "test_skeleton"
    try:
        register_skeleton(fqn, skeleton)
    except ValueError:
        # Already registered
        pass
    return fqn, skeleton


@pytest.fixture
def generated_config_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for generated config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def valid_config_path(
    generated_config_dir: Path, test_skeleton_registered: tuple
) -> Path:
    """Create a valid config file for testing.

    This config will be generated programmatically rather than using CLI
    to avoid circular dependencies during initial implementation.
    """
    fqn, skeleton = test_skeleton_registered

    # Generate valid config programmatically
    config = {
        "pipeline": {
            "name": "test_pipeline",
            "version": "1.0",
            "skeleton": fqn,
        },
        "test_pipeline": {  # Key must match skeleton.name
            "steps": {
                "generate_bars": {
                    "transform": "pipeline_app.transforms.generate_price_bars",
                    "params": {},
                },
                "compute_features": {
                    "transform": "pipeline_app.transforms.compute_feature_map",
                    "params": {
                        "annualization_factor": 252.0,
                    },
                },
            },
        },
    }

    config_path = generated_config_dir / "valid_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def invalid_config_missing_step(
    generated_config_dir: Path, test_skeleton_registered: tuple
) -> Path:
    """Create config with missing required step."""
    fqn, _ = test_skeleton_registered

    config = {
        "pipeline": {
            "name": "test_pipeline_invalid",
            "version": "1.0",
            "skeleton": fqn,
        },
        "test_pipeline": {
            "steps": {
                "generate_bars": {
                    "transform": "pipeline_app.transforms.generate_price_bars",
                    "params": {},
                },
                # Missing: compute_features (required step)
            },
        },
    }

    config_path = generated_config_dir / "invalid_missing_step.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def invalid_config_wrong_transform(
    generated_config_dir: Path, test_skeleton_registered: tuple
) -> Path:
    """Create config with non-existent transform FQN."""
    fqn, _ = test_skeleton_registered

    config = {
        "pipeline": {
            "name": "test_pipeline_invalid",
            "version": "1.0",
            "skeleton": fqn,
        },
        "test_pipeline": {
            "steps": {
                "generate_bars": {
                    "transform": "pipeline_app.transforms.non_existent_transform",  # Wrong FQN
                    "params": {},
                },
                "compute_features": {
                    "transform": "pipeline_app.transforms.compute_feature_map",
                    "params": {},
                },
            },
        },
    }

    config_path = generated_config_dir / "invalid_wrong_transform.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def invalid_config_wrong_params(
    generated_config_dir: Path, test_skeleton_registered: tuple
) -> Path:
    """Create config with invalid parameters."""
    fqn, _ = test_skeleton_registered

    config = {
        "pipeline": {
            "name": "test_pipeline_invalid",
            "version": "1.0",
            "skeleton": fqn,
        },
        "test_pipeline": {
            "steps": {
                "generate_bars": {
                    "transform": "pipeline_app.transforms.generate_price_bars",
                    "params": {},
                },
                "compute_features": {
                    "transform": "pipeline_app.transforms.compute_feature_map",
                    "params": {
                        "unknown_param": "invalid",  # Invalid parameter
                    },
                },
            },
        },
    }

    config_path = generated_config_dir / "invalid_wrong_params.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path
