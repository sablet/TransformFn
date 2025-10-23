"""Pytest fixtures for DAG tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Annotated

import pytest
import yaml

from xform_core import Check, ExampleValue, RegisteredType, transform
from xform_core.type_registry import TypeRegistryError, is_registered_schema

from .test_types import DataFrame as TestDataFrame
from .test_types import FeatureList as TestFeatureList
from .test_types import FeatureMap as TestFeatureMap
from .test_types import HLOCVSpec as TestHLOCVSpec


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

    try:
        import pipeline_app.transforms  # noqa: F401
    except (ImportError, TypeRegistryError):
        _setup_test_pipeline_modules()
    except Exception as exc:  # pragma: no cover - defensive logging
        underlying = exc.__cause__
        if isinstance(exc, TypeRegistryError) or isinstance(underlying, TypeRegistryError):
            _setup_test_pipeline_modules()
        else:
            raise


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
    # Use shared test types to avoid cross-module dependencies
    from .test_types import HLOCVSpec, FeatureMap

    skeleton = PipelineSkeleton(
        name="test_pipeline",
        steps=[
            PipelineStep(
                name="generate_bars",
                input_types=(HLOCVSpec,),
                output_type=TestDataFrame,
                default_transform="pipeline_app.transforms.generate_price_bars",
            ),
            PipelineStep(
                name="compute_features",
                input_types=(TestDataFrame,),
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


def check_test_dataframe(frame: TestDataFrame) -> None:
    if not frame.rows:
        raise ValueError("frame must contain at least one row")


def check_test_feature_map(feature_map: TestFeatureMap) -> None:
    if not feature_map.metrics:
        raise ValueError("feature map must contain metrics")


def check_test_feature_list(feature_list: TestFeatureList) -> None:
    if not feature_list.names:
        raise ValueError("feature list must include at least one entry")


def _setup_test_pipeline_modules() -> None:
    """Install lightweight test pipeline modules compliant with TR010."""

    # Remove partially imported modules, if any
    for name in (
        "pipeline_app",
        "pipeline_app.transforms",
        "pipeline_app.transforms_test",
    ):
        sys.modules.pop(name, None)

    pipeline_app_module = ModuleType("pipeline_app")
    transforms_module = ModuleType("pipeline_app.transforms")
    pipeline_app_module.transforms = transforms_module

    sys.modules["pipeline_app"] = pipeline_app_module
    sys.modules["pipeline_app.transforms"] = transforms_module

    # Register schema types if not already present
    if not is_registered_schema(TestHLOCVSpec):
        RegisteredType(TestHLOCVSpec).with_example(
            TestHLOCVSpec(length=8), "default_spec"
        ).register()

    if not is_registered_schema(TestDataFrame):
        RegisteredType(TestDataFrame).with_example(
            TestDataFrame(rows=(1.0, 2.0, 3.0)), "sample_frame"
        ).with_check(check_test_dataframe).register()

    if not is_registered_schema(TestFeatureMap):
        RegisteredType(TestFeatureMap).with_example(
            TestFeatureMap(metrics={"mean": 1.0, "vol": 0.5}), "sample_feature_map"
        ).with_check(check_test_feature_map).register()

    if not is_registered_schema(TestFeatureList):
        RegisteredType(TestFeatureList).with_example(
            TestFeatureList(names=("feature_1", "feature_2")), "feature_list"
        ).with_check(check_test_feature_list).register()

    # Ensure check functions resolve via pipeline_app.transforms.* FQN
    for check in (
        check_test_dataframe,
        check_test_feature_map,
        check_test_feature_list,
    ):
        check.__module__ = "pipeline_app.transforms"
        check.__qualname__ = check.__name__
        setattr(transforms_module, check.__name__, check)

    def generate_price_bars_impl(
        spec: Annotated[
            TestHLOCVSpec,
            ExampleValue(TestHLOCVSpec(length=4), "default_spec"),
        ]
    ) -> Annotated[
        TestDataFrame,
        Check("pipeline_app.transforms.check_test_dataframe"),
    ]:
        """Generate deterministic price bars for DAG tests."""

        values = tuple(float(idx + 1) for idx in range(spec.length))
        return TestDataFrame(rows=values)

    generate_price_bars_impl.__module__ = "pipeline_app.transforms"
    generate_price_bars_impl.__qualname__ = "generate_price_bars"
    generate_price_bars_impl.__name__ = "generate_price_bars"
    generate_price_bars = transform(generate_price_bars_impl)
    generate_price_bars.__module__ = "pipeline_app.transforms"
    generate_price_bars.__qualname__ = "generate_price_bars"
    transforms_module.generate_price_bars = generate_price_bars

    def compute_feature_map_impl(
        frame: Annotated[
            TestDataFrame,
            ExampleValue(TestDataFrame(rows=(1.0, 2.0, 3.0)), "frame_example"),
        ],
        *,
        annualization_factor: float = 252.0,
    ) -> Annotated[
        TestFeatureMap,
        Check("pipeline_app.transforms.check_test_feature_map"),
    ]:
        """Aggregate frame rows into a basic feature map."""

        mean = sum(frame.rows) / len(frame.rows)
        vol = (max(frame.rows) - min(frame.rows)) / max(annualization_factor, 1.0)
        return TestFeatureMap(metrics={"mean": mean, "vol": vol})

    compute_feature_map_impl.__module__ = "pipeline_app.transforms"
    compute_feature_map_impl.__qualname__ = "compute_feature_map"
    compute_feature_map_impl.__name__ = "compute_feature_map"
    compute_feature_map = transform(compute_feature_map_impl)
    compute_feature_map.__module__ = "pipeline_app.transforms"
    compute_feature_map.__qualname__ = "compute_feature_map"
    transforms_module.compute_feature_map = compute_feature_map

    def select_top_features_impl(
        feature_map: Annotated[
            TestFeatureMap,
            ExampleValue(
                TestFeatureMap(metrics={"alpha": 0.9}), "feature_map_example"
            ),
        ]
    ) -> Annotated[
        TestFeatureList,
        Check("pipeline_app.transforms.check_test_feature_list"),
    ]:
        """Return deterministic feature ranking."""

        sorted_features = tuple(sorted(feature_map.metrics.keys()))
        return TestFeatureList(names=sorted_features or ("default_feature",))

    select_top_features_impl.__module__ = "pipeline_app.transforms"
    select_top_features_impl.__qualname__ = "select_top_features"
    select_top_features_impl.__name__ = "select_top_features"
    select_top_features = transform(select_top_features_impl)
    select_top_features.__module__ = "pipeline_app.transforms"
    select_top_features.__qualname__ = "select_top_features"
    transforms_module.select_top_features = select_top_features



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
