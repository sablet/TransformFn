"""Test skeleton definitions using pipeline-app transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..test_types import DataFrame, FeatureList, FeatureMap, HLOCVSpec

if TYPE_CHECKING:  # pragma: no cover - typing support only
    from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep


def create_test_pipeline_skeleton() -> PipelineSkeleton:
    """Create a test pipeline skeleton using pipeline-app transforms.

    This skeleton defines a simple 3-step pipeline:
    1. generate_price_bars: HLOCVSpec -> pd.DataFrame
    2. compute_feature_map: pd.DataFrame -> FeatureMap
    3. select_top_features: FeatureMap -> list[str]
    """
    from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep

    return PipelineSkeleton(
        name="test_pipeline",
        steps=[
            PipelineStep(
                name="generate_bars",
                input_types=(HLOCVSpec,),
                output_type=DataFrame,
                default_transform="pipeline_app.transforms.generate_price_bars",
                required=True,
            ),
            PipelineStep(
                name="compute_features",
                input_types=(DataFrame,),
                output_type=FeatureMap,
                default_transform="pipeline_app.transforms.compute_feature_map",
                required=True,
            ),
            PipelineStep(
                name="select_features",
                input_types=(FeatureMap,),
                output_type=FeatureList,
                default_transform="pipeline_app.transforms.select_top_features",
                required=True,
            ),
        ],
    )


def create_test_skeleton_with_optional_steps() -> PipelineSkeleton:
    """Create a test skeleton with optional steps for testing validation."""
    from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep

    return PipelineSkeleton(
        name="test_pipeline_optional",
        steps=[
            PipelineStep(
                name="generate_bars",
                input_types=(HLOCVSpec,),
                output_type=DataFrame,
                default_transform="pipeline_app.transforms.generate_price_bars",
                required=True,
            ),
            PipelineStep(
                name="compute_features",
                input_types=(DataFrame,),
                output_type=FeatureMap,
                default_transform="pipeline_app.transforms.compute_feature_map",
                required=False,  # Optional step
            ),
            PipelineStep(
                name="select_features",
                input_types=(FeatureMap,),
                output_type=FeatureList,
                default_transform="pipeline_app.transforms.select_top_features",
                required=True,
            ),
        ],
    )
