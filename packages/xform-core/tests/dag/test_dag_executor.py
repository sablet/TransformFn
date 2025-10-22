"""Tests for DAG Executor and Resolver."""

from __future__ import annotations

import pytest

from xform_core.dag.executor import DAGExecutor
from xform_core.dag.resolver import TransformResolver
from xform_core.dag.skeleton import PipelineStep, PipelineSkeleton
from xform_core.dag.transform_registry import TransformRegistry, TransformSignature
from xform_core.dag.validator import ConfigurationValidator
from xform_core.models import (
    CodeRef,
    ParamField,
    ParamSchema,
    Schema,
    TransformFn,
)


class InputData:
    def __init__(self, value: int):
        self.value = value


class IntermediateData:
    def __init__(self, value: int):
        self.value = value


class OutputData:
    def __init__(self, value: int):
        self.value = value


def transform_step1(input_data: InputData) -> IntermediateData:
    return IntermediateData(input_data.value * 2)


def transform_step2(
    intermediate: IntermediateData, *, multiplier: int = 1
) -> OutputData:
    return OutputData(intermediate.value * multiplier)


def _attach_transform_metadata(
    func,
    *,
    input_schema_name: str,
    output_schema_name: str,
    param_fields: tuple[ParamField, ...] = (),
    parametric: bool | None = None,
) -> None:
    """Attach minimal TransformFn metadata for DAG validator tests."""

    func.__transform_fn__ = TransformFn(
        name=func.__name__,
        qualname=func.__qualname__,
        module=func.__module__,
        input_schema=Schema(name=input_schema_name),
        output_schema=Schema(name=output_schema_name),
        param_schema=ParamSchema(params=param_fields),
        code_ref=CodeRef(
            module=func.__module__,
            qualname=func.__qualname__,
            filepath=None,
            lineno=None,
            code_hash=f"dag-test-hash-{func.__name__}",
        ),
        parametric=parametric if parametric is not None else bool(param_fields),
    )


_attach_transform_metadata(
    transform_step1,
    input_schema_name="InputData",
    output_schema_name="IntermediateData",
    parametric=False,
)
_attach_transform_metadata(
    transform_step2,
    input_schema_name="IntermediateData",
    output_schema_name="OutputData",
    param_fields=(
        ParamField(
            name="multiplier",
            dtype="int",
            required=False,
            default=1,
        ),
    ),
)


@pytest.fixture
def registry() -> TransformRegistry:
    reg = TransformRegistry()
    reg.register(
        "test.step1",
        transform_step1,
        TransformSignature(
            input_types=(InputData,),
            output_type=IntermediateData,
            params={},
        ),
    )
    reg.register(
        "test.step2",
        transform_step2,
        TransformSignature(
            input_types=(IntermediateData,),
            output_type=OutputData,
            params={"multiplier": 1},
        ),
    )
    return reg


@pytest.fixture
def skeleton() -> PipelineSkeleton:
    return PipelineSkeleton(
        name="test_pipeline",
        steps=[
            PipelineStep(
                name="step1",
                input_types=(InputData,),
                output_type=IntermediateData,
                default_transform="test.step1",
                required=True,
            ),
            PipelineStep(
                name="step2",
                input_types=(IntermediateData,),
                output_type=OutputData,
                default_transform="test.step2",
                required=True,
            ),
        ],
    )


def test_EX_N_01_execute_multi_step_pipeline(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """EX-N-01: Execute multi-step pipeline with validation."""
    resolver = TransformResolver(registry)
    validator = ConfigurationValidator(registry, skeleton)
    executor = DAGExecutor(skeleton, resolver, validator)

    config = {
        "steps": {
            "step1": {"transform": "test.step1"},
            "step2": {"transform": "test.step2", "params": {"multiplier": 3}},
        }
    }

    initial_inputs = {"InputData": InputData(5)}

    result = executor.execute(config, initial_inputs)

    assert "step1" in result
    assert "step2" in result
    assert result["step2"].value == 30  # 5 * 2 * 3


def test_EX_E_01_execute_with_invalid_config(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """EX-E-01: Execute with invalid config (Fail Fast)."""
    resolver = TransformResolver(registry)
    validator = ConfigurationValidator(registry, skeleton)
    executor = DAGExecutor(skeleton, resolver, validator)

    config = {
        "steps": {
            "step1": {"transform": "test.nonexistent"},
        }
    }

    initial_inputs = {"InputData": InputData(5)}

    with pytest.raises(ValueError, match="Configuration validation failed"):
        executor.execute(config, initial_inputs)


def test_EX_E_02_execute_with_missing_input(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """EX-E-02: Execute with missing required input."""
    resolver = TransformResolver(registry)
    validator = ConfigurationValidator(registry, skeleton)
    executor = DAGExecutor(skeleton, resolver, validator)

    config = {
        "steps": {
            "step1": {"transform": "test.step1"},
            "step2": {"transform": "test.step2"},
        }
    }

    initial_inputs = {}  # Missing InputData

    with pytest.raises(RuntimeError, match="Required input type"):
        executor.execute(config, initial_inputs)


def test_RS_N_01_resolve_with_explicit_transform(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """RS-N-01: Resolve with explicit transform + params."""
    resolver = TransformResolver(registry)

    step = skeleton.steps[1]
    config = {"transform": "test.step2", "params": {"multiplier": 5}}

    func, params = resolver.resolve_step(step, config)

    assert func is transform_step2
    assert params == {"multiplier": 5}


def test_RS_N_02_resolve_using_default(
    registry: TransformRegistry, skeleton: PipelineSkeleton
) -> None:
    """RS-N-02: Resolve using default transform."""
    resolver = TransformResolver(registry)

    step = skeleton.steps[0]
    config = {}  # Use default

    func, params = resolver.resolve_step(step, config)

    assert func is transform_step1
    assert params == {}


def test_RS_E_01_resolve_with_no_transform_and_no_default(
    registry: TransformRegistry,
) -> None:
    """RS-E-01: Resolve with no transform and no default."""
    resolver = TransformResolver(registry)

    step = PipelineStep(
        name="step",
        input_types=(InputData,),
        output_type=OutputData,
        default_transform=None,
        required=True,
    )

    config = {}

    with pytest.raises(ValueError, match="No transform specified"):
        resolver.resolve_step(step, config)


# ========================================
# Multi-Input Tests (Features + Target)
# ========================================


class FeaturesData:
    """Test class representing features dataset."""

    def __init__(self, features: list[float]):
        self.features = features


class TargetData:
    """Test class representing target dataset."""

    def __init__(self, target: list[float]):
        self.target = target


class MetricsData:
    """Test class representing computed metrics."""

    def __init__(self, score: float):
        self.score = score


def transform_compute_metrics(
    features: FeaturesData, target: TargetData
) -> MetricsData:
    """Transform that takes two data inputs (features + target)."""
    # Simple dummy calculation: sum of features + sum of target
    score = sum(features.features) + sum(target.target)
    return MetricsData(score=score)


def transform_compute_metrics_with_params(
    features: FeaturesData, target: TargetData, *, weight: float = 1.0
) -> MetricsData:
    """Transform that takes two data inputs + parameter."""
    score = (sum(features.features) + sum(target.target)) * weight
    return MetricsData(score=score)


_attach_transform_metadata(
    transform_compute_metrics,
    input_schema_name="Features+Target",
    output_schema_name="MetricsData",
    parametric=False,
)
_attach_transform_metadata(
    transform_compute_metrics_with_params,
    input_schema_name="Features+Target",
    output_schema_name="MetricsData",
    param_fields=(
        ParamField(
            name="weight",
            dtype="float",
            required=False,
            default=1.0,
        ),
    ),
)


@pytest.fixture
def multi_input_registry() -> TransformRegistry:
    """Registry with multi-input transforms."""
    reg = TransformRegistry()
    reg.register(
        "test.compute_metrics",
        transform_compute_metrics,
        TransformSignature(
            input_types=(FeaturesData, TargetData),
            output_type=MetricsData,
            params={},
        ),
    )
    reg.register(
        "test.compute_metrics_with_params",
        transform_compute_metrics_with_params,
        TransformSignature(
            input_types=(FeaturesData, TargetData),
            output_type=MetricsData,
            params={"weight": 1.0},
        ),
    )
    return reg


@pytest.fixture
def multi_input_skeleton() -> PipelineSkeleton:
    """Skeleton with multi-input step."""
    return PipelineSkeleton(
        name="multi_input_pipeline",
        steps=[
            PipelineStep(
                name="compute_metrics",
                input_types=(FeaturesData, TargetData),
                output_type=MetricsData,
                default_transform="test.compute_metrics",
                required=True,
            ),
        ],
    )


def test_EX_N_02_execute_with_multiple_inputs(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """EX-N-02: Execute transform with multiple data inputs (features + target).

    This tests the critical case of transforms that require multiple datasets,
    such as ML training functions that need both features and target.
    """
    resolver = TransformResolver(multi_input_registry)
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)
    executor = DAGExecutor(multi_input_skeleton, resolver, validator)

    config = {
        "steps": {
            "compute_metrics": {"transform": "test.compute_metrics"},
        }
    }

    # Provide both features and target in initial inputs
    initial_inputs = {
        "FeaturesData": FeaturesData(features=[1.0, 2.0, 3.0]),
        "TargetData": TargetData(target=[4.0, 5.0]),
    }

    result = executor.execute(config, initial_inputs)

    assert "compute_metrics" in result
    assert result["compute_metrics"].score == 15.0  # (1+2+3) + (4+5) = 15


def test_EX_N_03_execute_with_multiple_inputs_and_params(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """EX-N-03: Execute transform with multiple inputs + parameters.

    This tests transforms that combine multiple datasets with configurable parameters.
    """
    resolver = TransformResolver(multi_input_registry)
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)
    executor = DAGExecutor(multi_input_skeleton, resolver, validator)

    config = {
        "steps": {
            "compute_metrics": {
                "transform": "test.compute_metrics_with_params",
                "params": {"weight": 2.0},
            },
        }
    }

    initial_inputs = {
        "FeaturesData": FeaturesData(features=[1.0, 2.0, 3.0]),
        "TargetData": TargetData(target=[4.0, 5.0]),
    }

    result = executor.execute(config, initial_inputs)

    assert "compute_metrics" in result
    assert result["compute_metrics"].score == 30.0  # ((1+2+3) + (4+5)) * 2.0 = 30


def test_EX_E_03_execute_with_missing_second_input(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """EX-E-03: Execute with missing second input (target missing).

    This tests error handling when one of multiple required inputs is missing.
    """
    resolver = TransformResolver(multi_input_registry)
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)
    executor = DAGExecutor(multi_input_skeleton, resolver, validator)

    config = {
        "steps": {
            "compute_metrics": {"transform": "test.compute_metrics"},
        }
    }

    # Only provide features, missing target
    initial_inputs = {
        "FeaturesData": FeaturesData(features=[1.0, 2.0, 3.0]),
    }

    with pytest.raises(RuntimeError, match="Required input type.*TargetData"):
        executor.execute(config, initial_inputs)


def test_EX_E_04_execute_with_missing_first_input(
    multi_input_registry: TransformRegistry,
    multi_input_skeleton: PipelineSkeleton,
) -> None:
    """EX-E-04: Execute with missing first input (features missing).

    This tests error handling when the first of multiple inputs is missing.
    """
    resolver = TransformResolver(multi_input_registry)
    validator = ConfigurationValidator(multi_input_registry, multi_input_skeleton)
    executor = DAGExecutor(multi_input_skeleton, resolver, validator)

    config = {
        "steps": {
            "compute_metrics": {"transform": "test.compute_metrics"},
        }
    }

    # Only provide target, missing features
    initial_inputs = {
        "TargetData": TargetData(target=[4.0, 5.0]),
    }

    with pytest.raises(RuntimeError, match="Required input type.*FeaturesData"):
        executor.execute(config, initial_inputs)


# ========================================
# Multi-Step Output Aggregation Tests
# ========================================


class BranchA:
    """Output from first branch."""

    def __init__(self, value_a: int):
        self.value_a = value_a


class BranchB:
    """Output from second branch."""

    def __init__(self, value_b: int):
        self.value_b = value_b


class AggregatedOutput:
    """Output aggregated from multiple branches."""

    def __init__(self, combined: int):
        self.combined = combined


def transform_branch_a(input_data: InputData) -> BranchA:
    """First branch: process input to BranchA."""
    return BranchA(value_a=input_data.value * 10)


def transform_branch_b(input_data: InputData) -> BranchB:
    """Second branch: process input to BranchB."""
    return BranchB(value_b=input_data.value * 100)


def transform_aggregate(branch_a: BranchA, branch_b: BranchB) -> AggregatedOutput:
    """Aggregate outputs from multiple upstream steps."""
    return AggregatedOutput(combined=branch_a.value_a + branch_b.value_b)


_attach_transform_metadata(
    transform_branch_a,
    input_schema_name="InputData",
    output_schema_name="BranchA",
    parametric=False,
)
_attach_transform_metadata(
    transform_branch_b,
    input_schema_name="InputData",
    output_schema_name="BranchB",
    parametric=False,
)
_attach_transform_metadata(
    transform_aggregate,
    input_schema_name="BranchA+BranchB",
    output_schema_name="AggregatedOutput",
    parametric=False,
)


@pytest.fixture
def multi_step_output_registry() -> TransformRegistry:
    """Registry with transforms that aggregate outputs from multiple steps."""
    reg = TransformRegistry()
    reg.register(
        "test.branch_a",
        transform_branch_a,
        TransformSignature(
            input_types=(InputData,),
            output_type=BranchA,
            params={},
        ),
    )
    reg.register(
        "test.branch_b",
        transform_branch_b,
        TransformSignature(
            input_types=(InputData,),
            output_type=BranchB,
            params={},
        ),
    )
    reg.register(
        "test.aggregate",
        transform_aggregate,
        TransformSignature(
            input_types=(BranchA, BranchB),
            output_type=AggregatedOutput,
            params={},
        ),
    )
    return reg


@pytest.fixture
def multi_step_output_skeleton() -> PipelineSkeleton:
    """Skeleton with parallel branches that merge into aggregation step."""
    return PipelineSkeleton(
        name="multi_branch_pipeline",
        steps=[
            PipelineStep(
                name="branch_a",
                input_types=(InputData,),
                output_type=BranchA,
                default_transform="test.branch_a",
                required=True,
            ),
            PipelineStep(
                name="branch_b",
                input_types=(InputData,),
                output_type=BranchB,
                default_transform="test.branch_b",
                required=True,
            ),
            PipelineStep(
                name="aggregate",
                input_types=(BranchA, BranchB),
                output_type=AggregatedOutput,
                default_transform="test.aggregate",
                required=True,
            ),
        ],
    )


def test_EX_N_04_execute_with_multiple_step_outputs_aggregation(
    multi_step_output_registry: TransformRegistry,
    multi_step_output_skeleton: PipelineSkeleton,
) -> None:
    """EX-N-04: Execute pipeline with step that aggregates outputs from multiple upstream steps.

    This tests the critical case where one step receives outputs from multiple
    previous steps (e.g., combining features from different branches).

    Pipeline structure:
        InputData → branch_a → BranchA ┐
        InputData → branch_b → BranchB ┴→ aggregate → AggregatedOutput
    """
    resolver = TransformResolver(multi_step_output_registry)
    validator = ConfigurationValidator(
        multi_step_output_registry, multi_step_output_skeleton
    )
    executor = DAGExecutor(multi_step_output_skeleton, resolver, validator)

    config = {
        "steps": {
            "branch_a": {"transform": "test.branch_a"},
            "branch_b": {"transform": "test.branch_b"},
            "aggregate": {"transform": "test.aggregate"},
        }
    }

    initial_inputs = {"InputData": InputData(value=5)}

    result = executor.execute(config, initial_inputs)

    assert "branch_a" in result
    assert "branch_b" in result
    assert "aggregate" in result
    assert result["branch_a"].value_a == 50  # 5 * 10
    assert result["branch_b"].value_b == 500  # 5 * 100
    assert result["aggregate"].combined == 550  # 50 + 500


def test_EX_E_05_execute_with_missing_upstream_branch(
    multi_step_output_registry: TransformRegistry,
) -> None:
    """EX-E-05: Execute with missing upstream branch output at runtime.

    This tests runtime error handling when aggregation step needs BranchB
    but branch_b step is optional and not configured.
    """
    # Create skeleton with branch_b as optional (required=False)
    skeleton_with_optional_branch = PipelineSkeleton(
        name="multi_branch_pipeline",
        steps=[
            PipelineStep(
                name="branch_a",
                input_types=(InputData,),
                output_type=BranchA,
                default_transform="test.branch_a",
                required=True,
            ),
            PipelineStep(
                name="branch_b",
                input_types=(InputData,),
                output_type=BranchB,
                default_transform="test.branch_b",
                required=False,  # Optional step
            ),
            PipelineStep(
                name="aggregate",
                input_types=(BranchA, BranchB),
                output_type=AggregatedOutput,
                default_transform="test.aggregate",
                required=True,
            ),
        ],
    )

    resolver = TransformResolver(multi_step_output_registry)
    validator = ConfigurationValidator(
        multi_step_output_registry, skeleton_with_optional_branch
    )
    executor = DAGExecutor(skeleton_with_optional_branch, resolver, validator)

    config = {
        "steps": {
            "branch_a": {"transform": "test.branch_a"},
            # branch_b is not configured (and is optional)
            "aggregate": {"transform": "test.aggregate"},
        }
    }

    initial_inputs = {"InputData": InputData(value=5)}

    # Should fail at runtime when aggregate step tries to find BranchB
    with pytest.raises(RuntimeError, match="Required input type.*BranchB"):
        executor.execute(config, initial_inputs)
