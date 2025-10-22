"""Tests for DAG Executor and Resolver."""

from __future__ import annotations

import pytest

from xform_core.dag.executor import DAGExecutor
from xform_core.dag.resolver import TransformResolver
from xform_core.dag.skeleton import PipelineStep, PipelineSkeleton
from xform_core.dag.transform_registry import TransformRegistry, TransformSignature
from xform_core.dag.validator import ConfigurationValidator


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
