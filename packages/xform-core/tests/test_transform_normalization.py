from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, TypedDict, cast

import pytest

from xform_core import (
    Check,
    ExampleValue,
    RegisteredContainer,
    RegisteredType,
    TransformFn,
    normalize_transform,
    transform,
)


class InputPayload(TypedDict):
    value: int


@dataclass
class OutputPayload:
    value: int


RegisteredType(InputPayload).register()
RegisteredType(OutputPayload).register()


class UnregisteredPayload(TypedDict):
    value: int


@dataclass
class ConfigParam:
    value: int


@dataclass
class UnregisteredOutput:
    value: int


PortfolioAllocations = RegisteredContainer["DictStrKeys", InputPayload](
    "PortfolioAllocations"
)


def validate_output(payload: OutputPayload) -> None:
    if not isinstance(payload.value, int):  # pragma: no cover - 防御チェック
        raise AssertionError("value must be int")


@transform
def add_one(
    data: Annotated[InputPayload, ExampleValue({"value": 1})],
    *,
    offset: int = 1,
) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
    """入力値に offset を加算するサンプル変換。"""

    return OutputPayload(value=data["value"] + offset)


def test_transform_decorator_attaches_metadata() -> None:
    assert hasattr(add_one, "__transform_fn__")
    transform_fn = cast(TransformFn, add_one.__transform_fn__)
    assert isinstance(transform_fn, TransformFn)
    assert transform_fn.input_schema.name == "InputPayload"
    assert transform_fn.output_schema.name == "OutputPayload"
    assert transform_fn.param_schema.params[0].name == "offset"
    assert transform_fn.output_checks == (f"{__name__}.validate_output",)
    assert transform_fn.parametric is True
    assert transform_fn.payload_parameters == ("data",)
    assert transform_fn.payload_schemas[0].name == "InputPayload"


def test_transform_supports_multiple_payload_parameters() -> None:
    @transform
    def combine(
        left: Annotated[InputPayload, ExampleValue({"value": 2})],
        right: Annotated[InputPayload, ExampleValue({"value": 3})],
        *,
        scale: int = 1,
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Aggregate two payload inputs."""

        value = (left["value"] + right["value"]) * scale
        return OutputPayload(value=value)

    transform_fn = cast(TransformFn, combine.__transform_fn__)
    assert transform_fn.payload_parameters == ("left", "right")
    assert len(transform_fn.payload_schemas) == 2
    assert transform_fn.param_schema.params[0].name == "scale"


def test_registered_container_payload_type() -> None:
    @transform
    def aggregate(
        allocations: Annotated[
            PortfolioAllocations,
            ExampleValue({"core": {"value": 1}}),
        ],
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Use registered container as payload."""

        return OutputPayload(value=allocations["core"]["value"])

    transform_fn = cast(TransformFn, aggregate.__transform_fn__)
    assert transform_fn.payload_parameters == ("allocations",)
    assert transform_fn.payload_schemas[0].name == "PortfolioAllocations"


def test_unregistered_container_rejected() -> None:
    def _invalid(
        allocations: Annotated[
            dict[str, InputPayload],
            ExampleValue({"core": {"value": 1}}),
        ],
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Dict payload without registration should fail."""

        return OutputPayload(value=allocations["core"]["value"])

    assert _run_and_capture_error(_invalid) == "TR010"


def test_normalize_transform_returns_consistent_model() -> None:
    normalized = normalize_transform(add_one)
    assert normalized.code_ref.module_path.endswith("add_one")
    assert normalized.input_metadata
    assert normalized.id
    assert normalized.version
    assert normalized.parametric is True


def test_transform_decorator_parametric_flag_override() -> None:
    @transform(parametric=False)
    def multiply(
        data: Annotated[InputPayload, ExampleValue({"value": 2})],
        *,
        multiplier: int = 3,
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """parametric フラグの上書きテスト。"""

        return OutputPayload(value=data["value"] * multiplier)

    transform_fn = cast(TransformFn, multiply.__transform_fn__)
    assert transform_fn.parametric is False


def _run_and_capture_error(
    func: Callable[..., object], *, auto_annotation: bool = False
) -> str:
    with pytest.raises(ValueError) as excinfo:
        normalize_transform(func, auto_annotation=auto_annotation)
    message = str(excinfo.value)
    return message.split(":", 1)[0]


def test_tr001_requires_first_argument() -> None:
    def _invalid() -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """No parameters."""

        return OutputPayload(value=1)

    assert _run_and_capture_error(_invalid) == "TR001"


def test_tr002_requires_annotated_first_argument() -> None:
    def _invalid(
        data: InputPayload,
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Missing Annotated wrapper."""

        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR002"


def test_tr003_requires_example_metadata() -> None:
    def _invalid(
        data: Annotated[InputPayload, "meta"],
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Annotated but missing Example metadata."""

        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR003"


def test_tr004_validates_example_value_type() -> None:
    def _invalid(
        data: Annotated[int, ExampleValue(cast(int, "oops"))],
    ) -> Annotated[int, Check(f"{__name__}.validate_output")]:
        """ExampleValue must match base type."""

        return data

    assert _run_and_capture_error(_invalid) == "TR004"


def test_tr005_requires_annotated_return() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
    ) -> OutputPayload:
        """Missing Annotated return."""

        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR005"


def test_tr006_requires_check_metadata() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
    ) -> Annotated[OutputPayload, "meta"]:
        """Missing Check metadata."""

        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR006"


def test_tr007_requires_literal_fqn() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
    ) -> Annotated[OutputPayload, Check("invalid")]:
        """Check must be FQN."""

        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR007"


def test_tr008_requires_existing_function() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
    ) -> Annotated[OutputPayload, Check("nonexistent.module.func")]:
        """Check must point to existing function."""

        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR008"


def test_tr009_requires_docstring() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        return OutputPayload(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR009"


def test_tr010_requires_registered_payload() -> None:
    def _invalid(
        data: Annotated[UnregisteredPayload, ExampleValue({"value": 1})],
        *,
        offset: int = 1,
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Unregistered payload should fail."""

        return OutputPayload(value=data["value"] + offset)

    assert _run_and_capture_error(_invalid) == "TR010"


def test_tr011_disallows_payload_after_boundary() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
        *,
        other: Annotated[InputPayload, ExampleValue({"value": 2})],
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Second payload should not be accepted."""

        return OutputPayload(value=data["value"] + other["value"])

    assert _run_and_capture_error(_invalid) == "TR011"


def test_tr012_requires_keyword_only_parameters() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
        offset: int = 1,
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Missing * boundary should fail."""

        return OutputPayload(value=data["value"] + offset)

    assert _run_and_capture_error(_invalid) == "TR012"


def test_tr013_restricts_parameter_type() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
        *,
        config: ConfigParam = ConfigParam(1),
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Custom config dataclass should be disallowed."""

        return OutputPayload(value=data["value"] + config.value)

    assert _run_and_capture_error(_invalid) == "TR013"


def test_tr014_requires_default_or_example_for_parameters() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
        *,
        threshold: int,
    ) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
        """Keyword parameter without default or Example should fail."""

        return OutputPayload(value=data["value"] + threshold)

    assert _run_and_capture_error(_invalid) == "TR014"


def test_tr015_requires_registered_return_type() -> None:
    def _invalid(
        data: Annotated[InputPayload, ExampleValue({"value": 1})],
    ) -> Annotated[UnregisteredOutput, Check(f"{__name__}.validate_output")]:
        """Return type must be registered."""

        return UnregisteredOutput(value=data["value"])

    assert _run_and_capture_error(_invalid) == "TR015"
