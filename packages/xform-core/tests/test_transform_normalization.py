from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, TypedDict, cast

import pytest

from xform_core import Check, ExampleValue, TransformFn, normalize_transform, transform


class InputPayload(TypedDict):
    value: int


@dataclass
class OutputPayload:
    value: int


def validate_output(payload: OutputPayload) -> None:
    if not isinstance(payload.value, int):  # pragma: no cover - 防御チェック
        raise AssertionError("value must be int")


@transform
def add_one(
    data: Annotated[InputPayload, ExampleValue({"value": 1})],
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


def test_normalize_transform_returns_consistent_model() -> None:
    normalized = normalize_transform(add_one)
    assert normalized.code_ref.module_path.endswith("add_one")
    assert normalized.input_metadata
    assert normalized.id
    assert normalized.version


def _run_and_capture_error(func: Callable[..., object]) -> str:
    with pytest.raises(ValueError) as excinfo:
        normalize_transform(func)
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
