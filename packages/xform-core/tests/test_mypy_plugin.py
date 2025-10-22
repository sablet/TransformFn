from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Callable, TypedDict, cast

from mypy.plugin import FunctionContext
from mypy.types import AnyType, TypeOfAny
from pytest import MonkeyPatch

from xform_core import (
    Check,
    ExampleValue,
    RegisteredType,
    register_check,
    register_example,
)
from xform_core.dtype_rules.plugin import transform_function_hook
from xform_core.transforms_core import PLUGIN_ENV_FLAG


class InputPayload(TypedDict):
    value: int


@dataclass
class OutputPayload:
    value: int


RegisteredType(InputPayload).register()
RegisteredType(OutputPayload).register()


def validate_output(payload: OutputPayload) -> None: ...


def valid_transform(
    data: Annotated[InputPayload, ExampleValue({"value": 1})],
) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
    """Valid transform used for plugin tests."""

    return OutputPayload(value=data["value"])


def invalid_transform(
    data: Annotated[InputPayload, "meta"],
) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
    """Invalid transform missing Example metadata."""

    return OutputPayload(value=data["value"])


def auto_registry_transform(data: InputPayload) -> OutputPayload:
    """Transform that relies on registry-provided annotations."""

    return OutputPayload(value=data["value"])


class DummyApi:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def fail(self, message: str, context: object) -> None:
        self.messages.append(message)


def _build_ctx(func: Callable[..., object]) -> tuple[FunctionContext, DummyApi]:
    api = DummyApi()
    node = SimpleNamespace(fullname=f"{func.__module__}.{func.__qualname__}")
    ctx = SimpleNamespace(
        arg_nodes=[[node]],
        default_return_type=AnyType(TypeOfAny.special_form),
        api=api,
        context=None,
        api_messages=api.messages,
    )
    return cast(FunctionContext, ctx), api


def test_plugin_allows_valid_transform(monkeypatch: MonkeyPatch) -> None:
    ctx, api = _build_ctx(valid_transform)
    monkeypatch.delenv(PLUGIN_ENV_FLAG, raising=False)

    result = transform_function_hook(ctx)

    assert isinstance(result, AnyType)
    assert not api.messages


def test_plugin_reports_tr003_for_missing_example(monkeypatch: MonkeyPatch) -> None:
    ctx, api = _build_ctx(invalid_transform)
    monkeypatch.delenv(PLUGIN_ENV_FLAG, raising=False)

    transform_function_hook(ctx)

    assert api.messages
    assert "TR003" in api.messages[0]


def test_plugin_allows_auto_annotation_via_registry(
    monkeypatch: MonkeyPatch,
) -> None:
    register_example(
        f"{InputPayload.__module__}.{InputPayload.__name__}",
        ExampleValue({"value": 11}),
    )
    register_check(
        f"{OutputPayload.__module__}.{OutputPayload.__name__}",
        Check(f"{__name__}.validate_output"),
    )

    ctx, api = _build_ctx(auto_registry_transform)
    monkeypatch.delenv(PLUGIN_ENV_FLAG, raising=False)

    result = transform_function_hook(ctx)

    assert isinstance(result, AnyType)
    assert not api.messages
