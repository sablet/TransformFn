"""Example メタデータから実行時値へ変換するユーティリティ。"""

from __future__ import annotations

import inspect
from typing import Callable

from xform_core import ExampleType, ExampleValue
from xform_core.registry import ExampleEntry

ProjectMaterializer = Callable[[ExampleValue[object]], object]
project_materialize_example: ProjectMaterializer | None

try:  # プロジェクト固有の materializer は任意依存
    from proj_dtypes.examples import (
        materialize_example as _project_materialize_example,
    )
except ModuleNotFoundError:  # pragma: no cover - 依存が無い場合
    project_materialize_example = None
else:
    project_materialize_example = _project_materialize_example


class ExampleMaterializationError(RuntimeError):
    """Example から値を生成できない場合に投げる例外。"""

    def __init__(self, message: str, *, parameter: str | None = None) -> None:
        super().__init__(message)
        self.parameter = parameter


def materialize_entry(entry: ExampleEntry) -> object:
    """ExampleEntry から実際に関数へ渡す値を構築する。"""

    for meta in entry.metadata:
        if isinstance(meta, ExampleValue):
            try:
                return _materialize_example_value(meta)
            except ExampleMaterializationError as exc:
                raise ExampleMaterializationError(
                    str(exc), parameter=entry.parameter
                ) from exc
    for meta in entry.metadata:
        if isinstance(meta, ExampleType):
            return _materialize_example_type(meta, parameter=entry.parameter)
    raise ExampleMaterializationError(
        "Parameter requires ExampleValue or ExampleType metadata",
        parameter=entry.parameter,
    )


def _materialize_example_value(example: ExampleValue[object]) -> object:
    if project_materialize_example is not None:
        try:
            return project_materialize_example(example)
        except Exception as exc:  # pragma: no cover - 依存側での失敗は稀
            raise ExampleMaterializationError(
                f"project materializer failed: {exc}",
            ) from exc

    value = example.value
    if callable(value) and not inspect.isclass(value):
        try:
            return value()
        except TypeError as exc:
            raise ExampleMaterializationError(
                "callable ExampleValue requires zero-arg callable",
            ) from exc
    return value


def _materialize_example_type(
    example_type: ExampleType[object], *, parameter: str | None
) -> object:
    factory = example_type.type
    try:
        return factory()
    except Exception as exc:
        target = f"{factory!r}"
        raise ExampleMaterializationError(
            f"failed to instantiate ExampleType {target}: {exc}",
            parameter=parameter,
        ) from exc
