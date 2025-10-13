from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

import pytest

from xform_core import (
    Check,
    ExampleValue,
    check_registry,
    example_registry,
    normalize_transform,
)


OFFSET_EXAMPLE = 2


class InputPayload(TypedDict):
    value: int


@dataclass
class OutputPayload:
    value: int


def validate_output(payload: OutputPayload) -> None:
    if payload.value < 0:  # pragma: no cover - 防御的チェック
        raise ValueError("value must be non-negative")


def sample_transform(
    data: Annotated[InputPayload, ExampleValue({"value": 10})],
    offset: Annotated[int, ExampleValue(OFFSET_EXAMPLE)] = 1,
) -> Annotated[OutputPayload, Check(f"{__name__}.validate_output")]:
    """レジストリ登録用のサンプル変換。"""

    return OutputPayload(value=data["value"] + offset)


def test_normalize_transform_populates_registries() -> None:
    transform_fn = normalize_transform(sample_transform)
    transform_fqn = f"{sample_transform.__module__}.{sample_transform.__qualname__}"

    example_entries = example_registry.get(transform_fqn)
    assert {entry.parameter for entry in example_entries} == {"data", "offset"}

    data_entry = next(entry for entry in example_entries if entry.parameter == "data")
    offset_entry = next(
        entry for entry in example_entries if entry.parameter == "offset"
    )
    assert isinstance(data_entry.metadata[0], ExampleValue)
    assert isinstance(offset_entry.metadata[0], ExampleValue)
    assert offset_entry.metadata[0].value == OFFSET_EXAMPLE

    assert example_registry.get_for_parameter(transform_fqn, "offset") == offset_entry
    assert {entry.parameter for entry in example_registry.iter_all()} == {
        "data",
        "offset",
    }

    check_entries = check_registry.get(transform_fqn)
    assert len(check_entries) == 1
    check_entry = check_entries[0]
    assert check_entry.target == f"{__name__}.validate_output"
    assert check_registry.resolve(check_entry.target) is validate_output

    assert transform_fn.output_checks == (check_entry.target,)


def test_example_registry_rejects_non_example_metadata() -> None:
    with pytest.raises(TypeError):
        example_registry.register_many(
            "pkg.transform",
            {"data": ("not-example",)},
        )


def test_check_registry_requires_overwrite_for_conflicting_registration() -> None:
    def check_one(_: object) -> None: ...

    def check_two(_: object) -> None: ...

    check_registry.register(
        "pkg.transform",
        (("pkg.check", check_one),),
    )

    with pytest.raises(ValueError):
        check_registry.register(
            "pkg.transform",
            (("pkg.check", check_two),),
        )

    check_registry.register(
        "pkg.transform",
        (("pkg.check", check_two),),
        overwrite=True,
    )
    assert check_registry.resolve("pkg.check") is check_two
