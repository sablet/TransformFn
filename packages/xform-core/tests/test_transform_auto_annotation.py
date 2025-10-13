from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, TypedDict

import pytest

from xform_core import (
    Check,
    ExampleValue,
    MissingCheckError,
    MissingExampleError,
    TransformFn,
    normalize_transform,
    register_check,
    register_example,
    transform,
)
from xform_core.transform_registry import example_registry as transform_example_registry


class Payload(TypedDict):
    value: int


@dataclass
class Stats:
    total: int


def validate_stats(stats: Stats) -> None:
    if stats.total < 0:
        raise ValueError("total must be non-negative")


def _register_defaults() -> None:
    register_example(
        f"{Payload.__module__}.{Payload.__name__}",
        ExampleValue({"value": 21}),
    )
    register_check(
        f"{Stats.__module__}.{Stats.__name__}",
        Check(f"{__name__}.validate_stats"),
    )


def test_auto_annotation_populates_metadata() -> None:
    _register_defaults()

    def accumulate(data: Payload) -> Stats:
        """Aggregate payload value."""

        return Stats(total=data["value"])

    decorated = transform(accumulate)
    transform_fn = decorated.__transform_fn__
    assert isinstance(transform_fn, TransformFn)
    assert transform_fn.input_metadata
    assert transform_fn.output_checks == (f"{__name__}.validate_stats",)

    transform_fqn = f"{decorated.__module__}.{decorated.__qualname__}"
    entries = transform_example_registry.get(transform_fqn)
    assert entries
    assert entries[0].metadata[0] == ExampleValue({"value": 21})


def test_missing_example_raises_resolution_error() -> None:
    _register_defaults()

    def faulty(data: Mapping[str, int]) -> Stats:
        """Missing registry entry for Mapping example."""

        return Stats(total=sum(data.values()))

    with pytest.raises(MissingExampleError) as excinfo:
        normalize_transform(faulty)

    assert "TR003" in str(excinfo.value)


def test_missing_check_raises_resolution_error() -> None:
    register_example(
        f"{Payload.__module__}.{Payload.__name__}",
        ExampleValue({"value": 5}),
    )

    def no_check(data: Payload) -> Iterable[int]:
        """Return type lacks registered check."""

        return [data["value"]]

    with pytest.raises(MissingCheckError) as excinfo:
        normalize_transform(no_check)

    assert "TR005" in str(excinfo.value)
