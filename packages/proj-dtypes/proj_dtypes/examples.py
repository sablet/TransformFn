"""Example materialization helpers for project-specific data types."""

from __future__ import annotations

import pandas as pd

from xform_core import ExampleValue

from .hlocv_spec import HLOCVSpec, gen_hlocv


def materialize_example(example: ExampleValue) -> object:
    """Convert an ExampleValue into a concrete runtime object.

    The helper understands project-specific specs such as ``HLOCVSpec`` and
    generates the corresponding DataFrame via :func:`gen_hlocv`. When the
    example already contains a concrete pandas object we return a defensive
    copy to avoid accidental in-place mutations leaking between runs.
    """

    if not isinstance(example, ExampleValue):
        raise TypeError("materialize_example expects an ExampleValue instance")

    return materialize_value(example.value)


def materialize_value(value: object) -> object:
    """Materialize raw example payloads into runtime-friendly values."""

    if isinstance(value, HLOCVSpec):
        return gen_hlocv(value)
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return value


__all__ = ["materialize_example", "materialize_value"]
