from __future__ import annotations

import pandas as pd
import pytest
from typing import cast

from xform_core import ExampleValue

from proj_dtypes import HLOCVSpec, materialize_example, materialize_value


SENTINEL_INT = 42


def test_materialize_example_from_spec() -> None:
    start = cast(pd.Timestamp, pd.Timestamp("2024-03-01"))
    spec = HLOCVSpec(n=5, seed=123, start=start)
    example = ExampleValue(spec)

    result = materialize_example(example)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == spec.n
    assert spec.start is not None
    assert result.loc[0, "timestamp"] == spec.start


def test_materialize_example_copies_dataframe() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})
    example = ExampleValue(frame)

    result = materialize_example(example)

    assert isinstance(result, pd.DataFrame)
    assert result is not frame
    pd.testing.assert_frame_equal(result, frame)


def test_materialize_value_passthrough_for_scalars() -> None:
    assert materialize_value(ExampleValue(SENTINEL_INT).value) == SENTINEL_INT


def test_materialize_example_requires_examplevalue() -> None:
    with pytest.raises(TypeError):
        materialize_example("not-an-example")  # type: ignore[arg-type]
