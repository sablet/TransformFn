from __future__ import annotations

import pandas as pd

from algo_trade_dtype.generators import HLOCVSpec
from algo_trade_dtype.materializers import materialize_algo_trade_value


def test_materialize_hlocv_spec_produces_dataframe() -> None:
    spec = HLOCVSpec(n=5, seed=11)
    result = materialize_algo_trade_value(spec)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == spec.n


def test_materialize_dataframe_returns_copy() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})
    materialized = materialize_algo_trade_value(frame)

    assert isinstance(materialized, pd.DataFrame)
    assert materialized is not frame
    pd.testing.assert_frame_equal(materialized, frame)


def test_materialize_scalar_passthrough() -> None:
    assert materialize_algo_trade_value(123) == 123
