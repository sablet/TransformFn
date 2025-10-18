from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from algo_trade_dtypes.generators import HLOCVSpec, gen_hlocv
from algo_trade_dtypes.types import HLOCV_COLUMN_ORDER


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def test_gen_hlocv_structure() -> None:
    spec = HLOCVSpec(n=16, seed=42, start=_ts("2024-02-01"))
    frame = gen_hlocv(spec)

    assert list(frame.columns) == list(HLOCV_COLUMN_ORDER)
    assert len(frame) == spec.n
    assert frame["open"].iloc[0] == pytest.approx(spec.start_price)
    assert np.allclose(frame["open"].to_numpy()[1:], frame["close"].to_numpy()[:-1])
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()


def test_gen_hlocv_timezone() -> None:
    spec = HLOCVSpec(n=3, seed=1, tz="UTC")
    frame = gen_hlocv(spec)

    assert frame["timestamp"].dt.tz is not None


@pytest.mark.parametrize(
    ("field", "value", "exc"),
    [
        ("n", 0, ValueError),
        ("start_price", -1.0, ValueError),
        ("sigma", -0.1, ValueError),
        ("base_volume", 0.0, ValueError),
        ("volume_scale", 0.0, ValueError),
        ("volume_jitter", 1.5, ValueError),
        ("spread_range", (0.02, 0.01), ValueError),
    ],
)
def test_invalid_spec_configuration(
    field: str, value: object, exc: type[Exception]
) -> None:
    kwargs = {field: value}
    with pytest.raises(exc):
        HLOCVSpec(**kwargs)  # type: ignore[arg-type]


def test_invalid_frequency_alias() -> None:
    with pytest.raises(ValueError):
        HLOCVSpec(freq="not-a-real-freq")


def test_invalid_start_type() -> None:
    with pytest.raises(TypeError):
        invalid_start = cast(pd.Timestamp, object())
        HLOCVSpec(start=invalid_start)
