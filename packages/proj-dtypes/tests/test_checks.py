from __future__ import annotations

import math

import pandas as pd
import pytest

from proj_dtypes import HLOCVSpec, check_feature_map, check_hlocv_dataframe, gen_hlocv


def test_check_hlocv_dataframe_accepts_valid_frame() -> None:
    frame = gen_hlocv(HLOCVSpec(n=8, seed=7))
    check_hlocv_dataframe(frame)


def test_check_hlocv_dataframe_rejects_missing_columns() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=3, freq="D")}
    )
    with pytest.raises(ValueError):
        check_hlocv_dataframe(frame)


def test_check_hlocv_dataframe_rejects_non_monotonic_timestamp() -> None:
    frame = gen_hlocv(HLOCVSpec(n=5, seed=2))
    frame.loc[3, "timestamp"] = frame.loc[1, "timestamp"]
    with pytest.raises(ValueError):
        check_hlocv_dataframe(frame)


def test_check_hlocv_dataframe_rejects_price_constraint_violation() -> None:
    frame = gen_hlocv(HLOCVSpec(n=4, seed=0))
    frame.loc[1, "high"] = frame.loc[1, "open"] * 0.5
    with pytest.raises(ValueError):
        check_hlocv_dataframe(frame)


def test_check_feature_map_accepts_valid_mapping() -> None:
    check_feature_map({"alpha": 0.1, "beta": -0.2})


def test_check_feature_map_rejects_invalid_entries() -> None:
    with pytest.raises(TypeError):
        check_feature_map({1: 0.5})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        check_feature_map({"flag": True})
    with pytest.raises(ValueError):
        check_feature_map({"explodes": math.inf})
    with pytest.raises(ValueError):
        check_feature_map({})
