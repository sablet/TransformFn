"""algo-trade-app向け検証関数集。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from xform_core.checks import (
    check_dataframe_has_columns,
    check_dataframe_not_empty,
    check_dataframe_notnull,
)

from .types import HLOCV_COLUMN_ORDER, PRICE_COLUMNS, VOLUME_COLUMN, MarketRegime


def check_hlocv_dataframe(frame: pd.DataFrame) -> None:
    """HLOCV DataFrameが全ての制約を満たすか検証する。"""
    check_hlocv_dataframe_length(frame)
    check_hlocv_dataframe_notnull(frame)
    _validate_price_columns(frame)
    _validate_price_relationships(frame)
    _validate_volume(frame)


def check_hlocv_dataframe_length(frame: pd.DataFrame) -> None:
    """必須列が存在し、行数が0でないことを検証する。"""
    check_dataframe_has_columns(frame, HLOCV_COLUMN_ORDER)
    check_dataframe_not_empty(frame)


def check_hlocv_dataframe_notnull(frame: pd.DataFrame) -> None:
    """タイムスタンプの性質と欠損値を検証する。"""
    check_dataframe_has_columns(frame, HLOCV_COLUMN_ORDER)
    timestamp = cast(pd.Series, frame["timestamp"])
    _validate_timestamp_column(timestamp)
    check_dataframe_notnull(frame)


def _validate_timestamp_column(timestamp: pd.Series) -> None:
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        raise TypeError("timestamp column must be datetime-like")
    if not timestamp.is_monotonic_increasing:
        raise ValueError("timestamps must be monotonic increasing")
    if timestamp.hasnans:
        raise ValueError("timestamps must not contain NaT")


def _validate_price_columns(frame: pd.DataFrame) -> None:
    data = frame.loc[:, list(PRICE_COLUMNS)]
    if (data <= 0).any().any():
        raise ValueError("price columns must be strictly positive")


def _validate_price_relationships(frame: pd.DataFrame) -> None:
    opens = frame["open"].to_numpy()
    closes = frame["close"].to_numpy()
    if not np.allclose(opens[1:], closes[:-1]):
        raise ValueError("open price must match previous close")

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    if np.any(highs < np.maximum(opens, closes)):
        raise ValueError("high price must be >= max(open, close)")
    if np.any(lows > np.minimum(opens, closes)):
        raise ValueError("low price must be <= min(open, close)")


def _validate_volume(frame: pd.DataFrame) -> None:
    volumes = frame[VOLUME_COLUMN].to_numpy()
    if np.any(~np.isfinite(volumes)):
        raise ValueError("volume must contain finite values")
    if np.any(volumes <= 0):
        raise ValueError("volume must be positive")


def check_feature_map(features: Mapping[str, float]) -> None:
    """FeatureMapが有限な数値値を持つか検証する。"""
    if not features:
        raise ValueError("feature map must contain at least one entry")

    for key, value in features.items():
        if not isinstance(key, str) or not key:
            raise TypeError("feature names must be non-empty strings")
        if isinstance(value, bool):
            raise TypeError("feature values must be numeric, not boolean")
        if not isinstance(value, (int, float)):
            raise TypeError("feature values must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("feature values must be finite numbers")


def check_market_regime_known(regime: MarketRegime | str) -> None:
    """マーケットレジームが列挙型のメンバか検証する。"""
    try:
        MarketRegime(regime)
    except ValueError as exc:
        raise ValueError(f"unknown market regime: {regime!r}") from exc


def check_ohlcv(df: pd.DataFrame) -> None:
    """FXドメイン向けの簡易OHLCV検証。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        return

    for col in ["open", "high", "low", "close"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric")

    if (df["high"] < df["low"]).any():
        raise ValueError("High must be >= Low")
    if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
        raise ValueError("High must be >= Open and Close")
    if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
        raise ValueError("Low must be <= Open and Close")


def check_target(df: pd.DataFrame) -> None:
    """ターゲットDataFrame構造を検証する。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    if "target" not in df.columns:
        raise ValueError("DataFrame must contain 'target' column")

    if not df.empty and not pd.api.types.is_numeric_dtype(df["target"]):
        raise TypeError("Target column must be numeric")


_EXPECTED_TUPLE_SIZE = 2


def check_aligned_data(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """特徴量とターゲットの整合性を検証する。"""
    if not isinstance(data, tuple) or len(data) != _EXPECTED_TUPLE_SIZE:
        raise TypeError("Expected tuple of 2 DataFrames")

    features, target = data

    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"Features must be pd.DataFrame, got {type(features)}")
    if not isinstance(target, pd.DataFrame):
        raise TypeError(f"Target must be pd.DataFrame, got {type(target)}")

    if features.empty or target.empty:
        return

    if len(features) != len(target):
        msg = (
            f"Features and target must have same length: "
            f"{len(features)} != {len(target)}"
        )
        raise ValueError(msg)

    if not features.index.equals(target.index):
        raise ValueError("Features and target must have aligned indices")


def check_prediction_result(result: dict[str, object]) -> None:
    """予測結果辞書のスキーマ検証。"""
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result)}")

    required_keys = {"timestamp", "predicted", "actual", "feature_importance"}
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    timestamp = result["timestamp"]
    predicted = result["predicted"]
    actual = result["actual"]

    if not isinstance(timestamp, list):
        raise TypeError("timestamp must be a list")
    if not isinstance(predicted, list):
        raise TypeError("predicted must be a list")
    if not isinstance(actual, list):
        raise TypeError("actual must be a list")

    if len(timestamp) != len(predicted) or len(timestamp) != len(actual):
        raise ValueError("timestamp, predicted, and actual must have same length")


__all__ = [
    "check_hlocv_dataframe",
    "check_hlocv_dataframe_length",
    "check_hlocv_dataframe_notnull",
    "check_feature_map",
    "check_market_regime_known",
    "check_ohlcv",
    "check_target",
    "check_aligned_data",
    "check_prediction_result",
]
