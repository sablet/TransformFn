"""Validation helpers for project data structures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd

from .types import HLOCV_COLUMN_ORDER, PRICE_COLUMNS, VOLUME_COLUMN, MarketRegime


def check_hlocv_dataframe(frame: pd.DataFrame) -> None:
    """Raise if the provided DataFrame violates HLOCV invariants."""

    check_hlocv_dataframe_length(frame)
    check_hlocv_dataframe_notnull(frame)
    _validate_price_columns(frame)
    _validate_price_relationships(frame)
    _validate_volume(frame)


def check_hlocv_dataframe_length(frame: pd.DataFrame) -> None:
    """Ensure dataframe contains required columns and at least one row."""

    _ensure_required_columns(frame)


def check_hlocv_dataframe_notnull(frame: pd.DataFrame) -> None:
    """Ensure dataframe timestamps are monotonic and values are not null."""

    _ensure_required_columns(frame)
    timestamp = cast(pd.Series, frame["timestamp"])
    _validate_timestamp_column(timestamp)
    if frame.isna().values.any():
        raise ValueError("HLOCV dataframe must not contain null values")


def check_feature_map(features: Mapping[str, float]) -> None:
    """Validate that the feature map contains finite numeric values."""

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
    """Ensure provided market regime value is part of known enumeration."""

    try:
        MarketRegime(regime)
    except ValueError as exc:
        raise ValueError(f"unknown market regime: {regime!r}") from exc


__all__ = [
    "check_feature_map",
    "check_hlocv_dataframe",
    "check_hlocv_dataframe_length",
    "check_hlocv_dataframe_notnull",
    "check_market_regime_known",
]


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in HLOCV_COLUMN_ORDER if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("HLOCV dataframe must not be empty")


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
