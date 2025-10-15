"""Training Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd

from xform_core import Check, transform

from algo_trade_dtype.types import CVMethod, SimpleCVConfig


@transform
def convert_nullable_dtypes(
    df: pd.DataFrame,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Convert nullable pandas dtypes to standard numpy dtypes for compatibility.

    This function converts nullable dtypes (Float64, Int64, boolean) to standard
    numpy dtypes (float64, int64, bool) to ensure compatibility with libraries
    like LightGBM that may not support nullable dtypes.
    """
    if df.empty:
        return df

    dtype_mapping = {"Float64": "float64", "Int64": "int64", "boolean": "bool"}

    converted_df = df.copy()
    for col in converted_df.columns:
        dtype_str = str(converted_df[col].dtype)
        if dtype_str in dtype_mapping:
            converted_df[col] = converted_df[col].astype(dtype_mapping[dtype_str])

    return converted_df


@transform
def get_cv_splits(
    n_samples: int,
    config: SimpleCVConfig | None = None,
) -> Annotated[
    list[tuple[list[int], list[int]]], Check("algo_trade_dtype.checks.check_cv_splits")
]:
    """Generate cross validation splits for time series data.

    Supports three CV methods:
    - TIME_SERIES: sklearn TimeSeriesSplit
    - EXPANDING_WINDOW: Expanding window split
    - SLIDING_WINDOW: Sliding window split
    """
    if config is None:
        config = {
            "method": CVMethod.TIME_SERIES,
            "n_splits": 5,
            "test_size": None,
            "gap": 0,
        }

    method = config.get("method", CVMethod.TIME_SERIES)
    n_splits = config.get("n_splits", 5)
    test_size = config.get("test_size", None)
    gap = config.get("gap", 0)

    if method == CVMethod.TIME_SERIES:
        return _time_series_split(n_samples, n_splits, test_size, gap)
    elif method == CVMethod.EXPANDING_WINDOW:
        return _expanding_window_split(n_samples, n_splits, test_size, gap)
    elif method == CVMethod.SLIDING_WINDOW:
        return _sliding_window_split(n_samples, n_splits, test_size, gap)
    else:
        raise ValueError(f"Unsupported CV method: {method}")


def _time_series_split(
    n_samples: int,
    n_splits: int,
    test_size: int | None,
    gap: int,
) -> list[tuple[list[int], list[int]]]:
    """Standard time series split implementation."""
    splits: list[tuple[list[int], list[int]]] = []

    if test_size is None:
        # Auto-calculate test_size
        test_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size

        test_end = min(test_end, n_samples)

        train_end = test_start - gap
        if train_end <= 0:
            continue

        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


def _expanding_window_split(
    n_samples: int,
    n_splits: int,
    test_size: int | None,
    gap: int,
) -> list[tuple[list[int], list[int]]]:
    """Expanding window split implementation."""
    splits: list[tuple[list[int], list[int]]] = []

    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        test_start = n_samples - test_size * (n_splits - i)
        test_end = test_start + test_size

        train_end = test_start - gap
        if train_end <= 0:
            continue

        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, min(test_end, n_samples)))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


def _sliding_window_split(
    n_samples: int,
    n_splits: int,
    test_size: int | None,
    gap: int,
) -> list[tuple[list[int], list[int]]]:
    """Sliding window split implementation."""
    splits: list[tuple[list[int], list[int]]] = []

    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    train_size = n_samples // 2  # Use half of data for training

    for i in range(n_splits):
        test_start = n_samples - test_size * (n_splits - i)
        test_end = test_start + test_size

        train_start = max(0, test_start - gap - train_size)
        train_end = test_start - gap

        if train_end <= train_start:
            continue

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, min(test_end, n_samples)))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


@transform
def calculate_rmse(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Annotated[float, Check("algo_trade_dtype.checks.check_nonnegative_float")]:
    """Calculate Root Mean Squared Error (RMSE) between true and predicted values."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        return 0.0

    if len(y_true_arr) != len(y_pred_arr):
        msg = (
            f"y_true and y_pred must have same length: "
            f"{len(y_true_arr)} != {len(y_pred_arr)}"
        )
        raise ValueError(msg)

    mse = np.mean((y_true_arr - y_pred_arr) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


__all__ = [
    "convert_nullable_dtypes",
    "get_cv_splits",
    "calculate_rmse",
]
