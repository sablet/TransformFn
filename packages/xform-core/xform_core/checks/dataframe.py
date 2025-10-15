"""汎用DataFrame検証関数群。"""

from __future__ import annotations

import pandas as pd


def check_dataframe_not_empty(df: pd.DataFrame) -> None:
    """DataFrameに最低1行存在することを検証する。"""
    if df.empty:
        raise ValueError("DataFrame must not be empty")


def check_dataframe_has_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
) -> None:
    """指定した列が全て存在することを検証する。"""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_dataframe_notnull(df: pd.DataFrame) -> None:
    """DataFrameに欠損値が含まれていないことを検証する。"""
    if df.isna().values.any():
        raise ValueError("DataFrame must not contain null values")


def check_column_monotonic(series: pd.Series, *, increasing: bool = True) -> None:
    """Seriesが単調増加/減少を満たすか検証する。"""
    if increasing and not series.is_monotonic_increasing:
        raise ValueError(f"Column {series.name} must be monotonic increasing")
    if not increasing and not series.is_monotonic_decreasing:
        raise ValueError(f"Column {series.name} must be monotonic decreasing")


def check_column_dtype(df: pd.DataFrame, column: str, expected_dtype: str) -> None:
    """列のdtypeが期待通りであることを検証する。"""
    actual_dtype = df[column].dtype
    if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
        raise TypeError(
            f"Column {column} has dtype {actual_dtype}, expected {expected_dtype}"
        )


def check_column_positive(df: pd.DataFrame, column: str) -> None:
    """列の値が全て正であることを検証する。"""
    if (df[column] <= 0).any():
        raise ValueError(f"Column {column} must contain only positive values")


def check_column_nonnegative(df: pd.DataFrame, column: str) -> None:
    """列の値が非負であることを検証する。"""
    if (df[column] < 0).any():
        raise ValueError(f"Column {column} must contain only non-negative values")


__all__ = [
    "check_dataframe_not_empty",
    "check_dataframe_has_columns",
    "check_dataframe_notnull",
    "check_column_monotonic",
    "check_column_dtype",
    "check_column_positive",
    "check_column_nonnegative",
]
