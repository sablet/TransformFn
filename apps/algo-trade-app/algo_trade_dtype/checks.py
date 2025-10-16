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


_EXPECTED_SPLIT_SIZE = 2


def check_cv_splits(splits: list[tuple[list[int], list[int]]]) -> None:
    """CV splits の構造検証。"""
    if not isinstance(splits, list):
        raise TypeError(f"Expected list of CV splits, got {type(splits)}")

    if len(splits) == 0:
        raise ValueError("CV splits must not be empty")

    for i, split in enumerate(splits):
        if not isinstance(split, tuple) or len(split) != _EXPECTED_SPLIT_SIZE:
            raise TypeError(f"Split {i} must be a tuple of (train_idx, valid_idx)")

        train_idx, valid_idx = split

        if not isinstance(train_idx, list):
            raise TypeError(f"Split {i}: train_idx must be a list")
        if not isinstance(valid_idx, list):
            raise TypeError(f"Split {i}: valid_idx must be a list")

        if len(train_idx) == 0:
            raise ValueError(f"Split {i}: train_idx must not be empty")
        if len(valid_idx) == 0:
            raise ValueError(f"Split {i}: valid_idx must not be empty")


def check_fold_result(result: dict[str, object]) -> None:
    """Fold 結果の検証。"""
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result)}")

    required_keys = {
        "fold_id",
        "train_indices",
        "valid_indices",
        "train_score",
        "valid_score",
        "predictions",
        "feature_importance",
    }
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    if not isinstance(result["fold_id"], int):
        raise TypeError("fold_id must be an integer")

    if not isinstance(result["train_score"], (int, float)):
        raise TypeError("train_score must be numeric")
    if not isinstance(result["valid_score"], (int, float)):
        raise TypeError("valid_score must be numeric")

    if not math.isfinite(float(result["train_score"])):
        raise ValueError("train_score must be finite")
    if not math.isfinite(float(result["valid_score"])):
        raise ValueError("valid_score must be finite")


def check_cv_result(result: dict[str, object]) -> None:
    """CV 結果全体の検証。"""
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result)}")

    required_keys = {"fold_results", "mean_score", "std_score", "oos_predictions"}
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    fold_results = result["fold_results"]
    if not isinstance(fold_results, list):
        raise TypeError("fold_results must be a list")

    if len(fold_results) == 0:
        raise ValueError("fold_results must not be empty")

    for fold_result in fold_results:
        check_fold_result(fold_result)

    if not isinstance(result["mean_score"], (int, float)):
        raise TypeError("mean_score must be numeric")
    if not isinstance(result["std_score"], (int, float)):
        raise TypeError("std_score must be numeric")

    if not math.isfinite(float(result["mean_score"])):
        raise ValueError("mean_score must be finite")
    if not math.isfinite(float(result["std_score"])):
        raise ValueError("std_score must be finite")


def check_nonnegative_float(value: float) -> None:
    """非負の浮動小数点数であることを検証する。"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric value, got {type(value)}")

    if not math.isfinite(value):
        raise ValueError("Value must be finite")

    if value < 0:
        raise ValueError(f"Value must be non-negative, got {value}")


def _validate_datetime_column(df: pd.DataFrame, col: str) -> None:
    """datetime列の検証を行うヘルパー関数。"""
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        raise TypeError(f"{col} column must be datetime-like")


def _validate_numeric_range(
    df: pd.DataFrame,
    col: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None:
    """数値列の範囲検証を行うヘルパー関数。"""
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"{col} must be numeric")

    if min_val is not None and (df[col] < min_val).any():
        raise ValueError(f"{col} must be >= {min_val}")
    if max_val is not None and (df[col] > max_val).any():
        raise ValueError(f"{col} must be <= {max_val}")


def check_ranked_predictions(data: list) -> None:
    """ランク付けされた予測結果の検証。"""
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")

    if not data:
        return

    required_keys = {
        "date",
        "currency_pair",
        "prediction",
        "actual_return",
        "prediction_rank_pct",
    }
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} must be dict, got {type(item)}")

        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            raise ValueError(f"Item {i}: missing required keys: {missing_keys}")

        rank_pct = item["prediction_rank_pct"]
        if not isinstance(rank_pct, (int, float)):
            raise TypeError(f"Item {i}: prediction_rank_pct must be numeric")

        if not (0 <= rank_pct <= 1):
            raise ValueError(
                f"Item {i}: prediction_rank_pct must be in [0, 1], got {rank_pct}"
            )


def check_selected_currencies(data: list) -> None:
    """選択された通貨ペアの検証。"""
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")

    if not data:
        return

    required_keys = {
        "date",
        "currency_pair",
        "prediction",
        "actual_return",
        "prediction_rank_pct",
        "signal",
    }
    valid_signals = {-1.0, 0.0, 1.0}

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} must be dict, got {type(item)}")

        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            raise ValueError(f"Item {i}: missing required keys: {missing_keys}")

        signal = item["signal"]
        if not isinstance(signal, (int, float)):
            raise TypeError(f"Item {i}: signal must be numeric")

        if signal not in valid_signals:
            raise ValueError(
                f"Item {i}: signal must be in {valid_signals}, got {signal}"
            )


def check_simulation_result(result: dict) -> None:
    """シミュレーション結果の検証。"""
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result)}")

    required_keys = {"date", "portfolio_return", "n_positions"}
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    dates = result.get("date", [])
    returns = result.get("portfolio_return", [])
    positions = result.get("n_positions", [])

    if not isinstance(dates, list):
        raise TypeError("date must be a list")
    if not isinstance(returns, list):
        raise TypeError("portfolio_return must be a list")
    if not isinstance(positions, list):
        raise TypeError("n_positions must be a list")

    if len(dates) != len(returns) or len(dates) != len(positions):
        raise ValueError(
            "date, portfolio_return, and n_positions must have same length"
        )

    for i, pos in enumerate(positions):
        if not isinstance(pos, int):
            raise TypeError(f"n_positions[{i}] must be int, got {type(pos)}")
        if pos < 0:
            raise ValueError(f"n_positions[{i}] must be non-negative, got {pos}")


def check_performance_metrics(metrics: dict[str, float]) -> None:
    """パフォーマンス指標の検証。"""
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected dict, got {type(metrics)}")

    required_keys = {
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    }
    missing_keys = required_keys - set(metrics.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    for key in required_keys:
        value = metrics[key]
        if not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric, got {type(value)}")
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value}")


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
    "check_cv_splits",
    "check_fold_result",
    "check_cv_result",
    "check_nonnegative_float",
    "check_ranked_predictions",
    "check_selected_currencies",
    "check_simulation_result",
    "check_performance_metrics",
]
