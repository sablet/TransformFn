"""Pure computation Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd

from xform_core import Check, transform

from algo_trade_dtype.types import ConvertType


@transform
def resample_ohlcv(
    df: pd.DataFrame,
    *,
    freq: str = "1h",
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Resample OHLCV DataFrame to specified frequency."""
    if df.empty:
        return df

    # Ensure DatetimeIndex for resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise TypeError(
                "DataFrame must have DatetimeIndex or 'timestamp' column for resampling"
            )

    resampled = df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    result = resampled.dropna()
    assert isinstance(result, pd.DataFrame)
    return result


@transform
def calculate_rsi(
    df: pd.DataFrame,
    *,
    period: int = 14,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate RSI (Relative Strength Index) indicator."""
    if df.empty or len(df) < period:
        result = df.copy()
        result["rsi"] = np.nan
        return result

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss_series = loss.rolling(window=period, min_periods=period).mean()
    avg_loss_replaced = avg_loss_series.replace(0, np.nan)

    rs = avg_gain / avg_loss_replaced
    rsi = 100 - (100 / (1 + rs))

    result = df.copy()
    result["rsi"] = rsi
    return result


@transform
def calculate_adx(
    df: pd.DataFrame,
    *,
    period: int = 14,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate ADX (Average Directional Index) indicator."""
    if df.empty or len(df) < period:
        result = df.copy()
        result["adx"] = np.nan
        return result

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.mask(plus_dm < 0, 0)
    minus_dm = minus_dm.mask(minus_dm < 0, 0)

    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / di_sum
    adx = dx.rolling(window=period, min_periods=period).mean()

    result = df.copy()
    result["adx"] = adx
    return result


@transform
def calculate_recent_return(
    df: pd.DataFrame,
    *,
    lookback: int = 5,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate recent return over lookback periods."""
    if df.empty or len(df) < lookback:
        result = df.copy()
        result["recent_return"] = np.nan
        return result

    recent_return = df["close"].pct_change(periods=lookback)
    result = df.copy()
    result["recent_return"] = recent_return
    return result


@transform
def calculate_future_return(
    df: pd.DataFrame,
    *,
    forward: int = 5,
    convert_type: ConvertType = ConvertType.RETURN,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_target")]:
    """Calculate future return or direction as target variable."""
    if df.empty or len(df) < forward:
        result = df.copy()
        result["target"] = np.nan
        return result

    future_close = df["close"].shift(-forward)
    current_close = df["close"]

    if convert_type == ConvertType.RETURN:
        target = (future_close - current_close) / current_close
    elif convert_type == ConvertType.LOG_RETURN:
        target = np.log(future_close / current_close)
    elif convert_type == ConvertType.DIRECTION:
        target = (future_close > current_close).astype(float)
    else:
        raise ValueError(f"Unsupported convert_type: {convert_type}")

    result = df.copy()
    result["target"] = target
    return result


@transform
def calculate_volatility(
    df: pd.DataFrame,
    *,
    window: int = 20,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate rolling volatility (standard deviation of returns)."""
    if df.empty or len(df) < window:
        result = df.copy()
        result["volatility"] = np.nan
        return result

    returns = df["close"].pct_change()
    volatility = returns.rolling(window=window, min_periods=window).std()

    result = df.copy()
    result["volatility"] = volatility
    return result


@transform
def clean_and_align(
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> Annotated[
    tuple[pd.DataFrame, pd.DataFrame],
    Check("algo_trade_dtype.checks.check_aligned_data"),
]:
    """Align features and target DataFrames by removing NaN rows."""
    if features.empty or target.empty:
        return features, target

    combined: pd.DataFrame = pd.concat([features, target], axis=1)
    combined_clean = combined.dropna()

    feature_cols = features.columns.tolist()
    target_cols = target.columns.tolist()

    features_clean = combined_clean[feature_cols].copy()
    target_clean = combined_clean[target_cols].copy()

    assert isinstance(features_clean, pd.DataFrame)
    assert isinstance(target_clean, pd.DataFrame)

    return features_clean, target_clean


__all__ = [
    "resample_ohlcv",
    "calculate_rsi",
    "calculate_adx",
    "calculate_recent_return",
    "calculate_future_return",
    "calculate_volatility",
    "clean_and_align",
]
