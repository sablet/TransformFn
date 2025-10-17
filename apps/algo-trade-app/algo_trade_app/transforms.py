"""Pure computation Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd

from xform_core import Check, transform
from xform_core.type_metadata import ExampleValue

from algo_trade_dtype.types import ConvertType
from algo_trade_dtype.generators import gen_sample_ohlcv
from algo_trade_dtype import checks


@transform
def resample_ohlcv(
    df: Annotated[pd.DataFrame, ExampleValue[pd.DataFrame](gen_sample_ohlcv())],
    *,
    freq: str = "1h",
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Resample OHLCV DataFrame to specified frequency."""
    if df.empty:
        return df

    # Work with MultiIndex if present (MultiAssetOHLCVFrame structure)
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        # Multi-asset DataFrame with MultiIndex (timestamp, symbol)
        # Resample each symbol group separately
        resampled_data = {}
        for symbol in df.index.get_level_values(1).unique():
            symbol_data = df.xs(symbol, level=1)
            if not isinstance(symbol_data.index, pd.DatetimeIndex):
                if "timestamp" in symbol_data.columns:
                    symbol_data = symbol_data.set_index("timestamp")
                else:
                    raise TypeError(
                        f"Symbol {symbol} DataFrame must have DatetimeIndex or 'timestamp' column for resampling"
                    )
            
            resampled_symbol = symbol_data.resample(freq).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            )
            for col in resampled_symbol.columns:
                resampled_data[(symbol, col)] = resampled_symbol[col]
        
        if resampled_data:
            result = pd.DataFrame(resampled_data)
            result.index.names = ["timestamp"]
            result = result.reorder_levels([1, 0], axis=1)  # (symbol, column)
            result = result.sort_index(axis=1)  # Sort by symbol, then column
        else:
            result = df  # Return original if no symbols found
    else:
        # Single asset DataFrame
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
    df: Annotated[pd.DataFrame, ExampleValue[pd.DataFrame](gen_sample_ohlcv())],
    *,
    period: int = 14,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate RSI (Relative Strength Index) indicator.

    Adds a new column named "rsi_{period}" to the DataFrame.
    Multiple calls with different periods create separate columns.
    """
    if df.empty or len(df) < period:
        result = df.copy()
        # For MultiIndex structure, we need to add the column for each symbol
        if isinstance(df.index, pd.MultiIndex):
            for symbol in df.index.get_level_values(1).unique():
                df[(symbol, f"rsi_{period}")] = np.nan
            return df
        else:
            result[f"rsi_{period}"] = np.nan
            return result

    result = df.copy()
    
    # Work with MultiIndex if present (MultiAssetOHLCVFrame structure)
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        # Calculate RSI for each symbol in the MultiIndex
        for symbol in df.index.get_level_values(1).unique():
            # Extract close prices for the specific symbol
            try:
                close_prices = df[(symbol, "close")]
            except KeyError:
                continue  # Skip if 'close' column doesn't exist for this symbol
            
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss_series = loss.rolling(window=period, min_periods=period).mean()
            avg_loss_replaced = avg_loss_series.replace(0, np.nan)

            rs = avg_gain / avg_loss_replaced
            rsi = 100 - (100 / (1 + rs))

            # Add the RSI column for this symbol with parameter in the name
            result[(symbol, f"rsi_{period}")] = rsi
    else:
        # Single asset DataFrame
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss_series = loss.rolling(window=period, min_periods=period).mean()
        avg_loss_replaced = avg_loss_series.replace(0, np.nan)

        rs = avg_gain / avg_loss_replaced
        rsi = 100 - (100 / (1 + rs))

        result[f"rsi_{period}"] = rsi

    return result


@transform
def calculate_adx(
    df: Annotated[pd.DataFrame, ExampleValue[pd.DataFrame](gen_sample_ohlcv())],
    *,
    period: int = 14,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate ADX (Average Directional Index) indicator.

    Adds a new column named "adx_{period}" to the DataFrame.
    Multiple calls with different periods create separate columns.
    """
    if df.empty or len(df) < period:
        result = df.copy()
        # For MultiIndex structure, we need to add the column for each symbol
        if isinstance(df.index, pd.MultiIndex):
            for symbol in df.index.get_level_values(1).unique():
                df[(symbol, f"adx_{period}")] = np.nan
            return df
        else:
            result[f"adx_{period}"] = np.nan
            return result

    result = df.copy()

    # Work with MultiIndex if present (MultiAssetOHLCVFrame structure)
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        # Calculate ADX for each symbol in the MultiIndex
        for symbol in df.index.get_level_values(1).unique():
            # Extract OHLC prices for the specific symbol
            try:
                high = df[(symbol, "high")]
                low = df[(symbol, "low")]
                close = df[(symbol, "close")]
            except KeyError:
                continue  # Skip if required columns don't exist for this symbol

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

            # Add the ADX column for this symbol with parameter in the name
            result[(symbol, f"adx_{period}")] = adx
    else:
        # Single asset DataFrame
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

        result[f"adx_{period}"] = adx

    return result


@transform
def calculate_recent_return(
    df: Annotated[pd.DataFrame, ExampleValue[pd.DataFrame](gen_sample_ohlcv())],
    *,
    lookback: int = 5,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate recent return over lookback periods.

    Adds a new column named "recent_return_{lookback}" to the DataFrame.
    Multiple calls with different lookbacks create separate columns.
    """
    if df.empty or len(df) < lookback:
        result = df.copy()
        # For MultiIndex structure, we need to add the column for each symbol
        if isinstance(df.index, pd.MultiIndex):
            for symbol in df.index.get_level_values(1).unique():
                df[(symbol, f"recent_return_{lookback}")] = np.nan
            return df
        else:
            result[f"recent_return_{lookback}"] = np.nan
            return result

    result = df.copy()

    # Work with MultiIndex if present (MultiAssetOHLCVFrame structure)
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        # Calculate recent return for each symbol in the MultiIndex
        for symbol in df.index.get_level_values(1).unique():
            # Extract close prices for the specific symbol
            try:
                close_prices = df[(symbol, "close")]
            except KeyError:
                continue  # Skip if 'close' column doesn't exist for this symbol

            recent_return = close_prices.pct_change(periods=lookback)

            # Add the recent_return column for this symbol with parameter in the name
            result[(symbol, f"recent_return_{lookback}")] = recent_return
    else:
        # Single asset DataFrame
        recent_return = df["close"].pct_change(periods=lookback)
        result[f"recent_return_{lookback}"] = recent_return

    return result


@transform
def calculate_future_return(
    df: Annotated[pd.DataFrame, ExampleValue[pd.DataFrame](gen_sample_ohlcv())],
    *,
    forward: int = 5,
    convert_type: ConvertType = ConvertType.RETURN,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_target")]:
    """Calculate future return or direction as target variable.

    Adds a new column named "target" to the DataFrame.
    """
    if df.empty or len(df) < forward:
        result = df.copy()
        # For MultiIndex structure, we need to add the target column for each symbol
        if isinstance(df.index, pd.MultiIndex):
            for symbol in df.index.get_level_values(1).unique():
                result[(symbol, "target")] = np.nan
        else:
            result["target"] = np.nan
        return result

    result = df.copy()

    # Work with MultiIndex if present (MultiAssetOHLCVFrame structure)
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        # Calculate future return for each symbol in the MultiIndex
        # For the target variable, we'll use the first symbol or a specific symbol if needed
        for symbol in df.index.get_level_values(1).unique():
            # Extract close prices for the specific symbol
            try:
                close_prices = df[(symbol, "close")]
            except KeyError:
                continue  # Skip if 'close' column doesn't exist for this symbol

            future_close = close_prices.shift(-forward)
            current_close = close_prices

            if convert_type == ConvertType.RETURN:
                target = (future_close - current_close) / current_close
            elif convert_type == ConvertType.LOG_RETURN:
                target = np.log(future_close / current_close)
            elif convert_type == ConvertType.DIRECTION:
                target = (future_close > current_close).astype(float)
            else:
                raise ValueError(f"Unsupported convert_type: {convert_type}")

            # Add the target column for this symbol
            result[(symbol, "target")] = target
    else:
        # Single asset DataFrame
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

        result["target"] = target

    return result


@transform
def calculate_volatility(
    df: Annotated[pd.DataFrame, ExampleValue[pd.DataFrame](gen_sample_ohlcv())],
    *,
    window: int = 20,
) -> Annotated[pd.DataFrame, Check("algo_trade_dtype.checks.check_ohlcv")]:
    """Calculate rolling volatility (standard deviation of returns).

    Adds a new column named "volatility_{window}" to the DataFrame.
    Multiple calls with different windows create separate columns.
    """
    if df.empty or len(df) < window:
        result = df.copy()
        # For MultiIndex structure, we need to add the column for each symbol
        if isinstance(df.index, pd.MultiIndex):
            for symbol in df.index.get_level_values(1).unique():
                df[(symbol, f"volatility_{window}")] = np.nan
            return df
        else:
            result[f"volatility_{window}"] = np.nan
            return result

    result = df.copy()

    # Work with MultiIndex if present (MultiAssetOHLCVFrame structure)
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        # Calculate volatility for each symbol in the MultiIndex
        for symbol in df.index.get_level_values(1).unique():
            # Extract close prices for the specific symbol
            try:
                close_prices = df[(symbol, "close")]
            except KeyError:
                continue  # Skip if 'close' column doesn't exist for this symbol

            returns = close_prices.pct_change()
            volatility = returns.rolling(window=window, min_periods=window).std()

            # Add the volatility column for this symbol with parameter in the name
            result[(symbol, f"volatility_{window}")] = volatility
    else:
        # Single asset DataFrame
        returns = df["close"].pct_change()
        volatility = returns.rolling(window=window, min_periods=window).std()

        result[f"volatility_{window}"] = volatility

    return result


def select_features(
    df: pd.DataFrame,
    feature_specs: list[tuple[str, str] | tuple[str, str, int] | tuple[str, str, int, ...]],
) -> pd.DataFrame:
    """Select specific indicators from specific assets for cross-asset modeling.

    This function enables flexible feature selection across multiple assets,
    preventing look-ahead bias by excluding price/volume data and allowing
    fine-grained control over which indicators to use for each asset.

    Helper function for pipeline composition - not a @transform function.

    Args:
        df: MultiAssetOHLCVFrame with MultiIndex[(symbol, column)] structure
        feature_specs: List of feature specifications in multiple formats:
            - 3-tuple (recommended): (symbol, indicator, param)
              Examples: ("USDJPY", "rsi", 14), ("SPY", "adx", 10)
            - 2-tuple (shorthand): (symbol, "indicator_param")
              Examples: ("USDJPY", "rsi_14"), ("SPY", "rsi_20")
            - N-tuple (multi-param): (symbol, indicator, param1, param2, ...)
              Example: ("USDJPY", "bollinger", 20, 2)

    Returns:
        FeatureFrame with only selected columns, flattened to
        "{symbol}_{indicator}_{params}" column names for ML compatibility

    Examples:
        >>> # Recommended: 3-tuple format with numeric parameters
        >>> features = select_features(df, [
        ...     ("USDJPY", "rsi", 14),
        ...     ("USDJPY", "adx", 14),
        ...     ("SPY", "rsi", 20),
        ...     ("GOLD", "adx", 10),
        ... ])
        >>> # Result columns: ["USDJPY_rsi_14", "USDJPY_adx_14", "SPY_rsi_20", "GOLD_adx_10"]

        >>> # Shorthand: 2-tuple format
        >>> features = select_features(df, [
        ...     ("USDJPY", "rsi_14"),
        ...     ("SPY", "rsi_20"),
        ... ])

        >>> # Dynamic parameter exploration (3-tuple format is natural)
        >>> for period in [4, 8, 14, 20]:
        ...     specs = [("USDJPY", "rsi", period)]
        ...     features = select_features(df, specs)
    """
    if df.empty:
        return pd.DataFrame()

    selected_data = {}
    for spec in feature_specs:
        if len(spec) == 2:
            # Shorthand: ("USDJPY", "rsi_14")
            symbol, indicator_with_param = spec
            col_name = f"{symbol}_{indicator_with_param}"
            df_col_name = indicator_with_param
        else:
            # Recommended: ("USDJPY", "rsi", 14) or ("USDJPY", "bollinger", 20, 2)
            symbol, indicator, *params = spec
            param_str = "_".join(map(str, params))
            df_col_name = f"{indicator}_{param_str}"
            col_name = f"{symbol}_{df_col_name}"

        # Use tuple to access MultiIndex column: (symbol, df_col_name)
        selected_data[col_name] = df[(symbol, df_col_name)]

    return pd.DataFrame(selected_data, index=df.index)


def extract_target(
    df: pd.DataFrame,
    symbol: str,
    indicator: str | None = None,
    column: str | None = None,
    **params,
) -> pd.DataFrame:
    """Extract target variable for a specific asset.

    This function selects the prediction target from a single asset,
    enabling cross-asset feature scenarios where features come from
    multiple assets but the target is from one specific asset.

    Helper function for pipeline composition - not a @transform function.

    Args:
        df: MultiAssetOHLCVFrame with MultiIndex[(symbol, column)] structure
        symbol: Symbol to use as prediction target (e.g., "USDJPY")
        indicator: Indicator name (e.g., "future_return") - used with params
        column: Full column name (e.g., "future_return_5") - shorthand format
        **params: Indicator parameters (e.g., forward=5)

    Returns:
        TargetFrame with single column named "target"

    Examples:
        >>> # Recommended: indicator + params format
        >>> target = extract_target(df, symbol="USDJPY", indicator="future_return", forward=5)
        >>> # Result: DataFrame with single "target" column

        >>> # Shorthand: column format
        >>> target = extract_target(df, symbol="USDJPY", column="future_return_5")

        >>> # Multi-parameter example
        >>> target = extract_target(df, symbol="USDJPY", indicator="bollinger_upper", window=20, std_dev=2)
    """
    if df.empty:
        return pd.DataFrame()

    if column is not None:
        # Shorthand format: column="future_return_5"
        df_col_name = column
    elif indicator is not None:
        # Recommended format: indicator="future_return", forward=5
        param_str = "_".join(str(v) for v in params.values())
        df_col_name = f"{indicator}_{param_str}" if param_str else indicator
    else:
        raise ValueError("Either 'indicator' or 'column' must be specified")

    # Use tuple to access MultiIndex column: (symbol, df_col_name)
    target_series = df[(symbol, df_col_name)]
    return pd.DataFrame({"target": target_series}, index=df.index)


def clean_and_align(
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
