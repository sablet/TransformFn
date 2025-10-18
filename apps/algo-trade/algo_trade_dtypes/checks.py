"""algo-trade向け検証関数集。"""

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

from .types import (
    CCXTExchange,
    CCXTConfig,
    Frequency,
    HLOCV_COLUMN_ORDER,
    MarketDataIngestionConfig,
    MarketDataProvider,
    MultiAssetOHLCVFrame,
    NormalizedOHLCVBundle,
    PositionSignal,
    PRICE_COLUMNS,
    ProviderBatchCollection,
    ProviderOHLCVBatch,
    VOLUME_COLUMN,
    MarketDataSnapshotMeta,
    MarketRegime,
    YahooFinanceConfig,
)


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


def check_finite_float(value: float) -> None:
    """有限の浮動小数点数であることを検証する（符号不問）。"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric value, got {type(value)}")

    if not math.isfinite(value):
        raise ValueError(f"Value must be finite, got {value}")


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


def ensure_rank_percent(value: float) -> None:
    """RankPercent が 0.0-1.0 範囲内であることを検証。"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric value, got {type(value)}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"RankPercent must be in [0.0, 1.0], got {value}")


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


def _validate_signal_value(signal: object, item_index: int) -> float:
    """シグナル値を検証し、数値に変換する共通ヘルパー関数。

    Args:
        signal: PositionSignal enum または数値
        item_index: エラーメッセージ用のアイテムインデックス

    Returns:
        検証済みのシグナル数値 (-1, 0, 1)

    Raises:
        TypeError: signal が PositionSignal または数値でない場合
        ValueError: signal 値が有効範囲外の場合
    """
    if isinstance(signal, PositionSignal):
        signal_value = float(signal.value)
    elif isinstance(signal, (int, float)):
        signal_value = float(signal)
    else:
        raise TypeError(f"Item {item_index}: signal must be PositionSignal or numeric")

    valid_signals = {-1, 0, 1, -1.0}
    if signal_value not in valid_signals:
        msg = (
            f"Item {item_index}: signal must be SHORT(-1), FLAT(0), or LONG(1), "
            f"got {signal_value}"
        )
        raise ValueError(msg)

    return signal_value


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

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} must be dict, got {type(item)}")

        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            raise ValueError(f"Item {i}: missing required keys: {missing_keys}")

        signal = item["signal"]
        _validate_signal_value(signal, i)


def _validate_trading_costs(item: dict, item_index: int, signal_value: float) -> None:
    """取引コスト項目の検証ヘルパー関数。"""
    # swap_rate validation
    swap_rate = item["swap_rate"]
    if not isinstance(swap_rate, (int, float)):
        raise TypeError(f"Item {item_index}: swap_rate must be numeric")
    if not math.isfinite(swap_rate):
        raise ValueError(f"Item {item_index}: swap_rate must be finite")

    # spread_cost validation (must be non-negative)
    spread_cost = item["spread_cost"]
    if not isinstance(spread_cost, (int, float)):
        raise TypeError(f"Item {item_index}: spread_cost must be numeric")
    if not math.isfinite(spread_cost):
        raise ValueError(f"Item {item_index}: spread_cost must be finite")
    if spread_cost < 0:
        raise ValueError(
            f"Item {item_index}: spread_cost must be non-negative, got {spread_cost}"
        )


def _validate_adjusted_return(
    item: dict, item_index: int, signal_value: float, tolerance: float
) -> None:
    """adjusted_return の計算整合性を検証する関数。"""
    actual_return = item["actual_return"]
    adjusted_return = item["adjusted_return"]
    swap_rate = item["swap_rate"]
    spread_cost = item["spread_cost"]

    expected_adjusted_return = (
        actual_return + signal_value * swap_rate - abs(signal_value) * spread_cost
    )

    if not math.isclose(adjusted_return, expected_adjusted_return, abs_tol=tolerance):
        msg = (
            f"Item {item_index}: adjusted_return calculation mismatch. "
            f"Expected {expected_adjusted_return:.10f}, got {adjusted_return:.10f} "
            f"(diff: {abs(adjusted_return - expected_adjusted_return):.10e})"
        )
        raise ValueError(msg)


def check_selected_currencies_with_costs(data: list) -> None:
    """取引コスト付き選択通貨データの検証。

    Validation:
        1. 必須キー存在確認
        2. signal が PositionSignal の列挙値
        3. swap_rate と spread_cost が非負
        4. prediction_rank_pct が RankPercent を満たす
        5. adjusted_return の計算整合性（許容誤差 1e-6）:
           adjusted_return ≈ actual_return + signal.value * swap_rate
                             - abs(signal.value) * spread_cost
    """
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
        "swap_rate",
        "spread_cost",
        "adjusted_return",
    }
    tolerance = 1e-6

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} must be dict, got {type(item)}")

        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            raise ValueError(f"Item {i}: missing required keys: {missing_keys}")

        # Signal validation
        signal = item["signal"]
        signal_value = _validate_signal_value(signal, i)

        # Trading costs validation
        _validate_trading_costs(item, i, signal_value)

        # prediction_rank_pct validation
        rank_pct = item["prediction_rank_pct"]
        if not isinstance(rank_pct, (int, float)):
            raise TypeError(f"Item {i}: prediction_rank_pct must be numeric")
        if not (0 <= rank_pct <= 1):
            raise ValueError(
                f"Item {i}: prediction_rank_pct must be in [0, 1], got {rank_pct}"
            )

        # adjusted_return calculation integrity check
        _validate_adjusted_return(item, i, signal_value, tolerance)


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


# Market Data Ingestion Phase 用のチェック関数


def _validate_config_common_fields(
    config: Mapping[str, object], config_type: str
) -> None:
    """共通フィールドの検証。"""
    if not isinstance(config, dict):
        raise TypeError(f"{config_type} config must be a dict")

    # 日付範囲の検証 (共通)
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    if start_date and end_date:
        if not isinstance(start_date, str) or not isinstance(end_date, str):
            raise TypeError(f"{config_type}: start_date and end_date must be strings")
        if start_date >= end_date:
            raise ValueError(f"{config_type}: start_date must be less than end_date")


def _check_yahoo_config(yahoo_config: YahooFinanceConfig) -> None:
    """Yahoo Finance 設定の検証。"""
    _validate_config_common_fields(yahoo_config, "yahoo")

    # tickers の検証
    tickers = yahoo_config.get("tickers", [])
    if not isinstance(tickers, list) or not all(isinstance(x, str) for x in tickers):
        raise TypeError("tickers must be a list of strings")

    # frequency の検証 (Yahoo Finance は最小粒度が日足)
    freq = yahoo_config.get("frequency")
    if freq:
        if isinstance(freq, str):
            freq = Frequency(freq)
        if freq not in [Frequency.DAY_1, Frequency.HOUR_1, Frequency.HOUR_4]:
            # Yahoo Finance最小粒度検証は実装時に行う
            pass  # ここでは具体的な制約を設けない


def _check_ccxt_config(ccxt_config: CCXTConfig) -> None:
    """CCXT 設定の検証。"""
    _validate_config_common_fields(ccxt_config, "ccxt")

    # symbols の検証
    symbols = ccxt_config.get("symbols", [])
    if not isinstance(symbols, list) or not all(isinstance(x, str) for x in symbols):
        raise TypeError("symbols must be a list of strings")

    # exchange の検証
    exchange = ccxt_config.get("exchange")
    if exchange and not isinstance(CCXTExchange(exchange), CCXTExchange):
        raise TypeError("exchange must be a valid CCXTExchange value")

    # rate_limit_ms の検証
    rate_limit = ccxt_config.get("rate_limit_ms", 1000)
    if not isinstance(rate_limit, int) or rate_limit < 0:
        raise TypeError("rate_limit_ms must be a non-negative integer")


def check_ingestion_config(config: MarketDataIngestionConfig) -> None:
    """日付範囲、対象シンボル、周波数設定の妥当性を検証。"""
    # 少なくとも yahoo または ccxt のいずれかが存在することを確認
    has_yahoo = "yahoo" in config
    has_ccxt = "ccxt" in config

    if not has_yahoo and not has_ccxt:
        raise ValueError(
            "At least one of 'yahoo' or 'ccxt' must be specified in the config"
        )

    if has_yahoo:
        _check_yahoo_config(config["yahoo"])

    if has_ccxt:
        _check_ccxt_config(config["ccxt"])


def check_batch_collection(collection: ProviderBatchCollection) -> None:
    """各プロバイダの取得結果が最低要件を満たすか検証。"""
    # provider 名の正当性
    provider = collection.get("provider")
    if provider is None:
        raise TypeError("provider cannot be None")
    if not isinstance(MarketDataProvider(provider), MarketDataProvider):
        raise TypeError(f"Invalid provider: {provider}")

    # batches の検証
    batches = collection.get("batches", [])
    if not isinstance(batches, list):
        raise TypeError("batches must be a list")

    for i, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise TypeError(f"Batch {i} must be a dict")
        check_provider_batch(batch)


def _check_provider_batch_frame(frame: pd.DataFrame) -> None:
    """DataFrame構造の検証。"""
    if frame.empty:
        return

    # timestamp が昇順であることを検証
    if "timestamp" in frame.columns:
        timestamp = frame["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(timestamp):
            raise TypeError("timestamp column must be datetime-like")
        if not timestamp.is_monotonic_increasing:
            raise ValueError("timestamps must be monotonic increasing")

    # OHLCV 列の存在を検証
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_columns if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in frame: {missing_cols}")

    # volume の有限性を検証
    if "volume" in frame.columns:
        volumes = frame["volume"].to_numpy()
        if np.any(~np.isfinite(volumes)):
            raise ValueError("volume must contain finite values")
        if np.any(volumes < 0):
            raise ValueError("volume must be non-negative")


def check_provider_batch(batch: ProviderOHLCVBatch) -> None:
    """個別バッチの DataFrame 構造を検証。"""
    # provider の検証
    provider = batch.get("provider")
    if provider is None:
        raise TypeError("provider cannot be None")
    if not isinstance(MarketDataProvider(provider), MarketDataProvider):
        raise TypeError(f"Invalid provider: {provider}")

    # symbol の検証
    symbol = batch.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise TypeError("symbol must be a non-empty string")

    # frame (DataFrame) の検証
    frame = batch.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame)}")

    _check_provider_batch_frame(frame)


def _validate_column_not_empty(frame: pd.DataFrame, column: str) -> None:
    """指定された列がnullや空文字でないことを検証。"""
    if column in frame.columns:
        series = frame[column]
        if series.isna().any() or (series == "").any():
            raise ValueError(f"{column} column must not contain null or empty values")


def _validate_positive_prices(frame: pd.DataFrame) -> None:
    """価格列がすべて正の値であることを検証。"""
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if col in frame.columns:
            if (frame[col] <= 0).any():
                raise ValueError(f"{col} column must contain positive values only")


def _check_normalized_bundle_frame(frame: pd.DataFrame) -> None:
    """正規化されたDataFrame構造の検証。"""
    if frame.empty:
        return

    # timestamp 列の検証
    if "timestamp" in frame.columns:
        timestamp = frame["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(timestamp):
            raise TypeError("timestamp column must be datetime-like")

    # provider, symbol の非空検証
    _validate_column_not_empty(frame, "provider")
    _validate_column_not_empty(frame, "symbol")

    # 価格 > 0 の検証
    _validate_positive_prices(frame)


def _validate_base_config_fields(
    config: Mapping[str, object], required_fields: list[str], config_type: str
) -> None:
    """基本的な設定フィールドの検証。"""
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    # 必須フィールドの検証
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # 型の検証は各関数で個別に行う


def _validate_config_type(
    config: Mapping[str, object], field: str, expected_type: type, error_msg: str
) -> None:
    """設定項目の型を検証する共通関数。"""
    if not isinstance(config.get(field), expected_type):
        raise TypeError(error_msg)


def check_yahoo_config(config: YahooFinanceConfig) -> None:
    """YahooFinanceConfigに対するチェック関数。"""
    required_fields = ["tickers", "start_date", "end_date", "frequency"]
    _validate_base_config_fields(config, required_fields, "yahoo")

    # 型の検証
    _validate_config_type(config, "tickers", list, "tickers must be a list")
    _validate_config_type(config, "start_date", str, "start_date must be a str")
    _validate_config_type(config, "end_date", str, "end_date must be a str")
    _validate_config_type(
        config, "frequency", Frequency, "frequency must be a Frequency enum"
    )


def check_ccxt_config(config: CCXTConfig) -> None:
    """CCXTConfigに対するチェック関数。"""
    required_fields = ["symbols", "start_date", "end_date", "frequency", "exchange"]
    _validate_base_config_fields(config, required_fields, "ccxt")

    # 型の検証
    _validate_config_type(config, "symbols", list, "symbols must be a list")
    _validate_config_type(config, "start_date", str, "start_date must be a str")
    _validate_config_type(config, "end_date", str, "end_date must be a str")
    _validate_config_type(
        config, "frequency", Frequency, "frequency must be a Frequency enum"
    )
    _validate_config_type(
        config, "exchange", CCXTExchange, "exchange must be a CCXTExchange enum"
    )


def check_normalized_bundle(bundle: NormalizedOHLCVBundle) -> None:
    """正規化済み DataFrame の一貫性を検証。"""
    # DataFrame 構造: timestamp, provider, symbol, open, high, low, close, volume
    frame = bundle.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame)}")

    required_columns = [
        "timestamp",
        "provider",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    missing_cols = [col for col in required_columns if col not in frame.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in normalized frame: {missing_cols}"
        )

    _check_normalized_bundle_frame(frame)


def _validate_multiindex_structure(frame: pd.DataFrame) -> None:
    """MultiIndex 構造を検証するヘルパー関数。"""
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("frame must have a MultiIndex with (timestamp, symbol)")

    # MultiIndex のレベル数を確認
    _EXPECTED_LEVELS = 2
    if frame.index.nlevels != _EXPECTED_LEVELS:
        raise ValueError(f"frame index must have exactly {_EXPECTED_LEVELS} levels")

    # MultiIndex の名前を確認
    level_names = frame.index.names
    if level_names != ["timestamp", "symbol"]:
        msg = f"frame index names must be ['timestamp', 'symbol'], got {level_names}"
        raise ValueError(msg)


def _validate_multiasset_columns(frame: pd.DataFrame) -> None:
    """MultiAsset の列とデータ型を検証するヘルパー関数。"""
    # 列セットの検証 (OHLCV スキーマ準拠)
    required_columns = {"open", "high", "low", "close", "volume"}
    actual_columns = set(frame.columns)
    missing_cols = required_columns - actual_columns
    if missing_cols:
        raise ValueError(f"Missing required columns in frame: {missing_cols}")

    # dtype 検証
    for col in required_columns:
        if col in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                msg = f"Column '{col}' must be numeric, got {frame[col].dtype}"
                raise TypeError(msg)

    # timestamp レベルの型を確認
    timestamp_index = frame.index.get_level_values("timestamp")
    if not pd.api.types.is_datetime64_any_dtype(timestamp_index):
        raise TypeError("timestamp level must be datetime-like")


def _validate_multiasset_metadata(
    frame: pd.DataFrame, metadata: Mapping[str, object]
) -> None:
    """MultiAsset のメタデータ整合性を検証するヘルパー関数。"""
    symbols = metadata.get("symbols")
    if not isinstance(symbols, list) or not all(
        isinstance(symbol, str) and symbol for symbol in symbols
    ):
        raise TypeError("symbols must be a list of non-empty strings")

    providers = metadata.get("providers")
    if not isinstance(providers, list) or not all(
        isinstance(provider, str) and provider for provider in providers
    ):
        raise TypeError("providers must be a list of non-empty strings")

    frame_symbols = set(
        str(symbol) for symbol in frame.index.get_level_values("symbol")
    )
    if frame_symbols and set(symbols) != frame_symbols:
        raise ValueError("symbols metadata must match frame index symbols")

    if "provider" in frame.columns:
        frame_providers = set(frame["provider"].astype(str).unique())
        if frame_providers and set(providers) != frame_providers:
            raise ValueError("providers metadata must match frame providers")


def check_multiasset_frame(
    frame_input: MultiAssetOHLCVFrame | pd.DataFrame,
) -> None:
    """MultiIndex DataFrame とメタ情報の整合性を検証する。"""
    metadata: Mapping[str, object] | None
    if isinstance(frame_input, pd.DataFrame):
        frame = frame_input
        metadata = None
    elif isinstance(frame_input, Mapping):
        metadata = frame_input
        frame = metadata.get("frame")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"frame must be a pandas DataFrame, got {type(frame)}")
    else:
        msg = "MultiAssetOHLCVFrame must be a pandas DataFrame or mapping with 'frame'"
        raise TypeError(msg)

    if frame.empty:
        return

    # 構造とデータの検証
    _validate_multiindex_structure(frame)
    _validate_multiasset_columns(frame)

    # メタ情報の検証（存在する場合）
    if metadata is not None:
        _validate_multiasset_metadata(frame, metadata)


def check_snapshot_meta(meta: MarketDataSnapshotMeta) -> None:
    """永続化メタ情報の整合性を検証。"""
    # snapshot_id の検証
    snapshot_id = meta.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise TypeError("snapshot_id must be a non-empty string")

    # record_count の検証
    record_count = meta.get("record_count")
    if not isinstance(record_count, int) or record_count < 0:
        raise TypeError("record_count must be a non-negative integer")

    # storage_path の検証
    storage_path = meta.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path:
        raise TypeError("storage_path must be a non-empty string")

    # created_at の検証
    created_at = meta.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise TypeError("created_at must be a non-empty string in ISO8601 format")


def check_feature_frame(df: pd.DataFrame) -> None:
    """特徴量DataFrameの検証（数値型列のみ、欠損値は許容）

    インジケータ計算（RSI、移動平均等）では初期期間に欠損が発生するため、
    欠損値チェックは clean_and_align 後の AlignedFeatureTarget で行う。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    if df.empty:
        return

    # 全列が数値型であることを確認
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric")


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
    "check_finite_float",
    "ensure_rank_percent",
    "check_ranked_predictions",
    "check_selected_currencies",
    "check_selected_currencies_with_costs",
    "check_simulation_result",
    "check_performance_metrics",
    # Market Data Ingestion
    "check_ingestion_config",
    "check_batch_collection",
    "check_provider_batch",
    "check_normalized_bundle",
    "check_multiasset_frame",
    "check_snapshot_meta",
    # Feature Engineering
    "check_feature_frame",
]
