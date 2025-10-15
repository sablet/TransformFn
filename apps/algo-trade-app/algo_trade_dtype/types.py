"""algo-trade-app向けの型定義集。"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias, TypedDict

import pandas as pd

HLOCV_COLUMN_ORDER: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMN: str = "volume"

PriceBarsFrame: TypeAlias = pd.DataFrame


class FeatureMap(TypedDict, total=False):
    """特徴量名とスコアのマッピング。"""

    mean_return: float
    volatility: float
    sharpe_ratio: float
    drawdown: float


class MarketRegime(StrEnum):
    """想定されるマーケットレジーム列挙。"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class DataProvider(StrEnum):
    """サポートするデータプロバイダ列挙。"""

    GMO_CLICK = "gmo_click"
    OANDA = "oanda"
    DUKASCOPY = "dukascopy"


class CurrencyPair(StrEnum):
    """サポートする通貨ペア列挙。"""

    USD_JPY = "USD_JPY"
    EUR_JPY = "EUR_JPY"
    EUR_USD = "EUR_USD"
    GBP_JPY = "GBP_JPY"
    AUD_JPY = "AUD_JPY"


class Frequency(StrEnum):
    """サポートする時間足列挙。"""

    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAY_1 = "1D"
    WEEK_1 = "1W"


class ConvertType(StrEnum):
    """ターゲット変換種別。"""

    RETURN = "return"
    DIRECTION = "direction"
    LOG_RETURN = "log_return"


class CVMethod(StrEnum):
    """クロスバリデーション方式。"""

    TIME_SERIES = "time_series"
    EXPANDING_WINDOW = "expanding_window"
    SLIDING_WINDOW = "sliding_window"


class OHLCVDataRequest(TypedDict, total=False):
    """OHLCV取得リクエスト仕様。"""

    provider: DataProvider
    currency_pairs: list[CurrencyPair]
    start_date: str
    end_date: str
    frequency: Frequency
    retry_count: int
    timeout_seconds: float


class FXDataSchema(TypedDict):
    """FX OHLCVデータの行スキーマ。"""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureGenerationRequest(TypedDict, total=False):
    """特徴量生成処理の要求仕様。"""

    currency_pair: CurrencyPair
    source_frequency: Frequency
    target_frequency: Frequency
    lookback_periods: int
    forward_periods: int
    convert_type: ConvertType


class TrainPredictRequest(TypedDict, total=False):
    """学習・推論フェーズの要求仕様。"""

    feature_data_id: str
    cv_method: CVMethod
    n_splits: int
    test_size: float
    lgbm_params: dict[str, object]
    output_path: str


class PredictionResult(TypedDict):
    """予測結果スキーマ。"""

    timestamp: list[str]
    predicted: list[float]
    actual: list[float]
    feature_importance: dict[str, float]


class ValidationResult(TypedDict):
    """データ検証結果スキーマ。"""

    is_valid: bool
    missing_count: int
    outlier_count: int
    correlation: float
    message: str


class SimpleCVConfig(TypedDict, total=False):
    """シンプルなクロスバリデーション設定。"""

    method: CVMethod
    n_splits: int
    test_size: int | None
    gap: int


class SimpleLGBMParams(TypedDict, total=False):
    """シンプルな LightGBM パラメータ設定。"""

    num_leaves: int
    learning_rate: float
    feature_fraction: float
    bagging_fraction: float
    min_child_samples: int
    n_estimators: int
    random_state: int


class FoldResult(TypedDict):
    """Fold 単位の学習結果。"""

    fold_id: int
    train_indices: list[int]
    valid_indices: list[int]
    train_score: float
    valid_score: float
    predictions: list[float]
    feature_importance: dict[str, float]


class CVResult(TypedDict):
    """クロスバリデーション結果全体。"""

    fold_results: list[FoldResult]
    mean_score: float
    std_score: float
    oos_predictions: list[float]


__all__ = [
    "HLOCV_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "PriceBarsFrame",
    "FeatureMap",
    "MarketRegime",
    "DataProvider",
    "CurrencyPair",
    "Frequency",
    "ConvertType",
    "CVMethod",
    "OHLCVDataRequest",
    "FXDataSchema",
    "FeatureGenerationRequest",
    "TrainPredictRequest",
    "PredictionResult",
    "ValidationResult",
    "SimpleCVConfig",
    "SimpleLGBMParams",
    "FoldResult",
    "CVResult",
]
