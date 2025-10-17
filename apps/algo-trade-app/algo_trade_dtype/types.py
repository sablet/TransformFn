"""algo-trade-app向けの型定義集。"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Annotated, List, Dict, TypeAlias, TypedDict

import pandas as pd

from xform_core import Check

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

RankPercent = Annotated[float, Check["algo_trade_dtype.checks.ensure_rank_percent"]]
"""Rank percentage in [0.0, 1.0] range."""


class PositionSignal(IntEnum):
    """Position signal enumeration."""

    SHORT = -1
    FLAT = 0
    LONG = 1


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


class TimeSeriesSplitConfig(TypedDict, total=False):
    """sklearn TimeSeriesSplit のクロスバリデーション設定。

    対応する sklearn パラメータ:
        n_splits: 分割数（デフォルト: 5）
        test_size: 各テストセットのサンプル数（固定、None の場合は自動計算）
        gap: train と test の間のギャップサンプル数（デフォルト: 0）

    参考: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    """

    n_splits: int
    test_size: int | None
    gap: int


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


class PerformanceMetrics(TypedDict):
    """ポートフォリオパフォーマンス指標。"""

    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float


class PredictionData(TypedDict):
    """予測値と実績値を含むデータスキーマ。"""

    date: str
    currency_pair: str
    prediction: float
    actual_return: float


class RankedPredictionData(TypedDict):
    """ランク付けされた予測データスキーマ。"""

    date: str
    currency_pair: str
    prediction: float
    actual_return: float
    prediction_rank_pct: RankPercent


class SelectedCurrencyData(TypedDict):
    """選択された通貨ペアデータスキーマ。"""

    date: str
    currency_pair: str
    prediction: float
    actual_return: float
    prediction_rank_pct: RankPercent
    signal: PositionSignal


class SimulationResult(TypedDict):
    """シミュレーション結果スキーマ。"""

    date: list[str]
    portfolio_return: list[float]
    n_positions: list[int]


# Market Data Ingestion Phase で使用される型定義
class SwapDataSource(StrEnum):
    """スワップデータソース識別子。"""

    FRED_POLICY_RATE = "fred_policy_rate"
    MANUAL = "manual"


class SpreadCalculationMethod(StrEnum):
    """スプレッド計算方法。"""

    CONSTANT = "constant"
    BID_ASK = "bid_ask"


class TradingCostConfig(TypedDict, total=False):
    """取引コスト計算設定。"""

    swap_source: SwapDataSource
    swap_cache_dir: str
    spread_method: SpreadCalculationMethod
    spread_constant_ratio: float | None


class SelectedCurrencyDataWithCosts(TypedDict):
    """取引コスト付きの選択通貨データ。"""

    date: str
    currency_pair: str
    prediction: float
    actual_return: float
    prediction_rank_pct: RankPercent
    signal: PositionSignal
    swap_rate: float
    spread_cost: float
    adjusted_return: float


class MarketDataProvider(StrEnum):
    """市場データプロバイダ識別子。"""

    YAHOO = "yahoo"
    CCXT = "ccxt"


class CCXTExchange(StrEnum):
    """CCXT で利用する取引所識別子。"""

    BINANCE = "binance"
    BYBIT = "bybit"
    KRAKEN = "kraken"


class YahooFinanceConfig(TypedDict):
    """Yahoo Finance データ取得設定。"""

    tickers: List[str]  # 例: ["AAPL", "MSFT"]
    start_date: str  # ISO8601 (YYYY-MM-DD)
    end_date: str  # ISO8601 (YYYY-MM-DD)
    frequency: Frequency  # 最小粒度: 1日 (Yahoo Finance の制約)
    use_adjusted_close: bool


class CCXTConfig(TypedDict):
    """CCXT データ取得設定。"""

    symbols: List[str]  # 例: ["BTC/USDT", "ETH/USDT"]
    start_date: str  # ISO8601 (YYYY-MM-DD)
    end_date: str  # ISO8601 (YYYY-MM-DD)
    frequency: Frequency  # 最小粒度: 1分 (取引所による)
    exchange: CCXTExchange
    rate_limit_ms: int  # レート制限 (ミリ秒)


class MarketDataIngestionConfig(TypedDict, total=False):
    """プロバイダ横断の取得条件（使用するプロバイダのみ指定）。"""

    yahoo: YahooFinanceConfig  # オプショナル
    ccxt: CCXTConfig  # オプショナル


class ProviderOHLCVBatch(TypedDict):
    """単一シンボルの取得結果とメタ情報。"""

    provider: MarketDataProvider
    symbol: str
    frame: pd.DataFrame  # 各行は FXDataSchema (OHLCVSchema) に準拠
    frequency: Frequency


class ProviderBatchCollection(TypedDict):
    """特定プロバイダの一括取得結果。"""

    provider: MarketDataProvider
    batches: List[ProviderOHLCVBatch]


class NormalizedOHLCVBundle(TypedDict):
    """正規化済み OHLCV データと メタデータ。

    frame: 正規化済み DataFrame
        - 列: timestamp, provider, symbol, open, high, low, close, volume
        - provider 列は MarketDataProvider enum の文字列表現
    metadata: リサンプリング設定などのメタ情報
    """

    frame: (
        pd.DataFrame
    )  # 列: timestamp, provider, symbol, open, high, low, close, volume
    metadata: Dict[str, str]


class MarketDataSnapshotMeta(TypedDict):
    """永続化済みスナップショットのメタ情報。

    注: この型は persist_market_data_snapshot の出力として、
    スナップショットの記録・追跡・監査に使用される。
    データ読み込み時は storage_path を直接 load_market_data に渡す。
    """

    snapshot_id: str
    record_count: int
    storage_path: str
    created_at: str  # ISO8601 (UTC)


# Feature Engineering Phase 用の型定義
class MultiAssetOHLCVFrame(TypedDict):
    """MultiIndex OHLCV DataFrame と関連メタ情報をまとめた構造体。

    frame:
        Index が (timestamp, symbol) の MultiIndex で、
        列は OHLCV スキーマ（open/high/low/close/volume など）に準拠する
        pandas.DataFrame。
    symbols:
        frame に含まれるシンボルのリスト（例: ["AAPL", "BTC/USDT"]）。
    providers:
        データ取得元プロバイダ識別子のリスト（例: ["yahoo", "ccxt"]）。
    """

    frame: pd.DataFrame
    symbols: List[str]
    providers: List[str]


FeatureFrame: TypeAlias = pd.DataFrame
"""特徴量DataFrame（数値型列のみ、インジケータ計算による適度な欠損を許容）

Structure:
- Index: DatetimeIndex (timestamp)
- Columns: Flattened "{symbol}_{indicator}" format
  - Examples: "USDJPY_rsi_14", "SPY_rsi_20", "GOLD_adx_10"
  - Selected from MultiAssetOHLCVFrame["frame"] via select_features()

Note: Column names include both symbol and parameter for cross-asset modeling.
"""

TargetFrame: TypeAlias = pd.DataFrame
"""ターゲットDataFrame（target列のみ、数値型）

Structure:
- Index: DatetimeIndex (timestamp)
- Columns: Single "target" column
  - Extracted from specific asset via extract_target()
  - Example: USDJPY's future_return_5 → "target" column
"""

AlignedFeatureTarget: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]
"""整列済み特徴量とターゲットのタプル（インデックス一致、欠損値なし）"""

CVSplits: TypeAlias = list[tuple[list[int], list[int]]]
"""CV分割の型定義: [(train_indices, validation_indices), ...]"""


__all__ = [
    "HLOCV_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "PriceBarsFrame",
    "RankPercent",
    "PositionSignal",
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
    "PerformanceMetrics",
    "PredictionData",
    "RankedPredictionData",
    "SelectedCurrencyData",
    "SimulationResult",
    "SwapDataSource",
    "SpreadCalculationMethod",
    "TradingCostConfig",
    "SelectedCurrencyDataWithCosts",
    # Market Data Ingestion Phase
    "MarketDataProvider",
    "CCXTExchange",
    "YahooFinanceConfig",
    "CCXTConfig",
    "MarketDataIngestionConfig",
    "ProviderOHLCVBatch",
    "ProviderBatchCollection",
    "NormalizedOHLCVBundle",
    "MultiAssetOHLCVFrame",
    "MarketDataSnapshotMeta",
    # Feature Engineering Phase
    "FeatureFrame",
    "TargetFrame",
    "AlignedFeatureTarget",
]
