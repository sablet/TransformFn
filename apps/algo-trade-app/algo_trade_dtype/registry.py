"""RegisteredType APIを用いた型メタデータ登録。"""

from __future__ import annotations

import pandas as pd
from xform_core import RegisteredType

from .checks import (
    check_aligned_data,
    check_cv_result,
    check_cv_splits,
    check_feature_frame,
    check_feature_map,
    check_fold_result,
    check_hlocv_dataframe_length,
    check_hlocv_dataframe_notnull,
    check_ingestion_config,
    check_market_regime_known,
    check_ohlcv,
    check_performance_metrics,
    check_prediction_result,
    check_ranked_predictions,
    check_selected_currencies,
    check_simulation_result,
    check_batch_collection,
    check_normalized_bundle,
    check_multiasset_frame,
    check_snapshot_meta,
    check_target,
    check_yahoo_config,
    check_ccxt_config,
)
from .generators import (
    HLOCVSpec,
    gen_aligned_feature_target,
    gen_cv_splits,
    gen_cv_result,
    gen_feature_frame,
    gen_fold_result,
    gen_hlocv,
    gen_multiasset_frame,
    gen_prediction_data,
    gen_ranked_prediction_data,
    gen_sample_ohlcv,
    gen_selected_currency_data,
    gen_simulation_result,
    gen_snapshot_meta,
    gen_target_frame,
    gen_ingestion_config,
    gen_yahoo_only_config,
    gen_ccxt_only_config,
    gen_mixed_frequency_config,
    gen_yahoo_batch_collection,
    gen_ccxt_batch_collection,
    gen_normalized_bundle,
)
from .types import (
    CCXTExchange,
    CCXTConfig,
    CVSplits,
    FeatureMap,
    MarketDataIngestionConfig,
    MarketDataProvider,
    MarketRegime,
    MultiAssetOHLCVFrame,
    NormalizedOHLCVBundle,
    PerformanceMetrics,
    PredictionData,
    PredictionResult,
    ProviderBatchCollection,
    RankedPredictionData,
    SelectedCurrencyData,
    SimulationResult,
    YahooFinanceConfig,
    MarketDataSnapshotMeta,
    FoldResult,
    CVResult,
    TimeSeriesSplitConfig,
)

HLOCVSpecReg = (
    RegisteredType(HLOCVSpec)
    .with_example(HLOCVSpec(n=32, seed=42), "raw_hlocv_spec_default")
    .with_example(HLOCVSpec(n=64, seed=99), "raw_hlocv_spec_large")
    .with_example(
        HLOCVSpec(n=128, sigma=0.02, seed=123), "raw_hlocv_spec_high_volatility"
    )
)

FeatureMapReg: RegisteredType[FeatureMap] = (
    RegisteredType(FeatureMap)
    .with_example(
        {
            "mean_return": 0.05,
            "volatility": 0.12,
            "sharpe_ratio": 0.4,
            "drawdown": 0.1,
        },
        "synthetic_feature_map",
    )
    .with_check(check_feature_map)  # type: ignore[arg-type]
)

MarketRegimeReg = (
    RegisteredType(MarketRegime)
    .with_example(MarketRegime.BULL, "bull_market")
    .with_example(MarketRegime.BEAR, "bear_market")
    .with_example(MarketRegime.SIDEWAYS, "sideways_market")
    .with_check(check_market_regime_known)
)

PredictionResultReg: RegisteredType[PredictionResult] = (
    RegisteredType(PredictionResult)
    .with_example(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "predicted": [0.01, 0.02],
            "actual": [0.015, 0.018],
            "feature_importance": {"rsi": 0.3, "adx": 0.2},
        },
        "synthetic_prediction",
    )
    .with_check(check_prediction_result)  # type: ignore[arg-type]
)

DataFrameReg: RegisteredType[object] = (
    RegisteredType("pandas.core.frame.DataFrame")
    .with_example(gen_hlocv(HLOCVSpec(n=32, seed=42)), "synthetic_hlocv_frame")
    .with_example(gen_sample_ohlcv(n=50, seed=99), "sample_ohlcv_frame")
    .with_check(check_hlocv_dataframe_length)
    .with_check(check_hlocv_dataframe_notnull)
    .with_check(check_ohlcv)
)

AlignedDataReg: RegisteredType[object] = RegisteredType("builtins.tuple").with_check(
    check_aligned_data
)

IntReg: RegisteredType[int] = RegisteredType(int).with_example(100, "sample_int")

SeriesReg: RegisteredType[object] = RegisteredType(
    "pandas.core.series.Series"
).with_example(gen_hlocv(HLOCVSpec(n=32, seed=42))["close"], "sample_series")

# Phase 4: Simulation types
# Note: builtins.list is registered WITHOUT examples to satisfy TR003,
# but actual examples are provided via ExampleValue[callable] in Transform annotations
ListReg: RegisteredType[object] = RegisteredType("builtins.list")

SimulationResultReg: RegisteredType[SimulationResult] = (
    RegisteredType(SimulationResult)
    .with_example(gen_simulation_result(n=3), "simulation_result")
    .with_check(check_simulation_result)  # type: ignore[arg-type]
)

PerformanceMetricsReg: RegisteredType[PerformanceMetrics] = RegisteredType(
    PerformanceMetrics
).with_check(check_performance_metrics)  # type: ignore[arg-type]

# Phase 5: Simulation types
PredictionDataReg: RegisteredType[PredictionData] = (
    RegisteredType(PredictionData)
    .with_example(gen_prediction_data()[0], "sample_prediction_data")
    .with_example(gen_prediction_data()[1], "sample_prediction_data2")
    .with_check(check_ranked_predictions)  # type: ignore[arg-type]
)

RankedPredictionDataReg: RegisteredType[RankedPredictionData] = (
    RegisteredType(RankedPredictionData)
    .with_example(gen_ranked_prediction_data()[0], "sample_ranked_prediction_data")
    .with_example(gen_ranked_prediction_data()[1], "sample_ranked_prediction_data2")
    .with_check(check_ranked_predictions)  # type: ignore[arg-type]
)

SelectedCurrencyDataReg: RegisteredType[SelectedCurrencyData] = (
    RegisteredType(SelectedCurrencyData)
    .with_example(gen_selected_currency_data()[0], "sample_selected_currency_data")
    .with_example(gen_selected_currency_data()[1], "sample_selected_currency_data2")
    .with_check(check_selected_currencies)  # type: ignore[arg-type]
)

# Market Data Ingestion Phase 用のRegisteredType宣言
MarketDataProviderReg = (
    RegisteredType(MarketDataProvider)
    .with_example(MarketDataProvider.YAHOO, "yahoo_provider")
    .with_example(MarketDataProvider.CCXT, "ccxt_provider")
)

CCXTExchangeReg = (
    RegisteredType(CCXTExchange)
    .with_example(CCXTExchange.BINANCE, "binance_exchange")
    .with_example(CCXTExchange.BYBIT, "bybit_exchange")
    .with_example(CCXTExchange.KRAKEN, "kraken_exchange")
)

YahooFinanceConfigReg: RegisteredType[YahooFinanceConfig] = (
    RegisteredType(YahooFinanceConfig)
    .with_example(gen_yahoo_only_config()["yahoo"], "yahoo_config_example")
    .with_check(check_yahoo_config)
)

CCXTConfigReg: RegisteredType[CCXTConfig] = (
    RegisteredType(CCXTConfig)
    .with_example(gen_ccxt_only_config()["ccxt"], "ccxt_config_example")
    .with_check(check_ccxt_config)
)

MarketDataIngestionConfigReg: RegisteredType[MarketDataIngestionConfig] = (
    RegisteredType(MarketDataIngestionConfig)
    .with_example(gen_ingestion_config(), "mixed_ingestion_config")
    .with_example(gen_yahoo_only_config(), "yahoo_only_config")
    .with_example(gen_ccxt_only_config(), "ccxt_only_config")
    .with_example(gen_mixed_frequency_config(), "mixed_frequency_config")
    .with_check(check_ingestion_config)
)

ProviderBatchCollectionReg: RegisteredType[ProviderBatchCollection] = (
    RegisteredType(ProviderBatchCollection)
    .with_example(gen_yahoo_batch_collection(), "yahoo_batch_collection")
    .with_example(gen_ccxt_batch_collection(), "ccxt_batch_collection")
    .with_check(check_batch_collection)
)

NormalizedOHLCVBundleReg: RegisteredType[NormalizedOHLCVBundle] = (
    RegisteredType(NormalizedOHLCVBundle)
    .with_example(gen_normalized_bundle(), "normalized_bundle")
    .with_check(check_normalized_bundle)
)

MultiAssetOHLCVFrameReg: RegisteredType[MultiAssetOHLCVFrame] = (
    RegisteredType(MultiAssetOHLCVFrame)
    .with_example(gen_multiasset_frame(), "multiasset_frame")
    .with_check(check_multiasset_frame)
)

MarketDataSnapshotMetaReg: RegisteredType[MarketDataSnapshotMeta] = (
    RegisteredType(MarketDataSnapshotMeta)
    .with_example(gen_snapshot_meta(), "snapshot_meta")
    .with_check(check_snapshot_meta)
)

# Feature Engineering Phase registered types
MultiAssetOHLCVFeatureFrameReg: RegisteredType[pd.DataFrame] = (
    RegisteredType(pd.DataFrame)
    .with_example(gen_multiasset_frame()["frame"], "multiasset_frame")
    .with_check(check_multiasset_frame)
)

FeatureFrameReg: RegisteredType[pd.DataFrame] = (
    RegisteredType(pd.DataFrame)
    .with_example(gen_feature_frame(), "feature_frame")
    .with_check(check_feature_frame)
)

TargetFrameReg: RegisteredType[pd.DataFrame] = (
    RegisteredType(pd.DataFrame)
    .with_example(gen_target_frame(), "target_frame")
    .with_check(check_target)
)

AlignedFeatureTargetReg: RegisteredType[tuple[pd.DataFrame, pd.DataFrame]] = (
    RegisteredType(tuple)
    .with_example(gen_aligned_feature_target(), "aligned_feature_target")
    .with_check(check_aligned_data)
)

# StringReg for storage_path in load_market_data
base_path = "output/data/snapshots/a3f2c8b1e4d6f9a0/2024-01-01_2024-01-10"
StringReg = RegisteredType(str).with_example(
    f"{base_path}/market.parquet",
    "sample_storage_path",
)

# Phase 3: Training & Prediction types
CVSplitsReg: RegisteredType[CVSplits] = (
    RegisteredType(CVSplits)
    .with_example(gen_cv_splits(), "cv_splits_example")
    .with_check(check_cv_splits)
)

FoldResultReg: RegisteredType[FoldResult] = (
    RegisteredType(FoldResult)
    .with_example(gen_fold_result(), "fold_result_example")
    .with_check(check_fold_result)  # type: ignore[arg-type]
)

CVResultReg: RegisteredType[CVResult] = (
    RegisteredType(CVResult)
    .with_example(gen_cv_result(), "cv_result_example")
    .with_check(check_cv_result)  # type: ignore[arg-type]
)

TimeSeriesSplitConfigReg: RegisteredType[TimeSeriesSplitConfig] = RegisteredType(
    TimeSeriesSplitConfig
).with_example(
    {"n_splits": 5, "test_size": 100, "gap": 10}, "timeseries_split_config_example"
)

ALL_REGISTERED_TYPES = [
    HLOCVSpecReg,
    FeatureMapReg,
    MarketRegimeReg,
    PredictionResultReg,
    DataFrameReg,
    AlignedDataReg,
    IntReg,
    SeriesReg,
    ListReg,
    SimulationResultReg,
    PerformanceMetricsReg,
    PredictionDataReg,
    RankedPredictionDataReg,
    SelectedCurrencyDataReg,
    # Market Data Ingestion
    MarketDataProviderReg,
    CCXTExchangeReg,
    YahooFinanceConfigReg,
    CCXTConfigReg,
    MarketDataIngestionConfigReg,
    ProviderBatchCollectionReg,
    NormalizedOHLCVBundleReg,
    MultiAssetOHLCVFrameReg,
    MarketDataSnapshotMetaReg,
    # Feature Engineering
    MultiAssetOHLCVFeatureFrameReg,
    FeatureFrameReg,
    TargetFrameReg,
    AlignedFeatureTargetReg,
    StringReg,
    # Phase 3: Training & Prediction
    CVSplitsReg,
    FoldResultReg,
    CVResultReg,
    TimeSeriesSplitConfigReg,
]


def register_all_types() -> None:
    """algo-trade-appの型メタデータを全て登録する。"""
    for registered_type in ALL_REGISTERED_TYPES:
        registered_type.register()  # type: ignore[attr-defined]


__all__ = [
    "ALL_REGISTERED_TYPES",
    "AlignedDataReg",
    "DataFrameReg",
    "FeatureMapReg",
    "HLOCVSpecReg",
    "IntReg",
    "ListReg",
    "MarketRegimeReg",
    "PerformanceMetricsReg",
    "PredictionDataReg",
    "PredictionResultReg",
    "RankedPredictionDataReg",
    "SelectedCurrencyDataReg",
    "SeriesReg",
    "SimulationResultReg",
    "MarketDataProviderReg",
    "CCXTExchangeReg",
    "YahooFinanceConfigReg",
    "CCXTConfigReg",
    "MarketDataIngestionConfigReg",
    "ProviderBatchCollectionReg",
    "NormalizedOHLCVBundleReg",
    "MultiAssetOHLCVFrameReg",
    "MultiAssetOHLCVFeatureFrameReg",
    "MarketDataSnapshotMetaReg",
    # Feature Engineering
    "FeatureFrameReg",
    "TargetFrameReg",
    "AlignedFeatureTargetReg",
    "StringReg",
    # Phase 3: Training & Prediction
    "CVSplitsReg",
    "FoldResultReg",
    "CVResultReg",
    "TimeSeriesSplitConfigReg",
    "register_all_types",
]
