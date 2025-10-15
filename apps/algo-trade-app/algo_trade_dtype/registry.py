"""RegisteredType APIを用いた型メタデータ登録。"""

from __future__ import annotations

from xform_core import RegisteredType

from .checks import (
    check_aligned_data,
    check_feature_map,
    check_hlocv_dataframe_length,
    check_hlocv_dataframe_notnull,
    check_market_regime_known,
    check_ohlcv,
    check_prediction_result,
)
from .generators import HLOCVSpec, gen_hlocv, gen_sample_ohlcv
from .types import FeatureMap, MarketRegime, PredictionResult

HLOCVSpecReg = (
    RegisteredType(HLOCVSpec)
    .with_example(HLOCVSpec(n=32, seed=42), "raw_hlocv_spec_default")
    .with_example(HLOCVSpec(n=64, seed=99), "raw_hlocv_spec_large")
    .with_example(
        HLOCVSpec(n=128, sigma=0.02, seed=123), "raw_hlocv_spec_high_volatility"
    )
)

FeatureMapReg = (
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
    .with_check(check_feature_map)
)

MarketRegimeReg = (
    RegisteredType(MarketRegime)
    .with_example(MarketRegime.BULL, "bull_market")
    .with_example(MarketRegime.BEAR, "bear_market")
    .with_example(MarketRegime.SIDEWAYS, "sideways_market")
    .with_check(check_market_regime_known)
)

PredictionResultReg = (
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
    .with_check(check_prediction_result)
)

DataFrameReg = (
    RegisteredType("pandas.core.frame.DataFrame")
    .with_example(gen_hlocv(HLOCVSpec(n=32, seed=42)), "synthetic_hlocv_frame")
    .with_example(gen_sample_ohlcv(n=50, seed=99), "sample_ohlcv_frame")
    .with_check(check_hlocv_dataframe_length)
    .with_check(check_hlocv_dataframe_notnull)
    .with_check(check_ohlcv)
)

AlignedDataReg = RegisteredType("builtins.tuple").with_check(check_aligned_data)

ALL_REGISTERED_TYPES = [
    HLOCVSpecReg,
    FeatureMapReg,
    MarketRegimeReg,
    PredictionResultReg,
    DataFrameReg,
    AlignedDataReg,
]


def register_all_types() -> None:
    """algo-trade-appの型メタデータを全て登録する。"""
    for registered_type in ALL_REGISTERED_TYPES:
        registered_type.register()


__all__ = [
    "ALL_REGISTERED_TYPES",
    "AlignedDataReg",
    "DataFrameReg",
    "FeatureMapReg",
    "HLOCVSpecReg",
    "MarketRegimeReg",
    "PredictionResultReg",
    "register_all_types",
]
