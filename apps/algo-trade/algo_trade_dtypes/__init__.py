"""Algo Trade Data Types Package (Layer 1: Foundation).

This package provides the fundamental data types, validation checks, example generators,
and type registration for the algorithmic trading pipeline.

Dependencies:
    - xform-core (shared infrastructure)

Dependents:
    - algo_trade_transforms (Layer 2: Transform functions)
    - algo_trade_dag (Layer 3: Pipeline orchestration)

Architecture:
    core → dtypes → transforms → dag

Exports:
    - Type definitions (FeatureMap, MarketRegime, etc.)
    - Validation checks (check_hlocv_dataframe, check_feature_map, etc.)
    - Example generators (gen_hlocv, gen_sample_ohlcv)
    - Type registration (register_all_types)
"""

from . import checks, generators, registry, types
from .checks import (
    check_aligned_data,
    check_feature_map,
    check_hlocv_dataframe,
    check_hlocv_dataframe_length,
    check_hlocv_dataframe_notnull,
    check_market_regime_known,
    check_ohlcv,
    check_prediction_result,
    check_target,
)
from .generators import HLOCVSpec, gen_hlocv, gen_sample_ohlcv
from .registry import register_all_types
from .types import (
    FeatureMap,
    HLOCV_COLUMN_ORDER,
    MarketRegime,
    PRICE_COLUMNS,
    VOLUME_COLUMN,
)

# Register all types on module import
register_all_types()

__all__ = [
    "checks",
    "generators",
    "registry",
    "types",
    "HLOCVSpec",
    "gen_hlocv",
    "gen_sample_ohlcv",
    "FeatureMap",
    "MarketRegime",
    "HLOCV_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "register_all_types",
    "check_hlocv_dataframe",
    "check_hlocv_dataframe_length",
    "check_hlocv_dataframe_notnull",
    "check_feature_map",
    "check_market_regime_known",
    "check_ohlcv",
    "check_target",
    "check_aligned_data",
    "check_prediction_result",
]
