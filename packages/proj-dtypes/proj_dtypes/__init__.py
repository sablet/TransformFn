"""Project-specific data types, example generators, and validation checks."""

from .checks import (
    check_feature_map,
    check_hlocv_dataframe,
    check_hlocv_dataframe_length,
    check_hlocv_dataframe_notnull,
    check_market_regime_known,
)
from .examples import materialize_example, materialize_value
from .hlocv_spec import HLOCVSpec, gen_hlocv
from .registry_setup import register_defaults
from .types import (
    FeatureMap,
    HLOCV_COLUMN_ORDER,
    MarketRegime,
    PRICE_COLUMNS,
    PriceBarsFrame,
    VOLUME_COLUMN,
)

register_defaults()

__all__ = [
    "FeatureMap",
    "HLOCVSpec",
    "HLOCV_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "MarketRegime",
    "PriceBarsFrame",
    "check_feature_map",
    "check_hlocv_dataframe",
    "check_hlocv_dataframe_length",
    "check_hlocv_dataframe_notnull",
    "check_market_regime_known",
    "gen_hlocv",
    "register_defaults",
    "materialize_example",
    "materialize_value",
]
