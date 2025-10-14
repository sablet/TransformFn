"""Default Example/Check registrations for project data types."""

from __future__ import annotations

from functools import lru_cache

from xform_core import Check, ExampleValue, register_check, register_example

from .checks import (
    check_feature_map,
    check_hlocv_dataframe_length,
    check_hlocv_dataframe_notnull,
    check_market_regime_known,
)
from .hlocv_spec import HLOCVSpec, gen_hlocv
from .types import FeatureMap, MarketRegime


_SPEC_KEY = f"{HLOCVSpec.__module__}.{HLOCVSpec.__qualname__}"
_FRAME_KEY = "pandas.core.frame.DataFrame"
_FEATURE_MAP_KEY = f"{FeatureMap.__module__}.{FeatureMap.__name__}"
_REGIME_KEY = f"{MarketRegime.__module__}.{MarketRegime.__qualname__}"


_DEFAULT_SPEC = HLOCVSpec(n=32, seed=42)
_DEFAULT_FRAME_EXAMPLE = ExampleValue(
    value=gen_hlocv(_DEFAULT_SPEC),
    description="synthetic_hlocv_frame",
)
_DEFAULT_SPEC_EXAMPLE = ExampleValue(
    value=_DEFAULT_SPEC,
    description="raw_hlocv_spec",
)
_DEFAULT_FEATURE_MAP_EXAMPLE = ExampleValue(
    value={
        "mean_return": 0.05,
        "volatility": 0.12,
        "sharpe_ratio": 0.4,
        "drawdown": 0.1,
    },
    description="synthetic_feature_map",
)
_DEFAULT_REGIME_EXAMPLE = ExampleValue(
    value=MarketRegime.BULL,
    description="default_market_regime",
)


@lru_cache(maxsize=1)
def register_defaults() -> None:
    """Register default Example/Check metadata for project data types."""

    # NOTE: ExampleValue[T] is invariant, causing mypy errors
    # for subtype assignments. These are intentional runtime
    # registrations for type-specific examples.
    register_example(_SPEC_KEY, _DEFAULT_SPEC_EXAMPLE)  # type: ignore[arg-type]
    register_example(_FRAME_KEY, _DEFAULT_FRAME_EXAMPLE)
    register_example(_FEATURE_MAP_KEY, _DEFAULT_FEATURE_MAP_EXAMPLE)  # type: ignore[arg-type]
    register_example(_REGIME_KEY, _DEFAULT_REGIME_EXAMPLE)  # type: ignore[arg-type]

    register_check(
        _FRAME_KEY,
        Check(
            f"{check_hlocv_dataframe_length.__module__}.{check_hlocv_dataframe_length.__name__}"
        ),
    )
    register_check(
        _FRAME_KEY,
        Check(
            f"{check_hlocv_dataframe_notnull.__module__}.{check_hlocv_dataframe_notnull.__name__}"
        ),
    )
    register_check(
        _FEATURE_MAP_KEY,
        Check(f"{check_feature_map.__module__}.{check_feature_map.__name__}"),
    )
    register_check(
        _REGIME_KEY,
        Check(
            f"{check_market_regime_known.__module__}.{check_market_regime_known.__name__}"
        ),
    )


def registered_keys() -> dict[str, tuple[str, ...]]:
    """Return a snapshot of registered example/check keys for diagnostics."""

    return {
        "examples": (_SPEC_KEY, _FRAME_KEY, _FEATURE_MAP_KEY, _REGIME_KEY),
        "checks": (
            f"{check_hlocv_dataframe_length.__module__}.{check_hlocv_dataframe_length.__name__}",
            f"{check_hlocv_dataframe_notnull.__module__}.{check_hlocv_dataframe_notnull.__name__}",
            f"{check_feature_map.__module__}.{check_feature_map.__name__}",
            f"{check_market_regime_known.__module__}.{check_market_regime_known.__name__}",
        ),
    }


__all__ = [
    "register_defaults",
    "registered_keys",
]
