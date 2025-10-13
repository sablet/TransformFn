"""Sample TransformFn implementations for the demo pipeline."""

from __future__ import annotations

import math
from typing import Annotated, Sequence, cast

import pandas as pd

from proj_dtypes.hlocv_spec import HLOCVSpec, gen_hlocv
from proj_dtypes.types import FeatureMap
from xform_core import Check, ExampleValue, transform

_ANNUALIZATION_FACTOR = 252.0
_EXAMPLE_SPEC = HLOCVSpec(n=32, seed=11)
_EXAMPLE_BARS = gen_hlocv(_EXAMPLE_SPEC)


def _calculate_feature_map(
    bars: pd.DataFrame, *, annualization_factor: float = _ANNUALIZATION_FACTOR
) -> FeatureMap:
    """Internal helper that computes feature statistics for a price series."""

    returns = bars["close"].pct_change().dropna()

    if returns.empty:
        mean_return = 0.0
        volatility = 0.0
    else:
        mean_raw = float(returns.mean() * annualization_factor)
        vol_raw = float(returns.std(ddof=0) * math.sqrt(annualization_factor))
        mean_return = 0.0 if math.isnan(mean_raw) else mean_raw
        volatility = 0.0 if math.isnan(vol_raw) else vol_raw

    sharpe_ratio = mean_return / volatility if volatility else 0.0
    rolling_max = bars["close"].cummax()
    drawdown_series = (rolling_max - bars["close"]) / rolling_max
    drawdown = float(drawdown_series.max()) if not drawdown_series.empty else 0.0

    features: FeatureMap = {
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "drawdown": drawdown,
    }
    return features


_EXAMPLE_FEATURES = _calculate_feature_map(_EXAMPLE_BARS)


@transform
def generate_price_bars(
    spec: Annotated[
        HLOCVSpec, ExampleValue(HLOCVSpec(n=16, seed=7), description="raw_hlocv_spec")
    ],
) -> Annotated[pd.DataFrame, Check("proj_dtypes.checks.check_hlocv_dataframe")]:
    """Materialize synthetic HLOCV bars from the provided specification."""

    return gen_hlocv(spec)


@transform
def compute_feature_map(
    bars: Annotated[pd.DataFrame, ExampleValue(_EXAMPLE_BARS.copy(deep=True))],
    *,
    annualization_factor: float = _ANNUALIZATION_FACTOR,
) -> Annotated[FeatureMap, Check("proj_dtypes.checks.check_feature_map")]:
    """Compute basic statistical features from price bars."""

    return _calculate_feature_map(bars, annualization_factor=annualization_factor)


def ensure_non_empty_selections(selections: Sequence[str]) -> None:
    """Check callback that enforces non-empty feature selections."""

    if not selections:
        raise ValueError("at least one feature must be selected")
    for item in selections:
        if not isinstance(item, str) or not item:
            raise ValueError("feature selections must be non-empty strings")


@transform
def select_top_features(
    features: Annotated[FeatureMap, ExampleValue(_EXAMPLE_FEATURES.copy())],
    *,
    top_n: int = 2,
    minimum_score: float = 0.0,
) -> Annotated[
    list[str],
    Check("pipeline_app.transforms.ensure_non_empty_selections"),
]:
    """Rank features by score and return the strongest entries."""

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    filtered_pairs: list[tuple[str, float]] = []
    for name, maybe_score in features.items():
        score = cast(float, maybe_score)
        if score >= minimum_score:
            filtered_pairs.append((name, score))
    sorted_pairs = sorted(filtered_pairs, key=lambda item: item[1], reverse=True)
    selected = [name for name, _score in sorted_pairs[:top_n]]

    ensure_non_empty_selections(selected)
    return selected


__all__ = [
    "compute_feature_map",
    "ensure_non_empty_selections",
    "generate_price_bars",
    "select_top_features",
]
