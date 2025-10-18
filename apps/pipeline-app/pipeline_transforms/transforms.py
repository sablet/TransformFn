"""Core transform function implementations."""

from __future__ import annotations

import math
from typing import Annotated, cast

import pandas as pd

from pipeline_dtype import FeatureMap, HLOCVSpec, gen_hlocv
from xform_core import Check, transform

_ANNUALIZATION_FACTOR = 252.0


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


@transform
def generate_price_bars(spec: HLOCVSpec) -> pd.DataFrame:
    """Materialize synthetic HLOCV bars from the provided specification."""
    return gen_hlocv(spec)


@transform
def compute_feature_map(
    bars: pd.DataFrame,
    *,
    annualization_factor: float = _ANNUALIZATION_FACTOR,
) -> FeatureMap:
    """Compute basic statistical features from price bars."""
    return _calculate_feature_map(bars, annualization_factor=annualization_factor)


@transform
def select_top_features(
    features: FeatureMap,
    *,
    top_n: int = 2,
    minimum_score: float = 0.0,
) -> Annotated[
    list[str],
    Check("pipeline_dtype.checks.ensure_non_empty_selections"),
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

    from pipeline_dtype.checks import ensure_non_empty_selections

    ensure_non_empty_selections(selected)
    return selected


@transform
def merge_feature_maps(
    features_a: FeatureMap,
    features_b: FeatureMap,
    *,
    prefix_a: str = "a_",
    prefix_b: str = "b_",
) -> Annotated[
    FeatureMap,
    Check("pipeline_dtype.checks.ensure_merged_features_non_empty"),
]:
    """Merge two feature maps with prefixes to avoid key collisions (n=2, m=1)."""
    merged: dict[str, float] = {}
    for key, value in features_a.items():
        merged[prefix_a + key] = cast(float, value)
    for key, value in features_b.items():
        merged[prefix_b + key] = cast(float, value)

    from pipeline_dtype.checks import ensure_merged_features_non_empty

    ensure_merged_features_non_empty(cast(FeatureMap, merged))
    return cast(FeatureMap, merged)


@transform
def compute_weighted_score(
    features: FeatureMap,
    weights: Annotated[FeatureMap, Check("algo_trade_dtypes.checks.check_feature_map")],
    *,
    normalize: bool = True,
) -> Annotated[
    float,
    Check("pipeline_dtype.checks.ensure_weighted_score_finite"),
]:
    """Compute weighted score from features and weights (n=3 with kwargs, m=1)."""
    total_score = 0.0
    total_weight = 0.0

    for name, feature_value in features.items():
        weight = cast(float, weights.get(name, 0.0))
        total_score += cast(float, feature_value) * weight
        total_weight += weight

    if normalize and total_weight > 0:
        return total_score / total_weight
    return total_score


@transform
def split_features_by_threshold(
    features: FeatureMap,
    threshold: float = 0.0,
) -> Annotated[
    tuple[FeatureMap, FeatureMap],
    Check("pipeline_dtype.checks.ensure_split_output_valid"),
]:
    """Split features into high and low based on threshold (n=2, m=2 as tuple)."""
    high_features: dict[str, float] = {}
    low_features: dict[str, float] = {}

    for name, value in features.items():
        score = cast(float, value)
        if score >= threshold:
            high_features[name] = score
        else:
            low_features[name] = score

    result = (cast(FeatureMap, high_features), cast(FeatureMap, low_features))

    from pipeline_dtype.checks import ensure_split_output_valid

    ensure_split_output_valid(result)
    return result


@transform
def compute_simple_stats(bars: pd.DataFrame) -> FeatureMap:
    """Compute basic stats from price bars (auto Check補完のテスト: m=1)."""
    return _calculate_feature_map(bars)


@transform
def extract_top_and_rest(
    features: FeatureMap,
    *,
    top_n: int = 1,
) -> tuple[FeatureMap, FeatureMap]:
    """Extract top N features and rest (auto Check補完のテスト: m=2 tuple)."""
    sorted_items = sorted(
        features.items(), key=lambda x: cast(float, x[1]), reverse=True
    )
    top_features: dict[str, float] = {
        k: cast(float, v) for k, v in sorted_items[:top_n]
    }
    rest_features: dict[str, float] = {
        k: cast(float, v) for k, v in sorted_items[top_n:]
    }

    return (cast(FeatureMap, top_features), cast(FeatureMap, rest_features))

