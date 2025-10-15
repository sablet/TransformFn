"""Sample TransformFn implementations for the demo pipeline."""

from __future__ import annotations

import math
from typing import Annotated, Sequence, cast

import pandas as pd

from algo_trade_dtype.generators import HLOCVSpec, gen_hlocv
from algo_trade_dtype.types import FeatureMap
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


def ensure_non_empty_selections(selections: Sequence[str]) -> None:
    """Check callback that enforces non-empty feature selections."""

    if not selections:
        raise ValueError("at least one feature must be selected")
    for item in selections:
        if not isinstance(item, str) or not item:
            raise ValueError("feature selections must be non-empty strings")


@transform
def select_top_features(
    features: FeatureMap,
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


def ensure_merged_features_non_empty(merged: FeatureMap) -> None:
    """Check callback that enforces non-empty merged feature map."""

    if not merged:
        raise ValueError("merged feature map must contain at least one entry")


@transform
def merge_feature_maps(
    features_a: FeatureMap,
    features_b: FeatureMap,
    *,
    prefix_a: str = "a_",
    prefix_b: str = "b_",
) -> Annotated[
    FeatureMap,
    Check("pipeline_app.transforms.ensure_merged_features_non_empty"),
]:
    """Merge two feature maps with prefixes to avoid key collisions (n=2, m=1)."""

    merged: dict[str, float] = {}
    for key, value in features_a.items():
        merged[prefix_a + key] = cast(float, value)
    for key, value in features_b.items():
        merged[prefix_b + key] = cast(float, value)
    ensure_merged_features_non_empty(cast(FeatureMap, merged))
    return cast(FeatureMap, merged)


def ensure_weighted_score_finite(score: float) -> None:
    """Check callback that enforces finite weighted score."""

    if not math.isfinite(score):
        raise ValueError("weighted score must be finite")


@transform
def compute_weighted_score(
    features: FeatureMap,
    weights: Annotated[FeatureMap, Check("algo_trade_dtype.checks.check_feature_map")],
    *,
    normalize: bool = True,
) -> Annotated[
    float,
    Check("pipeline_app.transforms.ensure_weighted_score_finite"),
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


def ensure_split_output_valid(split: tuple[FeatureMap, FeatureMap]) -> None:
    """Check callback that enforces valid split output."""

    high_features, low_features = split
    if not high_features and not low_features:
        raise ValueError("at least one of high or low features must be non-empty")


@transform
def split_features_by_threshold(
    features: FeatureMap,
    threshold: float = 0.0,
) -> Annotated[
    tuple[FeatureMap, FeatureMap],
    Check("pipeline_app.transforms.ensure_split_output_valid"),
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
