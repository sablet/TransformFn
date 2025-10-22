"""Check functions for pipeline application."""

from __future__ import annotations

import math
from typing import Sequence, cast

from algo_trade_dtypes.types import FeatureMap


def ensure_non_empty_selections(selections: Sequence[str]) -> None:
    """Check callback that enforces non-empty feature selections."""
    if not selections:
        raise ValueError("at least one feature must be selected")
    for item in selections:
        if not isinstance(item, str) or not item:
            raise ValueError("feature selections must be non-empty strings")


def ensure_merged_features_non_empty(merged: FeatureMap) -> None:
    """Check callback that enforces non-empty merged feature map."""
    if not merged:
        raise ValueError("merged feature map must contain at least one entry")


def ensure_weighted_score_finite(score: float) -> None:
    """Check callback that enforces finite weighted score."""
    if not math.isfinite(score):
        raise ValueError("weighted score must be finite")


def ensure_split_output_valid(split: tuple[FeatureMap, FeatureMap]) -> None:
    """Check callback that enforces valid split output."""
    high_features, low_features = split
    if not high_features and not low_features:
        raise ValueError("at least one of high or low features must be non-empty")
