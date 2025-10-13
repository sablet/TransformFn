"""Domain specific type declarations for project data."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict, TypeAlias

import pandas as pd

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


class FeatureMap(TypedDict, total=False):
    """Represents a mapping of feature names to numeric scores."""

    mean_return: float
    volatility: float
    sharpe_ratio: float
    drawdown: float


PriceBarsFrame: TypeAlias = pd.DataFrame


class MarketRegime(StrEnum):
    """Enumerates supported market regimes in the project domain."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


__all__ = [
    "FeatureMap",
    "HLOCV_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "PriceBarsFrame",
    "MarketRegime",
]
