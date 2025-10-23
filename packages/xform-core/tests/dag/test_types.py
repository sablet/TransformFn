"""Shared test-specific data types for DAG tests.

These lightweight dataclasses stand in for the real pipeline types while
remaining compatible with the stricter type registry rules (TR010).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HLOCVSpec:
    """Synthetic spec describing how many bars to generate."""

    length: int = 8


@dataclass(frozen=True)
class DataFrame:
    """Lightweight in-memory frame representation for tests."""

    rows: Tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class FeatureMap:
    """Aggregate metrics derived from the synthetic frame."""

    metrics: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metrics is None:
            object.__setattr__(self, "metrics", {"mean": 1.0})


@dataclass(frozen=True)
class FeatureList:
    """Selected feature identifiers."""

    names: Tuple[str, ...] = ("feature_1",)


__all__ = ["HLOCVSpec", "DataFrame", "FeatureMap", "FeatureList"]
