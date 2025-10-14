"""Transforms that intentionally violate checks for demonstration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from proj_dtypes.types import FeatureMap
from xform_core import (
    Check,
    ExampleValue,
    register_check,
    register_example,
    transform,
)


@dataclass(slots=True)
class _Tr004Input:
    """Dummy payload used to simulate mismatched ExampleValue registration."""

    value: int


@dataclass(slots=True)
class _Tr007Output:
    """Output type with improperly registered check target for TR007."""

    score: float


@dataclass(slots=True)
class _Tr008Output:
    """Output type pointing to a non-existent check target for TR008."""

    score: float


register_example(
    f"{_Tr004Input.__module__}.{_Tr004Input.__qualname__}",
    ExampleValue("invalid_payload"),
)
register_check(
    f"{_Tr007Output.__module__}.{_Tr007Output.__qualname__}",
    Check("proj_dtypes"),
)
register_check(
    f"{_Tr008Output.__module__}.{_Tr008Output.__qualname__}",
    Check("proj_dtypes.checks.non_existing_check"),
)


@transform
def produce_invalid_feature_map(bars: pd.DataFrame) -> FeatureMap:
    """Return values that break the FeatureMap check to showcase violations."""

    return {
        "mean_return": float("nan"),
        "volatility": math.inf,
        "sharpe_ratio": float("nan"),
        "drawdown": -1.0,
    }


@transform
def tr001_missing_first_argument() -> FeatureMap:
    """入力引数が存在しないため TR001 を誘発する。"""

    return {}


@transform
def tr002_non_annotated_input(bars: object) -> FeatureMap:
    """第1引数に型注釈が存在しないため TR002 を誘発する。"""

    return {}


@transform
def tr003_missing_example_metadata(bars: int) -> FeatureMap:
    """int 型に ExampleValue が登録されていないため TR003 を誘発する。"""

    return {}


@transform
def tr004_incompatible_example_value(data: _Tr004Input) -> FeatureMap:
    """ExampleValue の型が _Tr004Input と互換でないため TR004 を誘発する。"""

    return {}


@transform
def tr005_non_annotated_return(bars: pd.DataFrame):
    """戻り値に型注釈が存在しないため TR005 を誘発する。"""

    return {}


@transform
def tr006_missing_check_metadata(bars: pd.DataFrame) -> FeatureMap:
    """自動補完されたチェックに違反させて VIOLATION を誘発する。"""

    return {}


@transform
def tr007_non_literal_check_target(bars: pd.DataFrame) -> _Tr007Output:
    """Check のターゲットが FQN でないため TR007 を誘発する。"""

    return _Tr007Output(score=0.0)


@transform
def tr008_missing_check_target(bars: pd.DataFrame) -> _Tr008Output:
    """Check が存在しない関数を指すため TR008 を誘発する。"""

    return _Tr008Output(score=0.0)


@transform
def tr009_missing_docstring(bars: pd.DataFrame) -> FeatureMap:
    return {}
