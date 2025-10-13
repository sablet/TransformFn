"""Transforms that intentionally violate checks for demonstration."""

from __future__ import annotations

import math
from typing import Annotated

import pandas as pd

from proj_dtypes.hlocv_spec import HLOCVSpec, gen_hlocv
from proj_dtypes.types import FeatureMap
from xform_core import Check, ExampleValue, transform

_FAULTY_EXAMPLE_BARS = gen_hlocv(HLOCVSpec(n=12, seed=23))


@transform
def produce_invalid_feature_map(
    bars: Annotated[pd.DataFrame, ExampleValue(_FAULTY_EXAMPLE_BARS.copy(deep=True))],
) -> Annotated[FeatureMap, Check("proj_dtypes.checks.check_feature_map")]:
    """Return values that break the FeatureMap check to showcase violations."""

    return {
        "mean_return": float("nan"),
        "volatility": math.inf,
        "sharpe_ratio": float("nan"),
        "drawdown": -1.0,
    }


__all__ = [
    "produce_invalid_feature_map",
    "tr001_missing_first_argument",
    "tr002_non_annotated_input",
    "tr003_missing_example_metadata",
    "tr004_incompatible_example_value",
    "tr005_non_annotated_return",
    "tr006_missing_check_metadata",
    "tr007_non_literal_check_target",
    "tr008_missing_check_target",
    "tr009_missing_docstring",
]


@transform
def tr001_missing_first_argument() -> Annotated[
    FeatureMap, Check("proj_dtypes.checks.check_feature_map")
]:
    """Input 引数が存在しないため TR001 を誘発する。"""

    return {}
@transform
def tr002_non_annotated_input(
    bars: pd.DataFrame,
) -> Annotated[FeatureMap, Check("proj_dtypes.checks.check_feature_map")]:
    """最初の引数が Annotated ではないため TR002 を誘発する。"""

    return {}


@transform
def tr003_missing_example_metadata(
    bars: Annotated[pd.DataFrame, "no_example"],
) -> Annotated[FeatureMap, Check("proj_dtypes.checks.check_feature_map")]:
    """例示メタデータが欠落しているため TR003 を誘発する。"""

    return {}


@transform
def tr004_incompatible_example_value(
    bars: Annotated[pd.DataFrame, ExampleValue(42)],
) -> Annotated[FeatureMap, Check("proj_dtypes.checks.check_feature_map")]:
    """ExampleValue が DataFrame と互換でないため TR004 を誘発する。"""

    return {}


@transform
def tr005_non_annotated_return(
    bars: Annotated[
        pd.DataFrame, ExampleValue(_FAULTY_EXAMPLE_BARS.copy(deep=True))
    ]
) -> FeatureMap:
    """戻り値が Annotated ではないため TR005 を誘発する。"""

    return {}


@transform
def tr006_missing_check_metadata(
    bars: Annotated[
        pd.DataFrame, ExampleValue(_FAULTY_EXAMPLE_BARS.copy(deep=True))
    ]
) -> Annotated[FeatureMap, "no_check"]:
    """Check メタデータが欠落しているため TR006 を誘発する。"""

    return {}


@transform
def tr007_non_literal_check_target(
    bars: Annotated[
        pd.DataFrame, ExampleValue(_FAULTY_EXAMPLE_BARS.copy(deep=True))
    ]
) -> Annotated[FeatureMap, Check("proj_dtypes")]:
    """Check が FQN 文字列リテラルでないため TR007 を誘発する。"""

    return {}


@transform
def tr008_missing_check_target(
    bars: Annotated[
        pd.DataFrame, ExampleValue(_FAULTY_EXAMPLE_BARS.copy(deep=True))
    ]
) -> Annotated[
    FeatureMap, Check("proj_dtypes.checks.non_existing_check")
]:
    """Check が存在しない関数を指すため TR008 を誘発する。"""

    return {}


@transform
def tr009_missing_docstring(
    bars: Annotated[
        pd.DataFrame, ExampleValue(_FAULTY_EXAMPLE_BARS.copy(deep=True))
    ]
) -> Annotated[FeatureMap, Check("proj_dtypes.checks.check_feature_map")]:
    return {}
