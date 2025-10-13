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


__all__ = ["produce_invalid_feature_map"]
