from __future__ import annotations

from typing import TypedDict

import pandas as pd

from algo_trade_dtypes.generators import HLOCVSpec
from algo_trade_dtypes.registry import DataFrameReg, HLOCVSpecReg
from xform_core import TransformFn, transform


class SpecWrapper(TypedDict):
    spec: HLOCVSpec


def test_auto_annotation_uses_registered_examples() -> None:
    # Register HLOCVSpec examples and DataFrame checks for this test
    HLOCVSpecReg.register()
    DataFrameReg.register()

    def generate(spec: HLOCVSpec) -> pd.DataFrame:
        """Auto-annotation should pull ExampleValue from registry."""

        return pd.DataFrame()

    decorated = transform(generate)
    transform_fn = decorated.__transform_fn__
    assert isinstance(transform_fn, TransformFn)
    assert transform_fn.input_metadata
    assert transform_fn.output_checks
