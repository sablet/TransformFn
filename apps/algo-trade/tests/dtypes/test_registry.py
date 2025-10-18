from __future__ import annotations

from xform_core import resolve_checks, resolve_examples

from algo_trade_dtypes.registry import register_all_types
from algo_trade_dtypes.types import FeatureMap


def _qualname(obj: object) -> str:
    module = getattr(obj, "__module__", obj.__class__.__module__)
    qualname = getattr(obj, "__qualname__", obj.__class__.__name__)
    return f"{module}.{qualname}"


def test_register_all_types_populates_registries() -> None:
    register_all_types()

    spec_key = "algo_trade_dtypes.generators.HLOCVSpec"
    frame_key = "pandas.core.frame.DataFrame"
    feature_map_key = _qualname(FeatureMap)

    spec_examples = resolve_examples(spec_key)
    frame_examples = resolve_examples(frame_key)
    feature_examples = resolve_examples(feature_map_key)

    assert spec_examples and spec_examples[0].description
    assert frame_examples and frame_examples[0].value.shape[0] > 0
    assert feature_examples and "mean_return" in feature_examples[0].value

    frame_checks = resolve_checks(frame_key)
    check_targets = {check.target for check in frame_checks}
    assert any("check_hlocv_dataframe_length" in target for target in check_targets)
    assert any("check_ohlcv" in target for target in check_targets)


def test_register_all_types_is_idempotent() -> None:
    register_all_types()
    first = resolve_examples("pandas.core.frame.DataFrame")
    register_all_types()
    second = resolve_examples("pandas.core.frame.DataFrame")

    assert len(first) == len(second)
    assert id(first[0]) == id(second[0])
