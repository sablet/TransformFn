from __future__ import annotations

from xform_core import resolve_checks, resolve_examples

from proj_dtypes import FeatureMap, HLOCVSpec, MarketRegime, register_defaults
from proj_dtypes.registry_setup import registered_keys

_EXPECTED_SAMPLE_ROWS = 32


def _qualname(obj: object) -> str:
    module = getattr(obj, "__module__", obj.__class__.__module__)
    qualname = getattr(obj, "__qualname__", obj.__class__.__name__)
    return f"{module}.{qualname}"


def test_register_defaults_populates_example_registry() -> None:
    register_defaults()

    spec_key = _qualname(HLOCVSpec)
    frame_key = "pandas.core.frame.DataFrame"
    feature_map_key = _qualname(FeatureMap)
    regime_key = _qualname(MarketRegime)

    spec_examples = resolve_examples(spec_key)
    frame_examples = resolve_examples(frame_key)
    feature_examples = resolve_examples(feature_map_key)
    regime_examples = resolve_examples(regime_key)

    assert spec_examples and isinstance(spec_examples[0].value, HLOCVSpec)
    assert frame_examples and frame_examples[0].value.shape[0] == _EXPECTED_SAMPLE_ROWS
    assert feature_examples and "mean_return" in feature_examples[0].value
    assert regime_examples and regime_examples[0].value == MarketRegime.BULL


def test_register_defaults_populates_checks() -> None:
    register_defaults()

    frame_checks = resolve_checks("pandas.core.frame.DataFrame")
    feature_checks = resolve_checks(_qualname(FeatureMap))
    regime_checks = resolve_checks(_qualname(MarketRegime))

    check_targets = {check.target for check in frame_checks}
    assert any("check_hlocv_dataframe_length" in target for target in check_targets)
    assert any("check_hlocv_dataframe_notnull" in target for target in check_targets)

    assert any("check_feature_map" in check.target for check in feature_checks)
    assert any("check_market_regime_known" in check.target for check in regime_checks)


def test_register_defaults_is_idempotent() -> None:
    register_defaults()
    first = resolve_examples("pandas.core.frame.DataFrame")
    register_defaults()
    second = resolve_examples("pandas.core.frame.DataFrame")

    assert len(first) == len(second)
    assert id(first[0]) == id(second[0])
    snapshot = registered_keys()
    assert "examples" in snapshot and "checks" in snapshot
