from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "apps" / "pipeline-app"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import pytest  # noqa: E402

from pipeline_app import transforms  # noqa: E402
from algo_trade_dtype.generators import HLOCVSpec  # noqa: E402
from algo_trade_dtype.types import HLOCV_COLUMN_ORDER  # noqa: E402


def test_generate_price_bars_shapes_dataframe() -> None:
    spec = HLOCVSpec(n=24, seed=5)
    frame = transforms.generate_price_bars(spec)

    assert list(frame.columns) == list(HLOCV_COLUMN_ORDER)
    assert len(frame) == spec.n
    assert hasattr(transforms.generate_price_bars, "__transform_fn__")


def test_compute_feature_map_returns_expected_keys() -> None:
    spec = HLOCVSpec(n=48, seed=3)
    bars = transforms.generate_price_bars(spec)
    features = transforms.compute_feature_map(bars, annualization_factor=100.0)

    assert set(features.keys()) == {
        "mean_return",
        "volatility",
        "sharpe_ratio",
        "drawdown",
    }
    assert all(
        isinstance(value, float) and math.isfinite(value) for value in features.values()
    )


def test_select_top_features_respects_top_n() -> None:
    features = {
        "mean_return": 0.5,
        "volatility": 0.25,
        "sharpe_ratio": 1.5,
        "drawdown": 0.1,
    }

    selected = transforms.select_top_features(features, top_n=2)
    assert selected == ["sharpe_ratio", "mean_return"]


@pytest.mark.parametrize(
    "payload",
    [[], [""], [1]],
)
def test_ensure_non_empty_selections_validates(payload: list[object]) -> None:
    with pytest.raises(ValueError):
        transforms.ensure_non_empty_selections(payload)  # type: ignore[arg-type]
