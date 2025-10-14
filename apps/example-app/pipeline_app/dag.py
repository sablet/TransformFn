"""DAG description for the sample pipeline application."""

from __future__ import annotations

from proj_dtypes.hlocv_spec import HLOCVSpec
from xform_core import Pipeline
from xform_core.pipeline import Node, _resolve_transform

from . import transforms

__all__ = ["DEFAULT_PIPELINE_SPEC", "PIPELINE"]

DEFAULT_PIPELINE_SPEC = HLOCVSpec(n=128, seed=99)

PIPELINE = Pipeline(
    nodes=(
        Node(
            name="price_bars",
            func=transforms.generate_price_bars,
            transform=_resolve_transform(transforms.generate_price_bars),
            parameters=(("spec", DEFAULT_PIPELINE_SPEC),),
        ),
        Node(
            name="feature_map",
            func=transforms.compute_feature_map,
            transform=_resolve_transform(transforms.compute_feature_map),
            inputs=(("bars", "price_bars"),),
            parameters=(("annualization_factor", 252.0),),
        ),
        Node(
            name="top_features",
            func=transforms.select_top_features,
            transform=_resolve_transform(transforms.select_top_features),
            inputs=(("features", "feature_map"),),
            parameters=(("top_n", 3),),
        ),
    ),
)
