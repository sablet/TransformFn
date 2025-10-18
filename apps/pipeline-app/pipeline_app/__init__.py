"""Sample pipeline application showcasing TransformFn usage."""

from algo_trade_dtypes.registry import register_all_types

register_all_types()

# This package is intentionally minimal - public APIs are:
# - pipeline_app.transforms: @transform decorated functions
# - pipeline_app.dag: PIPELINE instance and DEFAULT_PIPELINE_SPEC
