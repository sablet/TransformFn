"""Algo-trade specific pipeline skeleton definitions."""

from xform_core.dag.skeleton import PipelineSkeleton, PipelineStep, register_skeleton

# Import types for skeleton definition
from algo_trade_dtypes.types import (
    MarketDataIngestionConfig,
    ProviderBatchCollection,
    NormalizedOHLCVBundle,
    MultiAssetOHLCVFrame,
    MarketDataSnapshotMeta,
)

# Phase 1 Skeleton: Market Data Ingestion
phase1_skeleton = PipelineSkeleton(
    name="phase1_market_data_ingestion",
    steps=[
        PipelineStep(
            name="fetch_data",
            input_types=(MarketDataIngestionConfig,),
            output_type=ProviderBatchCollection,
            default_transform="algo_trade_transforms.market_data.fetch_yahoo_finance_ohlcv",
        ),
        PipelineStep(
            name="normalize",
            input_types=(ProviderBatchCollection,),
            output_type=NormalizedOHLCVBundle,
            default_transform="algo_trade_transforms.market_data.normalize_multi_provider",
        ),
        PipelineStep(
            name="merge",
            input_types=(NormalizedOHLCVBundle,),
            output_type=MultiAssetOHLCVFrame,
            default_transform="algo_trade_transforms.market_data.merge_market_data_bundle",
        ),
        PipelineStep(
            name="persist",
            input_types=(MultiAssetOHLCVFrame, MarketDataIngestionConfig),
            output_type=MarketDataSnapshotMeta,
            default_transform="algo_trade_transforms.market_data.persist_market_data_snapshot",
        ),
    ],
)

# Register for dynamic lookup
register_skeleton("algo_trade_dag.skeleton.phase1_skeleton", phase1_skeleton)

# Note: phase2, phase3, phase4 skeletons can be added similarly when needed
