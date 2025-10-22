"""Algo Trade Transforms Package (Layer 2: Business Logic).

This package implements @transform decorated functions that handle data
transformations, feature engineering, model training, and simulation for the
algorithmic trading pipeline.

Dependencies:
    - xform-core (shared infrastructure)
    - algo_trade_dtypes (Layer 1: type definitions and checks)

Dependents:
    - algo_trade_dag (Layer 3: pipeline orchestration)

Architecture:
    core → dtypes → transforms → dag

Modules:
    - market_data: OHLCV data ingestion and preprocessing
    - training: Feature engineering and model training
    - simulation: Prediction ranking and buy scenario simulation

Note: This package automatically registers all dtypes on import.
"""

from algo_trade_dtypes.registry import register_all_types

__version__ = "0.1.0"

register_all_types()

__all__ = ["__version__", "register_all_types"]
