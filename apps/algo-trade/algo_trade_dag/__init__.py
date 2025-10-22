"""Algo Trade DAG Package (Layer 3: Pipeline Orchestration).

This package orchestrates the complete algorithmic trading pipeline by composing
@transform functions into a directed acyclic graph (DAG).

Dependencies:
    - xform-core (shared infrastructure)
    - algo_trade_dtypes (Layer 1: type definitions and checks)
    - algo_trade_transforms (Layer 2: transform functions)

Dependents:
    - None (top layer)

Architecture:
    core → dtypes → transforms → dag

Future Implementation:
    - DAG composition using TransformFn records
    - Pipeline execution orchestration
    - Result caching and artifact management
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
