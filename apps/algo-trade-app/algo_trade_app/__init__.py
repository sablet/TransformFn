"""Algorithmic trading pipeline application (algo_trade_v3 port)."""

from algo_trade_dtype.registry import register_all_types

__version__ = "0.1.0"

register_all_types()

__all__ = ["__version__", "register_all_types"]
