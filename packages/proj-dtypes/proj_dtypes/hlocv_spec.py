"""Utilities for generating synthetic HLOCV price data."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .types import HLOCV_COLUMN_ORDER, PRICE_COLUMNS, VOLUME_COLUMN

_DEFAULT_START = pd.Timestamp("2024-01-01", tz=None)
_MIN_PRICE = 1e-6


@dataclass(slots=True)
class HLOCVSpec:
    """Specification for generating synthetic price bars.

    Attributes
    ----------
    n:
        Number of rows (time steps) to generate.
    start_price:
        Opening price for the first bar.
    mu:
        Drift component used in the geometric Brownian motion model.
    sigma:
        Volatility component for the geometric Brownian motion model.
    freq:
        Pandas offset alias controlling the timestamp spacing.
    start:
        Timestamp for the first bar. Defaults to 2024-01-01 if not provided.
    tz:
        Optional timezone name applied to the generated timestamps.
    seed:
        RNG seed for deterministic output.
    base_volume:
        Baseline volume used when returns are flat.
    volume_scale:
        Multiplier applied to `abs(returns)` when computing volume.
    volume_jitter:
        Standard deviation of Gaussian noise applied to the volume series.
    spread_range:
        Tuple describing the minimum/maximum fractional spread added to
        high/low prices.
    """

    n: int = 128
    start_price: float = 100.0
    mu: float = 0.0005
    sigma: float = 0.01
    freq: str = "1D"
    start: Optional[pd.Timestamp] = None
    tz: Optional[str] = None
    seed: Optional[int] = None
    base_volume: float = 1_000_000.0
    volume_scale: float = 25.0
    volume_jitter: float = 0.05
    spread_range: tuple[float, float] = (0.001, 0.02)

    def __post_init__(self) -> None:
        _validate_numeric_constraints(self)
        _validate_volume_configuration(self.volume_jitter, self.spread_range)
        _validate_frequency_alias(self.freq)
        _normalize_start(self)
        _validate_timezone(self.tz)


def gen_hlocv(spec: HLOCVSpec) -> pd.DataFrame:
    """Generate a pandas DataFrame following the provided specification."""

    rng = np.random.default_rng(spec.seed)
    index = pd.date_range(start=spec.start, periods=spec.n, freq=spec.freq)
    index = cast(pd.DatetimeIndex, index)
    if spec.tz:
        index = index.tz_localize(spec.tz)

    returns = rng.normal(loc=spec.mu, scale=spec.sigma, size=spec.n)
    open_prices = np.empty(spec.n, dtype=float)
    close_prices = np.empty(spec.n, dtype=float)

    open_prices[0] = max(spec.start_price, _MIN_PRICE)
    close_prices[0] = max(open_prices[0] * exp(returns[0]), _MIN_PRICE)

    for i in range(1, spec.n):
        open_prices[i] = close_prices[i - 1]
        close_prices[i] = max(open_prices[i] * exp(returns[i]), _MIN_PRICE)

    base_max = np.maximum(open_prices, close_prices)
    base_min = np.minimum(open_prices, close_prices)

    spread = rng.uniform(
        low=spec.spread_range[0], high=spec.spread_range[1], size=spec.n
    )
    high_prices = base_max * (1.0 + spread)
    low_prices = base_min * (1.0 - spread)
    low_prices = np.maximum(low_prices, _MIN_PRICE)

    padded_close = np.concatenate(([open_prices[0]], close_prices[:-1]))
    pct_returns = np.abs(
        (close_prices - padded_close) / np.maximum(padded_close, _MIN_PRICE)
    )
    weekday_series = index.to_series().dt.dayofweek
    weekday_factor = 1.0 + weekday_series.to_numpy() / 10.0
    noise = rng.normal(loc=1.0, scale=spec.volume_jitter, size=spec.n)
    volume = spec.base_volume * (1.0 + spec.volume_scale * pct_returns * weekday_factor)
    volume = np.maximum(volume * noise, spec.base_volume * 0.1)
    volume = volume.astype(np.int64)

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        columns=pd.Index(HLOCV_COLUMN_ORDER),
    )

    _validate_generated_frame(frame)
    return frame


def _validate_generated_frame(frame: pd.DataFrame) -> None:
    """Internal assertion helper to guarantee generator invariants."""

    if frame.empty:  # pragma: no cover - safety guard
        raise AssertionError("generated frame should not be empty")

    for column in PRICE_COLUMNS:
        series = frame[column]
        if (series <= 0).any():
            raise AssertionError(f"{column} must remain positive")

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    opens = frame["open"].to_numpy()
    closes = frame["close"].to_numpy()

    if np.any(highs < np.maximum(opens, closes)):
        raise AssertionError("high price violates constraint")
    if np.any(lows > np.minimum(opens, closes)):
        raise AssertionError("low price violates constraint")
    if np.any(frame[VOLUME_COLUMN].to_numpy() <= 0):
        raise AssertionError("volume must be positive")
    if not np.allclose(opens[1:], closes[:-1]):
        raise AssertionError("open price must equal previous close")


__all__ = ["HLOCVSpec", "gen_hlocv"]


def _validate_numeric_constraints(spec: HLOCVSpec) -> None:
    checks = (
        (spec.n > 0, "n must be a positive integer"),
        (spec.start_price > 0, "start_price must be positive"),
        (spec.sigma >= 0, "sigma must be non-negative"),
        (spec.base_volume > 0, "base_volume must be positive"),
        (spec.volume_scale > 0, "volume_scale must be positive"),
    )
    for condition, message in checks:
        if not condition:
            raise ValueError(message)


def _validate_volume_configuration(
    volume_jitter: float, spread_range: tuple[float, float]
) -> None:
    if not 0 <= volume_jitter < 1:
        raise ValueError("volume_jitter must be in [0, 1)")
    low_spread, high_spread = spread_range
    if low_spread < 0 or high_spread <= 0 or low_spread >= high_spread:
        raise ValueError("spread_range must satisfy 0 <= low < high")


def _validate_frequency_alias(freq: str) -> None:
    try:
        to_offset(freq)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"invalid frequency alias: {freq!r}") from exc


def _normalize_start(spec: HLOCVSpec) -> None:
    if spec.start is None:
        object.__setattr__(spec, "start", _DEFAULT_START)
        return
    if not isinstance(spec.start, pd.Timestamp):
        raise TypeError("start must be a pandas.Timestamp or None")


def _validate_timezone(tz: Optional[str]) -> None:
    if tz is not None and not isinstance(tz, str):
        raise TypeError("tz must be a string timezone name or None")
