# proj-dtypes リファクタリング設計書

## 目的

現在の`packages/proj-dtypes/`パッケージを分析した結果、以下の問題が明らかになった:

1. **プロジェクト固有機能の混在**: HLOCV/金融特化の型・チェック・生成器が汎用パッケージに配置されている
2. **汎用機能の不足**: DataFrame検証など、実際には汎用的な機能が抽出されていない
3. **重複コード**: `apps/algo-trade-app`と`proj-dtypes`で同様の機能が重複実装されている
4. **register_defaults()の冗長性**: 手動でFQNを構築し、type: ignoreを多用する100行超のボイラープレート

本リファクタリングでは、以下を実現する:

- **汎用機能をxform-coreへ移動**: DataFrame検証、型メタデータ登録の基盤
- **プロジェクト固有機能をapps/[proj]/dtypeへ分離**: HLOCV関連の全機能を独立したdtypeパッケージに
- **宣言的なレジストリ登録**: `RegisteredType`クラスによるメソッドチェーン方式
- **xform-auditor CLIの互換性維持**: リファクタリング後もaudit機能が正常動作すること

## アーキテクチャ概要

```
┌──────────────────────────────────────────────────────────────┐
│ xform-core (汎用基盤)                                         │
├──────────────────────────────────────────────────────────────┤
│ • @transform デコレータ                                       │
│ • ExampleValue, Check メタ型                                 │
│ • registry (ExampleRegistry, CheckRegistry)                  │
│ • 🆕 type_metadata.py (RegisteredType, make_example)         │
│ • 🆕 checks/dataframe.py (汎用DataFrame検証関数)              │
│ • 🆕 materialization.py (Materializer protocol)              │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ apps/algo-trade-app/algo_trade_dtype/ (型・検証・生成器)      │
├──────────────────────────────────────────────────────────────┤
│ • types.py              # HLOCV + FX型定義                   │
│ • generators.py         # HLOCVSpec, gen_hlocv, etc.        │
│ • checks.py             # check_hlocv_*, check_feature_*    │
│ • materializers.py      # HLOCVSpec→DataFrame変換           │
│ • registry.py           # RegisteredType使用                │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ apps/algo-trade-app/algo_trade_app/ (transformers)           │
├──────────────────────────────────────────────────────────────┤
│ • transforms.py         # @transform関数群                   │
│ • dag.py                # Pipeline定義                       │
│ • runner.py             # 実行ロジック                       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ apps/pipeline-app/ (サンプルプロジェクト)                     │
├──────────────────────────────────────────────────────────────┤
│ • algo-trade-dtypeを参照する実装例                           │
│ • 別プロジェクトで独自dtype定義する場合のテンプレート        │
└──────────────────────────────────────────────────────────────┘

🗑️  packages/proj-dtypes/ → 削除
```

## 重要: xform-auditor CLI互換性

**最重要検証項目**: リファクタリング後も`xform-auditor`が正常動作すること

```bash
# リファクタリング前後で同じ結果が得られることを確認
uv run python -m xform_auditor apps/algo-trade-app/algo_trade_app
uv run python -m xform_auditor apps/pipeline-app/pipeline_app

# 期待される動作:
# 1. @transform関数の自動発見
# 2. ExampleValueからのテストデータ生成
# 3. Check関数による出力検証
# 4. OK / VIOLATION / ERROR / MISSING の正しいレポート
```

## 実装計画

### Phase 1: xform-coreに汎用機能を追加

#### 1.1. 型メタデータ登録基盤 (packages/xform-core/xform_core/type_metadata.py)

```python
"""Type metadata registration infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .metadata import Check, ExampleValue
from .registry import register_check, register_example

T = TypeVar("T")


@dataclass
class RegisteredType(Generic[T]):
    """Type wrapper that carries example values and check functions.

    This class provides a declarative way to register types with their
    associated examples and checks. It supports method chaining for
    fluent API style.

    Example:
        RegisteredType(HLOCVSpec) \\
            .with_example(HLOCVSpec(n=32, seed=42), "default_spec") \\
            .with_check(check_hlocv_spec_valid) \\
            .register()

        # Or with list-based initialization
        RegisteredType(
            type_=FeatureMap,
            examples=[make_example({...}, "default")],
            checks=[check_feature_map],
        ).register()
    """

    type_: type[T] | str  # Actual type or FQN for built-in types
    examples: list[ExampleValue[T]] = field(default_factory=list)
    checks: list[Callable[[T], None]] = field(default_factory=list)

    def with_example(self, value: T, description: str = "") -> RegisteredType[T]:
        """Add an example value (chainable).

        Args:
            value: The example value to register
            description: Human-readable description of this example

        Returns:
            Self for method chaining
        """
        self.examples.append(ExampleValue(value, description))
        return self

    def with_check(self, check_func: Callable[[T], None]) -> RegisteredType[T]:
        """Add a check function (chainable).

        Args:
            check_func: Validation function that raises on failure

        Returns:
            Self for method chaining
        """
        self.checks.append(check_func)
        return self

    def register(self) -> None:
        """Register all examples and checks to the global registry."""
        key = self._get_type_key()

        # Register examples
        for example in self.examples:
            register_example(key, example)  # type: ignore[arg-type]

        # Register checks
        for check_func in self.checks:
            check_fqn = f"{check_func.__module__}.{check_func.__name__}"
            register_check(key, Check(check_fqn))

    def _get_type_key(self) -> str:
        """Get the fully-qualified name for the type."""
        if isinstance(self.type_, str):
            return self.type_
        return f"{self.type_.__module__}.{self.type_.__qualname__}"

    def __repr__(self) -> str:
        key = self._get_type_key()
        return (
            f"RegisteredType({key}, "
            f"examples={len(self.examples)}, checks={len(self.checks)})"
        )


def make_example(value: T, description: str = "") -> ExampleValue[T]:
    """Helper to create ExampleValue instances.

    Args:
        value: The example value
        description: Human-readable description

    Returns:
        ExampleValue instance
    """
    return ExampleValue(value=value, description=description)


__all__ = ["RegisteredType", "make_example"]
```

#### 1.2. 汎用DataFrame検証関数 (packages/xform-core/xform_core/checks/dataframe.py)

```python
"""Generic DataFrame validation functions."""

from __future__ import annotations

import pandas as pd


def check_dataframe_not_empty(df: pd.DataFrame) -> None:
    """Ensure DataFrame contains at least one row.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If DataFrame is empty
    """
    if df.empty:
        raise ValueError("DataFrame must not be empty")


def check_dataframe_has_columns(
    df: pd.DataFrame, columns: tuple[str, ...] | list[str]
) -> None:
    """Ensure DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        columns: Required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_dataframe_notnull(df: pd.DataFrame) -> None:
    """Ensure DataFrame contains no null values.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If DataFrame contains any null values
    """
    if df.isna().values.any():
        raise ValueError("DataFrame must not contain null values")


def check_column_monotonic(
    series: pd.Series, *, increasing: bool = True
) -> None:
    """Ensure column is monotonically increasing or decreasing.

    Args:
        series: Series to validate
        increasing: If True, check monotonic increasing; otherwise decreasing

    Raises:
        ValueError: If column is not monotonic as specified
    """
    if increasing and not series.is_monotonic_increasing:
        raise ValueError(f"Column {series.name} must be monotonic increasing")
    if not increasing and not series.is_monotonic_decreasing:
        raise ValueError(f"Column {series.name} must be monotonic decreasing")


def check_column_dtype(
    df: pd.DataFrame, column: str, expected_dtype: str
) -> None:
    """Ensure column has the expected dtype.

    Args:
        df: DataFrame containing the column
        column: Column name to check
        expected_dtype: Expected pandas dtype string

    Raises:
        TypeError: If column dtype does not match expected
    """
    actual_dtype = df[column].dtype
    if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
        raise TypeError(
            f"Column {column} has dtype {actual_dtype}, expected {expected_dtype}"
        )


def check_column_positive(df: pd.DataFrame, column: str) -> None:
    """Ensure all values in column are strictly positive.

    Args:
        df: DataFrame containing the column
        column: Column name to check

    Raises:
        ValueError: If any values are <= 0
    """
    if (df[column] <= 0).any():
        raise ValueError(f"Column {column} must contain only positive values")


def check_column_nonnegative(df: pd.DataFrame, column: str) -> None:
    """Ensure all values in column are non-negative (>= 0).

    Args:
        df: DataFrame containing the column
        column: Column name to check

    Raises:
        ValueError: If any values are < 0
    """
    if (df[column] < 0).any():
        raise ValueError(f"Column {column} must contain only non-negative values")


__all__ = [
    "check_dataframe_not_empty",
    "check_dataframe_has_columns",
    "check_dataframe_notnull",
    "check_column_monotonic",
    "check_column_dtype",
    "check_column_positive",
    "check_column_nonnegative",
]
```

#### 1.3. `__init__.py` を更新

```python
# packages/xform-core/xform_core/checks/__init__.py (🆕)
"""Generic validation functions for common data types."""

from .dataframe import (
    check_column_dtype,
    check_column_monotonic,
    check_column_nonnegative,
    check_column_positive,
    check_dataframe_has_columns,
    check_dataframe_notnull,
    check_dataframe_not_empty,
)

__all__ = [
    "check_dataframe_not_empty",
    "check_dataframe_has_columns",
    "check_dataframe_notnull",
    "check_column_monotonic",
    "check_column_dtype",
    "check_column_positive",
    "check_column_nonnegative",
]
```

```python
# packages/xform-core/xform_core/__init__.py に追加
from .type_metadata import RegisteredType, make_example
from .checks.dataframe import (
    check_column_dtype,
    check_column_monotonic,
    check_column_nonnegative,
    check_column_positive,
    check_dataframe_has_columns,
    check_dataframe_notnull,
    check_dataframe_not_empty,
)

__all__ = [
    # ... 既存のエクスポート ...
    "RegisteredType",
    "make_example",
    "check_dataframe_not_empty",
    "check_dataframe_has_columns",
    "check_dataframe_notnull",
    "check_column_monotonic",
    "check_column_dtype",
    "check_column_positive",
    "check_column_nonnegative",
]
```

#### 1.4. Materialization基盤 (packages/xform-core/xform_core/materialization.py)

```python
"""Materialization protocol for converting specifications to concrete values."""

from __future__ import annotations

from typing import Protocol, TypeVar

import pandas as pd

T = TypeVar("T")


class Materializer(Protocol[T]):
    """Protocol for converting specifications to concrete values.

    Materializers are responsible for transforming declarative
    specifications (e.g., HLOCVSpec) into runtime-ready data
    (e.g., pandas DataFrame).
    """

    def materialize(self, spec: T) -> object:
        """Convert a specification object to its concrete representation.

        Args:
            spec: Specification object describing the desired data

        Returns:
            Concrete representation ready for use in transformations
        """
        ...


def default_materializer(value: object) -> object:
    """Default materializer that returns the value as-is or makes defensive copy.

    For pandas DataFrames, returns a deep copy to prevent accidental mutations.
    For all other types, returns the value unchanged.

    Args:
        value: Value to materialize

    Returns:
        Materialized value (copy for DataFrames, original otherwise)
    """
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return value


__all__ = ["Materializer", "default_materializer"]
```

### Phase 2: apps/algo-trade-app に機能を統合

#### 2.1. ディレクトリ構造の変更

```bash
# 現在の構造
apps/algo-trade-app/
└── algo_trade_app/          # transformers + types混在

# 変更後の構造
apps/algo-trade-app/
├── algo_trade_dtype/        # 🆕 型・検証・生成器パッケージ
│   ├── __init__.py
│   ├── types.py
│   ├── generators.py
│   ├── checks.py
│   ├── materializers.py
│   └── registry.py
└── algo_trade_app/          # transformers
    ├── __init__.py
    ├── transforms.py
    ├── dag.py
    └── runner.py
```

**注意**: `algo_trade_dtype`は別パッケージとして分離し、`algo_trade_app`から`import algo_trade_dtype`で利用する。

#### 2.2. 型定義の統合 (apps/algo-trade-app/algo_trade_dtype/types.py)

```python
"""Data type definitions for algorithmic trading pipeline.

This module consolidates all type definitions used in the algo-trade-app,
including both HLOCV (price bar) types and FX trading specific types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias, TypedDict

import pandas as pd

# ==================== HLOCV Price Bar Types ====================

HLOCV_COLUMN_ORDER: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMN: str = "volume"

PriceBarsFrame: TypeAlias = pd.DataFrame


class FeatureMap(TypedDict, total=False):
    """Represents a mapping of feature names to numeric scores."""
    mean_return: float
    volatility: float
    sharpe_ratio: float
    drawdown: float


class MarketRegime(StrEnum):
    """Enumerates supported market regimes in the project domain."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


# ==================== FX Trading Specific Types ====================

class DataProvider(StrEnum):
    """Enumeration of supported data providers."""
    GMO_CLICK = "gmo_click"
    OANDA = "oanda"
    DUKASCOPY = "dukascopy"


class CurrencyPair(StrEnum):
    """Enumeration of supported currency pairs."""
    USD_JPY = "USD_JPY"
    EUR_JPY = "EUR_JPY"
    EUR_USD = "EUR_USD"
    GBP_JPY = "GBP_JPY"
    AUD_JPY = "AUD_JPY"


class Frequency(StrEnum):
    """Enumeration of supported time frequencies."""
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAY_1 = "1D"
    WEEK_1 = "1W"


class ConvertType(StrEnum):
    """Enumeration of target conversion types."""
    RETURN = "return"
    DIRECTION = "direction"
    LOG_RETURN = "log_return"


class CVMethod(StrEnum):
    """Enumeration of cross-validation methods."""
    TIME_SERIES = "time_series"
    EXPANDING_WINDOW = "expanding_window"
    SLIDING_WINDOW = "sliding_window"


class OHLCVDataRequest(TypedDict, total=False):
    """Request schema for OHLCV data acquisition."""
    provider: DataProvider
    currency_pairs: list[CurrencyPair]
    start_date: str
    end_date: str
    frequency: Frequency
    retry_count: int
    timeout_seconds: float


class FXDataSchema(TypedDict):
    """Schema for FX OHLCV data validation."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureGenerationRequest(TypedDict, total=False):
    """Request schema for feature generation."""
    currency_pair: CurrencyPair
    source_frequency: Frequency
    target_frequency: Frequency
    lookback_periods: int
    forward_periods: int
    convert_type: ConvertType


class TrainPredictRequest(TypedDict, total=False):
    """Request schema for training and prediction."""
    feature_data_id: str
    cv_method: CVMethod
    n_splits: int
    test_size: float
    lgbm_params: dict[str, object]
    output_path: str


class PredictionResult(TypedDict):
    """Schema for prediction result."""
    timestamp: list[str]
    predicted: list[float]
    actual: list[float]
    feature_importance: dict[str, float]


class ValidationResult(TypedDict):
    """Schema for data validation result."""
    is_valid: bool
    missing_count: int
    outlier_count: int
    correlation: float
    message: str


__all__ = [
    # HLOCV types
    "HLOCV_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "FeatureMap",
    "PriceBarsFrame",
    "MarketRegime",
    # FX trading types
    "DataProvider",
    "CurrencyPair",
    "Frequency",
    "ConvertType",
    "CVMethod",
    "OHLCVDataRequest",
    "FXDataSchema",
    "FeatureGenerationRequest",
    "TrainPredictRequest",
    "PredictionResult",
    "ValidationResult",
]
```

#### 2.3. データ生成器の統合 (apps/algo-trade-app/algo_trade_dtype/generators.py)

```python
"""Data generators for algo-trade-app.

This module consolidates all data generation utilities, including:
- HLOCVSpec: Declarative specification for synthetic HLOCV price bars
- gen_hlocv: Generator function for creating realistic price data
- gen_sample_ohlcv: Simplified generator for testing purposes
"""

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

    This class uses a geometric Brownian motion model to generate
    realistic-looking OHLCV price data for testing and development.

    Attributes:
        n: Number of rows (time steps) to generate
        start_price: Opening price for the first bar
        mu: Drift component (daily return mean)
        sigma: Volatility component (daily return std dev)
        freq: Pandas offset alias (e.g., "1D", "1H")
        start: Timestamp for the first bar
        tz: Optional timezone name
        seed: RNG seed for deterministic output
        base_volume: Baseline volume when returns are flat
        volume_scale: Multiplier applied to abs(returns)
        volume_jitter: Std dev of Gaussian noise on volume
        spread_range: Min/max fractional spread for high/low prices
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
    """Generate a pandas DataFrame following the provided specification.

    Uses geometric Brownian motion for close prices, adds realistic
    high/low spreads, and generates correlated volume data.

    Args:
        spec: HLOCVSpec describing the desired price data

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    Raises:
        AssertionError: If generated data violates HLOCV invariants
    """

    rng = np.random.default_rng(spec.seed)
    index = pd.date_range(start=spec.start, periods=spec.n, freq=spec.freq)
    index = cast(pd.DatetimeIndex, index)
    if spec.tz:
        index = index.tz_localize(spec.tz)

    # Generate close prices via geometric Brownian motion
    returns = rng.normal(loc=spec.mu, scale=spec.sigma, size=spec.n)
    open_prices = np.empty(spec.n, dtype=float)
    close_prices = np.empty(spec.n, dtype=float)

    open_prices[0] = max(spec.start_price, _MIN_PRICE)
    close_prices[0] = max(open_prices[0] * exp(returns[0]), _MIN_PRICE)

    for i in range(1, spec.n):
        open_prices[i] = close_prices[i - 1]
        close_prices[i] = max(open_prices[i] * exp(returns[i]), _MIN_PRICE)

    # Generate high/low with random spreads
    base_max = np.maximum(open_prices, close_prices)
    base_min = np.minimum(open_prices, close_prices)

    spread = rng.uniform(
        low=spec.spread_range[0], high=spec.spread_range[1], size=spec.n
    )
    high_prices = base_max * (1.0 + spread)
    low_prices = base_min * (1.0 - spread)
    low_prices = np.maximum(low_prices, _MIN_PRICE)

    # Generate correlated volume
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


def gen_sample_ohlcv(
    n: int = 100,
    *,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate sample OHLCV data for testing (simplified API).

    This is a convenience wrapper around gen_hlocv with sensible defaults
    for quick test data generation.

    Args:
        n: Number of bars to generate
        start_price: Initial price
        volatility: Daily volatility (sigma)
        seed: RNG seed for reproducibility

    Returns:
        DataFrame with HLOCV columns and DatetimeIndex
    """
    spec = HLOCVSpec(
        n=n,
        start_price=start_price,
        sigma=volatility,
        seed=seed,
        freq="1h",
    )
    df = gen_hlocv(spec)
    df.set_index("timestamp", inplace=True)
    return df


# ==================== Validation Helpers ====================

def _validate_generated_frame(frame: pd.DataFrame) -> None:
    """Internal assertion helper to guarantee generator invariants."""

    if frame.empty:  # pragma: no cover
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
    except ValueError as exc:  # pragma: no cover
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


__all__ = ["HLOCVSpec", "gen_hlocv", "gen_sample_ohlcv"]
```

#### 2.4. 検証関数の統合 (apps/algo-trade-app/algo_trade_dtype/checks.py)

```python
"""Validation helpers for algo-trade-app data structures.

This module consolidates all check functions, combining:
- HLOCV-specific validation (from proj_dtypes)
- FX trading validation (existing algo_trade_app)
- Leveraging generic checks from xform_core
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from xform_core.checks import (
    check_dataframe_has_columns,
    check_dataframe_not_empty,
    check_dataframe_notnull,
)

from .types import HLOCV_COLUMN_ORDER, PRICE_COLUMNS, VOLUME_COLUMN, MarketRegime

# ==================== HLOCV Validation ====================


def check_hlocv_dataframe(frame: pd.DataFrame) -> None:
    """Raise if the provided DataFrame violates HLOCV invariants.

    Performs comprehensive validation including structure, relationships,
    and domain-specific constraints.

    Args:
        frame: DataFrame to validate

    Raises:
        ValueError: If any HLOCV invariants are violated
    """
    check_hlocv_dataframe_length(frame)
    check_hlocv_dataframe_notnull(frame)
    _validate_price_columns(frame)
    _validate_price_relationships(frame)
    _validate_volume(frame)


def check_hlocv_dataframe_length(frame: pd.DataFrame) -> None:
    """Ensure dataframe contains required columns and at least one row."""
    check_dataframe_has_columns(frame, HLOCV_COLUMN_ORDER)
    check_dataframe_not_empty(frame)


def check_hlocv_dataframe_notnull(frame: pd.DataFrame) -> None:
    """Ensure dataframe timestamps are monotonic and values are not null."""
    check_dataframe_has_columns(frame, HLOCV_COLUMN_ORDER)
    timestamp = cast(pd.Series, frame["timestamp"])
    _validate_timestamp_column(timestamp)
    check_dataframe_notnull(frame)


def _validate_timestamp_column(timestamp: pd.Series) -> None:
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        raise TypeError("timestamp column must be datetime-like")
    if not timestamp.is_monotonic_increasing:
        raise ValueError("timestamps must be monotonic increasing")
    if timestamp.hasnans:
        raise ValueError("timestamps must not contain NaT")


def _validate_price_columns(frame: pd.DataFrame) -> None:
    data = frame.loc[:, list(PRICE_COLUMNS)]
    if (data <= 0).any().any():
        raise ValueError("price columns must be strictly positive")


def _validate_price_relationships(frame: pd.DataFrame) -> None:
    """Validate HLOCV-specific price relationships."""
    opens = frame["open"].to_numpy()
    closes = frame["close"].to_numpy()
    if not np.allclose(opens[1:], closes[:-1]):
        raise ValueError("open price must match previous close")

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    if np.any(highs < np.maximum(opens, closes)):
        raise ValueError("high price must be >= max(open, close)")
    if np.any(lows > np.minimum(opens, closes)):
        raise ValueError("low price must be <= min(open, close)")


def _validate_volume(frame: pd.DataFrame) -> None:
    volumes = frame[VOLUME_COLUMN].to_numpy()
    if np.any(~np.isfinite(volumes)):
        raise ValueError("volume must contain finite values")
    if np.any(volumes <= 0):
        raise ValueError("volume must be positive")


def check_feature_map(features: Mapping[str, float]) -> None:
    """Validate that the feature map contains finite numeric values."""
    if not features:
        raise ValueError("feature map must contain at least one entry")

    for key, value in features.items():
        if not isinstance(key, str) or not key:
            raise TypeError("feature names must be non-empty strings")
        if isinstance(value, bool):
            raise TypeError("feature values must be numeric, not boolean")
        if not isinstance(value, (int, float)):
            raise TypeError("feature values must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("feature values must be finite numbers")


def check_market_regime_known(regime: MarketRegime | str) -> None:
    """Ensure provided market regime value is part of known enumeration."""
    try:
        MarketRegime(regime)
    except ValueError as exc:
        raise ValueError(f"unknown market regime: {regime!r}") from exc


# ==================== FX Trading Validation ====================


def check_ohlcv(df: pd.DataFrame) -> None:
    """Validate OHLCV DataFrame structure and constraints (FX version).

    This is a simpler version of check_hlocv_dataframe for FX data
    that may not have the same strict requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not df.empty:
        for col in ["open", "high", "low", "close"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be numeric")

        # High >= Low constraint
        if (df["high"] < df["low"]).any():
            raise ValueError("High must be >= Low")

        # High >= Open, Close constraint
        if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
            raise ValueError("High must be >= Open and Close")

        # Low <= Open, Close constraint
        if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
            raise ValueError("Low must be <= Open and Close")


def check_target(df: pd.DataFrame) -> None:
    """Validate target DataFrame structure."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    if "target" not in df.columns:
        raise ValueError("DataFrame must contain 'target' column")

    if not df.empty and not pd.api.types.is_numeric_dtype(df["target"]):
        raise TypeError("Target column must be numeric")


_EXPECTED_TUPLE_SIZE = 2


def check_aligned_data(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Validate aligned features and target DataFrames."""
    if not isinstance(data, tuple) or len(data) != _EXPECTED_TUPLE_SIZE:
        raise TypeError("Expected tuple of 2 DataFrames")

    features, target = data

    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"Features must be pd.DataFrame, got {type(features)}")

    if not isinstance(target, pd.DataFrame):
        raise TypeError(f"Target must be pd.DataFrame, got {type(target)}")

    if features.empty or target.empty:
        return

    if len(features) != len(target):
        msg = (
            f"Features and target must have same length: "
            f"{len(features)} != {len(target)}"
        )
        raise ValueError(msg)

    if not features.index.equals(target.index):
        raise ValueError("Features and target must have aligned indices")


def check_prediction_result(result: dict[str, object]) -> None:
    """Validate prediction result structure."""
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result)}")

    required_keys = {"timestamp", "predicted", "actual", "feature_importance"}
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    timestamp = result["timestamp"]
    predicted = result["predicted"]
    actual = result["actual"]

    if not isinstance(timestamp, list):
        raise TypeError("timestamp must be a list")

    if not isinstance(predicted, list):
        raise TypeError("predicted must be a list")

    if not isinstance(actual, list):
        raise TypeError("actual must be a list")

    if len(timestamp) != len(predicted) or len(timestamp) != len(actual):
        raise ValueError("timestamp, predicted, and actual must have same length")


__all__ = [
    # HLOCV checks
    "check_hlocv_dataframe",
    "check_hlocv_dataframe_length",
    "check_hlocv_dataframe_notnull",
    "check_feature_map",
    "check_market_regime_known",
    # FX trading checks
    "check_ohlcv",
    "check_target",
    "check_aligned_data",
    "check_prediction_result",
]
```

#### 2.5. Materializer (apps/algo-trade-app/algo_trade_dtype/materializers.py)

```python
"""Materializers for algo-trade-app example values.

Materializers convert declarative specifications (like HLOCVSpec)
into concrete runtime objects (like pandas DataFrame).
"""

from __future__ import annotations

import pandas as pd
from xform_core.materialization import default_materializer

from .generators import HLOCVSpec, gen_hlocv


def materialize_algo_trade_value(value: object) -> object:
    """Materialize algo-trade specific example values.

    Handles:
    - HLOCVSpec -> gen_hlocv(spec)
    - pandas.DataFrame -> defensive copy
    - Other types -> pass through

    Args:
        value: Value to materialize

    Returns:
        Materialized concrete value
    """
    if isinstance(value, HLOCVSpec):
        return gen_hlocv(value)
    return default_materializer(value)


__all__ = ["materialize_algo_trade_value"]
```

#### 2.6. レジストリ (apps/algo-trade-app/algo_trade_dtype/registry.py)

```python
"""Type registrations for algo-trade-app.

This module uses the RegisteredType declarative API from xform-core
to register all type metadata (examples and checks) used in this application.

Usage:
    from algo_trade_dtype.registry import register_all_types

    # Initialize registry (call once at application startup)
    register_all_types()
"""

from xform_core import RegisteredType, make_example

from .checks import (
    check_aligned_data,
    check_feature_map,
    check_hlocv_dataframe_length,
    check_hlocv_dataframe_notnull,
    check_market_regime_known,
    check_ohlcv,
    check_prediction_result,
    check_target,
)
from .generators import HLOCVSpec, gen_hlocv, gen_sample_ohlcv
from .types import FeatureMap, MarketRegime, PredictionResult

# ==================== Type Registrations ====================

# HLOCVSpec: Declarative specification for price data
HLOCVSpecReg = RegisteredType(HLOCVSpec) \
    .with_example(HLOCVSpec(n=32, seed=42), "default_spec") \
    .with_example(HLOCVSpec(n=64, seed=99), "large_spec") \
    .with_example(HLOCVSpec(n=128, sigma=0.02, seed=123), "high_volatility")

# FeatureMap: Dictionary of computed features
FeatureMapReg = RegisteredType(FeatureMap) \
    .with_example(
        {
            "mean_return": 0.05,
            "volatility": 0.12,
            "sharpe_ratio": 0.4,
            "drawdown": 0.1,
        },
        "synthetic_feature_map"
    ) \
    .with_check(check_feature_map)

# MarketRegime: Enumeration of market conditions
MarketRegimeReg = RegisteredType(MarketRegime) \
    .with_example(MarketRegime.BULL, "bull_market") \
    .with_example(MarketRegime.BEAR, "bear_market") \
    .with_example(MarketRegime.SIDEWAYS, "sideways_market") \
    .with_check(check_market_regime_known)

# PredictionResult: ML model output
PredictionResultReg = RegisteredType(PredictionResult) \
    .with_example(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "predicted": [0.01, 0.02],
            "actual": [0.015, 0.018],
            "feature_importance": {"rsi": 0.3, "adx": 0.2},
        },
        "synthetic_prediction"
    ) \
    .with_check(check_prediction_result)

# pandas DataFrame: HLOCV price bars
DataFrameReg = RegisteredType("pandas.core.frame.DataFrame") \
    .with_example(gen_hlocv(HLOCVSpec(n=32, seed=42)), "synthetic_hlocv_frame") \
    .with_example(gen_sample_ohlcv(n=50, seed=99), "sample_ohlcv_frame") \
    .with_check(check_hlocv_dataframe_length) \
    .with_check(check_hlocv_dataframe_notnull) \
    .with_check(check_ohlcv)

# Tuple[DataFrame, DataFrame]: Aligned features and target
AlignedDataReg = RegisteredType("builtins.tuple") \
    .with_check(check_aligned_data)


# Collection of all registered types
ALL_REGISTERED_TYPES = [
    HLOCVSpecReg,
    FeatureMapReg,
    MarketRegimeReg,
    PredictionResultReg,
    DataFrameReg,
    AlignedDataReg,
]


def register_all_types() -> None:
    """Register all algo-trade types to the global xform-core registry.

    Call this function once at application startup to populate the
    example and check registries.
    """
    for registered_type in ALL_REGISTERED_TYPES:
        registered_type.register()


__all__ = [
    "HLOCVSpecReg",
    "FeatureMapReg",
    "MarketRegimeReg",
    "PredictionResultReg",
    "DataFrameReg",
    "AlignedDataReg",
    "ALL_REGISTERED_TYPES",
    "register_all_types",
]
```

### Phase 3: proj-dtypes の削除とクリーンアップ

#### 3.1. proj-dtypes パッケージ削除

```bash
# ディレクトリ削除
rm -rf packages/proj-dtypes/

# pyproject.toml から依存削除
# [tool.uv.sources] セクションから proj-dtypes を削除
# [tool.pytest.ini_options] の testpaths から proj-dtypes を削除
# [tool.coverage.run] の source から proj-dtypes を削除
```

#### 3.2. pipeline-app の更新

```python
# apps/pipeline-app/pipeline_app/transforms.py の import を更新

# Before:
from proj_dtypes.hlocv_spec import HLOCVSpec, gen_hlocv
from proj_dtypes.types import FeatureMap

# After:
from algo_trade_dtype.generators import HLOCVSpec, gen_hlocv
from algo_trade_dtype.types import FeatureMap
```

#### 3.3. テストの更新

```python
# apps/pipeline-app/tests/test_transforms.py の import を更新

# Before:
from proj_dtypes.hlocv_spec import HLOCVSpec
from proj_dtypes.types import HLOCV_COLUMN_ORDER

# After:
from algo_trade_dtype.generators import HLOCVSpec
from algo_trade_dtype.types import HLOCV_COLUMN_ORDER
```

### Phase 4: ドキュメント更新

#### 4.1. CLAUDE.md 更新

```markdown
## Architecture & Dependencies

```
xform-core (common) ──▶ apps/* (@transform functions & DAG)
                        ▲
                        └── xform-auditor (common: annotation audit CLI)
```

- **xform-core**: `@transform` decorator, meta-types, mypy plugin, RegisteredType, generic checks
- **apps/**: Application-specific types, checks, generators, and @transform functions
- **xform-auditor**: CLI for testing via annotations

**Dependency Direction**: Unidirectional (`core` → `apps`). Apps can depend on each other if needed.
```

#### 4.2. README 作成 (apps/algo-trade-app/README.md)

```markdown
# algo-trade-app

Algorithmic trading pipeline implementation using TransformFn.

## Package Structure

### algo_trade_dtype/ (型・検証・生成器)

- `types.py`: HLOCV and FX trading type definitions
- `generators.py`: Synthetic data generators (HLOCVSpec, gen_hlocv)
- `checks.py`: Validation functions for all data types
- `materializers.py`: Convert specifications to concrete values
- `registry.py`: Type registration using RegisteredType

### algo_trade_app/ (transformers)

- `transforms.py`: @transform functions for feature engineering
- `dag.py`: Pipeline definition
- `runner.py`: Pipeline execution logic

## Type Registration

This package uses the declarative `RegisteredType` API:

```python
from algo_trade_dtype.registry import register_all_types

# Initialize registry at startup
register_all_types()
```

## Usage Example

```python
from algo_trade_dtype.generators import HLOCVSpec, gen_hlocv

# Generate synthetic price data
spec = HLOCVSpec(n=100, sigma=0.02, seed=42)
df = gen_hlocv(spec)
```
```

## 実装スケジュール

| Phase | タスク | 推定時間 |
|-------|--------|----------|
| 1.1 | xform-core: type_metadata.py 実装 | 2h |
| 1.2 | xform-core: checks/dataframe.py 実装 | 1h |
| 1.3 | xform-core: materialization.py 実装 | 30min |
| 1.4 | xform-core: __init__.py 更新 | 30min |
| 2.1 | algo-trade-app: types.py 統合 | 1h |
| 2.2 | algo-trade-app: generators.py 統合 | 1h |
| 2.3 | algo-trade-app: checks.py 統合 | 1.5h |
| 2.4 | algo-trade-app: materializers.py 作成 | 30min |
| 2.5 | algo-trade-app: registry.py 作成 | 1h |
| 3.1 | proj-dtypes 削除 + 設定更新 | 30min |
| 3.2 | pipeline-app import 更新 | 30min |
| 3.3 | テスト更新・実行 | 1h |
| 4.1 | ドキュメント更新 | 1h |
| **合計** | | **約12時間** |

## 検証計画

### 最重要: xform-auditor CLI 互換性検証

```bash
# リファクタリング前の結果を記録
uv run python -m xform_auditor apps/algo-trade-app/algo_trade_app > /tmp/audit_before.txt
uv run python -m xform_auditor apps/pipeline-app/pipeline_app > /tmp/audit_pipeline_before.txt

# リファクタリング実施

# リファクタリング後の結果を記録
uv run python -m xform_auditor apps/algo-trade-app/algo_trade_app > /tmp/audit_after.txt
uv run python -m xform_auditor apps/pipeline-app/pipeline_app > /tmp/audit_pipeline_after.txt

# 差分確認 (同一であることを期待)
diff /tmp/audit_before.txt /tmp/audit_after.txt
diff /tmp/audit_pipeline_before.txt /tmp/audit_pipeline_after.txt
```

**期待される動作:**
- @transform関数の自動発見が正常に動作
- ExampleValueからのテストデータ生成が正常
- Check関数による出力検証が正常
- OK / VIOLATION / ERROR / MISSING レポートが変わらない

### 静的解析・品質チェック

```bash
# すべての品質チェックを一括実行
make check

# 内訳:
# - make duplication  # コード重複検出
# - make format       # コードフォーマット
# - make lint         # Lintチェック
# - make typecheck    # 型チェック (mypy)
# - make complexity   # 複雑度チェック
```

### テスト実行 (補助的)

```bash
# 全テスト実行 (unitテストはauditの補助)
make test
```

## リスク管理

| リスク | 影響 | 対策 |
|--------|------|------|
| import path変更による既存コード破壊 | 高 | Phase 3で徹底的にgrep検索 |
| 型推論の破壊 | 中 | make typecheckで検証 |
| テストの失敗 | 中 | Phase 3.3で全テスト実行 |
| ドキュメントの不整合 | 低 | Phase 4で網羅的に更新 |

## 成功条件

### 必須 (MUST)

1. ✅ **xform-auditor CLI が正常動作**: リファクタリング前後で audit 結果が同一
2. ✅ **静的解析パス**: `make check` が全てパス (duplication, format, lint, typecheck, complexity)
3. ✅ **packages/proj-dtypes/ 完全削除**: 依存関係も含めて完全に削除されている

### 推奨 (SHOULD)

4. ✅ `RegisteredType`による宣言的レジストリが機能している
5. ✅ 全unit testがパス (`make test`)
6. ✅ ドキュメント(CLAUDE.md, README等)が最新の構造を反映している

## 参考資料

- [xform-core registry implementation](../packages/xform-core/xform_core/registry.py)
- [Current proj-dtypes implementation](../packages/proj-dtypes/proj_dtypes/)
- [algo-trade-app current structure](../apps/algo-trade-app/algo_trade_app/)
