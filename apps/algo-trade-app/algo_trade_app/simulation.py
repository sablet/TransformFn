"""Simulation and evaluation Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated, List, Literal

from xform_core import Check, ExampleValue

import numpy as np
import pandas as pd

from xform_core import transform

from algo_trade_dtype.types import (
    MultiAssetOHLCVFrame,
    PerformanceMetrics,
    PositionSignal,
    PredictionData,
    RankedPredictionData,
    SelectedCurrencyData,
    SelectedCurrencyDataWithCosts,
    SimulationResult,
    SpreadCalculationMethod,
    SwapDataSource,
    TradingCostConfig,
)
from algo_trade_dtype.generators import (
    gen_trading_cost_config,
    gen_selected_currency_data,
    gen_selected_currency_data_with_costs,
    gen_ranked_prediction_data,
)

_MAX_THRESHOLD_PCT = 0.5


@transform
def rank_predictions(
    predictions: Annotated[
        List[PredictionData],
        ExampleValue(
            [
                {
                    "date": "2024-01-01",
                    "currency_pair": "USD_JPY",
                    "prediction": 0.01,
                    "actual_return": 0.005,
                },
                {
                    "date": "2024-01-02",
                    "currency_pair": "EUR_JPY",
                    "prediction": 0.02,
                    "actual_return": 0.015,
                },
            ]
        ),
    ],
    *,
    method: Literal["percentile", "ordinal", "zscore", "minmax"] = "percentile",
    groupby: Literal["date", "currency_pair", "none"] = "date",
) -> Annotated[
    List[RankedPredictionData],
    Check("algo_trade_dtype.checks.check_ranked_predictions"),
]:
    """Rank predictions with configurable method and grouping.

    Args:
        predictions: List of prediction data for multiple currencies/dates
        method: Ranking method - "percentile" (0-1 scale), "ordinal" (1-N),
                "zscore" (standardized), "minmax" (0-1 normalized)
        groupby: Grouping strategy - "date" (within each date),
                 "currency_pair" (within each currency), "none" (all at once)

    Returns:
        List of ranked prediction data with prediction_rank_pct
    """
    if not predictions:
        return []

    df = pd.DataFrame(predictions)
    df["date"] = pd.to_datetime(df["date"])

    # Apply ranking based on method and grouping
    if groupby == "none":
        df = _apply_ranking_method(df, "prediction", method)
    elif groupby == "date":
        df = (
            df.groupby("date", group_keys=False)[df.columns.tolist()]
            .apply(
                lambda x: _apply_ranking_method(x, "prediction", method),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
    elif groupby == "currency_pair":
        df = (
            df.groupby("currency_pair", group_keys=False)[df.columns.tolist()]
            .apply(
                lambda x: _apply_ranking_method(x, "prediction", method),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

    result: list[RankedPredictionData] = []
    for _, row in df.iterrows():
        result.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "currency_pair": row["currency_pair"],
                "prediction": float(row["prediction"]),
                "actual_return": float(row["actual_return"]),
                "prediction_rank_pct": float(row["prediction_rank"]),
            }
        )

    return result


def _apply_ranking_method(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    """Apply the specified ranking method to a DataFrame column."""
    df = df.copy()
    values = df[column]

    if method == "percentile":
        # Use pandas rank with pct=True for percentile ranking
        df["prediction_rank"] = values.rank(pct=True, method="average")
    elif method == "ordinal":
        # Use pandas rank with ordinal method for 1-N ranking
        df["prediction_rank"] = values.rank(method="min").astype(int)
    elif method == "zscore":
        # Calculate z-scores
        mean_val = values.mean()
        std_val = values.std()
        if std_val == 0:
            df["prediction_rank"] = 0.0
        else:
            df["prediction_rank"] = (values - mean_val) / std_val
    elif method == "minmax":
        # Normalize to 0-1 scale
        min_val = values.min()
        max_val = values.max()
        if min_val == max_val:
            df["prediction_rank"] = 0.0
        else:
            df["prediction_rank"] = (values - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported ranking method: {method}")

    return df


@transform
def select_top_currency(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(gen_ranked_prediction_data()),
    ],
    threshold_pct: float = 0.03,
) -> Annotated[
    List[SelectedCurrencyData],
    Check("algo_trade_dtype.checks.check_selected_currencies"),
]:
    """Select top and bottom currencies based on prediction ranking.

    Selects currencies with rank_pct >= (1 - threshold_pct) for BUY signals
    and rank_pct <= threshold_pct for SELL signals.

    Args:
        ranked_predictions: List of ranked prediction data
        threshold_pct: Threshold for top/bottom selection (default 0.03 = 3%)

    Returns:
        List of selected currencies with signals
    """
    if not ranked_predictions:
        return []

    if not 0 < threshold_pct <= _MAX_THRESHOLD_PCT:
        msg = f"threshold_pct must be in (0, {_MAX_THRESHOLD_PCT}], got {threshold_pct}"
        raise ValueError(msg)

    result: list[SelectedCurrencyData] = []

    for item in ranked_predictions:
        rank_pct = item["prediction_rank_pct"]

        if rank_pct >= (1.0 - threshold_pct):
            signal = PositionSignal.LONG
        elif rank_pct <= threshold_pct:
            signal = PositionSignal.SHORT
        else:
            continue

        result.append(
            {
                "date": item["date"],
                "currency_pair": item["currency_pair"],
                "prediction": item["prediction"],
                "actual_return": item["actual_return"],
                "prediction_rank_pct": item["prediction_rank_pct"],
                "signal": signal,
            }
        )

    return result


@transform
def calculate_trading_costs(
    selected_currencies: Annotated[
        List[SelectedCurrencyData],
        ExampleValue(gen_selected_currency_data()),
    ],
    cost_config: Annotated[TradingCostConfig, ExampleValue(gen_trading_cost_config())],
    ohlcv_frame: MultiAssetOHLCVFrame,
) -> Annotated[
    List[SelectedCurrencyDataWithCosts],
    Check("algo_trade_dtype.checks.check_selected_currencies_with_costs"),
]:
    """FXペアの取引コスト（スワップ + スプレッド）をポジション方向に応じて計算。

    Logic:
        1. FXペア判定（USD_JPY形式かチェック）
        2. スワップ計算（FXのみ、日次変動）:
           - swap_source=FRED_POLICY_RATE の場合は NotImplementedError
           - その他は swap_rate=0.0
        3. スプレッド計算（全資産共通、保有シグナル時のみ）:
           - signal is FLAT の場合は spread_cost=0.0
           - SpreadCalculationMethod.CONSTANT: spread_constant_ratio を使用
           - SpreadCalculationMethod.BID_ASK: NotImplementedError
        4. ポジション方向を考慮したリターン調整:
           adjusted_return = actual_return + signal * swap_rate
                             - abs(signal) * spread_cost

    注意: signal は PositionSignal.value を使用
    """
    if not selected_currencies:
        return []

    # Extract config parameters
    swap_source = cost_config.get("swap_source", SwapDataSource.MANUAL)
    spread_method = cost_config.get("spread_method", SpreadCalculationMethod.CONSTANT)
    spread_constant_ratio = cost_config.get("spread_constant_ratio", 0.0)

    result: list[SelectedCurrencyDataWithCosts] = []

    for item in selected_currencies:
        currency_pair = item["currency_pair"]
        signal = item["signal"]
        actual_return = item["actual_return"]

        # Convert PositionSignal to numeric value
        signal_value = signal.value if isinstance(signal, PositionSignal) else signal

        # 1. FX pair detection (USD_JPY format)
        _FX_PAIR_COMPONENTS = 2
        is_fx_pair = (
            "_" in currency_pair
            and len(currency_pair.split("_")) == _FX_PAIR_COMPONENTS
        )

        # 2. Swap calculation (FX only)
        if is_fx_pair and signal_value != 0:
            if swap_source == SwapDataSource.FRED_POLICY_RATE:
                msg = (
                    "FRED policy rate swap calculation is not yet implemented "
                    "(Phase 4.1)"
                )
                raise NotImplementedError(msg)
            else:
                swap_rate = 0.0
        else:
            swap_rate = 0.0

        # 3. Spread calculation (all assets, only for non-FLAT positions)
        spread_cost: float
        if signal_value == 0:
            spread_cost = 0.0
        elif spread_method == SpreadCalculationMethod.BID_ASK:
            msg = "BID_ASK spread calculation is not yet implemented (Phase 4.1)"
            raise NotImplementedError(msg)
        elif spread_method == SpreadCalculationMethod.CONSTANT:
            if spread_constant_ratio is None:
                spread_cost = 0.0
            else:
                spread_cost = spread_constant_ratio

        # 4. Position-aware return adjustment
        adjusted_return = (
            actual_return + signal_value * swap_rate - abs(signal_value) * spread_cost
        )

        result.append(
            {
                "date": item["date"],
                "currency_pair": currency_pair,
                "prediction": item["prediction"],
                "actual_return": actual_return,
                "prediction_rank_pct": item["prediction_rank_pct"],
                "signal": signal,
                "swap_rate": swap_rate,
                "spread_cost": spread_cost,
                "adjusted_return": adjusted_return,
            }
        )

    return result


@transform
def simulate_buy_scenario(
    selected_currencies: Annotated[
        List[SelectedCurrencyDataWithCosts],
        ExampleValue(gen_selected_currency_data_with_costs()),
    ],
    *,
    apply_costs: bool = True,
    allocation_method: Literal[
        "equal", "prediction_weighted", "rank_weighted"
    ] = "equal",
    rebalance_freq: Literal["D", "W", "M"] = "D",
) -> Annotated[
    SimulationResult,
    Check("algo_trade_dtype.checks.check_simulation_result"),
]:
    """Simulate trading scenario with configurable allocation and rebalancing.

    Args:
        selected_currencies: List of selected currency data with costs and signals
        apply_costs: If True, use adjusted_return; if False, use actual_return
        allocation_method: How to weight positions - "equal" (equal weight),
                           "prediction_weighted" (weight by absolute prediction),
                           "rank_weighted" (weight by prediction rank)
        rebalance_freq: How often to rebalance - "D" (daily), "W" (weekly),
                        "M" (monthly)

    Returns:
        SimulationResult with keys: date, portfolio_return, n_positions
    """
    if not selected_currencies:
        return {
            "date": [],
            "portfolio_return": [],
            "n_positions": [],
        }

    df = pd.DataFrame(selected_currencies)
    df["date"] = pd.to_datetime(df["date"])

    # Apply rebalancing frequency by resampling to the specified frequency
    if rebalance_freq != "D":
        df = _apply_rebalancing_frequency(df, rebalance_freq)

    # Apply allocation method to calculate position weights
    df = _apply_allocation_method(df, allocation_method)

    # Select return column based on apply_costs parameter
    return_column = "adjusted_return" if apply_costs else "actual_return"

    # Calculate position returns based on weights
    # Convert PositionSignal enum to numeric value
    df["signal_value"] = df["signal"].apply(
        lambda s: s.value if isinstance(s, PositionSignal) else s
    )
    df["position_return"] = df["weight"] * df["signal_value"] * df[return_column]

    # Group by date and aggregate portfolio returns
    daily_returns = (
        df.groupby("date")
        .agg(
            portfolio_return=(
                "position_return",
                "sum",
            ),  # Sum instead of mean for weighted returns
            n_positions=("signal", lambda x: (x != 0).sum()),
        )
        .reset_index()
    )

    daily_returns["portfolio_return"] = daily_returns["portfolio_return"].fillna(0.0)

    return {
        "date": [d.strftime("%Y-%m-%d") for d in daily_returns["date"]],
        "portfolio_return": daily_returns["portfolio_return"].tolist(),
        "n_positions": daily_returns["n_positions"].astype(int).tolist(),
    }


def _apply_rebalancing_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Apply rebalancing frequency by resampling data."""
    if freq == "W":
        # Group by week
        df["date"] = df["date"].dt.to_period("W").dt.start_time
    elif freq == "M":
        # Group by month
        df["date"] = df["date"].dt.to_period("M").dt.start_time
    # For "D" (daily), no changes needed

    return df


def _apply_allocation_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply the specified allocation method to calculate position weights."""
    df = df.copy()

    if method == "equal":
        # Equal weight allocation - each position gets 1 / n_positions weight
        df["weight"] = 1.0 / df.groupby("date")["signal"].transform("size")
    elif method == "prediction_weighted":
        # Weight by absolute value of prediction
        df["abs_prediction"] = df["prediction"].abs()
        total_abs_predictions = df.groupby("date")["abs_prediction"].transform("sum")
        df["weight"] = df["abs_prediction"] / total_abs_predictions
        df = df.drop(columns=["abs_prediction"])  # Clean up temporary column
    elif method == "rank_weighted":
        # Weight by rank of prediction
        df["rank"] = df.groupby("date")["prediction"].rank(method="average")
        total_ranks = df.groupby("date")["rank"].transform("sum")
        df["weight"] = df["rank"] / total_ranks
        df = df.drop(columns=["rank"])  # Clean up temporary column
    else:
        raise ValueError(f"Unsupported allocation method: {method}")

    # Fill any missing weights with 0
    df["weight"] = df["weight"].fillna(0.0)

    return df


def _extract_prediction_arrays(
    ranked_predictions: list[RankedPredictionData],
) -> tuple[np.ndarray, np.ndarray]:
    """ランク付き予測データから予測値と実際の値を配列として抽出する共通ヘルパー関数。

    Args:
        ranked_predictions: ランク付き予測データのリスト

    Returns:
        (predictions, actuals): 予測値と実際の値の numpy 配列のタプル
    """
    predictions = np.array([item["prediction"] for item in ranked_predictions])
    actuals = np.array([item["actual_return"] for item in ranked_predictions])
    return predictions, actuals


@transform
def calculate_performance_metrics(
    simulation_result: SimulationResult,
    annual_periods: int = 252,
) -> PerformanceMetrics:
    """Calculate portfolio performance metrics from simulation results.

    Args:
        simulation_result: SimulationResult with keys: date, portfolio_return,
            n_positions
        annual_periods: Number of periods per year for annualization (default 252)

    Returns:
        PerformanceMetrics with annual_return, annual_volatility, sharpe_ratio,
        max_drawdown, and calmar_ratio
    """
    if not simulation_result or not simulation_result.get("portfolio_return"):
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    returns = np.array(simulation_result["portfolio_return"])

    if len(returns) == 0:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    annual_return = float(np.mean(returns) * annual_periods)
    annual_volatility = float(np.std(returns, ddof=1) * np.sqrt(annual_periods))

    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }


@transform
def filter_by_prediction_quantile(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(gen_ranked_prediction_data()),
    ],
    *,
    quantile_range: tuple[float, float] = (0.0, 1.0),
) -> Annotated[
    List[RankedPredictionData],
    Check("algo_trade_dtype.checks.check_ranked_predictions"),
]:
    """Filter predictions by quantile range of prediction_rank_pct.

    Parameters:
        quantile_range: (lower, upper) quantile bounds (0.0-1.0)
            - (0.0, 0.03): 下位3%
            - (0.97, 1.0): 上位3%
            - (0.0, 1.0): 全て
    """
    if not ranked_predictions:
        return []

    lower, upper = quantile_range
    if not (0.0 <= lower <= upper <= 1.0):
        msg = (
            f"quantile_range must satisfy 0.0 <= lower <= upper <= 1.0, "
            f"got {quantile_range}"
        )
        raise ValueError(msg)

    result: list[RankedPredictionData] = []
    for item in ranked_predictions:
        rank_pct = item["prediction_rank_pct"]
        if lower <= rank_pct <= upper:
            result.append(item)

    return result


@transform
def calculate_rmse_from_ranked(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(gen_ranked_prediction_data()),
    ],
) -> Annotated[float, Check["algo_trade_dtype.checks.check_nonnegative_float"]]:
    """Calculate RMSE from ranked prediction data.

    Formula: sqrt(mean((actual_return - prediction)^2))
    """
    if not ranked_predictions:
        return 0.0

    predictions, actuals = _extract_prediction_arrays(ranked_predictions)
    mse = np.mean((actuals - predictions) ** 2)
    return float(np.sqrt(mse))


@transform
def calculate_mae_from_ranked(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(gen_ranked_prediction_data()),
    ],
) -> Annotated[float, Check["algo_trade_dtype.checks.check_nonnegative_float"]]:
    """Calculate MAE from ranked prediction data.

    Formula: mean(abs(actual_return - prediction))
    """
    if not ranked_predictions:
        return 0.0

    predictions, actuals = _extract_prediction_arrays(ranked_predictions)
    mae = np.mean(np.abs(actuals - predictions))
    return float(mae)


@transform
def calculate_mse_from_ranked(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(gen_ranked_prediction_data()),
    ],
) -> Annotated[float, Check["algo_trade_dtype.checks.check_nonnegative_float"]]:
    """Calculate MSE from ranked prediction data.

    Formula: mean((actual_return - prediction)^2)
    """
    if not ranked_predictions:
        return 0.0

    predictions, actuals = _extract_prediction_arrays(ranked_predictions)
    mse = np.mean((actuals - predictions) ** 2)
    return float(mse)


@transform
def calculate_r2_from_ranked(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(gen_ranked_prediction_data()),
    ],
) -> Annotated[float, Check["algo_trade_dtype.checks.check_finite_float"]]:
    """Calculate R² score from ranked prediction data.

    Formula: 1 - SS_res / SS_tot
    where:
        SS_res = sum((actual_return - prediction)^2)
        SS_tot = sum((actual_return - mean(actual_return))^2)
    """
    if not ranked_predictions:
        return 0.0

    predictions, actuals = _extract_prediction_arrays(ranked_predictions)

    _MIN_SAMPLES_FOR_R2 = 2
    if len(actuals) < _MIN_SAMPLES_FOR_R2:
        return 0.0

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)

    if ss_tot == 0:
        return 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    # Clip to [0, 1] to handle floating-point rounding errors
    # (e.g., -2.22e-16 should be treated as 0.0)
    return float(np.clip(r2, 0.0, 1.0))


__all__ = [
    "rank_predictions",
    "select_top_currency",
    "calculate_trading_costs",
    "filter_by_prediction_quantile",
    "calculate_rmse_from_ranked",
    "calculate_mae_from_ranked",
    "calculate_mse_from_ranked",
    "calculate_r2_from_ranked",
    "simulate_buy_scenario",
    "calculate_performance_metrics",
]
